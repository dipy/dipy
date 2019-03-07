from __future__ import division, print_function, absolute_import

import logging
from dipy.workflows.workflow import Workflow
import numpy as np
from dipy.io.image import load_nifti
from scipy.spatial import cKDTree
from scipy.ndimage.interpolation import map_coordinates
from dipy.io.streamline import load_trk
from dipy.tracking.streamline import transform_streamlines
from dipy.tracking.streamline import set_number_of_points
from dipy.tracking.streamline import Streamlines
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import AveragePointwiseEuclideanMetric
import os
import pandas as pd
from scipy import spatial
from dipy.io.peaks import load_peaks
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


def plot(x, y, title, file_name):

    """ saves the simple plot with given x and y values

    """

    plt.plot(x, y)
    plt.xlabel('disk no')

    plt.ylabel("-log10(pvalues)")

    plt.title(title)

    plt.savefig(file_name)

    plt.clf()


def save_hdf5(dt, fname):

    """ saves the given input dataframe to .h5 file

    """

    df = pd.DataFrame(dt)
    filename_hdf5 = fname+'.h5'

    store = pd.HDFStore(filename_hdf5)
    store.append(fname, df, data_columns=True,
                 min_itemsize={"bundle": 5})
    store.close()


def peak_values(bundle, peaks, dt, pname, bname, subject, group, ind, dir):

    """ peak_values function finds the peak direction and peak value of a point
        on a streamline used while tracking (generating the tractogram) and
        save it in hd5 file.

        Parameters
        ----------
        bundle : string
            Name of bundle being analyzed
        peaks : peaks
            contains peak directions and values
        dt : DataFrame
            DataFrame to be populated
        pname : string
            Name of the dti metric
        bname : string
            Name of bundle being analyzed.
        subject : string
            subject number as a string (e.g. 10001)
        group : string
            which group subject belongs to (e.g. patient or control)
        ind : integer list
            ind tells which disk number a point belong.
        dir : string
            path of output directory

    """

    dt["bundle"] = []
    dt["disk#"] = []
    dt[pname] = []
    dt["subject"] = []
    dt["group"] = []

    point = 0
    shape = peaks.peak_dirs.shape
    for st in bundle:
        di = st[1:] - st[0:-1]
        dnorm = np.linalg.norm(di, axis=1)
        di = di/dnorm[:, None]
        count = 0
        for ip in range(len(st)-1):
            point += 1
            index = st[ip].astype(int)

            if (index[0] < shape[0] and index[1] < shape[1] and
               index[2] < shape[2]):

                dire = peaks.peak_dirs[index[0]][index[1]][index[2]]
                dval = peaks.peak_values[index[0]][index[1]][index[2]]

                res = []

                for i in range(len(dire)):
                    di2 = dire[i]
                    result = spatial.distance.cosine(di[ip], di2)
                    res.append(result)

                d_val = dval[res.index(min(res))]
                if d_val != 0.:
                    dt[pname].append(d_val)
                    dt["disk#"].append(ind[point]+1)
                    count += 1

        dt["bundle"].extend([bname]*count)
        dt["subject"].extend([subject]*count)
        dt["group"].extend([group]*count)

    save_hdf5(dt, os.path.join(dir, pname))


def dti_measures(bundle, metric, dt, pname, bname, subject, group, ind, dir):

    """ calculates dti measure (eg: FA, MD) per point on streamlines and
        save it in hd5 file.

        Parameters
        ----------
        bundle : string
            Name of bundle being analyzed
        metric : matrix of float values
            dti metric e.g. FA, MD
        dt : DataFrame
            DataFrame to be populated
        pname : string
            Name of the dti metric
        bname : string
            Name of bundle being analyzed.
        subject : string
            subject number as a string (e.g. 10001)
        group : string
            which group subject belongs to (e.g. patient or control)
        ind : integer list
            ind tells which disk number a point belong.
        dir : string
            path of output directory
    """

    dt["bundle"] = []
    dt["disk#"] = []
    dt["subject"] = []
    dt[pname] = []
    dt["group"] = []

    values = map_coordinates(metric, bundle._data.T,
                             order=1)

    dt["disk#"].extend(ind[list(range(len(values)))]+1)
    dt["bundle"].extend([bname]*len(values))
    dt["subject"].extend([subject]*len(values))
    dt["group"].extend([group]*len(values))
    dt[pname].extend(values)

    save_hdf5(dt, os.path.join(dir, pname))


def BundleAnalysis(model_bundle_files, bundle_files, orig_bundle_files,
                   dti_metric_files, group, no_disks=100, out_dir=''):

        """
        Applies statistical analysis on bundles and saves the results
        in a directory specified by ``out_dir``.

        Parameters
        ----------
        model_bundle_files : string
            Path to the input model bundle files. This path may contain
            wildcards to process multiple inputs at once.
        bundle_files : string
            Path to the input bundle files in common space. This path may
            contain wildcards to process multiple inputs at once.
        orig_files : string
            Path to the input bundle files in native space. This path may
            contain wildcards to process multiple inputs at once.
        dti_metric_files : string
            Path to the input dti metric or/and peak files. It will be used as
            metric for statistical analysis of bundles.
        group : string
            what group subject belongs to e.g. control or patient
        no_disks : integer, optional
            Number of disks used for dividing bundle into disks. (Default 100)
        out_dir : string, optional
            Output directory (default input file directory)

        References
        ----------
        Chandio, B.Q., S. Koudoro, D. Reagan, J. Harezlak, E. Garyfallidis,
        Bundle Analytics: a computational and statistical analyses framework
        for tractometric studies, Proceedings of: International Society of
        Magnetic Resonance in Medicine (ISMRM), Montreal, Canada, 2019.

        """

        dt = dict()

        mb = os.listdir(model_bundle_files)
        mb.sort()
        bd = os.listdir(bundle_files)
        bd.sort()
        org_bd = os.listdir(orig_bundle_files)
        org_bd.sort()
        n = len(org_bd)

        for io in range(n):
            mbundles, _ = load_trk(os.path.join(model_bundle_files, mb[io]))
            bundles, _ = load_trk(os.path.join(bundle_files, bd[io]))
            orig_bundles, _ = load_trk(os.path.join(orig_bundle_files,
                                       org_bd[io]))

            mbundle_streamlines = set_number_of_points(mbundles,
                                                       nb_points=no_disks)

            metric = AveragePointwiseEuclideanMetric()
            qb = QuickBundles(threshold=25., metric=metric)
            clusters = qb.cluster(mbundle_streamlines)
            centroids = Streamlines(clusters.centroids)

            print('Number of centroids ', len(centroids.data))
            print('Model bundle ', mb[io])
            print('number of streamlines in bundle in common space ',
                  len(bundles))
            print('Number of streamlines in bundle in original space ',
                  len(orig_bundles))

            _, indx = cKDTree(centroids.data, 1,
                              copy_data=True).query(bundles.data, k=1)

            dti_metric_files_names = os.listdir(dti_metric_files)
            _, affine = load_nifti(os.path.join(dti_metric_files, "fa.nii.gz"))

            affine_r = np.linalg.inv(affine)
            transformed_orig_bundles = transform_streamlines(orig_bundles,
                                                             affine_r)

            subject = org_bd[io][:4]
            for mn in range(0, len(dti_metric_files_names)):

                ind = np.array(indx)
                fm = dti_metric_files_names[mn][:2]
                bm = mb[io][:-4]
                dt = dict()
                metric_name = os.path.join(dti_metric_files,
                                           dti_metric_files_names[mn])

                if dti_metric_files_names[mn][2:] == '.nii.gz':
                    metric, _ = load_nifti(metric_name)

                    dti_measures(transformed_orig_bundles, metric, dt, fm,
                                 bm, subject, group, ind, out_dir)

                else:
                    fm = dti_metric_files_names[mn][:3]
                    metric = load_peaks(metric_name)
                    peak_values(bundles, metric, dt, fm, bm, subject, group,
                                ind, out_dir)


class BundleAnalysisPopulationFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return 'ba'

    def run(self, model_bundle_files, subject_files, no_disks=100, out_dir=''):
        """Workflow of bundle analytics.

        Applies statistical analysis on bundles of subjects and saves the
        results in a directory specified by ``out_dir``.

        Parameters
        ----------

        model_bundle_files : string
            Path to the input model bundle files. This path may
            contain wildcards to process multiple inputs at once.

        subject_files : string
            Path to the input subject folder. This path may contain
            wildcards to process multiple inputs at once.

        no_disks : integer, optional
            Number of disks used for dividing bundle into disks. (Default 100)

        out_dir : string, optional
            Output directory (default input file directory)

        """

        groups = os.listdir(subject_files)

        for group in groups:
            logging.info('group = {0}'.format(group))
            all_subjects = os.listdir(os.path.join(subject_files, group))

            for sub in all_subjects:

                pre = os.path.join(subject_files, group, sub)

                b = os.path.join(pre, "rec_bundles")
                c = os.path.join(pre, "org_bundles")
                d = os.path.join(pre, "dti_measures")
                BundleAnalysis(model_bundle_files, b, c, d, group,
                               no_disks, out_dir)


class LinearMixedModelsFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return 'lmm'

    def run(self, metric_files, no_disks=100, out_dir=''):
        """Workflow of linear Mixed Models.

        Applies linear Mixed Models on bundles of subjects and saves the
        results in a directory specified by ``out_dir``.

        Parameters
        ----------

        metric_files : string
            Path to the input metric files. This path may
            contain wildcards to process multiple inputs at once.

        no_disks : integer, optional
            Number of disks used for dividing bundle into disks. (Default 100)

        out_dir : string, optional
            Output directory (default input file directory)

        """

        all_files = os.listdir(metric_files)

        for file in all_files:

            logging.info('Applying metric {0}'.format(file))
            df = pd.read_hdf(os.path.join(metric_files, file))
            all_bundles = df.bundle.unique()
            # all_pvalues = []
            for bundle in all_bundles:
                sub_af = df[df['bundle'] == bundle]  # sub sample
                pvalues = np.zeros(no_disks)

                # run mixed linear model for every disk
                for i in range(no_disks):

                    sub = sub_af[sub_af['disk#'] == (i+1)]  # disk number

                    if len(sub) > 0:
                        criteria = file[:-3] + " ~ group"
                        md = smf.mixedlm(criteria, sub, groups=sub["subject"])

                        mdf = md.fit()

                        pvalues[i] = mdf.pvalues[1]

                x = list(range(1, len(pvalues)+1))
                y = -1*np.log10(pvalues)

                title = bundle+" on "+file[:-3]+" Values"
                file_name = os.path.join(out_dir, bundle+" "+file[:-3]+".png")
                plot(x, y, title, file_name)
