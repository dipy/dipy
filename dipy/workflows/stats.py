import logging
import numpy as np
import os
import json
import warnings
from time import time
from scipy.ndimage import binary_dilation
from dipy.utils.optpkg import optional_package
from dipy.io import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel
from dipy.io.peaks import load_peaks
from dipy.io.streamline import load_tractogram
from dipy.segment.mask import segment_from_cfa
from dipy.segment.mask import bounding_box
from dipy.tracking.streamline import transform_streamlines
from glob import glob
from dipy.workflows.workflow import Workflow
from dipy.segment.bundles import bundle_shape_similarity
from dipy.stats.analysis import assignment_map
from dipy.stats.analysis import anatomical_measures
from dipy.stats.analysis import peak_values

pd, have_pd, _ = optional_package("pandas")
smf, have_smf, _ = optional_package("statsmodels.formula.api")
matplt, have_matplotlib, _ = optional_package("matplotlib")
plt, have_plt, _ = optional_package("matplotlib.pyplot")


class SNRinCCFlow(Workflow):

    @classmethod
    def get_short_name(cls):
        return 'snrincc'

    def run(self, data_files, bvals_files, bvecs_files, mask_file,
            bbox_threshold=(0.6, 1, 0, 0.1, 0, 0.1), out_dir='',
            out_file='product.json', out_mask_cc='cc.nii.gz',
            out_mask_noise='mask_noise.nii.gz'):
        """Compute the signal-to-noise ratio in the corpus callosum.

        Parameters
        ----------
        data_files : string
            Path to the dwi.nii.gz file. This path may contain wildcards to
            process multiple inputs at once.
        bvals_files : string
            Path of bvals.
        bvecs_files : string
            Path of bvecs.
        mask_file : string
            Path of a brain mask file.
        bbox_threshold : variable float, optional
            Threshold for bounding box, values separated with commas for ex.
            [0.6,1,0,0.1,0,0.1].
        out_dir : string, optional
            Where the resulting file will be saved. (default current directory)
        out_file : string, optional
            Name of the result file to be saved.
        out_mask_cc : string, optional
            Name of the CC mask volume to be saved.
        out_mask_noise : string, optional
            Name of the mask noise volume to be saved.

        """
        io_it = self.get_io_iterator()

        for dwi_path, bvals_path, bvecs_path, mask_path, out_path, \
                cc_mask_path, mask_noise_path in io_it:
            data, affine = load_nifti(dwi_path)
            bvals, bvecs = read_bvals_bvecs(bvals_path, bvecs_path)
            gtab = gradient_table(bvals=bvals, bvecs=bvecs)

            mask, affine = load_nifti(mask_path)

            logging.info('Computing tensors...')
            tenmodel = TensorModel(gtab)
            tensorfit = tenmodel.fit(data, mask=mask)

            logging.info(
                'Computing worst-case/best-case SNR using the CC...')

            if np.ndim(data) == 4:
                CC_box = np.zeros_like(data[..., 0])
            elif np.ndim(data) == 3:
                CC_box = np.zeros_like(data)
            else:
                raise OSError('DWI data has invalid dimensions')

            mins, maxs = bounding_box(mask)
            mins = np.array(mins)
            maxs = np.array(maxs)
            diff = (maxs - mins) // 4
            bounds_min = mins + diff
            bounds_max = maxs - diff

            CC_box[bounds_min[0]:bounds_max[0],
                   bounds_min[1]:bounds_max[1],
                   bounds_min[2]:bounds_max[2]] = 1

            if len(bbox_threshold) != 6:
                raise OSError('bbox_threshold should have 6 float values')

            mask_cc_part, cfa = segment_from_cfa(tensorfit, CC_box,
                                                 bbox_threshold,
                                                 return_cfa=True)

            if not np.count_nonzero(mask_cc_part.astype(np.uint8)):
                logging.warning("Empty mask: corpus callosum not found."
                                " Update your data or your threshold")

            save_nifti(cc_mask_path, mask_cc_part.astype(np.uint8), affine)
            logging.info('CC mask saved as {0}'.format(cc_mask_path))

            masked_data = data[mask_cc_part]
            mean_signal = 0
            if masked_data.size:
                mean_signal = np.mean(masked_data, axis=0)
            mask_noise = binary_dilation(mask, iterations=10)
            mask_noise[..., :mask_noise.shape[-1]//2] = 1
            mask_noise = ~mask_noise

            save_nifti(mask_noise_path, mask_noise.astype(np.uint8), affine)
            logging.info('Mask noise saved as {0}'.format(mask_noise_path))

            noise_std = 0
            if np.count_nonzero(mask_noise.astype(np.uint8)):
                noise_std = np.std(data[mask_noise, :])

            logging.info('Noise standard deviation sigma= ' + str(noise_std))

            idx = np.sum(gtab.bvecs, axis=-1) == 0
            gtab.bvecs[idx] = np.inf
            axis_X = np.argmin(
                np.sum((gtab.bvecs-np.array([1, 0, 0])) ** 2, axis=-1))
            axis_Y = np.argmin(
                np.sum((gtab.bvecs-np.array([0, 1, 0])) ** 2, axis=-1))
            axis_Z = np.argmin(
                np.sum((gtab.bvecs-np.array([0, 0, 1])) ** 2, axis=-1))

            SNR_output = []
            SNR_directions = []
            for direction in ['b0', axis_X, axis_Y, axis_Z]:
                if direction == 'b0':
                    SNR = mean_signal[0]/noise_std if noise_std else 0
                    logging.info("SNR for the b=0 image is :" + str(SNR))
                else:
                    logging.info("SNR for direction " + str(direction) +
                                 " " + str(gtab.bvecs[direction]) + "is :" +
                                 str(SNR))
                    SNR_directions.append(direction)
                    SNR = mean_signal[direction]/noise_std if noise_std else 0
                SNR_output.append(SNR)

            data = [{
                'data': str(SNR_output[0]) + ' ' + str(SNR_output[1]) +
                        ' ' + str(SNR_output[2]) + ' ' + str(SNR_output[3]),
                'directions': 'b0' + ' ' + str(SNR_directions[0]) +
                              ' ' + str(SNR_directions[1]) + ' ' +
                              str(SNR_directions[2])
            }]

            with open(os.path.join(out_dir, out_path), 'w') as myfile:
                json.dump(data, myfile)


def buan_bundle_profiles(model_bundle_folder, bundle_folder,
                         orig_bundle_folder, metric_folder, group_id, subject,
                         no_disks=100, out_dir=''):
    """
    Applies statistical analysis on bundles and saves the results
    in a directory specified by ``out_dir``.

    Parameters
    ----------
    model_bundle_folder : string
        Path to the input model bundle files. This path may contain
        wildcards to process multiple inputs at once.
    bundle_folder : string
        Path to the input bundle files in common space. This path may
        contain wildcards to process multiple inputs at once.
    orig_folder : string
        Path to the input bundle files in native space. This path may
        contain wildcards to process multiple inputs at once.
    metric_folder : string
        Path to the input dti metric or/and peak files. It will be used as
        metric for statistical analysis of bundles.
    group_id : integer
        what group subject belongs to either 0 for control or 1 for patient.
    subject : string
        subject id e.g. 10001.
    no_disks : integer, optional
        Number of disks used for dividing bundle into disks.
    out_dir : string, optional
        Output directory. (default current directory)

    References
    ----------
    .. [Chandio2020] Chandio, B.Q., Risacher, S.L., Pestilli, F., Bullock, D.,
    Yeh, FC., Koudoro, S., Rokem, A., Harezlak, J., and Garyfallidis, E.
    Bundle analytics, a computational framework for investigating the
    shapes and profiles of brain pathways across populations.
    Sci Rep 10, 17149 (2020)

    """

    t = time()

    dt = dict()

    mb = glob(os.path.join(model_bundle_folder, "*.trk"))
    print(mb)

    mb.sort()

    bd = glob(os.path.join(bundle_folder, "*.trk"))

    bd.sort()
    print(bd)
    org_bd = glob(os.path.join(orig_bundle_folder, "*.trk"))
    org_bd.sort()
    print(org_bd)
    n = len(org_bd)
    n = len(mb)

    for io in range(n):

        mbundles = load_tractogram(mb[io], reference='same',
                                   bbox_valid_check=False).streamlines
        bundles = load_tractogram(bd[io], reference='same',
                                  bbox_valid_check=False).streamlines
        orig_bundles = load_tractogram(org_bd[io], reference='same',
                                       bbox_valid_check=False).streamlines

        if len(orig_bundles) > 5:

            indx = assignment_map(bundles, mbundles, no_disks)
            ind = np.array(indx)

            metric_files_names_dti = glob(os.path.join(metric_folder,
                                                       "*.nii.gz"))

            metric_files_names_csa = glob(os.path.join(metric_folder,
                                                       "*.pam5"))

            _, affine = load_nifti(metric_files_names_dti[0])

            affine_r = np.linalg.inv(affine)
            transformed_orig_bundles = transform_streamlines(orig_bundles,
                                                             affine_r)

            for mn in range(len(metric_files_names_dti)):

                ab = os.path.split(metric_files_names_dti[mn])
                metric_name = ab[1]

                fm = metric_name[:-7]
                bm = os.path.split(mb[io])[1][:-4]

                logging.info("bm = " + bm)

                dt = dict()

                logging.info("metric = " + metric_files_names_dti[mn])

                metric, _ = load_nifti(metric_files_names_dti[mn])

                anatomical_measures(transformed_orig_bundles, metric, dt, fm,
                                    bm, subject, group_id, ind, out_dir)

            for mn in range(len(metric_files_names_csa)):
                ab = os.path.split(metric_files_names_csa[mn])
                metric_name = ab[1]

                fm = metric_name[:-5]
                bm = os.path.split(mb[io])[1][:-4]

                logging.info("bm = " + bm)
                logging.info("metric = " + metric_files_names_csa[mn])
                dt = dict()
                metric = load_peaks(metric_files_names_csa[mn])

                peak_values(transformed_orig_bundles, metric, dt, fm, bm,
                            subject, group_id, ind, out_dir)

    print("total time taken in minutes = ", (-t + time())/60)


class BundleAnalysisTractometryFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return 'ba'

    def run(self, model_bundle_folder, subject_folder, no_disks=100,
            out_dir=''):
        """Workflow of bundle analytics.

        Applies statistical analysis on bundles of subjects and saves the
        results in a directory specified by ``out_dir``.

        Parameters
        ----------

        model_bundle_folder : string
            Path to the input model bundle files. This path may
            contain wildcards to process multiple inputs at once.

        subject_folder : string
            Path to the input subject folder. This path may contain
            wildcards to process multiple inputs at once.

        no_disks : integer, optional
            Number of disks used for dividing bundle into disks.

        out_dir : string, optional
            Output directory. (default current directory)

        References
        ----------
        .. [Chandio2020] Chandio, B.Q., Risacher, S.L., Pestilli, F.,
        Bullock, D., Yeh, FC., Koudoro, S., Rokem, A., Harezlak, J., and
        Garyfallidis, E. Bundle analytics, a computational framework for
        investigating the shapes and profiles of brain pathways across
        populations. Sci Rep 10, 17149 (2020)

        """

        if os.path.isdir(subject_folder) is False:
            raise ValueError("Invalid path to subjects")

        groups = os.listdir(subject_folder)
        groups.sort()
        for group in groups:
            if os.path.isdir(os.path.join(subject_folder, group)):
                logging.info('group = {0}'.format(group))
                all_subjects = os.listdir(os.path.join(subject_folder, group))
                all_subjects.sort()
                logging.info(all_subjects)
            if group.lower() == 'patient':
                group_id = 1  # 1 means patient
            elif group.lower() == 'control':
                group_id = 0  # 0 means control
            else:
                print(group)
                raise ValueError("Invalid group. Neither patient nor control")

            for sub in all_subjects:
                logging.info(sub)
                pre = os.path.join(subject_folder, group, sub)
                logging.info(pre)
                b = os.path.join(pre, "rec_bundles")
                c = os.path.join(pre, "org_bundles")
                d = os.path.join(pre, "anatomical_measures")
                buan_bundle_profiles(model_bundle_folder, b, c, d, group_id,
                                     sub, no_disks, out_dir)


class LinearMixedModelsFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return 'lmm'

    def get_metric_name(self, path):
        """ Splits the path string and returns name of anatomical measure
        (eg: fa), bundle name eg(AF_L) and bundle name with metric name
        (eg: AF_L_fa)

        Parameters
        ----------
        path : string
            Path to the input metric files. This path may
            contain wildcards to process multiple inputs at once.
        """

        head_tail = os.path.split(path)
        name = head_tail[1]
        count = 0
        i = len(name)-1
        while i > 0:
            if name[i] == '.':
                count = i
                break
            i = i-1

        for j in range(len(name)):
            if name[j] == '_':
                if name[j+1] != 'L' and name[j+1] != 'R' and name[j+1] != 'F':

                    return name[j+1:count], name[:j], name[:count]

        return " ", " ", " "

    def save_lmm_plot(self, plot_file, title, bundle_name, x, y):
        """ Saves LMM plot with segment/disk number on x-axis and
        -log10(pvalues) on y-axis in out_dir folder.

        Parameters
        ----------
        plot_file : string
            Path to the plot file. This path may
            contain wildcards to process multiple inputs at once.
        title : string
            Title for the plot.
        bundle_name : string
        x : list
            list containing segment/disk number for x-axis.
        y : list
            list containing -log10(pvalues) per segment/disk number for y-axis.

        """

        n = len(x)
        dotted = np.ones(n)
        dotted[:] = 2
        c1 = np.random.rand(1, 3)

        y_pos = np.arange(n)

        l1, = plt.plot(y_pos, dotted, color='red', marker='.',
                       linestyle='solid', linewidth=0.6,
                       markersize=0.7, label="p-value < 0.01")

        l2, = plt.plot(y_pos, dotted+1, color='black', marker='.',
                       linestyle='solid', linewidth=0.4,
                       markersize=0.4, label="p-value < 0.001")

        first_legend = plt.legend(handles=[l1, l2],
                                  loc='upper right')

        axes = plt.gca()
        axes.add_artist(first_legend)
        axes.set_ylim([0, 6])

        l3 = plt.bar(y_pos, y, color=c1, alpha=0.5,
                               label=bundle_name)
        plt.legend(handles=[l3], loc='upper left')
        plt.title(title.upper())
        plt.xlabel("Segment Number")
        plt.ylabel("-log10(Pvalues)")
        plt.savefig(plot_file)
        plt.clf()

    def run(self, h5_files, no_disks=100, out_dir=''):
        """Workflow of linear Mixed Models.

        Applies linear Mixed Models on bundles of subjects and saves the
        results in a directory specified by ``out_dir``.

        Parameters
        ----------

        h5_files : string
            Path to the input metric files. This path may
            contain wildcards to process multiple inputs at once.

        no_disks : integer, optional
            Number of disks used for dividing bundle into disks.

        out_dir : string, optional
            Output directory. (default current directory)

        """

        io_it = self.get_io_iterator()

        for file_path in io_it:

            logging.info('Applying metric {0}'.format(file_path))

            file_name, bundle_name, save_name = self.get_metric_name(file_path)
            logging.info(" file name = " + file_name)
            logging.info("file path = " + file_path)

            pvalues = np.zeros(no_disks)
            warnings.filterwarnings("ignore")
            # run mixed linear model for every disk
            for i in range(no_disks):
                disk_count = i+1
                df = pd.read_hdf(file_path, where='disk=disk_count')

                logging.info("read the dataframe for disk number " +
                             str(disk_count))
                # check if data has significant data to perform LMM
                if len(df) < 10:
                    raise ValueError("Dataset for Linear Mixed Model is too small")

                criteria = file_name + " ~ group"
                md = smf.mixedlm(criteria, df,
                                 groups=df["subject"])

                mdf = md.fit()

                pvalues[i] = mdf.pvalues[1]

            x = list(range(1, len(pvalues)+1))
            y = -1*np.log10(pvalues)

            save_file = os.path.join(out_dir, save_name + "_pvalues.npy")
            np.save(save_file, pvalues)

            save_file = os.path.join(out_dir, save_name + "_pvalues_log.npy")
            np.save(save_file, y)

            save_file = os.path.join(out_dir, save_name + ".png")
            self.save_lmm_plot(save_file, file_name, bundle_name, x, y)


class BundleShapeAnalysis(Workflow):
    @classmethod
    def get_short_name(cls):
        return 'BS'

    def run(self, subject_folder, clust_thr=(5, 3, 1.5), threshold=6,
            out_dir=''):
        """Workflow of bundle analytics.

        Applies bundle shape similarity analysis on bundles of subjects and
        saves the results in a directory specified by ``out_dir``.

        Parameters
        ----------

        subject_folder : string
            Path to the input subject folder. This path may contain
            wildcards to process multiple inputs at once.

        clust_thr : variable float, optional
            list of bundle clustering thresholds used in QuickBundlesX.

        threshold : float, optional
            Bundle shape similarity threshold.

        out_dir : string, optional
            Output directory. (default current directory)

        References
        ----------
        .. [Chandio2020] Chandio, B.Q., Risacher, S.L., Pestilli, F.,
        Bullock, D., Yeh, FC., Koudoro, S., Rokem, A., Harezlak, J., and
        Garyfallidis, E. Bundle analytics, a computational framework for
        investigating the shapes and profiles of brain pathways across
        populations. Sci Rep 10, 17149 (2020)

        """
        rng = np.random.default_rng()
        all_subjects = []
        if os.path.isdir(subject_folder):
            groups = os.listdir(subject_folder)
            groups.sort()
        else:
            raise ValueError("Not a directory")

        for group in groups:

            if os.path.isdir(os.path.join(subject_folder, group)):
                subjects = os.listdir(os.path.join(subject_folder, group))
                subjects.sort()
                logging.info("first " + str(len(subjects)) +
                             " subjects in matrix belong to " + group +
                             " group")

                for sub in subjects:
                    dpath = os.path.join(subject_folder, group, sub)
                    if os.path.isdir(dpath):
                        all_subjects.append(dpath)

        N = len(all_subjects)

        bundles = os.listdir(os.path.join(all_subjects[0], "rec_bundles"))
        for bun in bundles:
            # bundle shape similarity matrix
            ba_matrix = np.zeros((N, N))
            i = 0
            logging.info(bun)
            for sub in all_subjects:
                j = 0

                bundle1 = load_tractogram(os.path.join(sub, "rec_bundles",
                                                       bun), reference='same',
                                          bbox_valid_check=False).streamlines

                for subi in all_subjects:
                    logging.info(subi)

                    bundle2 = load_tractogram(os.path.join(subi, "rec_bundles",
                                                           bun),
                                              reference='same',
                                              bbox_valid_check=False).streamlines

                    ba_value = bundle_shape_similarity(bundle1, bundle2, rng,
                                                       clust_thr, threshold)

                    ba_matrix[i][j] = ba_value

                    j += 1
                i += 1
            logging.info("saving BA score matrix")
            np.save(os.path.join(out_dir, bun[:-4]+".npy"), ba_matrix)

            cmap = matplt.colormaps['Blues']
            plt.title(bun[:-4])
            plt.imshow(ba_matrix, cmap=cmap)
            plt.colorbar()
            plt.clim(0, 1)
            plt.savefig(os.path.join(out_dir, "SM_"+bun[:-4]))
            plt.clf()
