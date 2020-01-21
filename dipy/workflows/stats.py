
import logging
import numpy as np
import os
import json
from scipy.ndimage.morphology import binary_dilation
from dipy.utils.optpkg import optional_package
from dipy.io import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
from dipy.core.gradients import gradient_table
from dipy.segment.mask import median_otsu
from dipy.reconst.dti import TensorModel

from dipy.segment.mask import segment_from_cfa
from dipy.segment.mask import bounding_box

from dipy.workflows.workflow import Workflow

from dipy.viz.regtools import simple_plot
from dipy.stats.analysis import bundle_analysis
pd, have_pd, _ = optional_package("pandas")
smf, have_smf, _ = optional_package("statsmodels")
tables, have_tables, _ = optional_package("tables")

if have_pd:
    import pandas as pd

if have_smf:
    import statsmodels.formula.api as smf

if have_tables:
    import tables


class SNRinCCFlow(Workflow):

    @classmethod
    def get_short_name(cls):
        return 'snrincc'

    def run(self, data_files, bvals_files, bvecs_files, mask_file,
            bbox_threshold=[0.6, 1, 0, 0.1, 0, 0.1], out_dir='',
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
            [0.6,1,0,0.1,0,0.1]. (default (0.6, 1, 0, 0.1, 0, 0.1))
        out_dir : string, optional
            Where the resulting file will be saved. (default '')
        out_file : string, optional
            Name of the result file to be saved. (default 'product.json')
        out_mask_cc : string, optional
            Name of the CC mask volume to be saved (default 'cc.nii.gz')
        out_mask_noise : string, optional
            Name of the mask noise volume to be saved
            (default 'mask_noise.nii.gz')

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
                raise IOError('DWI data has invalid dimensions')

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
                raise IOError('bbox_threshold should have 6 float values')

            mask_cc_part, cfa = segment_from_cfa(tensorfit, CC_box,
                                                 bbox_threshold,
                                                 return_cfa=True)

            save_nifti(cc_mask_path, mask_cc_part.astype(np.uint8), affine)
            logging.info('CC mask saved as {0}'.format(cc_mask_path))

            mean_signal = np.mean(data[mask_cc_part], axis=0)
            mask_noise = binary_dilation(mask, iterations=10)
            mask_noise[..., :mask_noise.shape[-1]//2] = 1
            mask_noise = ~mask_noise

            save_nifti(mask_noise_path, mask_noise.astype(np.uint8), affine)
            logging.info('Mask noise saved as {0}'.format(mask_noise_path))

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
                    SNR = mean_signal[0]/noise_std
                    logging.info("SNR for the b=0 image is :" + str(SNR))
                else:
                    logging.info("SNR for direction " + str(direction) +
                                 " " + str(gtab.bvecs[direction]) + "is :" +
                                 str(SNR))
                    SNR_directions.append(direction)
                    SNR = mean_signal[direction]/noise_std
                SNR_output.append(SNR)

            data = []
            data.append({
                        'data': str(SNR_output[0]) + ' ' + str(SNR_output[1]) +
                        ' ' + str(SNR_output[2]) + ' ' + str(SNR_output[3]),
                        'directions': 'b0' + ' ' + str(SNR_directions[0]) +
                        ' ' + str(SNR_directions[1]) + ' ' +
                        str(SNR_directions[2])
                        })

            with open(os.path.join(out_dir, out_path), 'w') as myfile:
                json.dump(data, myfile)


class BundleAnalysisPopulationFlow(Workflow):
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
            Number of disks used for dividing bundle into disks. (Default 100)

        out_dir : string, optional
            Output directory (default input file directory)

        References
        ----------
        .. [Chandio19] Chandio, B.Q., S. Koudoro, D. Reagan, J. Harezlak,
        E. Garyfallidis, Bundle Analytics: a computational and statistical
        analyses framework for tractometric studies, Proceedings of:
        International Society of Magnetic Resonance in Medicine (ISMRM),
        Montreal, Canada, 2019.

        """

        groups = os.listdir(subject_folder)

        for group in groups:
            logging.info('group = {0}'.format(group))
            all_subjects = os.listdir(os.path.join(subject_folder, group))

            for sub in all_subjects:

                pre = os.path.join(subject_folder, group, sub)

                b = os.path.join(pre, "rec_bundles")
                c = os.path.join(pre, "org_bundles")
                d = os.path.join(pre, "measures")
                bundle_analysis(model_bundle_folder, b, c, d, group, sub,
                                no_disks, out_dir)


class LinearMixedModelsFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return 'lmm'

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
            Number of disks used for dividing bundle into disks. (Default 100)

        out_dir : string, optional
            Output directory (default input file directory)

        """

        io_it = self.get_io_iterator()

        for file_path in io_it:

            logging.info('Applying metric {0}'.format(file_path))
            file_name = os.path.basename(file_path)[:-3]
            df = pd.read_hdf(file_path)

            if len(df) < 100:
                raise ValueError("Dataset for Linear Mixed Model is too small")

            all_bundles = df.bundle.unique()
            # all_pvalues = []
            for bundle in all_bundles:
                sub_af = df[df['bundle'] == bundle]  # sub sample
                pvalues = np.zeros(no_disks)

                # run mixed linear model for every disk
                for i in range(no_disks):

                    sub = sub_af[sub_af['disk#'] == (i+1)]  # disk number

                    if len(sub) > 0:
                        criteria = file_name + " ~ group"
                        md = smf.mixedlm(criteria, sub, groups=sub["subject"])

                        mdf = md.fit()

                        pvalues[i] = mdf.pvalues[1]

                x = list(range(1, len(pvalues)+1))
                y = -1*np.log10(pvalues)

                title = bundle + " on " + file_name + " Values"
                plot_file = os.path.join(out_dir, bundle + "_" +
                                         file_name + ".png")

                simple_plot(plot_file, title, x, y, "disk no",
                            "-log10(pvalues)")
