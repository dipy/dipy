from __future__ import division, print_function, absolute_import

import logging
import numpy as np
import os.path
from ast import literal_eval

import nibabel as nib

from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.peaks import save_peaks, peaks_to_niftis
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response)
from dipy.reconst.dti import (TensorModel, color_fa, fractional_anisotropy,
                              geodesic_anisotropy, mean_diffusivity,
                              axial_diffusivity, radial_diffusivity,
                              lower_triangular, mode as get_mode)
from dipy.reconst.peaks import peaks_from_model
from dipy.reconst.shm import CsaOdfModel
from dipy.workflows.workflow import Workflow
from dipy.reconst.dki import DiffusionKurtosisModel, split_dki_param

from dipy.reconst import mapmri
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

class ReconstMAPMRIFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return 'mapmri'

    def run(self, data_file, data_bvecs, data_bvals, small_delta=0.0129, big_delta=0.0218
            save_metrics = [], out_dir='', out_mapmri='MAPMRI_maps_regularization.png'):
        """ Workflow for the app-dipy-mapmri on Brain-Life (www.brain-life.org).
        Generates RTOP graphs saved in a .png format in input files provided by
        `data_file` and saves the png file to an output directory specified by
        `out_dir`.

        Parameters
        ----------
        data_file : string
            Path to the input volume.
        data_bvecs : string
            Path to the bvec files.
        data_bvals :
            Path to the bval files.
        small_delta :
            Small delta value used in generation of gradient table of provided
            bval and bvec. (default: 0.0129)
        big_delta :
            Big delta value used in generation of gradient table of provided
            bval and bvec. (default: 0.0218)
        save_metrics :
            List of metrics to save.
            Possible values: mmri
            (default: [] (all))
        out_dir : string, optional
            Output directory (default: input file directory)
        out_mapmri : string, optional
            Name of the png file to be saved (default: MAPMRI_maps_regularization.png))
        """
        io_it = self.get_io_iterator()
        for dwi, bval, bvec, mapname in io_it:

            logging.info('Computing DTI metrics for {0}'.format(dwi))
            img = nib.load(dwi)
            data = img.get_data()
            bvals,bvecs = read_bvals_bvecs(bval, bvec)


            gtab = gradient_table(bvals=bvals, bvecs=bvecs,
                                small_delta=small_delta,
                                big_delta=big_delta, b0_threshold=50)


            data_small = data[60:85, 80:81, 60:85]

            if not save_metrics:
                save_metrics = ['mmri']

            # print('data.shape (%d, %d, %d, %d)' % data.shape)

            radial_order = 6
            map_model_laplacian_aniso = mapmri.MapmriModel(gtab, radial_order=radial_order,
                                                        laplacian_regularization=True,
                                                        laplacian_weighting=.2)

            map_model_positivity_aniso = mapmri.MapmriModel(gtab, radial_order=radial_order,
                                                            laplacian_regularization=False,
                                                            positivity_constraint=True)

            map_model_both_aniso = mapmri.MapmriModel(gtab, radial_order=radial_order,
                                                    laplacian_regularization=True,
                                                    laplacian_weighting=.05,
                                                    positivity_constraint=True)

            mapfit_laplacian_aniso = map_model_laplacian_aniso.fit(data_small)
            mapfit_positivity_aniso = map_model_positivity_aniso.fit(data_small)
            mapfit_both_aniso = map_model_both_aniso.fit(data_small)

            if 'mmri' in save_metrics:
                # generating RTOP plots
                fig = plt.figure(figsize=(10, 5))
                ax1 = fig.add_subplot(1, 3, 1, title=r'RTOP - Laplacian')
                ax1.set_axis_off()
                ind = ax1.imshow(mapfit_laplacian_aniso.rtop()[:, 0, :].T,
                                interpolation='nearest', origin='lower', cmap=plt.cm.gray,
                                vmin=0, vmax=5e7)

                ax2 = fig.add_subplot(1, 3, 2, title=r'RTOP - Positivity')
                ax2.set_axis_off()
                ind = ax2.imshow(mapfit_positivity_aniso.rtop()[:, 0, :].T,
                                interpolation='nearest', origin='lower', cmap=plt.cm.gray,
                                vmin=0, vmax=5e7)

                ax3 = fig.add_subplot(1, 3, 3, title=r'RTOP - Both')
                ax3.set_axis_off()
                ind = ax3.imshow(mapfit_both_aniso.rtop()[:, 0, :].T,
                                interpolation='nearest', origin='lower', cmap=plt.cm.gray,
                                vmin=0, vmax=5e7)
                divider = make_axes_locatable(ax3)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(ind, cax=cax)

                plt.savefig('MAPMRI_maps_regularization.png')

            logging.info('MAPMRI saved in {0}'.
                         format(os.path.dirname(mapname)))

from dipy.reconst import mapmri
# import matplotlib
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable

class ReconstMAPMRIFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return 'mapmri'

    def run(self, data_file, data_bvecs, data_bvals, small_delta=0.0129, big_delta=0.0218,
            model_type = [], save_metrics = [], out_dir='', out_lap_rtop = 'lap_rtop',
            out_lap_lapnorm = 'lap_lapnorm', out_lap_msd = 'lap_msd',
            out_lap_qiv = 'lap_qiv', out_lap_rtap = 'lap_rtap', out_lap_rtpp = 'lap_rtpp',
            out_pos_rtop = 'pos_rtop', out_pos_lapnorm = 'pos_lapnorm', out_pos_msd = 'pos_msd',
            out_pos_qiv = 'pos_qiv', out_pos_rtap = 'pos_rtap', out_pos_rtpp = 'pos_rtpp',
            out_both_rtop = 'both_rtop', out_both_lapnorm = 'both_lapnorm', out_both_msd = 'both_msd',
            out_both_qiv = 'both_qiv', out_both_rtap = 'both_rtap', out_both_rtpp = 'both_rtpp'):
        """ Workflow for the app-dipy-mapmri on Brain-Life (www.brain-life.org).
        Generates RTOP ??? saved in a ??? format in input files provided by
        `data_file` and saves the png file to an output directory specified by
        `out_dir`.

        Parameters
        ----------
        data_file : string
            Path to the input volume.
        data_bvecs : string
            Path to the bvec files.
        data_bvals :
            Path to the bval files.
        small_delta :
            Small delta value used in generation of gradient table of provided
            bval and bvec. (default: 0.0129)
        big_delta :
            Big delta value used in generation of gradient table of provided
            bval and bvec. (default: 0.0218)
        model_type:
            Model type to fit.
            Possible values: laplacian, positivity, both
            (default: [] (all))
        save_metrics :
            List of metrics to save.
            Possible values: rtop, laplacian_signal, msd, qiv, rtap, rtpp
            (default: [] (all))
        out_dir : string, optional
            Output directory (default: input file directory)
        out_lap_rtop : string, optional
            Name of the laplacian rtop to be saved (default 'lap_rtop')
        out_lap_lapnorm : string, optional
            Name of the laplacian norm of laplacian signal to be saved (default 'lap_lapnorm')
        out_lap_msd : string, optional
            Name of the laplacian msd to be saved (default 'lap_msd')
        out_lap_qiv : string, optional
            Name of the laplacian qiv to be saved (default 'lap_qiv')
        out_lap_rtap : string, optional
            Name of the laplacian rtap to be saved (default 'lap_rtap')
        out_lap_rtpp : string, optional
            Name of the laplacian rtpp to be saved (default 'lap_rtpp')
        out_pos_rtop : string, optional
            Name of the positivity rtop to be saved (default 'pos_rtop')
        out_pos_lapnorm : string, optional
            Name of the positivity norm of laplacian signal to be saved (default 'pos_lapnorm')
        out_pos_msd : string, optional
            Name of the positivity msd to be saved (default 'pos_msd')
        out_pos_qiv : string, optional
            Name of the positivity qiv to be saved (default 'pos_qiv')
        out_pos_rtap : string, optional
            Name of the positivity rtap to be saved (default 'pos_rtap')
        out_pos_rtpp : string, optional
            Name of the positivity rtpp to be saved (default 'pos_rtpp')
        out_both_rtop : string, optional
            Name of the both rtop to be saved (default 'both_rtop')
        out_both_lapnorm : string, optional
            Name of the both norm of laplacian signal to be saved (default 'both_lapnorm')
        out_both_msd : string, optional
            Name of the both msd to be saved (default 'both_msd')
        out_both_qiv : string, optional
            Name of the both qiv to be saved (default 'both_qiv')
        out_both_rtap : string, optional
            Name of the both rtap to be saved (default 'both_rtap')
        out_both_rtpp : string, optional
            Name of the both rtpp to be saved (default 'both_rtpp')
        """
        io_it = self.get_io_iterator()
        for dwi, bval, bvec, mapname in io_it:

            logging.info('Computing DTI metrics for {0}'.format(dwi))
            img = nib.load(dwi)
            data = img.get_data()
            bvals,bvecs = read_bvals_bvecs(bval, bvec)


            gtab = gradient_table(bvals=bvals, bvecs=bvecs,
                                small_delta=small_delta,
                                big_delta=big_delta, b0_threshold=50)


            data_small = data[60:85, 80:81, 60:85]

            if not model_type:
                model_type = ['laplacian', 'positivity', 'both']

            if not save_metrics:
                save_metrics = ['rtop', 'laplacian_signal', 'msd', 'qiv', 'rtap', 'rtpp']

            # print('data.shape (%d, %d, %d, %d)' % data.shape)

            radial_order = 6

            # How do I save the rtop in a useful form? A picture? A numpy array?

            if 'laplacian' in model_type:
                map_model_laplacian_aniso = mapmri.MapmriModel(gtab, radial_order=radial_order,
                                                            laplacian_regularization=True,
                                                            laplacian_weighting=.2)
                mapfit_laplacian_aniso = map_model_laplacian_aniso.fit(data_small)

                if 'rtop' in save_metrics:
                    return mapfit_laplacian_aniso.rtop()

                if 'laplacian_signal' in save_metrics:
                    return mapfit_laplacian_aniso.norm_of_laplacian_signal()

                if 'msd' in save_metrics:
                    return mapfit_laplacian_aniso.msd()

                if 'qiv' in save_metrics:
                    return mapfit_laplacian_aniso.qiv()

                if 'rtap' in save_metrics:
                    return mapfit_laplacian_aniso.rtap()

                if 'rtpp' in save_metrics:
                    return mapfit_laplacian_aniso.rtop()

            if 'positivity' in model_type:

                map_model_positivity_aniso = mapmri.MapmriModel(gtab, radial_order=radial_order,
                                                                laplacian_regularization=False,
                                                                positivity_constraint=True)
                mapfit_positivity_aniso = map_model_positivity_aniso.fit(data_small)

                if 'rtop' in save_metrics:
                    return mapfit_positivity_aniso.rtop()

                if 'laplacian_signal' in save_metrics:
                    return mapfit_positivity_aniso.norm_of_laplacian_signal()

                if 'msd' in save_metrics:
                    return mapfit_positivity_aniso.msd()

                if 'qiv' in save_metrics:
                    return mapfit_positivity_aniso.qiv()

                if 'rtap' in save_metrics:
                    return mapfit_positivity_aniso.rtap()

                if 'rtpp' in save_metrics:
                    return mapfit_positivity_aniso.rtop()

            if 'both' in model_type:
                map_model_both_aniso = mapmri.MapmriModel(gtab, radial_order=radial_order,
                                                        laplacian_regularization=True,
                                                        laplacian_weighting=.05,
                                                        positivity_constraint=True)
                mapfit_both_aniso = map_model_both_aniso.fit(data_small)

                if 'rtop' in save_metrics:
                    return mapfit_both_aniso.rtop()

                if 'laplacian_signal' in save_metrics:
                    return mapfit_both_aniso.norm_of_laplacian_signal()

                if 'msd' in save_metrics:
                    return mapfit_both_aniso.msd()

                if 'qiv' in save_metrics:
                    return mapfit_both_aniso.qiv()

                if 'rtap' in save_metrics:
                    return mapfit_both_aniso.rtap()

                if 'rtpp' in save_metrics:
                    return mapfit_both_aniso.rtop()

            logging.info('MAPMRI saved in {0}'.
                    format(os.path.dirname(mapname)))

            # if 'mmri' in save_metrics:
            #     # generating RTOP plots
            #     fig = plt.figure(figsize=(10, 5))
            #     ax1 = fig.add_subplot(1, 3, 1, title=r'RTOP - Laplacian')
            #     ax1.set_axis_off()
            #     ind = ax1.imshow(mapfit_laplacian_aniso.rtop()[:, 0, :].T,
            #                     interpolation='nearest', origin='lower', cmap=plt.cm.gray,
            #                     vmin=0, vmax=5e7)
            #
            #     ax2 = fig.add_subplot(1, 3, 2, title=r'RTOP - Positivity')
            #     ax2.set_axis_off()
            #     ind = ax2.imshow(mapfit_positivity_aniso.rtop()[:, 0, :].T,
            #                     interpolation='nearest', origin='lower', cmap=plt.cm.gray,
            #                     vmin=0, vmax=5e7)
            #
            #     ax3 = fig.add_subplot(1, 3, 3, title=r'RTOP - Both')
            #     ax3.set_axis_off()
            #     ind = ax3.imshow(mapfit_both_aniso.rtop()[:, 0, :].T,
            #                     interpolation='nearest', origin='lower', cmap=plt.cm.gray,
            #                     vmin=0, vmax=5e7)
            #     divider = make_axes_locatable(ax3)
            #     cax = divider.append_axes("right", size="5%", pad=0.05)
            #     plt.colorbar(ind, cax=cax)
            #
            #     plt.savefig('MAPMRI_maps_regularization.png')
            #
            # logging.info('MAPMRI saved in {0}'.
            #              format(os.path.dirname(mapname)))

from dipy.reconst import mapmri
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

class ReconstMAPMRIFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return 'mapmri'

    def run(self, data_file, data_bvecs, data_bvals, small_delta=0.0129, big_delta=0.0218
            save_metrics = [], out_dir='', out_mapmri='MAPMRI_maps_regularization.png'):
        """ Workflow for the app-dipy-mapmri on Brain-Life (www.brain-life.org).
        Generates RTOP graphs saved in a .png format in input files provided by
        `data_file` and saves the png file to an output directory specified by
        `out_dir`.

        Parameters
        ----------
        data_file : string
            Path to the input volume.
        data_bvecs : string
            Path to the bvec files.
        data_bvals :
            Path to the bval files.
        small_delta :
            Small delta value used in generation of gradient table of provided
            bval and bvec. (default: 0.0129)
        big_delta :
            Big delta value used in generation of gradient table of provided
            bval and bvec. (default: 0.0218)
        save_metrics :
            List of metrics to save.
            Possible values: mmri
            (default: [] (all))
        out_dir : string, optional
            Output directory (default: input file directory)
        out_mapmri : string, optional
            Name of the png file to be saved (default: MAPMRI_maps_regularization.png))
        """
        io_it = self.get_io_iterator()
        for dwi, bval, bvec, mapname in io_it:

            logging.info('Computing DTI metrics for {0}'.format(dwi))
            img = nib.load(dwi)
            data = img.get_data()
            bvals,bvecs = read_bvals_bvecs(bval, bvec)


            gtab = gradient_table(bvals=bvals, bvecs=bvecs,
                                small_delta=small_delta,
                                big_delta=big_delta, b0_threshold=50)


            data_small = data[60:85, 80:81, 60:85]

            if not save_metrics:
                save_metrics = ['mmri']

            # print('data.shape (%d, %d, %d, %d)' % data.shape)

            radial_order = 6
            map_model_laplacian_aniso = mapmri.MapmriModel(gtab, radial_order=radial_order,
                                                        laplacian_regularization=True,
                                                        laplacian_weighting=.2)

            map_model_positivity_aniso = mapmri.MapmriModel(gtab, radial_order=radial_order,
                                                            laplacian_regularization=False,
                                                            positivity_constraint=True)

            map_model_both_aniso = mapmri.MapmriModel(gtab, radial_order=radial_order,
                                                    laplacian_regularization=True,
                                                    laplacian_weighting=.05,
                                                    positivity_constraint=True)

            mapfit_laplacian_aniso = map_model_laplacian_aniso.fit(data_small)
            mapfit_positivity_aniso = map_model_positivity_aniso.fit(data_small)
            mapfit_both_aniso = map_model_both_aniso.fit(data_small)

            if 'mmri' in save_metrics:
                # generating RTOP plots
                fig = plt.figure(figsize=(10, 5))
                ax1 = fig.add_subplot(1, 3, 1, title=r'RTOP - Laplacian')
                ax1.set_axis_off()
                ind = ax1.imshow(mapfit_laplacian_aniso.rtop()[:, 0, :].T,
                                interpolation='nearest', origin='lower', cmap=plt.cm.gray,
                                vmin=0, vmax=5e7)

                ax2 = fig.add_subplot(1, 3, 2, title=r'RTOP - Positivity')
                ax2.set_axis_off()
                ind = ax2.imshow(mapfit_positivity_aniso.rtop()[:, 0, :].T,
                                interpolation='nearest', origin='lower', cmap=plt.cm.gray,
                                vmin=0, vmax=5e7)

                ax3 = fig.add_subplot(1, 3, 3, title=r'RTOP - Both')
                ax3.set_axis_off()
                ind = ax3.imshow(mapfit_both_aniso.rtop()[:, 0, :].T,
                                interpolation='nearest', origin='lower', cmap=plt.cm.gray,
                                vmin=0, vmax=5e7)
                divider = make_axes_locatable(ax3)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(ind, cax=cax)

                plt.savefig('MAPMRI_maps_regularization.png')

            logging.info('MAPMRI saved in {0}'.
                         format(os.path.dirname(mapname)))


class ReconstDtiFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return 'dti'

    def run(self, input_files, bvalues, bvectors, mask_files, b0_threshold=0.0,
            bvecs_tol=0.01,
            save_metrics=[],
            out_dir='', out_tensor='tensors.nii.gz', out_fa='fa.nii.gz',
            out_ga='ga.nii.gz', out_rgb='rgb.nii.gz', out_md='md.nii.gz',
            out_ad='ad.nii.gz', out_rd='rd.nii.gz', out_mode='mode.nii.gz',
            out_evec='evecs.nii.gz', out_eval='evals.nii.gz'):
        """ Workflow for tensor reconstruction and for computing DTI metrics.
        using Weighted Least-Squares.
        Performs a tensor reconstruction on the files by 'globing'
        ``input_files`` and saves the DTI metrics in a directory specified by
        ``out_dir``.

        Parameters
        ----------
        input_files : string
            Path to the input volumes. This path may contain wildcards to
            process multiple inputs at once.
        bvalues : string
            Path to the bvalues files. This path may contain wildcards to use
            multiple bvalues files at once.
        bvectors : string
            Path to the bvectors files. This path may contain wildcards to use
            multiple bvectors files at once.
        mask_files : string
            Path to the input masks. This path may contain wildcards to use
            multiple masks at once. (default: No mask used)
        b0_threshold : float, optional
            Threshold used to find b=0 directions (default 0.0)
        bvecs_tol : float, optional
            Threshold used to check that norm(bvec) = 1 +/- bvecs_tol
            b-vectors are unit vectors (default 0.01)
        save_metrics : variable string, optional
            List of metrics to save.
            Possible values: fa, ga, rgb, md, ad, rd, mode, tensor, evec, eval
            (default [] (all))
        out_dir : string, optional
            Output directory (default input file directory)
        out_tensor : string, optional
            Name of the tensors volume to be saved (default 'tensors.nii.gz')
        out_fa : string, optional
            Name of the fractional anisotropy volume to be saved
            (default 'fa.nii.gz')
        out_ga : string, optional
            Name of the geodesic anisotropy volume to be saved
            (default 'ga.nii.gz')
        out_rgb : string, optional
            Name of the color fa volume to be saved (default 'rgb.nii.gz')
        out_md : string, optional
            Name of the mean diffusivity volume to be saved
            (default 'md.nii.gz')
        out_ad : string, optional
            Name of the axial diffusivity volume to be saved
            (default 'ad.nii.gz')
        out_rd : string, optional
            Name of the radial diffusivity volume to be saved
            (default 'rd.nii.gz')
        out_mode : string, optional
            Name of the mode volume to be saved (default 'mode.nii.gz')
        out_evec : string, optional
            Name of the eigenvectors volume to be saved
            (default 'evecs.nii.gz')
        out_eval : string, optional
            Name of the eigenvalues to be saved (default 'evals.nii.gz')

        References
        ----------
        .. [1] Basser, P.J., Mattiello, J., LeBihan, D., 1994. Estimation of
           the effective self-diffusion tensor from the NMR spin echo. J Magn
           Reson B 103, 247-254.

        .. [2] Basser, P., Pierpaoli, C., 1996. Microstructural and
           physiological features of tissues elucidated by quantitative
           diffusion-tensor MRI.  Journal of Magnetic Resonance 111, 209-219.

        .. [3] Lin-Ching C., Jones D.K., Pierpaoli, C. 2005. RESTORE: Robust
           estimation of tensors by outlier rejection. MRM 53: 1088-1095

        .. [4] hung, SW., Lu, Y., Henry, R.G., 2006. Comparison of bootstrap
           approaches for estimation of uncertainties of DTI parameters.
           NeuroImage 33, 531-541.

        """
        io_it = self.get_io_iterator()

        for dwi, bval, bvec, mask, otensor, ofa, oga, orgb, omd, oad, orad, \
                omode, oevecs, oevals in io_it:

            logging.info('Computing DTI metrics for {0}'.format(dwi))
            img = nib.load(dwi)
            data = img.get_data()
            affine = img.affine

            if mask is not None:
                mask = nib.load(mask).get_data().astype(np.bool)

            tenfit, _ = self.get_fitted_tensor(data, mask, bval, bvec,
                                               b0_threshold, bvecs_tol)

            if not save_metrics:
                save_metrics = ['fa', 'md', 'rd', 'ad', 'ga', 'rgb', 'mode',
                                'evec', 'eval', 'tensor']

            FA = fractional_anisotropy(tenfit.evals)
            FA[np.isnan(FA)] = 0
            FA = np.clip(FA, 0, 1)

            if 'tensor' in save_metrics:
                tensor_vals = lower_triangular(tenfit.quadratic_form)
                correct_order = [0, 1, 3, 2, 4, 5]
                tensor_vals_reordered = tensor_vals[..., correct_order]
                fiber_tensors = nib.Nifti1Image(tensor_vals_reordered.astype(
                    np.float32), affine)
                nib.save(fiber_tensors, otensor)

            if 'fa' in save_metrics:
                fa_img = nib.Nifti1Image(FA.astype(np.float32), affine)
                nib.save(fa_img, ofa)

            if 'ga' in save_metrics:
                GA = geodesic_anisotropy(tenfit.evals)
                ga_img = nib.Nifti1Image(GA.astype(np.float32), affine)
                nib.save(ga_img, oga)

            if 'rgb' in save_metrics:
                RGB = color_fa(FA, tenfit.evecs)
                rgb_img = nib.Nifti1Image(np.array(255 * RGB, 'uint8'), affine)
                nib.save(rgb_img, orgb)

            if 'md' in save_metrics:
                MD = mean_diffusivity(tenfit.evals)
                md_img = nib.Nifti1Image(MD.astype(np.float32), affine)
                nib.save(md_img, omd)

            if 'ad' in save_metrics:
                AD = axial_diffusivity(tenfit.evals)
                ad_img = nib.Nifti1Image(AD.astype(np.float32), affine)
                nib.save(ad_img, oad)

            if 'rd' in save_metrics:
                RD = radial_diffusivity(tenfit.evals)
                rd_img = nib.Nifti1Image(RD.astype(np.float32), affine)
                nib.save(rd_img, orad)

            if 'mode' in save_metrics:
                MODE = get_mode(tenfit.quadratic_form)
                mode_img = nib.Nifti1Image(MODE.astype(np.float32), affine)
                nib.save(mode_img, omode)

            if 'evec' in save_metrics:
                evecs_img = nib.Nifti1Image(tenfit.evecs.astype(np.float32), affine)
                nib.save(evecs_img, oevecs)

            if 'eval' in save_metrics:
                evals_img = nib.Nifti1Image(tenfit.evals.astype(np.float32), affine)
                nib.save(evals_img, oevals)

            dname_ = os.path.dirname(oevals)
            if dname_ == '':
                logging.info('DTI metrics saved in current directory')
            else:
                logging.info(
                        'DTI metrics saved in {0}'.format(dname_))

    def get_tensor_model(self, gtab):
        return TensorModel(gtab, fit_method="WLS")

    def get_fitted_tensor(self, data, mask, bval, bvec,
                          b0_threshold=0, bvecs_tol=0.01):

        logging.info('Tensor estimation...')
        bvals, bvecs = read_bvals_bvecs(bval, bvec)
        gtab = gradient_table(bvals, bvecs, b0_threshold=b0_threshold,
                              atol=bvecs_tol)

        tenmodel = self.get_tensor_model(gtab)
        tenfit = tenmodel.fit(data, mask)

        return tenfit, gtab


class ReconstDtiRestoreFlow(ReconstDtiFlow):
    @classmethod
    def get_short_name(cls):
        return 'dti_restore'

    def run(self, input_files, bvalues, bvectors, mask_files, sigma,
            b0_threshold=0.0, save_metrics=[], jacobian=True,
            out_dir='', out_tensor='tensors.nii.gz', out_fa='fa.nii.gz',
            out_ga='ga.nii.gz', out_rgb='rgb.nii.gz', out_md='md.nii.gz',
            out_ad='ad.nii.gz', out_rd='rd.nii.gz', out_mode='mode.nii.gz',
            out_evec='evecs.nii.gz', out_eval='evals.nii.gz'):

        """ Workflow for tensor reconstruction and for computing DTI metrics.
            Performs a tensor reconstruction on the files by 'globing'
            ``input_files`` and saves the DTI metrics in a directory specified by
            ``out_dir``.

            Parameters
            ----------
            input_files : string
                Path to the input volumes. This path may contain wildcards to
                process multiple inputs at once.
            bvalues : string
                Path to the bvalues files. This path may contain wildcards to use
                multiple bvalues files at once.
            bvectors : string
                Path to the bvectors files. This path may contain wildcards to use
                multiple bvectors files at once.
            mask_files : string
                Path to the input masks. This path may contain wildcards to use
                multiple masks at once. (default: No mask used)
            sigma : float
                An estimate of the variance.
            b0_threshold : float, optional
                Threshold used to find b=0 directions (default 0.0)
            save_metrics : variable string, optional
                List of metrics to save.
                Possible values: fa, ga, rgb, md, ad, rd, mode, tensor, evec, eval
                (default [] (all))
            jacobian : bool, optional
                Whether to use the Jacobian of the tensor to speed the
                non-linear optimization procedure used to fit the tensor
                parameters (default True)
            out_dir : string, optional
                Output directory (default input file directory)
            out_tensor : string, optional
                Name of the tensors volume to be saved (default 'tensors.nii.gz')
            out_fa : string, optional
                Name of the fractional anisotropy volume to be saved
                (default 'fa.nii.gz')
            out_ga : string, optional
                Name of the geodesic anisotropy volume to be saved
                (default 'ga.nii.gz')
            out_rgb : string, optional
                Name of the color fa volume to be saved (default 'rgb.nii.gz')
            out_md : string, optional
                Name of the mean diffusivity volume to be saved
                (default 'md.nii.gz')
            out_ad : string, optional
                Name of the axial diffusivity volume to be saved
                (default 'ad.nii.gz')
            out_rd : string, optional
                Name of the radial diffusivity volume to be saved
                (default 'rd.nii.gz')
            out_mode : string, optional
                Name of the mode volume to be saved (default 'mode.nii.gz')
            out_evec : string, optional
                Name of the eigenvectors volume to be saved
                (default 'evecs.nii.gz')
            out_eval : string, optional
                Name of the eigenvalues to be saved (default 'evals.nii.gz')
            """
        self.sigma = sigma
        self.jacobian = jacobian

        super(ReconstDtiRestoreFlow, self).\
            run(input_files, bvalues, bvectors, mask_files, b0_threshold,
                save_metrics, out_dir, out_tensor, out_fa, out_ga, out_rgb,
                out_md, out_ad, out_rd, out_mode, out_evec, out_eval)


class ReconstCSDFlow(Workflow):

    @classmethod
    def get_short_name(cls):
        return 'csd'

    def run(self, input_files, bvalues, bvectors, mask_files,
            b0_threshold=0.0,
            bvecs_tol=0.01,
            roi_center=None,
            roi_radius=10,
            fa_thr=0.7,
            frf=None, extract_pam_values=False,
            sh_order=8,
            odf_to_sh_order=8,
            out_dir='',
            out_pam='peaks.pam5', out_shm='shm.nii.gz',
            out_peaks_dir='peaks_dirs.nii.gz',
            out_peaks_values='peaks_values.nii.gz',
            out_peaks_indices='peaks_indices.nii.gz', out_gfa='gfa.nii.gz'):
        """ Constrained spherical deconvolution

        Parameters
        ----------
        input_files : string
            Path to the input volumes. This path may contain wildcards to
            process multiple inputs at once.
        bvalues : string
            Path to the bvalues files. This path may contain wildcards to use
            multiple bvalues files at once.
        bvectors : string
            Path to the bvectors files. This path may contain wildcards to use
            multiple bvectors files at once.
        mask_files : string
            Path to the input masks. This path may contain wildcards to use
            multiple masks at once. (default: No mask used)
        b0_threshold : float, optional
            Threshold used to find b=0 directions
        bvecs_tol : float, optional
            Bvecs should be unit vectors. (default:0.01)
        roi_center : variable int, optional
            Center of ROI in data. If center is None, it is assumed that it is
            the center of the volume with shape `data.shape[:3]` (default None)
        roi_radius : int, optional
            radius of cubic ROI in voxels (default 10)
        fa_thr : float, optional
            FA threshold for calculating the response function (default 0.7)
        frf : variable float, optional
            Fiber response function can be for example inputed as 15 4 4
            (from the command line) or [15, 4, 4] from a Python script to be
            converted to float and mutiplied by 10**-4 . If None
            the fiber response function will be computed automatically
            (default: None).
        extract_pam_values : bool, optional
            Save or not to save pam volumes as single nifti files.
        sh_order : int, optional
            Spherical harmonics order (default 6) used in the CSA fit.
        odf_to_sh_order : int, optional
            Spherical harmonics order used for peak_from_model to compress
            the ODF to spherical harmonics coefficients (default 8)
        out_dir : string, optional
            Output directory (default input file directory)
        out_pam : string, optional
            Name of the peaks volume to be saved (default 'peaks.pam5')
        out_shm : string, optional
            Name of the shperical harmonics volume to be saved
            (default 'shm.nii.gz')
        out_peaks_dir : string, optional
            Name of the peaks directions volume to be saved
            (default 'peaks_dirs.nii.gz')
        out_peaks_values : string, optional
            Name of the peaks values volume to be saved
            (default 'peaks_values.nii.gz')
        out_peaks_indices : string, optional
            Name of the peaks indices volume to be saved
            (default 'peaks_indices.nii.gz')
        out_gfa : string, optional
            Name of the generalise fa volume to be saved (default 'gfa.nii.gz')


        References
        ----------
        .. [1] Tournier, J.D., et al. NeuroImage 2007. Robust determination of
           the fibre orientation distribution in diffusion MRI: Non-negativity
           constrained super-resolved spherical deconvolution.
        """
        io_it = self.get_io_iterator()

        for (dwi, bval, bvec, maskfile, opam, oshm, opeaks_dir, opeaks_values,
             opeaks_indices, ogfa) in io_it:

            logging.info('Loading {0}'.format(dwi))
            img = nib.load(dwi)
            data = img.get_data()
            affine = img.affine

            bvals, bvecs = read_bvals_bvecs(bval, bvec)
            gtab = gradient_table(bvals, bvecs, b0_threshold=b0_threshold,
                                  atol=bvecs_tol)
            mask_vol = nib.load(maskfile).get_data().astype(np.bool)

            sh_order = 8
            if data.shape[-1] < 15:
                raise ValueError(
                    'You need at least 15 unique DWI volumes to '
                    'compute fiber odfs. You currently have: {0}'
                    ' DWI volumes.'.format(data.shape[-1]))
            elif data.shape[-1] < 30:
                sh_order = 6

            if frf is None:
                logging.info('Computing response function')
                if roi_center is not None:
                    logging.info('Response ROI center:\n{0}'
                                 .format(roi_center))
                    logging.info('Response ROI radius:\n{0}'
                                 .format(roi_radius))
                response, ratio, nvox = auto_response(
                        gtab, data,
                        roi_center=roi_center,
                        roi_radius=roi_radius,
                        fa_thr=fa_thr,
                        return_number_of_voxels=True)
                response = list(response)

            else:
                logging.info('Using response function')
                if isinstance(frf, str):
                    l01 = np.array(literal_eval(frf), dtype=np.float64)
                else:
                    l01 = np.array(frf, dtype=np.float64)

                l01 *= 10 ** -4
                response = np.array([l01[0], l01[1], l01[1]])
                ratio = l01[1] / l01[0]
                response = (response, ratio)

            logging.info(
                'Eigenvalues for the response of the input data are:\n{0}'
                .format(response[0]))

            peaks_sphere = get_sphere('repulsion724')

            logging.info('CSD computation started.')
            csd_model = ConstrainedSphericalDeconvModel(gtab, response,
                                                        sh_order=sh_order)

            peaks_csd = peaks_from_model(model=csd_model,
                                         data=data,
                                         sphere=peaks_sphere,
                                         relative_peak_threshold=.5,
                                         min_separation_angle=25,
                                         mask=mask_vol,
                                         return_sh=True,
                                         sh_order=sh_order,
                                         normalize_peaks=True,
                                         parallel=False)
            peaks_csd.affine = affine

            save_peaks(opam, peaks_csd)

            logging.info('CSD computation completed.')

            if extract_pam_values:
                peaks_to_niftis(peaks_csd, oshm, opeaks_dir, opeaks_values,
                                opeaks_indices, ogfa, reshape_dirs=True)

            dname_ = os.path.dirname(opam)
            if dname_ == '':
                logging.info('Pam5 file saved in current directory')
            else:
                logging.info(
                        'Pam5 file saved in {0}'.format(dname_))

            return io_it


class ReconstCSAFlow(Workflow):

    @classmethod
    def get_short_name(cls):
        return 'csa'

    def run(self, input_files, bvalues, bvectors, mask_files, sh_order=6,
            odf_to_sh_order=8, b0_threshold=0.0, bvecs_tol=0.01,
            extract_pam_values=False,
            out_dir='',
            out_pam='peaks.pam5', out_shm='shm.nii.gz',
            out_peaks_dir='peaks_dirs.nii.gz',
            out_peaks_values='peaks_values.nii.gz',
            out_peaks_indices='peaks_indices.nii.gz',
            out_gfa='gfa.nii.gz'):
        """ Constant Solid Angle.

        Parameters
        ----------
        input_files : string
            Path to the input volumes. This path may contain wildcards to
            process multiple inputs at once.
        bvalues : string
            Path to the bvalues files. This path may contain wildcards to use
            multiple bvalues files at once.
        bvectors : string
            Path to the bvectors files. This path may contain wildcards to use
            multiple bvectors files at once.
        mask_files : string
            Path to the input masks. This path may contain wildcards to use
            multiple masks at once. (default: No mask used)
        sh_order : int, optional
            Spherical harmonics order (default 6) used in the CSA fit.
        odf_to_sh_order : int, optional
            Spherical harmonics order used for peak_from_model to compress
            the ODF to spherical harmonics coefficients (default 8)
        b0_threshold : float, optional
            Threshold used to find b=0 directions
        bvecs_tol : float, optional
            Threshold used so that norm(bvec)=1 (default 0.01)
        extract_pam_values : bool, optional
            Wheter or not to save pam volumes as single nifti files.
        out_dir : string, optional
            Output directory (default input file directory)
        out_pam : string, optional
            Name of the peaks volume to be saved (default 'peaks.pam5')
        out_shm : string, optional
            Name of the shperical harmonics volume to be saved
            (default 'shm.nii.gz')
        out_peaks_dir : string, optional
            Name of the peaks directions volume to be saved
            (default 'peaks_dirs.nii.gz')
        out_peaks_values : string, optional
            Name of the peaks values volume to be saved
            (default 'peaks_values.nii.gz')
        out_peaks_indices : string, optional
            Name of the peaks indices volume to be saved
            (default 'peaks_indices.nii.gz')
        out_gfa : string, optional
            Name of the generalise fa volume to be saved (default 'gfa.nii.gz')


        References
        ----------
        .. [1] Aganj, I., et. al. 2009. ODF Reconstruction in Q-Ball Imaging
           with Solid Angle Consideration.
        """
        io_it = self.get_io_iterator()

        for (dwi, bval, bvec, maskfile, opam, oshm, opeaks_dir,
             opeaks_values, opeaks_indices, ogfa) in io_it:

            logging.info('Loading {0}'.format(dwi))
            vol = nib.load(dwi)
            data = vol.get_data()
            affine = vol.affine

            bvals, bvecs = read_bvals_bvecs(bval, bvec)
            gtab = gradient_table(bvals, bvecs,
                                  b0_threshold=b0_threshold, atol=bvecs_tol)
            mask_vol = nib.load(maskfile).get_data().astype(np.bool)

            peaks_sphere = get_sphere('repulsion724')

            logging.info('Starting CSA computations {0}'.format(dwi))

            csa_model = CsaOdfModel(gtab, sh_order)

            peaks_csa = peaks_from_model(model=csa_model,
                                         data=data,
                                         sphere=peaks_sphere,
                                         relative_peak_threshold=.5,
                                         min_separation_angle=25,
                                         mask=mask_vol,
                                         return_sh=True,
                                         sh_order=odf_to_sh_order,
                                         normalize_peaks=True,
                                         parallel=False)
            peaks_csa.affine = affine

            save_peaks(opam, peaks_csa)

            logging.info('Finished CSA {0}'.format(dwi))

            if extract_pam_values:
                peaks_to_niftis(peaks_csa, oshm, opeaks_dir,
                                opeaks_values,
                                opeaks_indices, ogfa, reshape_dirs=True)

            dname_ = os.path.dirname(opam)
            if dname_ == '':
                logging.info('Pam5 file saved in current directory')
            else:
                logging.info(
                        'Pam5 file saved in {0}'.format(dname_))

            return io_it


class ReconstDkiFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return 'dki'

    def run(self, input_files, bvalues, bvectors, mask_files, b0_threshold=0.0,
            save_metrics=[],
            out_dir='', out_dt_tensor='dti_tensors.nii.gz', out_fa='fa.nii.gz',
            out_ga='ga.nii.gz', out_rgb='rgb.nii.gz', out_md='md.nii.gz',
            out_ad='ad.nii.gz', out_rd='rd.nii.gz', out_mode='mode.nii.gz',
            out_evec='evecs.nii.gz', out_eval='evals.nii.gz',
            out_dk_tensor="dki_tensors.nii.gz",
            out_mk="mk.nii.gz", out_ak="ak.nii.gz", out_rk="rk.nii.gz"):
        """ Workflow for Diffusion Kurtosis reconstruction and for computing
        DKI metrics. Performs a DKI reconstruction on the files by 'globing'
        ``input_files`` and saves the DTI metrics in a directory specified by
        ``out_dir``.

        Parameters
        ----------
        input_files : string
            Path to the input volumes. This path may contain wildcards to
            process multiple inputs at once.
        bvalues : string
            Path to the bvalues files. This path may contain wildcards to use
            multiple bvalues files at once.
        bvectors : string
            Path to the bvalues files. This path may contain wildcards to use
            multiple bvalues files at once.
        mask_files : string
            Path to the input masks. This path may contain wildcards to use
            multiple masks at once. (default: No mask used)
        b0_threshold : float, optional
            Threshold used to find b=0 directions (default 0.0)
        save_metrics : variable string, optional
            List of metrics to save.
            Possible values: fa, ga, rgb, md, ad, rd, mode, tensor, evec, eval
            (default [] (all))
        out_dir : string, optional
            Output directory (default input file directory)
        out_dt_tensor : string, optional
            Name of the tensors volume to be saved
            (default: 'dti_tensors.nii.gz')
        out_dk_tensor : string, optional
            Name of the tensors volume to be saved
            (default 'dki_tensors.nii.gz')
        out_fa : string, optional
            Name of the fractional anisotropy volume to be saved
            (default 'fa.nii.gz')
        out_ga : string, optional
            Name of the geodesic anisotropy volume to be saved
            (default 'ga.nii.gz')
        out_rgb : string, optional
            Name of the color fa volume to be saved (default 'rgb.nii.gz')
        out_md : string, optional
            Name of the mean diffusivity volume to be saved
            (default 'md.nii.gz')
        out_ad : string, optional
            Name of the axial diffusivity volume to be saved
            (default 'ad.nii.gz')
        out_rd : string, optional
            Name of the radial diffusivity volume to be saved
            (default 'rd.nii.gz')
        out_mode : string, optional
            Name of the mode volume to be saved (default 'mode.nii.gz')
        out_evec : string, optional
            Name of the eigenvectors volume to be saved
            (default 'evecs.nii.gz')
        out_eval : string, optional
            Name of the eigenvalues to be saved (default 'evals.nii.gz')
        out_mk : string, optional
            Name of the mean kurtosis to be saved (default: 'mk.nii.gz')
        out_ak : string, optional
            Name of the axial kurtosis to be saved (default: 'ak.nii.gz')
        out_rk : string, optional
            Name of the radial kurtosis to be saved (default: 'rk.nii.gz')

        References
        ----------

        .. [1] Tabesh, A., Jensen, J.H., Ardekani, B.A., Helpern, J.A., 2011.
           Estimation of tensors and tensor-derived measures in diffusional
           kurtosis imaging. Magn Reson Med. 65(3), 823-836

        .. [2] Jensen, Jens H., Joseph A. Helpern, Anita Ramani, Hanzhang Lu,
           and Kyle Kaczynski. 2005. Diffusional Kurtosis Imaging: The
           Quantification of Non-Gaussian Water Diffusion by Means of Magnetic
           Resonance Imaging. MRM 53 (6):1432-40.
        """
        io_it = self.get_io_iterator()

        for (dwi, bval, bvec, mask, otensor, ofa, oga, orgb, omd, oad, orad,
             omode, oevecs, oevals, odk_tensor, omk, oak, ork) in io_it:

            logging.info('Computing DKI metrics for {0}'.format(dwi))
            img = nib.load(dwi)
            data = img.get_data()
            affine = img.affine

            if mask is not None:
                mask = nib.load(mask).get_data().astype(np.bool)

            dkfit, _ = self.get_fitted_tensor(data, mask, bval, bvec,
                                              b0_threshold)

            if not save_metrics:
                save_metrics = ['mk', 'rk', 'ak', 'fa', 'md', 'rd', 'ad', 'ga',
                                'rgb', 'mode', 'evec', 'eval', 'dt_tensor',
                                'dk_tensor']

            evals, evecs, kt = split_dki_param(dkfit.model_params)
            FA = fractional_anisotropy(evals)
            FA[np.isnan(FA)] = 0
            FA = np.clip(FA, 0, 1)

            if 'dt_tensor' in save_metrics:
                tensor_vals = lower_triangular(dkfit.quadratic_form)
                correct_order = [0, 1, 3, 2, 4, 5]
                tensor_vals_reordered = tensor_vals[..., correct_order]
                fiber_tensors = nib.Nifti1Image(tensor_vals_reordered.astype(
                    np.float32), affine)
                nib.save(fiber_tensors, otensor)

            if 'dk_tensor' in save_metrics:
                kt_img = nib.Nifti1Image(dkfit.kt.astype(np.float32), affine)
                nib.save(kt_img, odk_tensor)

            if 'fa' in save_metrics:
                fa_img = nib.Nifti1Image(FA.astype(np.float32), affine)
                nib.save(fa_img, ofa)

            if 'ga' in save_metrics:
                GA = geodesic_anisotropy(dkfit.evals)
                ga_img = nib.Nifti1Image(GA.astype(np.float32), affine)
                nib.save(ga_img, oga)

            if 'rgb' in save_metrics:
                RGB = color_fa(FA, dkfit.evecs)
                rgb_img = nib.Nifti1Image(np.array(255 * RGB, 'uint8'), affine)
                nib.save(rgb_img, orgb)

            if 'md' in save_metrics:
                MD = mean_diffusivity(dkfit.evals)
                md_img = nib.Nifti1Image(MD.astype(np.float32), affine)
                nib.save(md_img, omd)

            if 'ad' in save_metrics:
                AD = axial_diffusivity(dkfit.evals)
                ad_img = nib.Nifti1Image(AD.astype(np.float32), affine)
                nib.save(ad_img, oad)

            if 'rd' in save_metrics:
                RD = radial_diffusivity(dkfit.evals)
                rd_img = nib.Nifti1Image(RD.astype(np.float32), affine)
                nib.save(rd_img, orad)

            if 'mode' in save_metrics:
                MODE = get_mode(dkfit.quadratic_form)
                mode_img = nib.Nifti1Image(MODE.astype(np.float32), affine)
                nib.save(mode_img, omode)

            if 'evec' in save_metrics:
                evecs_img = nib.Nifti1Image(dkfit.evecs.astype(np.float32),
                                            affine)
                nib.save(evecs_img, oevecs)

            if 'eval' in save_metrics:
                evals_img = nib.Nifti1Image(dkfit.evals.astype(np.float32),
                                            affine)
                nib.save(evals_img, oevals)

            if 'mk' in save_metrics:
                mk_img = nib.Nifti1Image(dkfit.mk().astype(np.float32),
                                         affine)
                nib.save(mk_img, omk)

            if 'ak' in save_metrics:
                ak_img = nib.Nifti1Image(dkfit.ak().astype(np.float32),
                                         affine)
                nib.save(ak_img, oak)

            if 'rk' in save_metrics:
                rk_img = nib.Nifti1Image(dkfit.rk().astype(np.float32),
                                         affine)
                nib.save(rk_img, ork)

            logging.info('DKI metrics saved in {0}'.
                         format(os.path.dirname(oevals)))

    def get_dki_model(self, gtab):
        return DiffusionKurtosisModel(gtab)

    def get_fitted_tensor(self, data, mask, bval, bvec, b0_threshold=0):
        logging.info('Diffusion kurtosis estimation...')
        bvals, bvecs = read_bvals_bvecs(bval, bvec)
        gtab = gradient_table(bvals, bvecs, b0_threshold=b0_threshold)

        dkmodel = self.get_dki_model(gtab)
        dkfit = dkmodel.fit(data, mask)

        return dkfit, gtab
