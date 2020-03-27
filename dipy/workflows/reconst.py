
import logging
import numpy as np
import os.path
from ast import literal_eval
from warnings import warn

import nibabel as nib

from dipy.core.gradients import gradient_table
from dipy.data import default_sphere
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.peaks import save_peaks, peaks_to_niftis
from dipy.io.image import load_nifti, save_nifti, load_nifti_data
from dipy.io.utils import nifti1_symmat
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response)
from dipy.reconst.dti import (TensorModel, color_fa, fractional_anisotropy,
                              geodesic_anisotropy, mean_diffusivity,
                              axial_diffusivity, radial_diffusivity,
                              lower_triangular, mode as get_mode)
from dipy.direction.peaks import peaks_from_model
from dipy.reconst.shm import CsaOdfModel
from dipy.workflows.workflow import Workflow
from dipy.reconst.dki import DiffusionKurtosisModel, split_dki_param
from dipy.reconst.ivim import IvimModel

from dipy.reconst import mapmri


class ReconstMAPMRIFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return 'mapmri'

    def run(self, data_files, bvals_files, bvecs_files, small_delta, big_delta,
            b0_threshold=50.0, laplacian=True, positivity=True,
            bval_threshold=2000, save_metrics=[],
            laplacian_weighting=0.05, radial_order=6, out_dir='',
            out_rtop='rtop.nii.gz', out_lapnorm='lapnorm.nii.gz',
            out_msd='msd.nii.gz', out_qiv='qiv.nii.gz',
            out_rtap='rtap.nii.gz',
            out_rtpp='rtpp.nii.gz', out_ng='ng.nii.gz',
            out_perng='perng.nii.gz',
            out_parng='parng.nii.gz'):
        """ Workflow for fitting the MAPMRI model (with optional Laplacian
        regularization). Generates rtop, lapnorm, msd, qiv, rtap, rtpp,
        non-gaussian (ng), parallel ng, perpendicular ng saved in a nifti
        format in input files provided by `data_files` and saves the nifti
        files to an output directory specified by `out_dir`.

        In order for the MAPMRI workflow to work in the way
        intended either the Laplacian or positivity or both must
        be set to True.

        Parameters
        ----------
        data_files : string
            Path to the input volume.
        bvals_files : string
            Path to the bval files.
        bvecs_files : string
            Path to the bvec files.
        small_delta : float
            Small delta value used in generation of gradient table of provided
            bval and bvec.
        big_delta : float
            Big delta value used in generation of gradient table of provided
            bval and bvec.
        b0_threshold : float, optional
            Threshold used to find b=0 directions (default 0.0)
        laplacian : bool, optional
            Regularize using the Laplacian of the MAP-MRI basis (default True)
        positivity : bool, optional
            Constrain the propagator to be positive. (default True)
        bval_threshold : float, optional
            Sets the b-value threshold to be used in the scale factor
            estimation. In order for the estimated non-Gaussianity to have
            meaning this value should set to a lower value (b<2000 s/mm^2)
            such that the scale factors are estimated on signal points that
            reasonably represent the spins at Gaussian diffusion.
            (default: 2000)
        save_metrics : variable string, optional
            List of metrics to save.
            Possible values: rtop, laplacian_signal, msd, qiv, rtap, rtpp,
            ng, perng, parng
            (default: [] (all))
        laplacian_weighting : float, optional
            Weighting value used in fitting the MAPMRI model in the Laplacian
            and both model types. (default: 0.05)
        radial_order : unsigned int, optional
            Even value used to set the order of the basis
            (default: 6)
        out_dir : string, optional
            Output directory (default: input file directory)
        out_rtop : string, optional
            Name of the rtop to be saved
        out_lapnorm : string, optional
            Name of the norm of Laplacian signal to be saved
        out_msd : string, optional
            Name of the msd to be saved
        out_qiv : string, optional
            Name of the qiv to be saved
        out_rtap : string, optional
            Name of the rtap to be saved
        out_rtpp : string, optional
            Name of the rtpp to be saved
        out_ng : string, optional
            Name of the Non-Gaussianity to be saved
        out_perng :  string, optional
            Name of the Non-Gaussianity perpendicular to be saved
        out_parng : string, optional
            Name of the Non-Gaussianity parallel to be saved
        """
        io_it = self.get_io_iterator()
        for (dwi, bval, bvec, out_rtop, out_lapnorm, out_msd, out_qiv,
             out_rtap, out_rtpp, out_ng, out_perng, out_parng) in io_it:

            logging.info('Computing MAPMRI metrics for {0}'.format(dwi))
            data, affine = load_nifti(dwi)

            bvals, bvecs = read_bvals_bvecs(bval, bvec)
            if b0_threshold < bvals.min():
                warn("b0_threshold (value: {0}) is too low, increase your "
                     "b0_threshold. It should be higher than the first b0 value "
                     "({1}).".format(b0_threshold, bvals.min()))
            gtab = gradient_table(bvals=bvals, bvecs=bvecs,
                                  small_delta=small_delta,
                                  big_delta=big_delta,
                                  b0_threshold=b0_threshold)

            if not save_metrics:
                save_metrics = ['rtop', 'laplacian_signal', 'msd',
                                'qiv', 'rtap', 'rtpp',
                                'ng', 'perng', 'parng']

            if laplacian and positivity:
                map_model_aniso = mapmri.MapmriModel(
                            gtab,
                            radial_order=radial_order,
                            laplacian_regularization=True,
                            laplacian_weighting=laplacian_weighting,
                            positivity_constraint=True,
                            bval_threshold=bval_threshold)

                mapfit_aniso = map_model_aniso.fit(data)

            elif positivity:
                map_model_aniso = mapmri.MapmriModel(
                            gtab,
                            radial_order=radial_order,
                            laplacian_regularization=False,
                            positivity_constraint=True,
                            bval_threshold=bval_threshold)
                mapfit_aniso = map_model_aniso.fit(data)

            elif laplacian:
                map_model_aniso = mapmri.MapmriModel(
                            gtab,
                            radial_order=radial_order,
                            laplacian_regularization=True,
                            laplacian_weighting=laplacian_weighting,
                            bval_threshold=bval_threshold)
                mapfit_aniso = map_model_aniso.fit(data)

            else:
                map_model_aniso = mapmri.MapmriModel(
                            gtab,
                            radial_order=radial_order,
                            laplacian_regularization=False,
                            positivity_constraint=False,
                            bval_threshold=bval_threshold)
                mapfit_aniso = map_model_aniso.fit(data)

            if 'rtop' in save_metrics:
                r = mapfit_aniso.rtop()
                save_nifti(out_rtop, r.astype(np.float32), affine)

            if 'laplacian_signal' in save_metrics:
                ll = mapfit_aniso.norm_of_laplacian_signal()
                save_nifti(out_lapnorm, ll.astype(np.float32), affine)

            if 'msd' in save_metrics:
                m = mapfit_aniso.msd()
                save_nifti(out_msd, m.astype(np.float32), affine)

            if 'qiv' in save_metrics:
                q = mapfit_aniso.qiv()
                save_nifti(out_qiv, q.astype(np.float32), affine)

            if 'rtap' in save_metrics:
                r = mapfit_aniso.rtap()
                save_nifti(out_rtap, r.astype(np.float32), affine)

            if 'rtpp' in save_metrics:
                r = mapfit_aniso.rtpp()
                save_nifti(out_rtpp, r.astype(np.float32), affine)

            if 'ng' in save_metrics:
                n = mapfit_aniso.ng()
                save_nifti(out_ng, n.astype(np.float32), affine)

            if 'perng' in save_metrics:
                n = mapfit_aniso.ng_perpendicular()
                save_nifti(out_perng, n.astype(np.float32), affine)

            if 'parng' in save_metrics:
                n = mapfit_aniso.ng_parallel()
                save_nifti(out_parng, n.astype(np.float32), affine)

            logging.info('MAPMRI saved in {0}'.
                         format(os.path.dirname(out_dir)))


class ReconstDtiFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return 'dti'

    def run(self, input_files, bvalues_files, bvectors_files, mask_files,
            b0_threshold=50, bvecs_tol=0.01, save_metrics=[],
            out_dir='', out_tensor='tensors.nii.gz', out_fa='fa.nii.gz',
            out_ga='ga.nii.gz', out_rgb='rgb.nii.gz', out_md='md.nii.gz',
            out_ad='ad.nii.gz', out_rd='rd.nii.gz', out_mode='mode.nii.gz',
            out_evec='evecs.nii.gz', out_eval='evals.nii.gz', nifti_tensor=True):
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
        bvalues_files : string
            Path to the bvalues files. This path may contain wildcards to use
            multiple bvalues files at once.
        bvectors_files : string
            Path to the bvectors files. This path may contain wildcards to use
            multiple bvectors files at once.
        mask_files : string
            Path to the input masks. This path may contain wildcards to use
            multiple masks at once.
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
            Name of the tensors volume to be saved (default 'tensors.nii.gz').
            Per default, this will be saved following the nifti standard:
            with the tensor elements as Dxx, Dxy, Dyy, Dxz, Dyz, Dzz on the
            last (5th) dimension of the volume (shape: (i, j, k, 1, 6)). If
            `nifti_tensor` is False, this will be saved in an alternate format
            that is used by other software (e.g., FSL): a
            4-dimensional volume (shape (i, j, k, 6)) with Dxx, Dxy, Dxz, Dyy,
            Dyz, Dzz on the last dimension.
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
        nifti_tensor : bool, optional
            Whether the tensor is saved in the standard Nifti format or in an
            alternate format
            that is used by other software (e.g., FSL): a
            4-dimensional volume (shape (i, j, k, 6)) with
            Dxx, Dxy, Dxz, Dyy, Dyz, Dzz on the last dimension. Default: True

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
            data, affine = load_nifti(dwi)

            if mask is not None:
                mask = load_nifti_data(mask).astype(np.bool)

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

                if nifti_tensor:
                    ten_img = nifti1_symmat(tensor_vals, affine=affine)
                else:
                    alt_order = [0, 1, 3, 2, 4, 5]
                    ten_img = nib.Nifti1Image(
                            tensor_vals[..., alt_order].astype(np.float32),
                            affine)

                nib.save(ten_img, otensor)

            if 'fa' in save_metrics:
                save_nifti(ofa, FA.astype(np.float32), affine)

            if 'ga' in save_metrics:
                GA = geodesic_anisotropy(tenfit.evals)
                save_nifti(oga, GA.astype(np.float32), affine)

            if 'rgb' in save_metrics:
                RGB = color_fa(FA, tenfit.evecs)
                save_nifti(orgb, np.array(255 * RGB, 'uint8'), affine)

            if 'md' in save_metrics:
                MD = mean_diffusivity(tenfit.evals)
                save_nifti(omd, MD.astype(np.float32), affine)

            if 'ad' in save_metrics:
                AD = axial_diffusivity(tenfit.evals)
                save_nifti(oad, AD.astype(np.float32), affine)

            if 'rd' in save_metrics:
                RD = radial_diffusivity(tenfit.evals)
                save_nifti(orad, RD.astype(np.float32), affine)

            if 'mode' in save_metrics:
                MODE = get_mode(tenfit.quadratic_form)
                save_nifti(omode, MODE.astype(np.float32), affine)

            if 'evec' in save_metrics:
                save_nifti(oevecs, tenfit.evecs.astype(np.float32), affine)

            if 'eval' in save_metrics:
                save_nifti(oevals, tenfit.evals.astype(np.float32), affine)

            dname_ = os.path.dirname(oevals)
            if dname_ == '':
                logging.info('DTI metrics saved in current directory')
            else:
                logging.info(
                        'DTI metrics saved in {0}'.format(dname_))

    def get_tensor_model(self, gtab):
        return TensorModel(gtab, fit_method="WLS")

    def get_fitted_tensor(self, data, mask, bval, bvec,
                          b0_threshold=50, bvecs_tol=0.01):

        logging.info('Tensor estimation...')
        bvals, bvecs = read_bvals_bvecs(bval, bvec)
        gtab = gradient_table(bvals, bvecs, b0_threshold=b0_threshold,
                              atol=bvecs_tol)

        tenmodel = self.get_tensor_model(gtab)
        tenfit = tenmodel.fit(data, mask)

        return tenfit, gtab


class ReconstCSDFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return 'csd'

    def run(self, input_files, bvalues_files, bvectors_files, mask_files,
            b0_threshold=50.0, bvecs_tol=0.01, roi_center=None, roi_radius=10,
            fa_thr=0.7, frf=None, extract_pam_values=False, sh_order=8,
            odf_to_sh_order=8, parallel=False, nbr_processes=None,
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
        bvalues_files : string
            Path to the bvalues files. This path may contain wildcards to use
            multiple bvalues files at once.
        bvectors_files : string
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
            converted to float and multiplied by 10**-4 . If None
            the fiber response function will be computed automatically
            (default: None).
        extract_pam_values : bool, optional
            Save or not to save pam volumes as single nifti files.
        sh_order : int, optional
            Spherical harmonics order (default 6) used in the CSA fit.
        odf_to_sh_order : int, optional
            Spherical harmonics order used for peak_from_model to compress
            the ODF to spherical harmonics coefficients (default 8)
        parallel : bool, optional
            Whether to use parallelization in peak-finding during the
            calibration procedure. Default: False
        nbr_processes : int, optional
            If `parallel` is True, the number of subprocesses to use
            (default multiprocessing.cpu_count()).
        out_dir : string, optional
            Output directory (default input file directory)
        out_pam : string, optional
            Name of the peaks volume to be saved (default 'peaks.pam5')
        out_shm : string, optional
            Name of the spherical harmonics volume to be saved
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
            Name of the generalized FA volume to be saved (default 'gfa.nii.gz')


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
            data, affine = load_nifti(dwi)

            bvals, bvecs = read_bvals_bvecs(bval, bvec)
            print(b0_threshold, bvals.min())
            if b0_threshold < bvals.min():
                warn("b0_threshold (value: {0}) is too low, increase your "
                     "b0_threshold. It should be higher than the first b0 value "
                     "({1}).".format(b0_threshold, bvals.min()))
            gtab = gradient_table(bvals, bvecs, b0_threshold=b0_threshold,
                                  atol=bvecs_tol)
            mask_vol = load_nifti_data(maskfile).astype(np.bool)

            n_params = ((sh_order + 1) * (sh_order + 2)) / 2
            if data.shape[-1] < n_params:
                raise ValueError(
                    'You need at least {0} unique DWI volumes to '
                    'compute fiber odfs. You currently have: {1}'
                    ' DWI volumes.'.format(n_params, data.shape[-1]))

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

            logging.info("Eigenvalues for the frf of the input"
                         " data are :{0}".format(response[0]))
            logging.info('Ratio for smallest to largest eigen value is {0}'
                         .format(ratio))

            peaks_sphere = default_sphere

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
                                         parallel=parallel,
                                         nbr_processes=nbr_processes)
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

    def run(self, input_files, bvalues_files, bvectors_files, mask_files,
            sh_order=6, odf_to_sh_order=8, b0_threshold=50.0, bvecs_tol=0.01,
            extract_pam_values=False, parallel=False, nbr_processes=None,
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
        bvalues_files : string
            Path to the bvalues files. This path may contain wildcards to use
            multiple bvalues files at once.
        bvectors_files : string
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
        parallel : bool, optional
            Whether to use parallelization in peak-finding during the
            calibration procedure. Default: False
        nbr_processes : int, optional
            If `parallel` is True, the number of subprocesses to use
            (default multiprocessing.cpu_count()).
        out_dir : string, optional
            Output directory (default input file directory)
        out_pam : string, optional
            Name of the peaks volume to be saved (default 'peaks.pam5')
        out_shm : string, optional
            Name of the spherical harmonics volume to be saved
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
            Name of the generalized FA volume to be saved (default 'gfa.nii.gz')

        References
        ----------
        .. [1] Aganj, I., et al. 2009. ODF Reconstruction in Q-Ball Imaging
           with Solid Angle Consideration.
        """
        io_it = self.get_io_iterator()

        for (dwi, bval, bvec, maskfile, opam, oshm, opeaks_dir,
             opeaks_values, opeaks_indices, ogfa) in io_it:

            logging.info('Loading {0}'.format(dwi))
            data, affine = load_nifti(dwi)

            bvals, bvecs = read_bvals_bvecs(bval, bvec)
            if b0_threshold < bvals.min():
                warn("b0_threshold (value: {0}) is too low, increase your "
                     "b0_threshold. It should be higher than the first b0 value "
                     "({1}).".format(b0_threshold, bvals.min()))
            gtab = gradient_table(bvals, bvecs,
                                  b0_threshold=b0_threshold, atol=bvecs_tol)
            mask_vol = load_nifti_data(maskfile).astype(np.bool)

            peaks_sphere = default_sphere

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
                                         parallel=parallel,
                                         nbr_processes=nbr_processes)
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

    def run(self, input_files, bvalues_files, bvectors_files, mask_files,
            b0_threshold=50.0, save_metrics=[],
            out_dir='', out_dt_tensor='dti_tensors.nii.gz', out_fa='fa.nii.gz',
            out_ga='ga.nii.gz', out_rgb='rgb.nii.gz', out_md='md.nii.gz',
            out_ad='ad.nii.gz', out_rd='rd.nii.gz', out_mode='mode.nii.gz',
            out_evec='evecs.nii.gz', out_eval='evals.nii.gz',
            out_dk_tensor="dki_tensors.nii.gz",
            out_mk="mk.nii.gz", out_ak="ak.nii.gz", out_rk="rk.nii.gz"):
        """ Workflow for Diffusion Kurtosis reconstruction and for computing
        DKI metrics. Performs a DKI reconstruction on the files by 'globing'
        ``input_files`` and saves the DKI metrics in a directory specified by
        ``out_dir``.

        Parameters
        ----------
        input_files : string
            Path to the input volumes. This path may contain wildcards to
            process multiple inputs at once.
        bvalues_files : string
            Path to the bvalues files. This path may contain wildcards to use
            multiple bvalues files at once.
        bvectors_files : string
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
            data, affine = load_nifti(dwi)

            if mask is not None:
                mask = load_nifti_data(mask).astype(np.bool)

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
                save_nifti(otensor, tensor_vals_reordered.astype(np.float32),
                           affine)

            if 'dk_tensor' in save_metrics:
                save_nifti(odk_tensor, dkfit.kt.astype(np.float32), affine)

            if 'fa' in save_metrics:
                save_nifti(ofa, FA.astype(np.float32), affine)

            if 'ga' in save_metrics:
                GA = geodesic_anisotropy(dkfit.evals)
                save_nifti(oga, GA.astype(np.float32), affine)

            if 'rgb' in save_metrics:
                RGB = color_fa(FA, dkfit.evecs)
                save_nifti(orgb, np.array(255 * RGB, 'uint8'), affine)

            if 'md' in save_metrics:
                MD = mean_diffusivity(dkfit.evals)
                save_nifti(omd, MD.astype(np.float32), affine)

            if 'ad' in save_metrics:
                AD = axial_diffusivity(dkfit.evals)
                save_nifti(oad, AD.astype(np.float32), affine)

            if 'rd' in save_metrics:
                RD = radial_diffusivity(dkfit.evals)
                save_nifti(orad, RD.astype(np.float32), affine)

            if 'mode' in save_metrics:
                MODE = get_mode(dkfit.quadratic_form)
                save_nifti(omode, MODE.astype(np.float32), affine)

            if 'evec' in save_metrics:
                save_nifti(oevecs, dkfit.evecs.astype(np.float32), affine)

            if 'eval' in save_metrics:
                save_nifti(oevals, dkfit.evals.astype(np.float32), affine)

            if 'mk' in save_metrics:
                save_nifti(omk, dkfit.mk().astype(np.float32), affine)

            if 'ak' in save_metrics:
                save_nifti(oak, dkfit.ak().astype(np.float32), affine)

            if 'rk' in save_metrics:
                save_nifti(ork, dkfit.rk().astype(np.float32), affine)

            logging.info('DKI metrics saved in {0}'.
                         format(os.path.dirname(oevals)))

    def get_dki_model(self, gtab):
        return DiffusionKurtosisModel(gtab)

    def get_fitted_tensor(self, data, mask, bval, bvec, b0_threshold=50):
        logging.info('Diffusion kurtosis estimation...')
        bvals, bvecs = read_bvals_bvecs(bval, bvec)
        if b0_threshold < bvals.min():
            warn("b0_threshold (value: {0}) is too low, increase your "
                 "b0_threshold. It should be higher than the first b0 value "
                 "({1}).".format(b0_threshold, bvals.min()))

        gtab = gradient_table(bvals, bvecs, b0_threshold=b0_threshold)
        dkmodel = self.get_dki_model(gtab)
        dkfit = dkmodel.fit(data, mask)

        return dkfit, gtab


class ReconstIvimFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return 'ivim'

    def run(self, input_files, bvalues_files, bvectors_files, mask_files,
            split_b_D=400, split_b_S0=200, b0_threshold=0, save_metrics=[],
            out_dir='', out_S0_predicted='S0_predicted.nii.gz',
            out_perfusion_fraction='perfusion_fraction.nii.gz',
            out_D_star='D_star.nii.gz', out_D='D.nii.gz'):
        """ Workflow for Intra-voxel Incoherent Motion reconstruction and for
        computing IVIM metrics. Performs a IVIM reconstruction on the files
        by 'globing' ``input_files`` and saves the IVIM metrics in a directory
        specified by ``out_dir``.

        Parameters
        ----------
        input_files : string
            Path to the input volumes. This path may contain wildcards to
            process multiple inputs at once.
        bvalues_files : string
            Path to the bvalues files. This path may contain wildcards to use
            multiple bvalues files at once.
        bvectors_files : string
            Path to the bvalues files. This path may contain wildcards to use
            multiple bvalues files at once.
        mask_files : string
            Path to the input masks. This path may contain wildcards to use
            multiple masks at once. (default: No mask used)
        split_b_D : int, optional
            Value to split the bvals to estimate D for the two-stage process of
            fitting
            (default 400)
        split_b_S0 : int, optional
            Value to split the bvals to estimate S0 for the two-stage process
            of fitting.
            (default 200)
        b0_threshold : int, optional
            Threshold value for the b0 bval.
            (default 0)
        save_metrics : variable string, optional
            List of metrics to save.
            Possible values: S0_predicted, perfusion_fraction, D_star, D
            (default [] (all))
        out_dir : string, optional
            Output directory (default input file directory)
        out_S0_predicted : string, optional
            Name of the S0 signal estimated to be saved
            (default: 'S0_predicted.nii.gz')
        out_perfusion_fraction : string, optional
            Name of the estimated volume fractions to be saved
            (default 'perfusion_fraction.nii.gz')
        out_D_star : string, optional
            Name of the estimated pseudo-diffusion parameter to be saved
            (default 'D_star.nii.gz')
        out_D : string, optional
            Name of the estimated diffusion parameter to be saved
            (default 'D.nii.gz')

        References
        ----------

        .. [Stejskal65] Stejskal, E. O.; Tanner, J. E. (1 January 1965).
                        "Spin Diffusion Measurements: Spin Echoes in the
                        Presence of a Time-Dependent Field Gradient". The
                        Journal of Chemical Physics 42 (1): 288.
                        Bibcode: 1965JChPh..42..288S. doi:10.1063/1.1695690.

        .. [LeBihan84] Le Bihan, Denis, et al. "Separation of diffusion
                       and perfusion in intravoxel incoherent motion MR
                       imaging." Radiology 168.2 (1988): 497-505.
        """

        io_it = self.get_io_iterator()

        for (dwi, bval, bvec, mask, oS0_predicted, operfusion_fraction,
             oD_star, oD) in io_it:

            logging.info('Computing IVIM metrics for {0}'.format(dwi))
            data, affine = load_nifti(dwi)

            if mask is not None:
                mask = load_nifti_data(mask).astype(np.bool)

            ivimfit, _ = self.get_fitted_ivim(data, mask, bval, bvec,
                                              b0_threshold)

            if not save_metrics:
                save_metrics = ['S0_predicted', 'perfusion_fraction', 'D_star',
                                'D']

            if 'S0_predicted' in save_metrics:
                save_nifti(oS0_predicted,
                           ivimfit.S0_predicted.astype(np.float32), affine)

            if 'perfusion_fraction' in save_metrics:
                save_nifti(operfusion_fraction,
                           ivimfit.perfusion_fraction.astype(np.float32),
                           affine)

            if 'D_star' in save_metrics:
                save_nifti(oD_star, ivimfit.D_star.astype(np.float32), affine)

            if 'D' in save_metrics:
                save_nifti(oD, ivimfit.D.astype(np.float32), affine)

            logging.info('IVIM metrics saved in {0}'.
                         format(os.path.dirname(oD)))

    def get_fitted_ivim(self, data, mask, bval, bvec, b0_threshold=50):
        logging.info('Intra-Voxel Incoherent Motion Estimation...')
        bvals, bvecs = read_bvals_bvecs(bval, bvec)
        if b0_threshold < bvals.min():
            warn("b0_threshold (value: {0}) is too low, increase your "
                 "b0_threshold. It should be higher than the first b0 value "
                 "({1}).".format(b0_threshold, bvals.min()))

        gtab = gradient_table(bvals, bvecs, b0_threshold=b0_threshold)
        ivimmodel = IvimModel(gtab)
        ivimfit = ivimmodel.fit(data, mask)

        return ivimfit, gtab
