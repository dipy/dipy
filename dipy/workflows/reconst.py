from ast import literal_eval
import logging
import os.path
from warnings import warn

import nibabel as nib
import numpy as np

from dipy.core.gradients import gradient_table, mask_non_weighted_bvals
from dipy.core.ndindex import ndindex
from dipy.data import default_sphere, get_sphere
from dipy.direction.peaks import peak_directions, peaks_from_model
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, load_nifti_data, save_nifti
from dipy.io.peaks import niftis_to_pam, pam_to_niftis, save_pam, tensor_to_pam
from dipy.io.utils import nifti1_symmat
from dipy.reconst import mapmri
from dipy.reconst.csdeconv import (
    ConstrainedSDTModel,
    ConstrainedSphericalDeconvModel,
    auto_response_ssst,
)
from dipy.reconst.dki import DiffusionKurtosisModel, split_dki_param
from dipy.reconst.dsi import DiffusionSpectrumDeconvModel, DiffusionSpectrumModel
from dipy.reconst.dti import (
    TensorModel,
    axial_diffusivity,
    color_fa,
    fractional_anisotropy,
    geodesic_anisotropy,
    lower_triangular,
    mean_diffusivity,
    mode as get_mode,
    radial_diffusivity,
)
from dipy.reconst.forecast import ForecastModel
from dipy.reconst.gqi import GeneralizedQSamplingModel
from dipy.reconst.ivim import IvimModel
from dipy.reconst.rumba import RumbaSDModel
from dipy.reconst.sfm import SparseFascicleModel
from dipy.reconst.shm import CsaOdfModel, OpdtModel, QballModel
from dipy.testing.decorators import warning_for_keywords
from dipy.utils.deprecator import deprecated_params
from dipy.workflows.workflow import Workflow


class ReconstMAPMRIFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "mapmri"

    def run(
        self,
        data_files,
        bvals_files,
        bvecs_files,
        small_delta,
        big_delta,
        b0_threshold=50.0,
        laplacian=True,
        positivity=True,
        bval_threshold=2000,
        save_metrics=(),
        laplacian_weighting=0.05,
        radial_order=6,
        sphere_name=None,
        relative_peak_threshold=0.5,
        min_separation_angle=25,
        npeaks=5,
        normalize_peaks=False,
        extract_pam_values=False,
        out_dir="",
        out_rtop="rtop.nii.gz",
        out_lapnorm="lapnorm.nii.gz",
        out_msd="msd.nii.gz",
        out_qiv="qiv.nii.gz",
        out_rtap="rtap.nii.gz",
        out_rtpp="rtpp.nii.gz",
        out_ng="ng.nii.gz",
        out_perng="perng.nii.gz",
        out_parng="parng.nii.gz",
        out_pam="mapmri_peaks.pam5",
        out_peaks_dir="mapmri_peaks_dirs.nii.gz",
        out_peaks_values="mapmri_peaks_values.nii.gz",
        out_peaks_indices="mapmri_peaks_indices.nii.gz",
    ):
        """Workflow for fitting the MAPMRI model (with optional Laplacian
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
            Threshold used to find b0 volumes.
        laplacian : bool, optional
            Regularize using the Laplacian of the MAP-MRI basis.
        positivity : bool, optional
            Constrain the propagator to be positive.
        bval_threshold : float, optional
            Sets the b-value threshold to be used in the scale factor
            estimation. In order for the estimated non-Gaussianity to have
            meaning this value should set to a lower value (b<2000 s/mm^2)
            such that the scale factors are estimated on signal points that
            reasonably represent the spins at Gaussian diffusion.
        save_metrics : variable string, optional
            List of metrics to save.
            Possible values: rtop, laplacian_signal, msd, qiv, rtap, rtpp,
            ng, perng, parng
        laplacian_weighting : float, optional
            Weighting value used in fitting the MAPMRI model in the Laplacian
            and both model types.
        radial_order : unsigned int, optional
            Even value used to set the order of the basis.
        sphere_name : string, optional
            Sphere name on which to reconstruct the fODFs.
        relative_peak_threshold : float, optional
            Only return peaks greater than ``relative_peak_threshold * m``
            where m is the largest peak.
        min_separation_angle : float, optional
            The minimum distance between directions. If two peaks are too close
            only the larger of the two is returned.
        npeaks : int, optional
            Maximum number of peaks found.
        normalize_peaks : bool, optional
            If true, all peak values are calculated relative to `max(odf)`.
        extract_pam_values : bool, optional
            Save or not to save pam volumes as single nifti files.
        out_dir : string, optional
            Output directory.
        out_rtop : string, optional
            Name of the rtop to be saved.
        out_lapnorm : string, optional
            Name of the norm of Laplacian signal to be saved.
        out_msd : string, optional
            Name of the msd to be saved.
        out_qiv : string, optional
            Name of the qiv to be saved.
        out_rtap : string, optional
            Name of the rtap to be saved.
        out_rtpp : string, optional
            Name of the rtpp to be saved.
        out_ng : string, optional
            Name of the Non-Gaussianity to be saved.
        out_perng :  string, optional
            Name of the Non-Gaussianity perpendicular to be saved.
        out_parng : string, optional
            Name of the Non-Gaussianity parallel to be saved.
        out_pam : string, optional
            Name of the peaks volume to be saved.
        out_peaks_dir : string, optional
            Name of the peaks directions volume to be saved.
        out_peaks_values : string, optional
            Name of the peaks values volume to be saved.
        out_peaks_indices : string, optional
            Name of the peaks indices volume to be saved.

        """
        io_it = self.get_io_iterator()
        for (
            dwi,
            bval,
            bvec,
            out_rtop,
            out_lapnorm,
            out_msd,
            out_qiv,
            out_rtap,
            out_rtpp,
            out_ng,
            out_perng,
            out_parng,
            opam,
            opeaks_dir,
            opeaks_values,
            opeaks_indices,
        ) in io_it:
            logging.info(f"Computing MAPMRI metrics for {dwi}")
            data, affine = load_nifti(dwi)

            bvals, bvecs = read_bvals_bvecs(bval, bvec)
            # If all b-values are smaller or equal to the b0 threshold, it is
            # assumed that no thresholding is requested
            if any(mask_non_weighted_bvals(bvals, b0_threshold)):
                if b0_threshold < bvals.min():
                    warn(
                        f"b0_threshold (value: {b0_threshold}) is too low, "
                        "increase your b0_threshold. It should be higher than the "
                        f"first b0 value ({bvals.min()}).",
                        stacklevel=2,
                    )
            gtab = gradient_table(
                bvals=bvals,
                bvecs=bvecs,
                small_delta=small_delta,
                big_delta=big_delta,
                b0_threshold=b0_threshold,
            )

            if not save_metrics:
                save_metrics = [
                    "rtop",
                    "laplacian_signal",
                    "msd",
                    "qiv",
                    "rtap",
                    "rtpp",
                    "ng",
                    "perng",
                    "parng",
                ]

            kwargs = {
                "laplacian_regularization": laplacian,
                "positivity_constraint": positivity,
            }
            map_model_aniso = mapmri.MapmriModel(
                gtab,
                radial_order=radial_order,
                laplacian_weighting=laplacian_weighting,
                bval_threshold=bval_threshold,
                **kwargs,
            )
            mapfit_aniso = map_model_aniso.fit(data)

            for name, fname, func in [
                ("rtop", out_rtop, mapfit_aniso.rtop),
                (
                    "laplacian_signal",
                    out_lapnorm,
                    mapfit_aniso.norm_of_laplacian_signal,
                ),
                ("msd", out_msd, mapfit_aniso.msd),
                ("qiv", out_qiv, mapfit_aniso.qiv),
                ("rtap", out_rtap, mapfit_aniso.rtap),
                ("rtpp", out_rtpp, mapfit_aniso.rtpp),
                ("ng", out_ng, mapfit_aniso.ng),
                ("perng", out_perng, mapfit_aniso.ng_perpendicular),
                ("parng", out_parng, mapfit_aniso.ng_parallel),
            ]:
                if name in save_metrics:
                    r = func()
                    save_nifti(fname, r.astype(np.float32), affine)

            logging.info(f"MAPMRI saved in {os.path.abspath(out_dir)}")

            sphere = default_sphere
            if sphere_name:
                sphere = get_sphere(sphere_name)

            shape = data.shape[:-1]
            peak_dirs = np.zeros((shape + (npeaks, 3)))
            peak_values = np.zeros((shape + (npeaks,)))
            peak_indices = np.zeros((shape + (npeaks,)), dtype=np.int32)
            peak_indices.fill(-1)

            odf = mapfit_aniso.odf(sphere)
            for idx in ndindex(shape):
                # Get peaks of odf
                direction, pk, ind = peak_directions(
                    odf[idx],
                    sphere,
                    relative_peak_threshold=relative_peak_threshold,
                    min_separation_angle=min_separation_angle,
                )

                # Calculate peak metrics
                if pk.shape[0] != 0:
                    n = min(npeaks, pk.shape[0])

                    peak_dirs[idx][:n] = direction[:n]
                    peak_indices[idx][:n] = ind[:n]
                    peak_values[idx][:n] = pk[:n]

                    if normalize_peaks:
                        peak_values[idx][:n] /= pk[0]
                        peak_dirs[idx] *= peak_values[idx][:, None]

            pam = niftis_to_pam(
                affine, peak_dirs, peak_values, peak_indices, odf=odf, sphere=sphere
            )
            save_pam(opam, pam)

            if extract_pam_values:
                pam_to_niftis(
                    pam,
                    fname_peaks_dir=opeaks_dir,
                    fname_peaks_values=opeaks_values,
                    fname_peaks_indices=opeaks_indices,
                    reshape_dirs=True,
                )


class ReconstDtiFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "dti"

    def run(
        self,
        input_files,
        bvalues_files,
        bvectors_files,
        mask_files,
        fit_method="WLS",
        b0_threshold=50,
        bvecs_tol=0.01,
        npeaks=1,
        sigma=None,
        save_metrics=None,
        nifti_tensor=True,
        extract_pam_values=False,
        out_dir="",
        out_tensor="tensors.nii.gz",
        out_fa="fa.nii.gz",
        out_ga="ga.nii.gz",
        out_rgb="rgb.nii.gz",
        out_md="md.nii.gz",
        out_ad="ad.nii.gz",
        out_rd="rd.nii.gz",
        out_mode="mode.nii.gz",
        out_evec="evecs.nii.gz",
        out_eval="evals.nii.gz",
        out_pam="peaks.pam5",
        out_peaks_dir="peaks_dirs.nii.gz",
        out_peaks_values="peaks_values.nii.gz",
        out_peaks_indices="peaks_indices.nii.gz",
        out_sphere="sphere.txt",
        out_qa="qa.nii.gz",
    ):
        """Workflow for tensor reconstruction and for computing DTI metrics
        using Weighted  Least-Squares.

        Performs a tensor reconstruction :footcite:p:`Basser1994b`,
        :footcite:p:`Basser1996` on the files by 'globing' ``input_files`` and
        saves the DTI metrics in a directory specified by ``out_dir``.

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
        fit_method : string, optional
            can be one of the following:
            'WLS' for weighted least squares :footcite:p:`Chung2006`
            'LS' or 'OLS' for ordinary least squares :footcite:p:`Chung2006`
            'NLLS' for non-linear least-squares
            'RT' or 'restore' or 'RESTORE' for RESTORE robust tensor fitting
            :footcite:p:`Chang2005`.
        b0_threshold : float, optional
            Threshold used to find b0 volumes.
        bvecs_tol : float, optional
            Threshold used to check that norm(bvec) = 1 +/- bvecs_tol
        npeaks : int, optional
            Number of peaks/eigen vectors to save in each voxel. DTI generates
            3 eigen values and eigen vectors. The principal eigenvector is
            saved by default.
        sigma : float, optional
            An estimate of the variance. :footcite:t:`Chang2005` recommend to
            use 1.5267 * std(background_noise), where background_noise is
            estimated from some part of the image known to contain no signal
            (only noise) b-vectors are unit vectors.
        save_metrics : variable string, optional
            List of metrics to save.
            Possible values: fa, ga, rgb, md, ad, rd, mode, tensor, evec, eval
        nifti_tensor : bool, optional
            Whether the tensor is saved in the standard Nifti format or in an
            alternate format that is used by other software (e.g., FSL): a
            4-dimensional volume (shape (i, j, k, 6)) with
            Dxx, Dxy, Dxz, Dyy, Dyz, Dzz on the last dimension.
        extract_pam_values : bool, optional
            Save or not to save pam volumes as single nifti files.
        out_dir : string, optional
            Output directory.
        out_tensor : string, optional
            Name of the tensors volume to be saved.
            Per default, this will be saved following the nifti standard:
            with the tensor elements as Dxx, Dxy, Dyy, Dxz, Dyz, Dzz on the
            last (5th) dimension of the volume (shape: (i, j, k, 1, 6)). If
            `nifti_tensor` is False, this will be saved in an alternate format
            that is used by other software (e.g., FSL): a
            4-dimensional volume (shape (i, j, k, 6)) with Dxx, Dxy, Dxz, Dyy,
            Dyz, Dzz on the last dimension.
        out_fa : string, optional
            Name of the fractional anisotropy volume to be saved.
        out_ga : string, optional
            Name of the geodesic anisotropy volume to be saved.
        out_rgb : string, optional
            Name of the color fa volume to be saved.
        out_md : string, optional
            Name of the mean diffusivity volume to be saved.
        out_ad : string, optional
            Name of the axial diffusivity volume to be saved.
        out_rd : string, optional
            Name of the radial diffusivity volume to be saved.
        out_mode : string, optional
            Name of the mode volume to be saved.
        out_evec : string, optional
            Name of the eigenvectors volume to be saved.
        out_eval : string, optional
            Name of the eigenvalues to be saved.
        out_pam : string, optional
            Name of the peaks volume to be saved.
        out_peaks_dir : string, optional
            Name of the peaks directions volume to be saved.
        out_peaks_values : string, optional
            Name of the peaks values volume to be saved.
        out_peaks_indices : string, optional
            Name of the peaks indices volume to be saved.
        out_sphere : string, optional
            Sphere vertices name to be saved.
        out_qa : string, optional
            Name of the Quantitative Anisotropy to be saved.

        References
        ----------
        .. footbibliography::

        """
        save_metrics = save_metrics or []

        io_it = self.get_io_iterator()

        for (
            dwi,
            bval,
            bvec,
            mask,
            otensor,
            ofa,
            oga,
            orgb,
            omd,
            oad,
            orad,
            omode,
            oevecs,
            oevals,
            opam,
            opeaks_dir,
            opeaks_values,
            opeaks_indices,
            osphere,
            oqa,
        ) in io_it:
            logging.info(f"Computing DTI metrics for {dwi}")
            data, affine = load_nifti(dwi)

            if mask is not None:
                mask = load_nifti_data(mask).astype(bool)

            optional_args = {}
            if fit_method in ["RT", "restore", "RESTORE", "NLLS"]:
                optional_args["sigma"] = sigma

            tenfit, tenmodel, _ = self.get_fitted_tensor(
                data,
                mask,
                bval,
                bvec,
                b0_threshold=b0_threshold,
                bvecs_tol=bvecs_tol,
                fit_method=fit_method,
                optional_args=optional_args,
            )

            if not save_metrics:
                save_metrics = [
                    "fa",
                    "md",
                    "rd",
                    "ad",
                    "ga",
                    "rgb",
                    "mode",
                    "evec",
                    "eval",
                    "tensor",
                ]

            FA = fractional_anisotropy(tenfit.evals)
            FA[np.isnan(FA)] = 0
            FA = np.clip(FA, 0, 1)

            if "tensor" in save_metrics:
                tensor_vals = lower_triangular(tenfit.quadratic_form)

                if nifti_tensor:
                    ten_img = nifti1_symmat(tensor_vals, affine=affine)
                else:
                    alt_order = [0, 1, 3, 2, 4, 5]
                    ten_img = nib.Nifti1Image(
                        tensor_vals[..., alt_order].astype(np.float32), affine
                    )

                nib.save(ten_img, otensor)

            if "fa" in save_metrics:
                save_nifti(ofa, FA.astype(np.float32), affine)

            if "ga" in save_metrics:
                GA = geodesic_anisotropy(tenfit.evals)
                save_nifti(oga, GA.astype(np.float32), affine)

            if "rgb" in save_metrics:
                RGB = color_fa(FA, tenfit.evecs)
                save_nifti(orgb, np.array(255 * RGB, "uint8"), affine)

            if "md" in save_metrics:
                MD = mean_diffusivity(tenfit.evals)
                save_nifti(omd, MD.astype(np.float32), affine)

            if "ad" in save_metrics:
                AD = axial_diffusivity(tenfit.evals)
                save_nifti(oad, AD.astype(np.float32), affine)

            if "rd" in save_metrics:
                RD = radial_diffusivity(tenfit.evals)
                save_nifti(orad, RD.astype(np.float32), affine)

            if "mode" in save_metrics:
                MODE = get_mode(tenfit.quadratic_form)
                save_nifti(omode, MODE.astype(np.float32), affine)

            if "evec" in save_metrics:
                save_nifti(oevecs, tenfit.evecs.astype(np.float32), affine)

            if "eval" in save_metrics:
                save_nifti(oevals, tenfit.evals.astype(np.float32), affine)

            if save_metrics:
                msg = f"DTI metrics saved to {os.path.abspath(out_dir)}"
                logging.info(msg)
                for metric in save_metrics:
                    logging.info(self.last_generated_outputs[f"out_{metric}"])

            pam = tensor_to_pam(
                tenfit.evals.astype(np.float32),
                tenfit.evecs.astype(np.float32),
                affine,
                sphere=default_sphere,
                generate_peaks_indices=False,
                npeaks=npeaks,
            )

            save_pam(opam, pam)

            if extract_pam_values:
                pam_to_niftis(
                    pam,
                    fname_peaks_dir=opeaks_dir,
                    fname_peaks_values=opeaks_values,
                    fname_peaks_indices=opeaks_indices,
                    fname_sphere=osphere,
                    fname_qa=oqa,
                    reshape_dirs=True,
                )

    def get_fitted_tensor(
        self,
        data,
        mask,
        bval,
        bvec,
        b0_threshold=50,
        bvecs_tol=0.01,
        fit_method="WLS",
        optional_args=None,
    ):
        logging.info("Tensor estimation...")
        bvals, bvecs = read_bvals_bvecs(bval, bvec)
        gtab = gradient_table(
            bvals, bvecs=bvecs, b0_threshold=b0_threshold, atol=bvecs_tol
        )

        tenmodel = TensorModel(gtab, fit_method=fit_method, **optional_args)
        tenfit = tenmodel.fit(data, mask=mask)

        return tenfit, tenmodel, gtab


class ReconstDsiFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "dsi"

    def run(
        self,
        input_files,
        bvalues_files,
        bvectors_files,
        mask_files,
        qgrid_size=17,
        r_start=2.1,
        r_end=6.0,
        r_step=0.2,
        filter_width=32,
        remove_convolution=False,
        normalize_peaks=False,
        sphere_name=None,
        relative_peak_threshold=0.5,
        min_separation_angle=25,
        sh_order_max=8,
        extract_pam_values=False,
        parallel=False,
        num_processes=None,
        out_dir="",
        out_pam="peaks.pam5",
        out_shm="shm.nii.gz",
        out_peaks_dir="peaks_dirs.nii.gz",
        out_peaks_values="peaks_values.nii.gz",
        out_peaks_indices="peaks_indices.nii.gz",
        out_gfa="gfa.nii.gz",
        out_sphere="sphere.txt",
        out_b="B.nii.gz",
        out_qa="qa.nii.gz",
    ):
        """Diffusion Spectrum Imaging (DSI) reconstruction workflow.

        In DSI, the diffusion signal is sampled on a Cartesian grid in q-space.
        When using remove_convolution=True, the convolution on the DSI propagator that
        is caused by the truncation of the q-space in the DSI sampling is removed.

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
        qgrid_size : int, optional
            has to be an odd number. Sets the size of the q_space grid.
            For example if qgrid_size is 17 then the shape of the grid will be
            ``(17, 17, 17)``.
        r_start : float, optional
            ODF is sampled radially in the PDF. This parameters shows where the
            sampling should start.
        r_end : float, optional
            Radial endpoint of ODF sampling
        r_step : float, optional
            Step size of the ODf sampling from r_start to r_end
        filter_width : float, optional
            Strength of the hanning filter
        remove_convolution : bool, optional
            Whether to remove the convolution on the DSI propagator that is
            caused by the truncation of the q-space in the DSI sampling.
        normalize_peaks : bool, optional
            Whether to normalize the peaks
        sphere_name : string, optional
            Sphere name on which to reconstruct the fODFs.
        relative_peak_threshold : float, optional
            Only return peaks greater than ``relative_peak_threshold * m``
            where m is the largest peak.
        min_separation_angle : float, optional
            The minimum distance between directions. If two peaks are too close
            only the larger of the two is returned.
        sh_order_max : int, optional
            Spherical harmonics order (l) used in the DKI fit.
        extract_pam_values : bool, optional
            Save or not to save pam volumes as single nifti files.
        parallel : bool, optional
            Whether to use parallelization in peak-finding during the
            calibration procedure.
        num_processes : int, optional
            If `parallel` is True, the number of subprocesses to use
            (default multiprocessing.cpu_count()). If < 0 the maximal number
            of cores minus ``num_processes + 1`` is used (enter -1 to use as
            many cores as possible). 0 raises an error.
        out_dir : string, optional
            Output directory.
        out_pam : string, optional
            Name of the peaks volume to be saved.
        out_shm : string, optional
            Name of the spherical harmonics volume to be saved.
        out_peaks_dir : string, optional
            Name of the peaks directions volume to be saved.
        out_peaks_values : string, optional
            Name of the peaks values volume to be saved.
        out_peaks_indices : string, optional
            Name of the peaks indices volume to be saved.
        out_gfa : string, optional
            Name of the generalized FA volume to be saved.
        out_sphere : string, optional
            Sphere vertices name to be saved.
        out_b : string, optional
            Name of the B Matrix to be saved.
        out_qa : string, optional
            Name of the Quantitative Anisotropy to be saved.
        """
        io_it = self.get_io_iterator()

        if remove_convolution:
            filter_width = np.inf

        for (
            dwi,
            bval,
            bvec,
            mask,
            opam,
            oshm,
            opeaks_dir,
            opeaks_values,
            opeaks_indices,
            ogfa,
            osphere,
            ob,
            oqa,
        ) in io_it:
            logging.info(f"Computing DSI Model for {dwi}")
            data, affine = load_nifti(dwi)

            bvals, bvecs = read_bvals_bvecs(bval, bvec)
            gtab = gradient_table(bvals, bvecs=bvecs)
            mask = load_nifti_data(mask).astype(bool)

            DSIModel = (
                DiffusionSpectrumDeconvModel
                if remove_convolution
                else DiffusionSpectrumModel
            )
            dsi_model = DSIModel(
                gtab,
                qgrid_size=qgrid_size,
                r_start=r_start,
                r_end=r_end,
                r_step=r_step,
                filter_width=filter_width,
                normalize_peaks=normalize_peaks,
            )

            peaks_sphere = default_sphere
            if sphere_name is not None:
                peaks_sphere = get_sphere(name=sphere_name)

            peaks_dsi = peaks_from_model(
                model=dsi_model,
                data=data,
                sphere=peaks_sphere,
                relative_peak_threshold=relative_peak_threshold,
                min_separation_angle=min_separation_angle,
                mask=mask,
                return_sh=True,
                sh_order_max=sh_order_max,
                normalize_peaks=normalize_peaks,
                parallel=parallel,
                num_processes=num_processes,
            )
            peaks_dsi.affine = affine

            save_pam(opam, peaks_dsi)

            logging.info("DSI computation completed.")

            if extract_pam_values:
                pam_to_niftis(
                    peaks_dsi,
                    fname_shm=oshm,
                    fname_peaks_dir=opeaks_dir,
                    fname_peaks_values=opeaks_values,
                    fname_peaks_indices=opeaks_indices,
                    fname_gfa=ogfa,
                    fname_sphere=osphere,
                    fname_b=ob,
                    fname_qa=oqa,
                    reshape_dirs=True,
                )

            logging.info(f"DSI metrics saved to {os.path.abspath(out_dir)}")


class ReconstCSDFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "csd"

    def run(
        self,
        input_files,
        bvalues_files,
        bvectors_files,
        mask_files,
        b0_threshold=50.0,
        bvecs_tol=0.01,
        roi_center=None,
        roi_radii=10,
        fa_thr=0.7,
        frf=None,
        sphere_name=None,
        relative_peak_threshold=0.5,
        min_separation_angle=25,
        sh_order_max=8,
        parallel=False,
        extract_pam_values=False,
        num_processes=None,
        out_dir="",
        out_pam="peaks.pam5",
        out_shm="shm.nii.gz",
        out_peaks_dir="peaks_dirs.nii.gz",
        out_peaks_values="peaks_values.nii.gz",
        out_peaks_indices="peaks_indices.nii.gz",
        out_gfa="gfa.nii.gz",
        out_sphere="sphere.txt",
        out_b="B.nii.gz",
        out_qa="qa.nii.gz",
    ):
        """Constrained spherical deconvolution.

        See :footcite:p:`Tournier2007` for further details about the method.

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
            Threshold used to find b0 volumes.
        bvecs_tol : float, optional
            Bvecs should be unit vectors.
        roi_center : variable int, optional
            Center of ROI in data. If center is None, it is assumed that it is
            the center of the volume with shape `data.shape[:3]`.
        roi_radii : int or array-like, optional
            radii of cuboid ROI in voxels.
        fa_thr : float, optional
            FA threshold for calculating the response function.
        frf : variable float, optional
            Fiber response function can be for example inputted as 15 4 4
            (from the command line) or [15, 4, 4] from a Python script to be
            converted to float and multiplied by 10**-4 . If None
            the fiber response function will be computed automatically.
        sphere_name : string, optional
            Sphere name on which to reconstruct the fODFs.
        relative_peak_threshold : float, optional
            Only return peaks greater than ``relative_peak_threshold * m``
            where m is the largest peak.
        min_separation_angle : float, optional
            The minimum distance between directions. If two peaks are too close
            only the larger of the two is returned.
        sh_order_max : int, optional
            Spherical harmonics order (l) used in the CSA fit.
        parallel : bool, optional
            Whether to use parallelization in peak-finding during the
            calibration procedure.
        extract_pam_values : bool, optional
            Save or not to save pam volumes as single nifti files.
        num_processes : int, optional
            If `parallel` is True, the number of subprocesses to use
            (default multiprocessing.cpu_count()). If < 0 the maximal number
            of cores minus ``num_processes + 1`` is used (enter -1 to use as
            many cores as possible). 0 raises an error.
        out_dir : string, optional
            Output directory.
        out_pam : string, optional
            Name of the peaks volume to be saved.
        out_shm : string, optional
            Name of the spherical harmonics volume to be saved.
        out_peaks_dir : string, optional
            Name of the peaks directions volume to be saved.
        out_peaks_values : string, optional
            Name of the peaks values volume to be saved.
        out_peaks_indices : string, optional
            Name of the peaks indices volume to be saved.
        out_gfa : string, optional
            Name of the generalized FA volume to be saved.
        out_sphere : string, optional
            Sphere vertices name to be saved.
        out_b : string, optional
            Name of the B Matrix to be saved.
        out_qa : string, optional
            Name of the Quantitative Anisotropy to be saved.


        References
        ----------
        .. footbibliography::
        """
        io_it = self.get_io_iterator()

        for (
            dwi,
            bval,
            bvec,
            maskfile,
            opam,
            oshm,
            opeaks_dir,
            opeaks_values,
            opeaks_indices,
            ogfa,
            osphere,
            ob,
            oqa,
        ) in io_it:
            logging.info(f"Loading {dwi}")
            data, affine = load_nifti(dwi)

            bvals, bvecs = read_bvals_bvecs(bval, bvec)

            # If all b-values are smaller or equal to the b0 threshold, it is
            # assumed that no thresholding is requested
            if any(mask_non_weighted_bvals(bvals, b0_threshold)):
                if b0_threshold < bvals.min():
                    warn(
                        f"b0_threshold (value: {b0_threshold}) is too low, "
                        "increase your b0_threshold. It should be higher than the "
                        f"first b0 value ({bvals.min()}).",
                        stacklevel=2,
                    )
            gtab = gradient_table(
                bvals, bvecs=bvecs, b0_threshold=b0_threshold, atol=bvecs_tol
            )
            mask_vol = load_nifti_data(maskfile).astype(bool)

            n_params = ((sh_order_max + 1) * (sh_order_max + 2)) / 2
            if data.shape[-1] < n_params:
                raise ValueError(
                    f"You need at least {n_params} unique DWI volumes to "
                    f"compute fiber odfs. You currently have: {data.shape[-1]}"
                    " DWI volumes."
                )

            if frf is None:
                logging.info("Computing response function")
                if roi_center is not None:
                    logging.info(f"Response ROI center:\n{roi_center}")
                    logging.info(f"Response ROI radii:\n{roi_radii}")
                response, ratio = auto_response_ssst(
                    gtab,
                    data,
                    roi_center=roi_center,
                    roi_radii=roi_radii,
                    fa_thr=fa_thr,
                )
                response = list(response)

            else:
                logging.info("Using response function")
                if isinstance(frf, str):
                    l01 = np.array(literal_eval(frf), dtype=np.float64)
                else:
                    l01 = np.array(frf, dtype=np.float64)

                l01 *= 10**-4
                response = np.array([l01[0], l01[1], l01[1]])
                ratio = l01[1] / l01[0]
                response = (response, ratio)

            logging.info(
                f"Eigenvalues for the frf of the input data are :{response[0]}"
            )
            logging.info(f"Ratio for smallest to largest eigen value is {ratio}")

            peaks_sphere = default_sphere
            if sphere_name is not None:
                peaks_sphere = get_sphere(name=sphere_name)

            logging.info("CSD computation started.")
            csd_model = ConstrainedSphericalDeconvModel(
                gtab, response, sh_order_max=sh_order_max
            )

            peaks_csd = peaks_from_model(
                model=csd_model,
                data=data,
                sphere=peaks_sphere,
                relative_peak_threshold=relative_peak_threshold,
                min_separation_angle=min_separation_angle,
                mask=mask_vol,
                return_sh=True,
                sh_order_max=sh_order_max,
                normalize_peaks=True,
                parallel=parallel,
                num_processes=num_processes,
            )
            peaks_csd.affine = affine

            save_pam(opam, peaks_csd)

            logging.info("CSD computation completed.")

            if extract_pam_values:
                pam_to_niftis(
                    peaks_csd,
                    fname_shm=oshm,
                    fname_peaks_dir=opeaks_dir,
                    fname_peaks_values=opeaks_values,
                    fname_peaks_indices=opeaks_indices,
                    fname_gfa=ogfa,
                    fname_sphere=osphere,
                    fname_b=ob,
                    fname_qa=oqa,
                    reshape_dirs=True,
                )

            dname_ = os.path.dirname(opam)
            if dname_ == "":
                logging.info("Pam5 file saved in current directory")
            else:
                logging.info(f"Pam5 file saved in {dname_}")

            return io_it


class ReconstQBallBaseFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "qballbase"

    @deprecated_params("sh_order", new_name="sh_order_max", since="1.9", until="2.0")
    def run(
        self,
        input_files,
        bvalues_files,
        bvectors_files,
        mask_files,
        *,
        method="csa",
        smooth=0.006,
        min_signal=1e-5,
        assume_normed=False,
        b0_threshold=50.0,
        bvecs_tol=0.01,
        sphere_name=None,
        relative_peak_threshold=0.5,
        min_separation_angle=25,
        sh_order_max=8,
        parallel=False,
        extract_pam_values=False,
        num_processes=None,
        out_dir="",
        out_pam="peaks.pam5",
        out_shm="shm.nii.gz",
        out_peaks_dir="peaks_dirs.nii.gz",
        out_peaks_values="peaks_values.nii.gz",
        out_peaks_indices="peaks_indices.nii.gz",
        out_sphere="sphere.txt",
        out_gfa="gfa.nii.gz",
        out_b="B.nii.gz",
        out_qa="qa.nii.gz",
    ):
        """Constant Solid Angle.

        See :footcite:p:`Aganj2009` for further details about the method.

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
        method : string, optional
            Method to use for the reconstruction. Can be one of the following:
            'csa' for Constant Solid Angle reconstruction
            'qball' for Q-Ball reconstruction
            'opdt' for Orientation Probability Density Transform reconstruction
        smooth : float, optional
            The regularization parameter of the model.
        min_signal : float, optional
            During fitting, all signal values less than `min_signal` are
            clipped to `min_signal`. This is done primarily to avoid values
            less than or equal to zero when taking logs.
        assume_normed : bool, optional
            If True, clipping and normalization of the data with respect to the
            mean B0 signal are skipped during mode fitting. This is an advanced
            feature and should be used with care.
        b0_threshold : float, optional
            Threshold used to find b0 volumes.
        bvecs_tol : float, optional
            Threshold used so that norm(bvec)=1.
        sphere_name : string, optional
            Sphere name on which to reconstruct the fODFs.
        relative_peak_threshold : float, optional
            Only return peaks greater than ``relative_peak_threshold * m``
            where m is the largest peak.
        min_separation_angle : float, optional
            The minimum distance between directions. If two peaks are too close
            only the larger of the two is returned.
        sh_order_max : int, optional
            Spherical harmonics order (l) used in the CSA fit.
        parallel : bool, optional
            Whether to use parallelization in peak-finding during the
            calibration procedure.
        extract_pam_values : bool, optional
            Whether or not to save pam volumes as single nifti files.
        num_processes : int, optional
            If `parallel` is True, the number of subprocesses to use
            (default multiprocessing.cpu_count()). If < 0 the maximal number
            of cores minus ``num_processes + 1`` is used (enter -1 to use as
            many cores as possible). 0 raises an error.
        out_dir : string, optional
            Output directory.
        out_pam : string, optional
            Name of the peaks volume to be saved.
        out_shm : string, optional
            Name of the spherical harmonics volume to be saved.
        out_peaks_dir : string, optional
            Name of the peaks directions volume to be saved.
        out_peaks_values : string, optional
            Name of the peaks values volume to be saved.
        out_peaks_indices : string, optional
            Name of the peaks indices volume to be saved.
        out_sphere : string, optional
            Sphere vertices name to be saved.
        out_gfa : string, optional
            Name of the generalized FA volume to be saved.
        out_b : string, optional
            Name of the B Matrix to be saved.
        out_qa : string, optional
            Name of the Quantitative Anisotropy to be saved.

        References
        ----------
        .. footbibliography::

        """
        io_it = self.get_io_iterator()

        if method.lower() not in ["csa", "qball", "opdt"]:
            raise ValueError(
                f"Method {method} not recognized. "
                "Please choose between 'csa', 'qball', 'opdt'."
            )

        model_list = {
            "csa": CsaOdfModel,
            "qball": QballModel,
            "opdt": OpdtModel,
        }

        for (
            dwi,
            bval,
            bvec,
            maskfile,
            opam,
            oshm,
            opeaks_dir,
            opeaks_values,
            opeaks_indices,
            osphere,
            ogfa,
            ob,
            oqa,
        ) in io_it:
            logging.info(f"Loading {dwi}")
            data, affine = load_nifti(dwi)

            bvals, bvecs = read_bvals_bvecs(bval, bvec)
            # If all b-values are smaller or equal to the b0 threshold, it is
            # assumed that no thresholding is requested
            if any(mask_non_weighted_bvals(bvals, b0_threshold)):
                if b0_threshold < bvals.min():
                    warn(
                        f"b0_threshold (value: {b0_threshold}) is too low, "
                        "increase your b0_threshold. It should be higher than the "
                        f"first b0 value ({bvals.min()}).",
                        stacklevel=2,
                    )
            gtab = gradient_table(
                bvals, bvecs=bvecs, b0_threshold=b0_threshold, atol=bvecs_tol
            )
            mask_vol = load_nifti_data(maskfile).astype(bool)

            peaks_sphere = default_sphere
            if sphere_name is not None:
                peaks_sphere = get_sphere(name=sphere_name)

            logging.info(f"Starting {method.upper()} computations {dwi}")

            qball_base_model = model_list[method.lower()](
                gtab,
                sh_order_max,
                smooth=smooth,
                min_signal=min_signal,
                assume_normed=assume_normed,
            )

            peaks_qballbase = peaks_from_model(
                model=qball_base_model,
                data=data,
                sphere=peaks_sphere,
                relative_peak_threshold=relative_peak_threshold,
                min_separation_angle=min_separation_angle,
                mask=mask_vol,
                return_sh=True,
                sh_order_max=sh_order_max,
                normalize_peaks=True,
                parallel=parallel,
                num_processes=num_processes,
            )
            peaks_qballbase.affine = affine

            save_pam(opam, peaks_qballbase)

            logging.info(f"Finished {method.upper()} {dwi}")

            if extract_pam_values:
                pam_to_niftis(
                    peaks_qballbase,
                    fname_shm=oshm,
                    fname_peaks_dir=opeaks_dir,
                    fname_peaks_values=opeaks_values,
                    fname_peaks_indices=opeaks_indices,
                    fname_sphere=osphere,
                    fname_gfa=ogfa,
                    fname_b=ob,
                    fname_qa=oqa,
                    reshape_dirs=True,
                )

            dname_ = os.path.dirname(opam)
            if dname_ == "":
                logging.info("Pam5 file saved in current directory")
            else:
                logging.info(f"Pam5 file saved in {dname_}")

            return io_it


class ReconstDkiFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "dki"

    def run(
        self,
        input_files,
        bvalues_files,
        bvectors_files,
        mask_files,
        fit_method="WLS",
        b0_threshold=50.0,
        sigma=None,
        save_metrics=None,
        extract_pam_values=False,
        npeaks=5,
        out_dir="",
        out_dt_tensor="dti_tensors.nii.gz",
        out_fa="fa.nii.gz",
        out_ga="ga.nii.gz",
        out_rgb="rgb.nii.gz",
        out_md="md.nii.gz",
        out_ad="ad.nii.gz",
        out_rd="rd.nii.gz",
        out_mode="mode.nii.gz",
        out_evec="evecs.nii.gz",
        out_eval="evals.nii.gz",
        out_dk_tensor="dki_tensors.nii.gz",
        out_mk="mk.nii.gz",
        out_ak="ak.nii.gz",
        out_rk="rk.nii.gz",
        out_pam="peaks.pam5",
        out_peaks_dir="peaks_dirs.nii.gz",
        out_peaks_values="peaks_values.nii.gz",
        out_peaks_indices="peaks_indices.nii.gz",
        out_sphere="sphere.txt",
    ):
        """Workflow for Diffusion Kurtosis reconstruction and for computing
        DKI metrics.

        Performs a DKI reconstruction :footcite:p:`Tabesh2011`,
        :footcite:p:`Jensen2005` on the files by 'globing' ``input_files`` and
        saves the DKI metrics in a directory specified by ``out_dir``.

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
        fit_method : string, optional
            can be one of the following:
            'OLS' or 'ULLS' for ordinary least squares
            'WLS' or 'UWLLS' for weighted ordinary least squares
        b0_threshold : float, optional
            Threshold used to find b0 volumes.
        sigma : float, optional
            An estimate of the variance. :footcite:t:`Chang2005` recommend to
            use 1.5267 * std(background_noise), where background_noise is
            estimated from some part of the image known to contain no signal
            (only noise)
        save_metrics : variable string, optional
            List of metrics to save.
            Possible values: fa, ga, rgb, md, ad, rd, mode, tensor, evec, eval
        extract_pam_values : bool, optional
            Save or not to save pam volumes as single nifti files.
        npeaks : int, optional
            Number of peaks to fit in each voxel.
        out_dir : string, optional
            Output directory.
        out_dt_tensor : string, optional
            Name of the tensors volume to be saved.
        out_dk_tensor : string, optional
            Name of the tensors volume to be saved.
        out_fa : string, optional
            Name of the fractional anisotropy volume to be saved.
        out_ga : string, optional
            Name of the geodesic anisotropy volume to be saved.
        out_rgb : string, optional
            Name of the color fa volume to be saved.
        out_md : string, optional
            Name of the mean diffusivity volume to be saved.
        out_ad : string, optional
            Name of the axial diffusivity volume to be saved.
        out_rd : string, optional
            Name of the radial diffusivity volume to be saved.
        out_mode : string, optional
            Name of the mode volume to be saved.
        out_evec : string, optional
            Name of the eigenvectors volume to be saved.
        out_eval : string, optional
            Name of the eigenvalues to be saved.
        out_mk : string, optional
            Name of the mean kurtosis to be saved.
        out_ak : string, optional
            Name of the axial kurtosis to be saved.
        out_rk : string, optional
            Name of the radial kurtosis to be saved.
        out_pam : string, optional
            Name of the peaks volume to be saved.
        out_peaks_dir : string, optional
            Name of the peaks directions volume to be saved.
        out_peaks_values : string, optional
            Name of the peaks values volume to be saved.
        out_peaks_indices : string, optional
            Name of the peaks indices volume to be saved.
        out_sphere : string, optional
            Sphere vertices name to be saved.

        References
        ----------
        .. footbibliography::

        """
        save_metrics = save_metrics or []

        io_it = self.get_io_iterator()

        for (
            dwi,
            bval,
            bvec,
            mask,
            otensor,
            ofa,
            oga,
            orgb,
            omd,
            oad,
            orad,
            omode,
            oevecs,
            oevals,
            odk_tensor,
            omk,
            oak,
            ork,
            opam,
            opeaks_dir,
            opeaks_values,
            opeaks_indices,
            osphere,
        ) in io_it:
            logging.info(f"Computing DKI metrics for {dwi}")
            data, affine = load_nifti(dwi)

            if mask is not None:
                mask = load_nifti_data(mask).astype(bool)

            optional_args = {}
            if fit_method in ["RT", "restore", "RESTORE", "NLLS"]:
                optional_args["sigma"] = sigma

            dkfit, dkmodel, _ = self.get_fitted_tensor(
                data,
                mask,
                bval,
                bvec,
                b0_threshold=b0_threshold,
                fit_method=fit_method,
                optional_args=optional_args,
            )

            if not save_metrics:
                save_metrics = [
                    "mk",
                    "rk",
                    "ak",
                    "fa",
                    "md",
                    "rd",
                    "ad",
                    "ga",
                    "rgb",
                    "mode",
                    "evec",
                    "eval",
                    "dt_tensor",
                    "dk_tensor",
                ]

            evals, evecs, kt = split_dki_param(dkfit.model_params)
            FA = fractional_anisotropy(evals)
            FA[np.isnan(FA)] = 0
            FA = np.clip(FA, 0, 1)

            if "dt_tensor" in save_metrics:
                tensor_vals = lower_triangular(dkfit.quadratic_form)
                correct_order = [0, 1, 3, 2, 4, 5]
                tensor_vals_reordered = tensor_vals[..., correct_order]
                save_nifti(otensor, tensor_vals_reordered.astype(np.float32), affine)

            if "dk_tensor" in save_metrics:
                save_nifti(odk_tensor, dkfit.kt.astype(np.float32), affine)

            if "fa" in save_metrics:
                save_nifti(ofa, FA.astype(np.float32), affine)

            if "ga" in save_metrics:
                GA = geodesic_anisotropy(dkfit.evals)
                save_nifti(oga, GA.astype(np.float32), affine)

            if "rgb" in save_metrics:
                RGB = color_fa(FA, dkfit.evecs)
                save_nifti(orgb, np.array(255 * RGB, "uint8"), affine)

            if "md" in save_metrics:
                MD = mean_diffusivity(dkfit.evals)
                save_nifti(omd, MD.astype(np.float32), affine)

            if "ad" in save_metrics:
                AD = axial_diffusivity(dkfit.evals)
                save_nifti(oad, AD.astype(np.float32), affine)

            if "rd" in save_metrics:
                RD = radial_diffusivity(dkfit.evals)
                save_nifti(orad, RD.astype(np.float32), affine)

            if "mode" in save_metrics:
                MODE = get_mode(dkfit.quadratic_form)
                save_nifti(omode, MODE.astype(np.float32), affine)

            if "evec" in save_metrics:
                save_nifti(oevecs, dkfit.evecs.astype(np.float32), affine)

            if "eval" in save_metrics:
                save_nifti(oevals, dkfit.evals.astype(np.float32), affine)

            if "mk" in save_metrics:
                save_nifti(omk, dkfit.mk().astype(np.float32), affine)

            if "ak" in save_metrics:
                save_nifti(oak, dkfit.ak().astype(np.float32), affine)

            if "rk" in save_metrics:
                save_nifti(ork, dkfit.rk().astype(np.float32), affine)

            logging.info(f"DKI metrics saved in {os.path.dirname(oevals)}")

            pam = tensor_to_pam(
                dkfit.evals.astype(np.float32),
                dkfit.evecs.astype(np.float32),
                affine,
                sphere=default_sphere,
                generate_peaks_indices=False,
                npeaks=npeaks,
            )

            save_pam(opam, pam)

            if extract_pam_values:
                pam_to_niftis(
                    pam,
                    fname_peaks_dir=opeaks_dir,
                    fname_peaks_values=opeaks_values,
                    fname_peaks_indices=opeaks_indices,
                    fname_sphere=osphere,
                    reshape_dirs=True,
                )

    def get_fitted_tensor(
        self,
        data,
        mask,
        bval,
        bvec,
        b0_threshold=50,
        fit_method="WLS",
        optional_args=None,
    ):
        logging.info("Diffusion kurtosis estimation...")
        bvals, bvecs = read_bvals_bvecs(bval, bvec)
        # If all b-values are smaller or equal to the b0 threshold, it is
        # assumed that no thresholding is requested
        if any(mask_non_weighted_bvals(bvals, b0_threshold)):
            if b0_threshold < bvals.min():
                warn(
                    f"b0_threshold (value: {b0_threshold}) is too low, "
                    "increase your b0_threshold. It should be higher than the "
                    f"first b0 value ({bvals.min()}).",
                    stacklevel=2,
                )

        gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=b0_threshold)
        dkmodel = DiffusionKurtosisModel(gtab, fit_method=fit_method, **optional_args)
        dkfit = dkmodel.fit(data, mask=mask)

        return dkfit, dkmodel, gtab


class ReconstIvimFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "ivim"

    def run(
        self,
        input_files,
        bvalues_files,
        bvectors_files,
        mask_files,
        split_b_D=400,
        split_b_S0=200,
        b0_threshold=0,
        save_metrics=None,
        out_dir="",
        out_S0_predicted="S0_predicted.nii.gz",
        out_perfusion_fraction="perfusion_fraction.nii.gz",
        out_D_star="D_star.nii.gz",
        out_D="D.nii.gz",
    ):
        """Workflow for Intra-voxel Incoherent Motion reconstruction and for
        computing IVIM metrics.

        Performs a IVIM reconstruction :footcite:p:`LeBihan1988`,
        :footcite:p:`Stejskal1965` on the files by 'globing' ``input_files`` and
        saves the IVIM metrics in a directory specified by ``out_dir``.

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
            fitting.
        split_b_S0 : int, optional
            Value to split the bvals to estimate S0 for the two-stage process
            of fitting.
        b0_threshold : int, optional
            Threshold value for the b0 bval.
        save_metrics : variable string, optional
            List of metrics to save.
            Possible values: S0_predicted, perfusion_fraction, D_star, D
        out_dir : string, optional
            Output directory.
        out_S0_predicted : string, optional
            Name of the S0 signal estimated to be saved.
        out_perfusion_fraction : string, optional
            Name of the estimated volume fractions to be saved.
        out_D_star : string, optional
            Name of the estimated pseudo-diffusion parameter to be saved.
        out_D : string, optional
            Name of the estimated diffusion parameter to be saved.

        References
        ----------
        .. footbibliography::
        """
        save_metrics = save_metrics or []

        io_it = self.get_io_iterator()

        for (
            dwi,
            bval,
            bvec,
            mask,
            oS0_predicted,
            operfusion_fraction,
            oD_star,
            oD,
        ) in io_it:
            logging.info(f"Computing IVIM metrics for {dwi}")
            data, affine = load_nifti(dwi)

            if mask is not None:
                mask = load_nifti_data(mask).astype(bool)

            ivimfit, _ = self.get_fitted_ivim(
                data, mask, bval, bvec, b0_threshold=b0_threshold
            )

            if not save_metrics:
                save_metrics = ["S0_predicted", "perfusion_fraction", "D_star", "D"]

            if "S0_predicted" in save_metrics:
                save_nifti(
                    oS0_predicted, ivimfit.S0_predicted.astype(np.float32), affine
                )

            if "perfusion_fraction" in save_metrics:
                save_nifti(
                    operfusion_fraction,
                    ivimfit.perfusion_fraction.astype(np.float32),
                    affine,
                )

            if "D_star" in save_metrics:
                save_nifti(oD_star, ivimfit.D_star.astype(np.float32), affine)

            if "D" in save_metrics:
                save_nifti(oD, ivimfit.D.astype(np.float32), affine)

            logging.info(f"IVIM metrics saved in {os.path.dirname(oD)}")

    @warning_for_keywords()
    def get_fitted_ivim(self, data, mask, bval, bvec, *, b0_threshold=50):
        logging.info("Intra-Voxel Incoherent Motion Estimation...")
        bvals, bvecs = read_bvals_bvecs(bval, bvec)
        # If all b-values are smaller or equal to the b0 threshold, it is
        # assumed that no thresholding is requested
        if any(mask_non_weighted_bvals(bvals, b0_threshold)):
            if b0_threshold < bvals.min():
                warn(
                    f"b0_threshold (value: {b0_threshold}) is too low, "
                    "increase your b0_threshold. It should be higher than the "
                    f"first b0 value ({bvals.min()}).",
                    stacklevel=2,
                )

        gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=b0_threshold)
        ivimmodel = IvimModel(gtab)
        ivimfit = ivimmodel.fit(data, mask=mask)

        return ivimfit, gtab


class ReconstRUMBAFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "rumba"

    @deprecated_params("sh_order", new_name="sh_order_max", since="1.9", until="2.0")
    def run(
        self,
        input_files,
        bvalues_files,
        bvectors_files,
        mask_files,
        *,
        b0_threshold=50.0,
        bvecs_tol=0.01,
        roi_center=None,
        roi_radii=10,
        fa_thr=0.7,
        extract_pam_values=False,
        sh_order_max=8,
        parallel=True,
        num_processes=None,
        gm_response=0.8e-3,
        csf_response=3.0e-3,
        n_iter=600,
        recon_type="smf",
        n_coils=1,
        R=1,
        voxelwise=True,
        use_tv=False,
        sphere_name="repulsion724",
        verbose=False,
        relative_peak_threshold=0.5,
        min_separation_angle=25,
        out_dir="",
        out_pam="peaks.pam5",
        out_shm="shm.nii.gz",
        out_peaks_dir="peaks_dirs.nii.gz",
        out_peaks_values="peaks_values.nii.gz",
        out_peaks_indices="peaks_indices.nii.gz",
        out_gfa="gfa.nii.gz",
        out_sphere="sphere.txt",
        out_b="B.nii.gz",
        out_qa="qa.nii.gz",
    ):
        """Reconstruct the fiber local orientations using the Robust and
        Unbiased Model-BAsed Spherical Deconvolution (RUMBA-SD) model.

        The fiber response function (FRF) is computed using the single-shell,
        single-tissue model, and the voxel-wise fitting procedure is used for
        RUMBA-SD :footcite:p:`CanalesRodriguez2015`.

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
            Threshold used to find b0 volumes.
        bvecs_tol : float, optional
            Bvecs should be unit vectors.
        roi_center : variable int, optional
            Center of ROI in data. If center is None, it is assumed that it is
            the center of the volume with shape `data.shape[:3]`.
        roi_radii : variable int, optional
            radii of cuboid ROI in voxels.
        fa_thr : float, optional
            FA threshold to compute the WM response function.
        extract_pam_values : bool, optional
            Save or not to save pam volumes as single nifti files.
        sh_order : int, optional
            Spherical harmonics order (l) used in the RUMBA fit.
        parallel : bool, optional
            Whether to use parallelization in peak-finding during the
            calibration procedure.
        num_processes : int, optional
            If `parallel` is True, the number of subprocesses to use
            (default multiprocessing.cpu_count()). If < 0 the maximal number
            of cores minus ``num_processes + 1`` is used (enter -1 to use as
            many cores as possible). 0 raises an error.
        gm_response : float, optional
            Mean diffusivity for GM compartment. If `None`, then grey
            matter volume fraction is not computed.
        csf_response : float, optional
            Mean diffusivity for CSF compartment. If `None`, then CSF
            volume fraction is not computed.
        n_iter : int, optional
            Number of iterations for fODF estimation. Must be a positive int.
        recon_type : str, optional
            MRI reconstruction method type: spatial matched filter (`smf`) or
            sum-of-squares (`sos`). SMF reconstruction generates Rician noise
            while SoS reconstruction generates Noncentral Chi noise.
        n_coils : int, optional
            Number of coils in MRI scanner -- only relevant in SoS
            reconstruction. Must be a positive int. Default: 1
        R : int, optional
            Acceleration factor of the acquisition. For SIEMENS,
            R = iPAT factor. For GE, R = ASSET factor. For PHILIPS,
            R = SENSE factor. Typical values are 1 or 2. Must be a positive
            integer.
        voxelwise : bool, optional
            If true, performs a voxelwise fit. If false, performs a global fit
            on the entire brain at once. The global fit requires a 4D brain
            volume in `fit`.
        use_tv : bool, optional
            If true, applies total variation regularization. This only takes
            effect in a global fit (`voxelwise` is set to `False`). TV can only
            be applied to 4D brain volumes with no singleton dimensions.
        sphere_name : str, optional
            Sphere name on which to reconstruct the fODFs.
        verbose : bool, optional
            If true, logs updates on estimated signal-to-noise ratio after each
            iteration. This only takes effect in a global fit (`voxelwise` is
            set to `False`).
        relative_peak_threshold : float, optional
            Only return peaks greater than ``relative_peak_threshold * m``
             where m is the largest peak.
        min_separation_angle : float, optional
            The minimum distance between directions. If two peaks are too close
            only the larger of the two is returned.
        out_dir : string, optional
            Output directory.
        out_pam : string, optional
            Name of the peaks volume to be saved.
        out_shm : string, optional
            Name of the spherical harmonics volume to be saved.
        out_peaks_dir : string, optional
            Name of the peaks directions volume to be saved.
        out_peaks_values : string, optional
            Name of the peaks values volume to be saved.
        out_peaks_indices : string, optional
            Name of the peaks indices volume to be saved.
        out_gfa : string, optional
            Name of the generalized FA volume to be saved.
        out_sphere : string, optional
            Sphere vertices name to be saved.
        out_b : string, optional
            Name of the B Matrix to be saved.
        out_qa : string, optional
            Name of the Quantitative Anisotropy to be saved.

        References
        ----------
        .. footbibliography::
        """

        io_it = self.get_io_iterator()

        for (
            dwi,
            bval,
            bvec,
            maskfile,
            opam,
            oshm,
            opeaks_dir,
            opeaks_values,
            opeaks_indices,
            ogfa,
            osphere,
            ob,
            oqa,
        ) in io_it:
            # Read the data
            logging.info(f"Loading {dwi}")
            data, affine = load_nifti(dwi)

            bvals, bvecs = read_bvals_bvecs(bval, bvec)

            mask_vol = load_nifti_data(maskfile).astype(bool)

            # If all b-values are smaller or equal to the b0 threshold, it is
            # assumed that no thresholding is requested
            if any(mask_non_weighted_bvals(bvals, b0_threshold)):
                if b0_threshold < bvals.min():
                    warn(
                        f"b0_threshold (value: {b0_threshold}) is too low, "
                        "increase your b0_threshold. It should be higher than the "
                        f"first b0 value ({bvals.min()}).",
                        stacklevel=2,
                    )

            gtab = gradient_table(
                bvals, bvecs=bvecs, b0_threshold=b0_threshold, atol=bvecs_tol
            )

            sphere = get_sphere(name=sphere_name)

            # Compute the FRF
            wm_response, _ = auto_response_ssst(
                gtab, data, roi_center=roi_center, roi_radii=roi_radii, fa_thr=fa_thr
            )

            # Instantiate the RUMBA-SD reconstruction model
            rumba = RumbaSDModel(
                gtab,
                wm_response=wm_response[0],
                gm_response=gm_response,
                csf_response=csf_response,
                n_iter=n_iter,
                recon_type=recon_type,
                n_coils=n_coils,
                R=R,
                voxelwise=voxelwise,
                use_tv=use_tv,
                sphere=sphere,
                verbose=verbose,
            )

            rumba_peaks = peaks_from_model(
                model=rumba,
                data=data,
                sphere=sphere,
                relative_peak_threshold=relative_peak_threshold,
                min_separation_angle=min_separation_angle,
                mask=mask_vol,
                return_sh=True,
                sh_order_max=sh_order_max,
                normalize_peaks=True,
                parallel=parallel,
                num_processes=num_processes,
            )

            logging.info("Peak computation completed.")

            rumba_peaks.affine = affine

            save_pam(opam, rumba_peaks)

            if extract_pam_values:
                pam_to_niftis(
                    rumba_peaks,
                    fname_shm=oshm,
                    fname_peaks_dir=opeaks_dir,
                    fname_peaks_values=opeaks_values,
                    fname_peaks_indices=opeaks_indices,
                    fname_gfa=ogfa,
                    fname_sphere=osphere,
                    fname_b=ob,
                    fname_qa=oqa,
                    reshape_dirs=True,
                )

            dname_ = os.path.dirname(opam)
            if dname_ == "":
                logging.info("Pam5 file saved in current directory")
            else:
                logging.info(f"Pam5 file saved in {dname_}")

            return io_it


class ReconstSDTFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "sdt"

    def run(
        self,
        input_files,
        bvalues_files,
        bvectors_files,
        mask_files,
        *,
        ratio=None,
        roi_center=None,
        roi_radii=10,
        fa_thr=0.7,
        sphere_name=None,
        sh_order_max=8,
        lambda_=1.0,
        tau=0.1,
        b0_threshold=50.0,
        bvecs_tol=0.01,
        relative_peak_threshold=0.5,
        min_separation_angle=25,
        parallel=False,
        extract_pam_values=False,
        num_processes=None,
        out_dir="",
        out_pam="peaks.pam5",
        out_shm="shm.nii.gz",
        out_peaks_dir="peaks_dirs.nii.gz",
        out_peaks_values="peaks_values.nii.gz",
        out_peaks_indices="peaks_indices.nii.gz",
        out_gfa="gfa.nii.gz",
        out_sphere="sphere.txt",
        out_b="B.nii.gz",
        out_qa="qa.nii.gz",
    ):
        """Workflow for Spherical Deconvolution Transform (SDT)

        See :footcite:p:`Descoteaux2009` for further details about the method.

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
        ratio : float, optional
            Ratio of the smallest to largest eigenvalue used in the response
            function estimation. If None, the response function will be
            estimated automatically.
        roi_center : variable int, optional
            Center of ROI in data. If center is None, it is assumed that it is
            the center of the volume with shape `data.shape[:3]`.
        roi_radii : variable int, optional
            radii of cuboid ROI in voxels.
        fa_thr : float, optional
            FA threshold to compute the WM response function.
        sphere_name : str, optional
            Sphere name on which to reconstruct the fODFs.
        sh_order_max : int, optional
            Maximum spherical harmonics order (l) used in the SDT fit.
        lambda_ : float, optional
            Regularization parameter.
        tau : float, optional
            Diffusion time.
        b0_threshold : float, optional
            Threshold used to find b0 volumes.
        bvecs_tol : float, optional
            Bvecs should be unit vectors.
        relative_peak_threshold : float, optional
            Only return peaks greater than ``relative_peak_threshold * m``
            where m is the largest peak.
        min_separation_angle : float, optional
            The angle tolerance between directions.
        parallel : bool, optional
            Whether to use parallelization in peak-finding.
        extract_pam_values : bool, optional
            Save or not to save pam volumes as single nifti files.
        num_processes : int, optional
            If `parallel` is True, the number of subprocesses to use
        out_dir : string, optional
            Output directory.
        out_pam : string, optional
            Name of the peaks volume to be saved.
        out_shm : string, optional
            Name of the spherical harmonics volume to be saved.
        out_peaks_dir : string, optional
            Name of the peaks directions volume to be saved.
        out_peaks_values : string, optional
            Name of the peaks values volume to be saved.
        out_peaks_indices : string, optional
            Name of the peaks indices volume to be saved.
        out_gfa : string, optional
            Name of the generalized FA volume to be saved.
        out_sphere : string, optional
            Sphere vertices name to be saved.
        out_b : string, optional
            Name of the B Matrix to be saved.
        out_qa : string, optional
            Name of the Quantitative Anisotropy to be saved.

        References
        ----------
        .. footbibliography::
        """
        io_it = self.get_io_iterator()

        for (
            dwi,
            bval,
            bvec,
            maskfile,
            opam,
            oshm,
            opeaks_dir,
            opeaks_values,
            opeaks_indices,
            ogfa,
            osphere,
            ob,
            oqa,
        ) in io_it:
            logging.info(f"Loading {dwi}")
            data, affine = load_nifti(dwi)
            bvals, bvecs = read_bvals_bvecs(bval, bvec)

            # If all b-values are smaller or equal to the b0 threshold, it is
            # assumed that no thresholding is requested
            if any(mask_non_weighted_bvals(bvals, b0_threshold)):
                if b0_threshold < bvals.min():
                    warn(
                        f"b0_threshold (value: {b0_threshold}) is too low, "
                        "increase your b0_threshold. It should be higher than the "
                        f"first b0 value ({bvals.min()}).",
                        stacklevel=2,
                    )
            gtab = gradient_table(
                bvals, bvecs=bvecs, b0_threshold=b0_threshold, atol=bvecs_tol
            )
            mask_vol = load_nifti_data(maskfile).astype(bool)

            n_params = ((sh_order_max + 1) * (sh_order_max + 2)) / 2
            if data.shape[-1] < n_params:
                raise ValueError(
                    f"You need at least {n_params} unique DWI volumes to "
                    f"compute fiber odfs. You currently have: {data.shape[-1]}"
                    " DWI volumes."
                )

            if ratio is None:
                logging.info("Computing response function")
                _, ratio = auto_response_ssst(
                    gtab,
                    data,
                    roi_center=roi_center,
                    roi_radii=roi_radii,
                    fa_thr=fa_thr,
                )

            logging.info(f"Ratio for smallest to largest eigen value is {ratio}")

            peaks_sphere = default_sphere
            if sphere_name is not None:
                peaks_sphere = get_sphere(name=sphere_name)

            logging.info("SDT computation started.")
            sdt_model = ConstrainedSDTModel(
                gtab,
                ratio,
                sh_order_max=sh_order_max,
                reg_sphere=peaks_sphere,
                lambda_=lambda_,
                tau=tau,
            )

            peaks_sdt = peaks_from_model(
                model=sdt_model,
                data=data,
                sphere=peaks_sphere,
                relative_peak_threshold=relative_peak_threshold,
                min_separation_angle=min_separation_angle,
                mask=mask_vol,
                return_sh=True,
                sh_order_max=sh_order_max,
                normalize_peaks=True,
                parallel=parallel,
                num_processes=num_processes,
            )
            peaks_sdt.affine = affine

            save_pam(opam, peaks_sdt)

            logging.info("SDT computation completed.")

            if extract_pam_values:
                pam_to_niftis(
                    peaks_sdt,
                    fname_shm=oshm,
                    fname_peaks_dir=opeaks_dir,
                    fname_peaks_values=opeaks_values,
                    fname_peaks_indices=opeaks_indices,
                    fname_gfa=ogfa,
                    fname_sphere=osphere,
                    fname_b=ob,
                    fname_qa=oqa,
                    reshape_dirs=True,
                )

            dname_ = os.path.dirname(opam)
            if dname_ == "":
                logging.info("Pam5 file saved in current directory")
            else:
                logging.info(f"Pam5 file saved in {dname_}")

            return io_it


class ReconstSFMFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "sfm"

    def run(
        self,
        input_files,
        bvalues_files,
        bvectors_files,
        mask_files,
        *,
        sphere_name=None,
        response=None,
        solver="ElasticNet",
        l1_ratio=0.5,
        alpha=0.001,
        seed=42,
        b0_threshold=50.0,
        bvecs_tol=0.01,
        sh_order_max=8,
        relative_peak_threshold=0.5,
        min_separation_angle=25,
        parallel=False,
        extract_pam_values=False,
        num_processes=None,
        out_dir="",
        out_pam="peaks.pam5",
        out_shm="shm.nii.gz",
        out_peaks_dir="peaks_dirs.nii.gz",
        out_peaks_values="peaks_values.nii.gz",
        out_peaks_indices="peaks_indices.nii.gz",
        out_gfa="gfa.nii.gz",
        out_sphere="sphere.txt",
        out_b="B.nii.gz",
        out_qa="qa.nii.gz",
    ):
        """Workflow for Sparse Fascicle Model (SFM)

        See :footcite:p:`Rokem2015` for further details about the method.

        Parameters
        ----------
        input_files : string
            Path to the input volumes. This path may contain wildcards to
        bvalues_files : string
            Path to the bvalues files. This path may contain wildcards to use
            multiple bvalues files at once.
        bvectors_files : string
            Path to the bvalues files. This path may contain wildcards to use
        mask_files : string
            Path to the input masks. This path may contain wildcards to use
        sphere_name : string, optional
            Sphere name on which to reconstruct the fODFs.
        response : variable int, optional
            Response function to use. If None, the response function will be
            defined automatically.
        solver : str, optional
            This will determine the algorithm used to solve the set of linear
            equations underlying this model. It needs to be one of the following:
            {'ElasticNet', 'NNLS'}
        l1_ratio : float, optional
            The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1. For l1_ratio = 0
            the penalty is an L2 penalty. For l1_ratio = 1 it is an L1 penalty. For
            0 < l1_ratio < 1, the penalty is a combination of L1 and L2.
        alpha : float, optional
            Sets the balance between least-squares error and L1/L2
            regularization in ElasticNet :footcite:p`Zou2005`.
        seed : int, optional
            Seed for the random number generator.
        b0_threshold : float, optional
            Threshold used to find b0 volumes.
        bvecs_tol : float, optional
            Bvecs should be unit vectors.
        sh_order_max : int, optional
            Maximum spherical harmonics order (l) used in the SFM fit.
        relative_peak_threshold : float, optional
            Only return peaks greater than ``relative_peak_threshold * m``
            where m is the largest peak.
        min_separation_angle : float, optional
            The angle tolerance between directions.
        parallel : bool, optional
            Whether to use parallelization in peak-finding.
        extract_pam_values : bool, optional
            Save or not to save pam volumes as single nifti files.
        num_processes : int, optional
            If `parallel` is True, the number of subprocesses to use
        out_dir : string, optional
            Output directory.
        out_pam : string, optional
            Name of the peaks volume to be saved.
        out_shm : string, optional
            Name of the spherical harmonics volume to be saved.
        out_peaks_dir : string, optional
            Name of the peaks directions volume to be saved.
        out_peaks_values : string, optional
            Name of the peaks values volume to be saved.
        out_peaks_indices : string, optional
            Name of the peaks indices volume to be saved.
        out_gfa : string, optional
            Name of the generalized FA volume to be saved.
        out_sphere : string, optional
            Sphere vertices name to be saved.
        out_b : string, optional
            Name of the B Matrix to be saved.
        out_qa : string, optional
            Name of the Quantitative Anisotropy to be saved.

        References
        ----------
        .. footbibliography::
        """
        io_it = self.get_io_iterator()
        response = response or (0.0015, 0.0005, 0.0005)

        for (
            dwi,
            bval,
            bvec,
            maskfile,
            opam,
            oshm,
            opeaks_dir,
            opeaks_values,
            opeaks_indices,
            ogfa,
            osphere,
            ob,
            oqa,
        ) in io_it:
            logging.info(f"Loading {dwi}")
            data, affine = load_nifti(dwi)
            bvals, bvecs = read_bvals_bvecs(bval, bvec)

            # If all b-values are smaller or equal to the b0 threshold, it is
            # assumed that no thresholding is requested
            if any(mask_non_weighted_bvals(bvals, b0_threshold)):
                if b0_threshold < bvals.min():
                    warn(
                        f"b0_threshold (value: {b0_threshold}) is too low, "
                        "increase your b0_threshold. It should be higher than the "
                        f"first b0 value ({bvals.min()}).",
                        stacklevel=2,
                    )
            gtab = gradient_table(
                bvals, bvecs=bvecs, b0_threshold=b0_threshold, atol=bvecs_tol
            )
            mask_vol = load_nifti_data(maskfile).astype(bool)

            n_params = ((sh_order_max + 1) * (sh_order_max + 2)) / 2
            if data.shape[-1] < n_params:
                raise ValueError(
                    f"You need at least {n_params} unique DWI volumes to "
                    f"compute fiber odfs. You currently have: {data.shape[-1]}"
                    " DWI volumes."
                )

            peaks_sphere = (
                default_sphere if sphere_name is None else get_sphere(name=sphere_name)
            )

            logging.info("SFM computation started.")
            sfm_model = SparseFascicleModel(
                gtab,
                sphere=peaks_sphere,
                response=response,
                solver=solver,
                l1_ratio=l1_ratio,
                alpha=alpha,
                seed=seed,
            )

            peaks_sfm = peaks_from_model(
                model=sfm_model,
                data=data,
                sphere=peaks_sphere,
                relative_peak_threshold=relative_peak_threshold,
                min_separation_angle=min_separation_angle,
                mask=mask_vol,
                return_sh=True,
                sh_order_max=sh_order_max,
                normalize_peaks=True,
                parallel=parallel,
                num_processes=num_processes,
            )
            peaks_sfm.affine = affine

            save_pam(opam, peaks_sfm)

            logging.info("SFM computation completed.")

            if extract_pam_values:
                pam_to_niftis(
                    peaks_sfm,
                    fname_shm=oshm,
                    fname_peaks_dir=opeaks_dir,
                    fname_peaks_values=opeaks_values,
                    fname_peaks_indices=opeaks_indices,
                    fname_gfa=ogfa,
                    fname_sphere=osphere,
                    fname_b=ob,
                    fname_qa=oqa,
                    reshape_dirs=True,
                )

            dname_ = os.path.dirname(opam)
            msg = (
                "Pam5 file saved in current directory"
                if dname_ == ""
                else f"Pam5 file saved in {dname_}"
            )
            logging.info(msg)

            return io_it


class ReconstGQIFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "gqi"

    def run(
        self,
        input_files,
        bvalues_files,
        bvectors_files,
        mask_files,
        *,
        method="gqi2",
        sampling_length=1.2,
        normalize_peaks=False,
        sphere_name=None,
        b0_threshold=50.0,
        bvecs_tol=0.01,
        sh_order_max=8,
        relative_peak_threshold=0.5,
        min_separation_angle=25,
        parallel=False,
        extract_pam_values=False,
        num_processes=None,
        out_dir="",
        out_pam="peaks.pam5",
        out_shm="shm.nii.gz",
        out_peaks_dir="peaks_dirs.nii.gz",
        out_peaks_values="peaks_values.nii.gz",
        out_peaks_indices="peaks_indices.nii.gz",
        out_gfa="gfa.nii.gz",
        out_sphere="sphere.txt",
        out_b="B.nii.gz",
        out_qa="qa.nii.gz",
    ):
        """Workflow for Generalized Q-Sampling Imaging (GQI)

        See :footcite:p:`Yeh2010` for further details about the method.

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
            multiple masks at once.
        method : str, optional
            Method used to compute the ODFs. It can be 'standard' or 'gqi2'.
        sampling_length : float, optional
            The maximum length of the sampling fibers.
        normalize_peaks : bool, optional
            If True, the peaks are normalized to 1.
        sphere_name : str, optional
            Sphere name on which to reconstruct the fODFs.
        b0_threshold : float, optional
            Threshold used to find b0 volumes.
        bvecs_tol : float, optional
            Bvecs should be unit vectors.
        sh_order_max : int, optional
            Maximum spherical harmonics order (l) used in the SFM fit.
        relative_peak_threshold : float, optional
            Only return peaks greater than ``relative_peak_threshold * m``
            where m is the largest peak.
        min_separation_angle : float, optional
            The angle tolerance between directions.
        parallel : bool, optional
            Whether to use parallelization in peak-finding.
        extract_pam_values : bool, optional
            Save or not to save pam volumes as single nifti files.
        num_processes : int, optional
            If `parallel` is True, the number of subprocesses to use
        out_dir : string, optional
            Output directory.
        out_pam : string, optional
            Name of the peaks volume to be saved.
        out_shm : string, optional
            Name of the spherical harmonics volume to be saved.
        out_peaks_dir : string, optional
            Name of the peaks directions volume to be saved.
        out_peaks_values : string, optional
            Name of the peaks values volume to be saved.
        out_peaks_indices : string, optional
            Name of the peaks indices volume to be saved.
        out_gfa : string, optional
            Name of the generalized FA volume to be saved.
        out_sphere : string, optional
            Sphere vertices name to be saved.
        out_b : string, optional
            Name of the B Matrix to be saved.
        out_qa : string, optional
            Name of the Quantitative Anisotropy to be saved.

        References
        ----------
        .. footbibliography::
        """
        io_it = self.get_io_iterator()

        for (
            dwi,
            bval,
            bvec,
            maskfile,
            opam,
            oshm,
            opeaks_dir,
            opeaks_values,
            opeaks_indices,
            ogfa,
            osphere,
            ob,
            oqa,
        ) in io_it:
            logging.info(f"Loading {dwi}")
            data, affine = load_nifti(dwi)
            bvals, bvecs = read_bvals_bvecs(bval, bvec)

            # If all b-values are smaller or equal to the b0 threshold, it is
            # assumed that no thresholding is requested
            if any(mask_non_weighted_bvals(bvals, b0_threshold)):
                if b0_threshold < bvals.min():
                    warn(
                        f"b0_threshold (value: {b0_threshold}) is too low, "
                        "increase your b0_threshold. It should be higher than the "
                        f"first b0 value ({bvals.min()}).",
                        stacklevel=2,
                    )
            gtab = gradient_table(
                bvals, bvecs=bvecs, b0_threshold=b0_threshold, atol=bvecs_tol
            )
            mask_vol = load_nifti_data(maskfile).astype(bool)

            n_params = ((sh_order_max + 1) * (sh_order_max + 2)) / 2
            if data.shape[-1] < n_params:
                raise ValueError(
                    f"You need at least {n_params} unique DWI volumes to "
                    f"compute fiber odfs. You currently have: {data.shape[-1]}"
                    " DWI volumes."
                )

            peaks_sphere = (
                default_sphere if sphere_name is None else get_sphere(name=sphere_name)
            )

            logging.info("GQI computation started.")
            gqi_model = GeneralizedQSamplingModel(
                gtab,
                method=method,
                sampling_length=sampling_length,
                normalize_peaks=normalize_peaks,
            )

            peaks_gqi = peaks_from_model(
                model=gqi_model,
                data=data,
                sphere=peaks_sphere,
                relative_peak_threshold=relative_peak_threshold,
                min_separation_angle=min_separation_angle,
                mask=mask_vol,
                return_sh=True,
                sh_order_max=sh_order_max,
                normalize_peaks=normalize_peaks,
                parallel=parallel,
                num_processes=num_processes,
            )
            peaks_gqi.affine = affine

            save_pam(opam, peaks_gqi)

            logging.info("GQI computation completed.")

            if extract_pam_values:
                pam_to_niftis(
                    peaks_gqi,
                    fname_shm=oshm,
                    fname_peaks_dir=opeaks_dir,
                    fname_peaks_values=opeaks_values,
                    fname_peaks_indices=opeaks_indices,
                    fname_gfa=ogfa,
                    fname_sphere=osphere,
                    fname_b=ob,
                    fname_qa=oqa,
                    reshape_dirs=True,
                )

            dname_ = os.path.dirname(opam)
            msg = (
                "Pam5 file saved in current directory"
                if dname_ == ""
                else f"Pam5 file saved in {dname_}"
            )
            logging.info(msg)

            return io_it


class ReconstForecastFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "forecast"

    def run(
        self,
        input_files,
        bvalues_files,
        bvectors_files,
        mask_files,
        *,
        lambda_lb=1e-3,
        dec_alg="CSD",
        lambda_csd=1.0,
        sphere_name=None,
        b0_threshold=50.0,
        bvecs_tol=0.01,
        sh_order_max=8,
        relative_peak_threshold=0.5,
        min_separation_angle=25,
        parallel=False,
        extract_pam_values=False,
        num_processes=None,
        out_dir="",
        out_pam="peaks.pam5",
        out_shm="shm.nii.gz",
        out_peaks_dir="peaks_dirs.nii.gz",
        out_peaks_values="peaks_values.nii.gz",
        out_peaks_indices="peaks_indices.nii.gz",
        out_gfa="gfa.nii.gz",
        out_sphere="sphere.txt",
        out_b="B.nii.gz",
        out_qa="qa.nii.gz",
    ):
        """Workflow for Fiber ORientation Estimated using Continuous Axially Symmetric
        Tensors (FORECAST).

        FORECAST :footcite:p:`Anderson2005`, :footcite:p:`Kaden2016a`,
        :footcite:p:`Zucchelli2017` is a Spherical Deconvolution reconstruction
        model for multi-shell diffusion data which enables the calculation of a
        voxel adaptive response function using the Spherical Mean Technique (SMT)
        :footcite:p:`Kaden2016a`, :footcite:p:`Zucchelli2017`.

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
            multiple bvalues files at once.
        mask_files : string
            Path to the input masks. This path may contain wildcards to use
            multiple masks at once. (default: No mask used)
        lambda_lb : float, optional
            Regularization parameter for the Laplacian-Beltrami operator.
        dec_alg : str, optional
            Spherical deconvolution algorithm. The possible values are Weighted Least
            Squares ('WLS'),
            Positivity Constraints using CVXPY ('POS') and the Constraint
            Spherical Deconvolution algorithm ('CSD').
        lambda_csd : float, optional
            Regularization parameter for the CSD algorithm.
        sphere_name : str, optional
            Sphere name on which to reconstruct the fODFs.
        b0_threshold : float, optional
            Threshold used to find b0 volumes.
        bvecs_tol : float, optional
            Bvecs should be unit vectors.
        sh_order_max : int, optional
            Maximum spherical harmonics order (l) used in the SFM fit.
        relative_peak_threshold : float, optional
            Only return peaks greater than ``relative_peak_threshold * m``
            where m is the largest peak.
        min_separation_angle : float, optional
            The angle tolerance between directions.
        parallel : bool, optional
            Whether to use parallelization in peak-finding.
        extract_pam_values : bool, optional
            Save or not to save pam volumes as single nifti files.
        num_processes : int, optional
            If `parallel` is True, the number of subprocesses to use
        out_dir : string, optional
            Output directory.
        out_pam : string, optional
            Name of the peaks volume to be saved.
        out_shm : string, optional
            Name of the spherical harmonics volume to be saved.
        out_peaks_dir : string, optional
            Name of the peaks directions volume to be saved.
        out_peaks_values : string, optional
            Name of the peaks values volume to be saved.
        out_peaks_indices : string, optional
            Name of the peaks indices volume to be saved.
        out_gfa : string, optional
            Name of the generalized FA volume to be saved.
        out_sphere : string, optional
            Sphere vertices name to be saved.
        out_b : string, optional
            Name of the B Matrix to be saved.
        out_qa : string, optional
            Name of the Quantitative Anisotropy to be saved.

        References
        ----------
        .. footbibliography::
        """
        io_it = self.get_io_iterator()

        for (
            dwi,
            bval,
            bvec,
            maskfile,
            opam,
            oshm,
            opeaks_dir,
            opeaks_values,
            opeaks_indices,
            ogfa,
            osphere,
            ob,
            oqa,
        ) in io_it:
            logging.info(f"Loading {dwi}")
            data, affine = load_nifti(dwi)
            bvals, bvecs = read_bvals_bvecs(bval, bvec)

            # If all b-values are smaller or equal to the b0 threshold, it is
            # assumed that no thresholding is requested
            if any(mask_non_weighted_bvals(bvals, b0_threshold)):
                if b0_threshold < bvals.min():
                    warn(
                        f"b0_threshold (value: {b0_threshold}) is too low, "
                        "increase your b0_threshold. It should be higher than the "
                        f"first b0 value ({bvals.min()}).",
                        stacklevel=2,
                    )
            gtab = gradient_table(
                bvals, bvecs=bvecs, b0_threshold=b0_threshold, atol=bvecs_tol
            )
            mask_vol = load_nifti_data(maskfile).astype(bool)

            n_params = ((sh_order_max + 1) * (sh_order_max + 2)) / 2
            if data.shape[-1] < n_params:
                raise ValueError(
                    f"You need at least {n_params} unique DWI volumes to "
                    f"compute fiber odfs. You currently have: {data.shape[-1]}"
                    " DWI volumes."
                )

            peaks_sphere = (
                default_sphere if sphere_name is None else get_sphere(name=sphere_name)
            )

            logging.info("FORECAST computation started.")
            forecast_model = ForecastModel(
                gtab,
                sh_order_max=sh_order_max,
                lambda_lb=lambda_lb,
                dec_alg=dec_alg,
                sphere=peaks_sphere.vertices,
                lambda_csd=lambda_csd,
            )

            peaks_forecast = peaks_from_model(
                model=forecast_model,
                data=data,
                sphere=peaks_sphere,
                relative_peak_threshold=relative_peak_threshold,
                min_separation_angle=min_separation_angle,
                mask=mask_vol,
                return_sh=True,
                sh_order_max=sh_order_max,
                normalize_peaks=True,
                parallel=parallel,
                num_processes=num_processes,
            )
            peaks_forecast.affine = affine

            save_pam(opam, peaks_forecast)

            logging.info("FORECAST computation completed.")

            if extract_pam_values:
                pam_to_niftis(
                    peaks_forecast,
                    fname_shm=oshm,
                    fname_peaks_dir=opeaks_dir,
                    fname_peaks_values=opeaks_values,
                    fname_peaks_indices=opeaks_indices,
                    fname_gfa=ogfa,
                    fname_sphere=osphere,
                    fname_b=ob,
                    fname_qa=oqa,
                    reshape_dirs=True,
                )

            dname_ = os.path.dirname(opam)
            msg = (
                "Pam5 file saved in current directory"
                if dname_ == ""
                else f"Pam5 file saved in {dname_}"
            )
            logging.info(msg)

            return io_it
