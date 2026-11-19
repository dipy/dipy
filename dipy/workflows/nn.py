from pathlib import Path
import sys

import numpy as np

from dipy.align.disco import synb0_syn
from dipy.core.gradients import gradient_table
from dipy.denoise.bias_correction import bias_field_correction
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, load_nifti_data, save_nifti
from dipy.nn.deepn4 import DeepN4
from dipy.nn.evac import EVACPlus
from dipy.utils.logging import logger
from dipy.workflows.workflow import Workflow


class EVACPlusFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "evacplus"

    def run(
        self,
        input_files,
        save_masked=False,
        out_dir="",
        out_mask="brain_mask.nii.gz",
        out_masked="brain_masked.nii.gz",
    ):
        """Extract brain using EVAC+.

        See :footcite:p:`Park2024` for further details about EVAC+.

        Parameters
        ----------
        input_files : string or Path
            Path to the input volumes. This path may contain wildcards to
            process multiple inputs at once.
        save_masked : bool, optional
            Save mask.
        out_dir : string or Path, optional
            Output directory.
        out_mask : string, optional
            Name of the mask volume to be saved.
        out_masked : string, optional
            Name of the masked volume to be saved.

        References
        ----------
        .. footbibliography::
        """
        io_it = self.get_io_iterator()
        empty_flag = True

        for fpath, mask_out_path, masked_out_path in io_it:
            logger.info(f"Applying evac+ brain extraction on {fpath}")

            data, affine, img, voxsize = load_nifti(
                fpath, return_img=True, return_voxsize=True
            )
            evac = EVACPlus()
            mask_volume = evac.predict(data, affine)
            masked_volume = mask_volume * data

            save_nifti(mask_out_path, mask_volume.astype(np.float64), affine)

            logger.info(f"Mask saved as {mask_out_path}")

            if save_masked:
                save_nifti(masked_out_path, masked_volume, affine, hdr=img.header)

                logger.info(f"Masked volume saved as {masked_out_path}")
            empty_flag = False
        if empty_flag:
            raise ValueError(
                "All output paths exists."
                " If you want to overwrite "
                "please use the --force option."
            )

        return io_it


class BiasFieldCorrectionFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return "bias_field_correction"

    def run(
        self,
        input_files,
        bval=None,
        bvec=None,
        mask=None,
        method="bspline",
        threshold=0.5,
        use_cuda=False,
        verbose=False,
        order=3,
        n_control_points=8,
        pyramid_levels="4,2,1",
        n_iter=4,
        lambda_reg=1e-3,
        robust=True,
        gradient_weighting=True,
        smoothness=10.0,
        sharpen=True,
        max_iter=50,
        convergence_threshold=1e-3,
        shrink_factor=2,
        zero_background=False,
        out_dir="",
        out_corrected="biasfield_corrected.nii.gz",
        out_bias_field="bias_field.nii.gz",
    ):
        """Correct bias field.

        Parameters
        ----------
        input_files : string or Path
            Path to the input volumes. This path may contain wildcards to
            process multiple inputs at once.
        bval : string or Path, optional
            Path to the b-value file.
        bvec : string or Path, optional
            Path to the b-vector file.
        mask : string or Path, optional
            Path to the brain mask file. If not provided, a mask will be estimated
            from the input volume using Otsu's method for 'poly' and 'bspline' methods.
        method : string, optional
            Bias field correction method. Choose from:
                - 'n4': DeepN4 bias field correction.
                  See :footcite:p:`Kanakaraj2024` for more details.
                - 'poly': Legendre polynomial regression bias correction.
                - 'bspline': Cubic B-spline regression bias correction.
                - 'auto': Deprecated since 1.13.0, removed in 2.0.0. Run both
                  poly and bspline, return the one with lower Coefficient of
                  Variation within the brain mask.

            'n4' method is recommended for T1-weighted images. 'poly' and 'bspline'
            methods are recommended for diffusion-weighted images.
        threshold : float, optional
            Threshold for cleaning the final correction field in DeepN4 method.
        use_cuda : bool, optional
            Use CUDA for DeepN4 bias field correction.
        verbose : bool, optional
            Print verbose output.
        order : int, optional
            Maximum Legendre polynomial degree (used with method='poly').
        n_control_points : int, optional
            Control grid size per axis (used with method='bspline').
        pyramid_levels : string, optional
            Comma-separated downsampling factors for coarse-to-fine pyramid,
            e.g. '4,2,1' (used with method='poly' or 'bspline').
        n_iter : int, optional
            Reweighting iterations per pyramid level (poly/bspline methods).
        lambda_reg : float, optional
            Ridge regularization strength (poly/bspline methods).
        robust : bool, optional
            Apply Tukey biweight robust reweighting (poly/bspline methods).
        gradient_weighting : bool, optional
            Apply gradient-based edge suppression (poly/bspline methods).
        smoothness : float, optional
            Bending energy penalty on the B-spline control lattice, relative
            to the data term. 0 disables it (bspline method).
        sharpen : bool, optional
            Iterate the regression inside the N4 histogram-sharpening loop
            so tissue contrast is not absorbed into the field. If False, a
            single direct regression of the log b0 is used (poly/bspline
            methods).
        max_iter : int, optional
            Maximum number of sharpening iterations (poly/bspline methods).
        convergence_threshold : float, optional
            Sharpening stops when the coefficient of variation of the field
            update inside the mask falls below this value (poly/bspline
            methods).
        shrink_factor : int, optional
            Downsampling factor used during the sharpening iterations
            (poly/bspline methods).
        zero_background : bool, optional
            If True, set the bias field to 1.0 outside the brain mask so
            background voxels are left untouched. If False, the in-mask
            field is extrapolated to the background and the whole volume is
            corrected (poly/bspline methods).
        out_dir : string or Path, optional
            Output directory.
        out_corrected : string, optional
            Name of the corrected volume to be saved.
        out_bias_field : string, optional
            Name of the bias field volume to be saved (poly/bspline methods).

        References
        ----------
        .. footbibliography::

        """
        io_it = self.get_io_iterator()

        if method.lower() not in ["n4", "poly", "bspline", "auto"]:
            logger.error(
                "Unknown bias field correction method. "
                "Choose from 'n4', 'poly', 'bspline', 'auto'."
            )
            sys.exit(1)

        prefix = "t1" if method.lower() == "n4" else "dwi"
        for i, name in enumerate(self.flat_outputs):
            if str(name).endswith("biasfield_corrected.nii.gz"):
                self.flat_outputs[i] = Path(name).parent / Path(
                    "biasfield_corrected.nii.gz"
                ).with_name(f"{prefix}_biasfield_corrected.nii.gz")

        if io_it:
            self.update_flat_outputs(self.flat_outputs, io_it)
        else:
            if isinstance(self.last_generated_outputs, dict):
                self.last_generated_outputs = dict(
                    zip(self.last_generated_outputs.keys(), self.flat_outputs)
                )
            else:
                self.last_generated_outputs = self.flat_outputs
        for fpath, corrected_out_path, obf in io_it:
            logger.info(f"Applying bias field correction on {fpath}")

            data, affine, img, voxsize = load_nifti(
                fpath, return_img=True, return_voxsize=True
            )

            corrected_data = None
            if method.lower() == "n4":
                deepn4_model = DeepN4(verbose=verbose, use_cuda=use_cuda)
                deepn4_model.fetch_default_weights()
                corrected_data = deepn4_model.predict(data, affine, threshold=threshold)
            elif method.lower() in ["poly", "bspline", "auto"]:
                bvals, bvecs = read_bvals_bvecs(bval, bvec)
                gtab = gradient_table(bvals, bvecs=bvecs)
                levels = tuple(int(x.strip()) for x in pyramid_levels.split(","))
                n_ctrl = (int(n_control_points),) * 3
                mask_arr = load_nifti_data(mask) if mask is not None else None
                corrected_data, bias = bias_field_correction(
                    data,
                    gtab,
                    mask=mask_arr,
                    method=method.lower(),
                    order=int(order),
                    n_control_points=n_ctrl,
                    pyramid_levels=levels,
                    n_iter=int(n_iter),
                    lambda_reg=float(lambda_reg),
                    robust=bool(robust),
                    gradient_weighting=bool(gradient_weighting),
                    smoothness=float(smoothness),
                    sharpen=bool(sharpen),
                    max_iter=int(max_iter),
                    convergence_threshold=float(convergence_threshold),
                    shrink_factor=int(shrink_factor),
                    return_bias_field=True,
                    zero_background=bool(zero_background),
                )
                save_nifti(str(obf), bias.astype(np.float32), affine)
                logger.info(f"Bias field saved as {obf}")

            save_nifti(corrected_out_path, corrected_data, affine, hdr=img.header)
            logger.info(f"Corrected volume saved as {corrected_out_path}")

        return io_it


class Synb0SynFlow(Workflow):
    """Workflow for Synb0-SyN susceptibility distortion correction."""

    @classmethod
    def get_short_name(cls):
        return "synb0_syn"

    def run(
        self,
        dwi_files,
        t1_files,
        b0_index=0,
        pe_axis=1,
        dwi_mask_file="",
        t1_mask_file="",
        apply_skull_strip=False,
        apply_deepn4=False,
        b0_to_t1_method="bbr",
        num_threads=None,
        out_dir="",
        out_corrected="corrected.nii.gz",
        out_field="field.nii.gz",
    ):
        """Synb0-SyN distortion correction for diffusion MRI data.

        This workflow uses Synb0-SyN to correct
        susceptibility-induced distortions in diffusion MRI data
        :footcite:p:`Chigurupati2024`, :footcite:p:`Schilling2019`,
        :footcite:p:`Schilling2020`.

        Parameters
        ----------
        dwi_files : string or Path
            Path to the input diffusion MRI volumes. This path may contain
            wildcards to process multiple inputs at once.
        t1_files : string or Path
            Path to the T1-weighted structural image files. This path may
            contain wildcards to use multiple T1 files at once.
        b0_index : int, optional
            The index of the b0 volume used for field estimation when
            the DWI data is 4D.
        pe_axis : int, optional
            The phase-encoding axis, as a voxel axis of the DWI (0, 1 or 2),
            not a world axis. The returned field is the displacement in mm
            along this axis.
        dwi_mask_file : string or Path, optional
            Path to the DWI brain mask file. If empty, no masking is applied.
        t1_mask_file : string or Path, optional
            Path to the T1 brain mask file. If empty, no masking is applied.
        apply_skull_strip : bool, optional
            If True, apply MindGrab skull stripping to both the T1 and
            DWI b0 images before registration.
        apply_deepn4 : bool, optional
            If True, apply DeepN4 bias field correction to the T1 image
            before the first affine registration. DeepN4 requires PyTorch
            or TensorFlow.
        b0_to_t1_method : string, optional
            How the b0 is aligned to the T1: 'mi' for a rigid registration
            maximizing mutual information, 'bbr' to refine it with a rigid
            boundary-based registration :footcite:p:`Greve2009`, or
            'bbr_scaling' to also estimate a scaling along each axis, for a
            T1 and a DWI that do not share the same geometry.
        num_threads : int, optional
            Number of threads used by the registrations, MindGrab, Synb0 and
            the final warping. If None, the default of each step is used; a
            negative value uses all the threads minus $|num_threads + 1|$.
        out_dir : string or Path, optional
            Directory to save the corrected image and field map.
        out_corrected : string, optional
            Name for the saved distortion-corrected DWI image.
        out_field : string, optional
            Name for the saved estimated deformation field.

        References
        ----------
        .. footbibliography::
        """
        io_it = self.get_io_iterator()

        for dwi_path, t1_path, ocorrected, ofield in io_it:
            logger.info(f"Loading DWI file {dwi_path}")
            dwi_data, dwi_affine = load_nifti(dwi_path)

            logger.info(f"Loading T1 file {t1_path}")
            t1_data, t1_affine = load_nifti(t1_path)

            dwi_mask = None
            if dwi_mask_file:
                logger.info(f"Loading DWI mask file {dwi_mask_file}")
                dwi_mask, _ = load_nifti(dwi_mask_file)

            t1_mask = None
            if t1_mask_file:
                logger.info(f"Loading T1 mask file {t1_mask_file}")
                t1_mask, _ = load_nifti(t1_mask_file)

            corrected_dwi, field = synb0_syn(
                dwi=dwi_data,
                T1=t1_data,
                dwi_affine=dwi_affine,
                T1_affine=t1_affine,
                b0_index=b0_index,
                dwi_mask=dwi_mask,
                T1_mask=t1_mask,
                pe_axis=pe_axis,
                return_field=True,
                apply_skull_strip=apply_skull_strip,
                apply_deepn4=apply_deepn4,
                b0_to_t1_method=b0_to_t1_method,
                num_threads=num_threads,
            )

            logger.info(f"Saving corrected DWI to {ocorrected}")
            save_nifti(ocorrected, corrected_dwi, dwi_affine)

            logger.info(f"Saving deformation field to {ofield}")
            save_nifti(ofield, field, dwi_affine)
