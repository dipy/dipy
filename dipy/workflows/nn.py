from pathlib import Path
import sys

import numpy as np

from dipy.core.gradients import gradient_table
from dipy.denoise.bias_correction import bias_field_correction
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
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
            mask_volume = evac.predict(data, affine, voxsize=voxsize)
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
        method="n4",
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
        method : string, optional
            Bias field correction method. Choose from:
                - 'n4': DeepN4 bias field correction.
                  See :footcite:p:`Kanakaraj2024` for more details.
                - 'poly': Legendre polynomial regression bias correction.
                - 'bspline': Cubic B-spline regression bias correction.
                - 'auto': Run both poly and bspline, return the one with
                  lower Coefficient of Variation within the brain mask.

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
        zero_background : bool, optional
            If True, set the saved bias field to 1.0 outside the brain mask,
            suppressing extrapolation artifacts in the background
            (poly/bspline methods).
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

        self.update_flat_outputs(self.flat_outputs, io_it)
        for fpath, corrected_out_path in io_it:
            logger.info(f"Applying bias field correction on {fpath}")

            data, affine, img, voxsize = load_nifti(
                fpath, return_img=True, return_voxsize=True
            )

            corrected_data = None
            if method.lower() == "n4":
                deepn4_model = DeepN4(verbose=verbose, use_cuda=use_cuda)
                deepn4_model.fetch_default_weights()
                corrected_data = deepn4_model.predict(
                    data, affine, voxsize=voxsize, threshold=threshold
                )
            elif method.lower() in ["poly", "bspline", "auto"]:
                bvals, bvecs = read_bvals_bvecs(bval, bvec)
                gtab = gradient_table(bvals, bvecs=bvecs)
                levels = tuple(int(x.strip()) for x in pyramid_levels.split(","))
                n_ctrl = (int(n_control_points),) * 3
                corrected_data, bias = bias_field_correction(
                    data,
                    gtab,
                    method=method.lower(),
                    order=int(order),
                    n_control_points=n_ctrl,
                    pyramid_levels=levels,
                    n_iter=int(n_iter),
                    lambda_reg=float(lambda_reg),
                    robust=bool(robust),
                    gradient_weighting=bool(gradient_weighting),
                    return_bias_field=True,
                    zero_background=bool(zero_background),
                )
                bias_out_path = Path(corrected_out_path).parent / out_bias_field
                save_nifti(str(bias_out_path), bias.astype(np.float32), affine)
                logger.info(f"Bias field saved as {bias_out_path}")

            save_nifti(corrected_out_path, corrected_data, affine, hdr=img.header)
            logger.info(f"Corrected volume saved as {corrected_out_path}")

        return io_it
