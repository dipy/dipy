import logging
import sys

import numpy as np

from dipy.core.gradients import extract_b0, gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
from dipy.nn.deepn4 import DeepN4
from dipy.nn.evac import EVACPlus
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
        out_masked="dwi_masked.nii.gz",
    ):
        """Extract brain using EVAC+.

        See :footcite:p:`Park2024` for further details about EVAC+.

        Parameters
        ----------
        input_files : string
            Path to the input volumes. This path may contain wildcards to
            process multiple inputs at once.
        save_masked : bool, optional
            Save mask.
        out_dir : string, optional
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
            logging.info(f"Applying evac+ brain extraction on {fpath}")

            data, affine, img, voxsize = load_nifti(
                fpath, return_img=True, return_voxsize=True
            )
            evac = EVACPlus()
            mask_volume = evac.predict(data, affine, voxsize=voxsize)
            masked_volume = mask_volume * data

            save_nifti(mask_out_path, mask_volume.astype(np.float64), affine)

            logging.info(f"Mask saved as {mask_out_path}")

            if save_masked:
                save_nifti(masked_out_path, masked_volume, affine, hdr=img.header)

                logging.info(f"Masked volume saved as {masked_out_path}")
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
        out_dir="",
        out_corrected="biasfield_corrected.nii.gz",
    ):
        """Correct bias field.

        Parameters
        ----------
        input_files : string
            Path to the input volumes. This path may contain wildcards to
            process multiple inputs at once.
        bval : string, optional
            Path to the b-value file.
        bvec : string, optional
            Path to the b-vector file.
        method : string, optional
            Bias field correction method. Choose from:
                - 'n4': DeepN4 bias field correction.
                  See :footcite:p:`Kanakaraj2024` for more details.
                - 'b0': B0 bias field correction via normalization.

            'n4' method is recommended for T1-weighted images where 'b0' method
            is recommended for diffusion-weighted images.
        threshold : float, optional
            Threshold for cleaning the final correction field in DeepN4 method.
        use_cuda : bool, optional
            Use CUDA for DeepN4 bias field correction.
        verbose : bool, optional
            Print verbose output.
        out_dir : string, optional
            Output directory.
        out_corrected : string, optional
            Name of the corrected volume to be saved.

        References
        ----------
        .. footbibliography::

        """
        io_it = self.get_io_iterator()

        if method.lower() not in ["n4", "b0"]:
            logging.error("Unknown bias field correction method. Choose from 'n4, b0'.")
            sys.exit(1)

        prefix = "t1" if method.lower() == "n4" else "dwi"
        for i, name in enumerate(self.flat_outputs):
            if name.endswith("biasfield_corrected.nii.gz"):
                self.flat_outputs[i] = name.replace(
                    "biasfield_corrected.nii.gz", f"{prefix}_biasfield_corrected.nii.gz"
                )

        self.update_flat_outputs(self.flat_outputs, io_it)
        for fpath, corrected_out_path in io_it:
            logging.info(f"Applying bias field correction on {fpath}")

            data, affine, img, voxsize = load_nifti(
                fpath, return_img=True, return_voxsize=True
            )

            corrected_data = None
            if method.lower() == "b0":
                bvals, bvecs = read_bvals_bvecs(bval, bvec)
                gtab = gradient_table(bvals, bvecs=bvecs)
                b0 = extract_b0(data, gtab.b0s_mask)
                for i in range(data.shape[-1]):
                    data[..., i] = np.divide(
                        data[..., i],
                        b0,
                        out=np.zeros_like(data[..., i]).astype(float),
                        where=b0 != 0,
                    )
                corrected_data = data
            elif method.lower() == "n4":
                deepn4_model = DeepN4(verbose=verbose, use_cuda=use_cuda)
                deepn4_model.fetch_default_weights()
                corrected_data = deepn4_model.predict(
                    data, affine, voxsize=voxsize, threshold=threshold
                )

            save_nifti(corrected_out_path, corrected_data, affine, hdr=img.header)
            logging.info(f"Corrected volume saved as {corrected_out_path}")

        return io_it
