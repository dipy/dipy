import logging

import numpy as np

from dipy.io.image import save_nifti, load_nifti
from dipy.nn.evac import EVACPlus
from dipy.workflows.workflow import Workflow


class EVACPlusFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return 'evacplus'

    def run(self, input_files, save_masked=False, out_dir='',
            out_mask='brain_mask.nii.gz', out_masked='dwi_masked.nii.gz'):
        """Extract brain using EVAC+.

        Parameters
        ----------
        input_files : string
            Path to the input volumes. This path may contain wildcards to
            process multiple inputs at once.
        save_masked : bool, optional
            Save mask.
        out_dir : string, optional
            Output directory. (default current directory)
        out_mask : string, optional
            Name of the mask volume to be saved.
        out_masked : string, optional
            Name of the masked volume to be saved.

        References
        ----------
        ..  [Park2022] Park, J.S., Fadnavis, S., & Garyfallidis, E. (2022).
        EVAC+: Multi-scale V-net with Deep Feature
        CRF Layers for Brain Extraction.

        """
        io_it = self.get_io_iterator()
        empty_flag = True

        for fpath, mask_out_path, masked_out_path in io_it:
            logging.info('Applying evac+ brain extraction on {0}'.
                         format(fpath))

            data, affine, img, voxsize = load_nifti(fpath, return_img=True,
                                                    return_voxsize=True)
            evac = EVACPlus()
            mask_volume = evac.predict(data, affine, voxsize)
            masked_volume = mask_volume * data

            save_nifti(mask_out_path, mask_volume.astype(np.float64), affine)

            logging.info('Mask saved as {0}'.format(mask_out_path))

            if save_masked:
                save_nifti(masked_out_path, masked_volume, affine,
                           img.header)

                logging.info('Masked volume saved as {0}'.
                             format(masked_out_path))
            empty_flag = False
        if empty_flag:
            raise ValueError("All output paths exists."
                             " If you want to overwrite "
                             "please use the --force option.")

        return io_it
