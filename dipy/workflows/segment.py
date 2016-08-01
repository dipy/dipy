from __future__ import division, print_function, absolute_import

import logging
import numpy as np

import nibabel as nib

from dipy.segment.mask import median_otsu
from dipy.workflows.workflow import Workflow


class MedianOtsuFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return 'medotsu'


    def run(self, input_files, save_masked=False, median_radius=2, numpass=5,
            autocrop=False, vol_idx=None, dilate=None, out_dir='',
            out_mask='brain_mask.nii.gz', out_masked='dwi_masked.nii.gz'):
        """ Workflow wrapping the median_otsu segmentation method.

        Applies median_otsu segmentation on each file found by 'globing'
        ``input_files`` and saves the results in a directory specified by
        ``out_dir``.

        Parameters
        ----------
        input_files : string
            Path to the input volumes. This path may contain wildcards to
            process multiple inputs at once.
        save_masked : bool
            Save mask
        median_radius : int, optional
            Radius (in voxels) of the applied median filter (default 2)
        numpass : int, optional
            Number of pass of the median filter (default 5)
        autocrop : bool, optional
            If True, the masked input_volumes will also be cropped using the
            bounding box defined by the masked data. For example, if diffusion
            images are of 1x1x1 (mm^3) or higher resolution auto-cropping could
            reduce their size in memory and speed up some of the analysis.
            (default False)
        vol_idx : string, optional
            1D array representing indices of ``axis=3`` of a 4D `input_volume`
            'None' (the default) corresponds to ``(0,)`` (assumes first volume
            in 4D array)
        dilate : string, optional
            number of iterations for binary dilation (default 'None')
        out_dir : string, optional
            Output directory (default input file directory)
        out_mask : string, optional
            Name of the mask volume to be saved (default 'brain_mask.nii.gz')
        out_masked : string, optional
            Name of the masked volume to be saved (default 'dwi_masked.nii.gz')
        """
        io_it = self.get_io_iterator()

        for fpath, mask_out_path, masked_out_path in io_it:
            logging.info('Applying median_otsu segmentation on {0}'.
                         format(fpath))

            img = nib.load(fpath)
            volume = img.get_data()

            masked_volume, mask_volume = median_otsu(volume, median_radius,
                                                     numpass, autocrop, vol_idx,
                                                     dilate)

            mask_img = nib.Nifti1Image(mask_volume.astype(np.float32),
                                       img.get_affine())

            mask_img.to_filename(mask_out_path)
            logging.info('Mask saved as {0}'.format(mask_out_path))

            if save_masked:
                masked_img = nib.Nifti1Image(masked_volume, img.get_affine(),
                                             img.get_header())

                masked_img.to_filename(masked_out_path)
                logging.info('Masked volume saved as {0}'.
                             format(masked_out_path))

        return io_it
