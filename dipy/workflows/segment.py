from __future__ import division, print_function, absolute_import

import logging
from glob import glob
from os.path import join, basename, splitext

import nibabel as nib
import numpy as np

from dipy.workflows.utils import choose_create_out_dir
from dipy.segment.mask import median_otsu


def median_otsu_flow(input_files, out_dir='', mask='brain_mask.nii.gz',
                     masked='dwi_2x2x2_masked.nii.gz', save_masked=False,
                     median_radius=4, numpass=4, autocrop=False,
                     vol_idx=None, dilate=None):
    """ Workflow wrapping the median_otsu segmentation method.

    It applies median_otsu segmentation on each file found by 'globing'
    ``input_files`` and saves the results in a directory specified by
    ``out_dir``.

    Parameters
    ----------
    input_files : string
        Path to the input volumes. This path may contain wildcards to process
        multiple inputs at once.
    out_dir : string, optional
        Output directory (default input file directory)
    mask : string, optional
        Name of the mask volume to be saved (default 'brain_mask.nii.gz')
    masked : string, optional
        Name of the masked volume to be saved (default 'dwi_2x2x2_masked.nii.gz')
    save_masked : bool
        Save mask
    median_radius : int, optional
        Radius (in voxels) of the applied median filter(default 4)
    numpass : int, optional
        Number of pass of the median filter (default 4)
    autocrop : bool, optional
        If True, the masked input_volumes will also be cropped using the
        bounding box defined by the masked data. Should be on if DWI is
        upsampled to 1x1x1 resolution. (default False)
    vol_idx : string, optional
        1D array representing indices of ``axis=3`` of a 4D `input_volume`
        'None' (the default) corresponds to ``(0,)`` (assumes first volume in
        4D array)
    dilate : string, optional
        number of iterations for binary dilation (default 'None')

    Outputs
    -------
    mask : Nifti File
           Binary volume representing the computed mask.
    masked : Nifti File
            Volume representing the masked input. This file is saved
            save_masked is True.
    """
    for fpath in glob(input_files):
        logging.info('Applying median_otsu segmentation on {0}'.format(fpath))
        img = nib.load(fpath)
        volume = img.get_data()

        masked_volume, mask_volume = median_otsu(volume, median_radius,
                                   numpass, autocrop,
                                   vol_idx, dilate)

        out_dir_path = choose_create_out_dir(out_dir, fpath)

        mask_img = nib.Nifti1Image(mask_volume.astype(np.float32), img.get_affine())
        mask_out_path = join(out_dir_path, mask)
        mask_img.to_filename(mask_out_path)
        logging.info('Mask saved as {0}'.format(mask_out_path))

        if save_masked:
            masked_img = nib.Nifti1Image(masked_volume, img.get_affine(), img.get_header())
            masked_out_path = join(out_dir_path, masked)
            masked_img.to_filename(masked_out_path)
            logging.info('Masked volume saved as {0}'.format(masked_out_path))
