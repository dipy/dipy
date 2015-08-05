from __future__ import division, print_function, absolute_import

from glob import glob
from os.path import join, basename, splitext

import nibabel as nib
import numpy as np

from dipy.workflows.utils import choose_create_out_dir, int_param, bool_param, \
    int_list_param
from dipy.segment.mask import median_otsu


def median_otsu_flow(input_files, out_dir, save_masked=str(False),
                     median_radius=str(4), numpass=str(4), autocrop=str(False),
                     vol_idx=str(None), dilate=str(None)):
    """ Workflow wrapping the median_otsu segmentation method.

    It applies median_otsu segmentation on each file found by 'globing'
    ``input_files`` and saves the results in a directory specified by ``out_dir``

    Parameters
    ----------
    input_files : string
        Path to the input volumes. This path may contain wildcards to process
        multiple inputs at once.
    median_radius : string, optional
        Radius (in voxels) of the applied median filter(default 4)
    numpass: string, optional
        Number of pass of the median filter (default 4)
    autocrop: string, optional
        if True, the masked input_volumes will also be cropped using the
        bounding box defined by the masked data. Should be on if DWI is
        upsampled to 1x1x1 resolution. (default False)
    vol_idx : string, optional
        1D array representing indices of ``axis=3`` of a 4D `input_volume`
        'None' (the default) corresponds to ``(0,)`` (assumes first volume in
        4D array)
    dilate : string, optional
        number of iterations for binary dilation (default 'None')
    """

    for fpath in glob(input_files):
        print('')
        print('Applying median_otsu segmentation on {0}'.format(fpath))
        img = nib.load(fpath)
        volume = img.get_data()

        masked, mask = median_otsu(volume, int_param(median_radius),
                                   int_param(numpass), bool_param(autocrop),
                                   int_list_param(vol_idx), int_param(dilate))

        fname, ext = splitext(basename(fpath))
        if(fname.endswith('.nii')):
            fname, _ = splitext(fname)
            ext = '.nii.gz'

        mask_fname = fname + '_mask' + ext

        out_dir_path = choose_create_out_dir(out_dir, fpath)

        mask_img = nib.Nifti1Image(mask.astype(np.float32), img.get_affine())
        mask_out_path = join(out_dir_path, mask_fname)
        mask_img.to_filename(mask_out_path)
        print('Mask saved as {0}'.format(mask_out_path))

        if bool_param(save_masked):
            masked_fname = fname + '_bet' + ext
            masked_img = nib.Nifti1Image(masked, img.get_affine(), img.get_header())
            masked_out_path = join(out_dir_path, masked_fname)
            masked_img.to_filename(masked_out_path)
            print('Masked volume saved as {0}'.format(masked_out_path))
