from __future__ import division, print_function, absolute_import

from glob import glob
from os.path import join, basename, splitext

import nibabel as nib
import numpy as np

from dipy.segment.mask import median_otsu


def median_otsu_bet(input_file, out_dir, save_masked=False, median_radius=4,
                numpass=4, autocrop=False, vol_idx=None, dilate=None):

    inputs = glob(input_file)
    out_dirs = glob(out_dir)
    if len(out_dirs) == 1:
        out_dirs = list(out_dirs) * len(inputs)

    for (fpath, out_dir_path) in zip(inputs, out_dirs):
        img = nib.load(fpath)
        volume = img.get_data()
        masked, mask = median_otsu(volume, median_radius, numpass, autocrop,
                                   vol_idx, dilate)

        fname, ext = splitext(basename(fpath))
        if(fname.endswith('.nii')):
            fname, _ = splitext(fname)
            ext = '.nii.gz'

        mask_fname = fname + '_mask' + ext

        mask_img = nib.Nifti1Image(mask.astype(np.float32), img.get_affine())
        mask_img.to_filename(join(out_dir_path, mask_fname))

        if save_masked:
            masked_fname = fname + '_masked' + ext
            masked_img = nib.Nifti1Image(masked, img.get_affine(), img.get_header())
            masked_img.to_filename(join(out_dir_path, masked_fname))
