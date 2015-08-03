from __future__ import division, print_function, absolute_import

from glob import glob
from os.path import join, basename, splitext, dirname, isabs, exists
from os import makedirs

import nibabel as nib
import numpy as np

from dipy.segment.mask import median_otsu


def median_otsu_bet(input_file, out_dir, save_masked=False, median_radius=4,
                    numpass=4, autocrop=False, vol_idx=None, dilate=None):

    for fpath in glob(input_file):
        img = nib.load(fpath)
        volume = img.get_data()
        masked, mask = median_otsu(volume, median_radius, numpass, autocrop,
                                   vol_idx, dilate)

        fname, ext = splitext(basename(fpath))
        if(fname.endswith('.nii')):
            fname, _ = splitext(fname)
            ext = '.nii.gz'

        mask_fname = fname + '_mask' + ext

        if out_dir == '':
            out_dir_path = dirname(fpath)
        elif not isabs(out_dir):
            out_dir_path = join(dirname(fpath), out_dir)
            if not exists(out_dir_path):
                makedirs(out_dir_path)
        else:
            out_dir_path = out_dir

        mask_img = nib.Nifti1Image(mask.astype(np.float32), img.get_affine())
        mask_img.to_filename(join(out_dir_path, mask_fname))

        if save_masked:
            masked_fname = fname + '_masked' + ext
            masked_img = nib.Nifti1Image(masked, img.get_affine(), img.get_header())
            masked_img.to_filename(join(out_dir_path, masked_fname))
