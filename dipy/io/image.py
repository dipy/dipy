from __future__ import division, print_function, absolute_import

import nibabel as nib
import numpy as np


def load_nifti(fname, return_img=False, return_voxsize=False):
    img = nib.load(fname)
    hdr = img.get_header()
    data = img.get_data()
    vox_size = hdr.get_zooms()[:3]

    ret_val = [data, img.get_affine()]
    if return_voxsize:
        ret_val.append(vox_size)
    if return_img:
        ret_val.append(img)

    return tuple(ret_val)