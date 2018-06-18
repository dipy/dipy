from __future__ import division, print_function, absolute_import

import nibabel as nib
import numpy as np


def load_nifti(fname, return_img=False, return_voxsize=False,
               return_coords=False):
    img = nib.load(fname)
    data = img.get_data()
    vox_size = img.header.get_zooms()[:3]

    ret_val = [data, img.affine]

    if return_img:
        ret_val.append(img)
    if return_voxsize:
        ret_val.append(vox_size)
    if return_coords:
        ret_val.append(nib.aff2axcodes(img.affine))

    return tuple(ret_val)


def save_nifti(fname, data, affine, hdr=None):
    result_img = nib.Nifti1Image(data, affine, header=hdr)
    result_img.to_filename(fname)


def save_affine_matrix(fname, affine):
    """
    Parameters
    ---------
    fname : str
        File name to save the affine matrix.
    affine : numpy array
        The object containing the affine matrix.
    """
    np.savetxt(fname, affine)
