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


def save_qa_metric(fname, xopt, fopt):
    """Save Quality Assurance metrics.

    Parameters
    ----------
    fname: string
        File name to save the metric values.
    xopt: numpy array
        The metric containing the
        optimal parameters for
        image registration.
    fopt: int
        The distance between the registered images.
    """
    np.savetxt(fname, xopt, header="Optimal Parameter metric")
    with open(fname, 'a') as f:
        f.write('# Distance after registration\n')
        f.write(str(fopt))
