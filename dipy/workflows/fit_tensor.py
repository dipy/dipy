#!/usr/bin/env python
from __future__ import division, print_function, absolute_import

import os
import numpy as np
import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel
from dipy.io.utils import nifti1_symmat
from dipy.io.bvectxt import read_bvec_file, orientation_to_string
from nibabel.trackvis import empty_header, write
from dipy.segment.mask import median_otsu


def saveqform(img, filename):
    """Some DTI/tools require the qform code to be 1. We set the affine, qform,
    and sfrom to be the same for maximum portibility.
    """
    affine = img.get_affine()
    img.set_sform(affine, 1)
    img.set_qform(affine, 1)
    nib.save(img, filename)


ft_out = {"t2di" : "%s_t2di.nii.gz",
          "fa_map" : "%s_fa.nii.gz",
          "dirfa_map" : "%s_dirFA.nii.gz",
          "md_map" : "%s_md.nii.gz",
          "ad_map" : "%s_ad.nii.gz",
          "rd_map" : "%s_rd.nii.gz",
          "dummy_trk_file" : "%s_dummy.trk",
          "mask_file" : "%s_mask.nii.gz",
          "tensor" : "%s_tensor.nii.gz"}


def fit_tensor(dwi_data, bvec=None, root=None, min_signal=1., scale=1.,
               mask='median_otsu', save_tensor=False):

    gzip_exts = set([".gz"])
    if root is None:
        pth, file = os.path.split(dwi_data)
        root, ext = os.path.splitext(dwi_data)
        if ext.lower() in gzip_exts:
            root, _ = os.path.splitext(root)
        root = os.path.join(pth, root)
    else:
        root = root

    if bvec is None:
        bvec = root + '.bvec'

    out = dict((key, path % root) for key, path in ft_out.items())

    img = nib.load(dwi_data)
    affine = img.get_affine()
    voxel_size = img.get_header().get_zooms()[:3]
    data = img.get_data()
    bvec, bval = read_bvec_file(bvec)
    gtab = gradient_table(bval / scale, bvec)

    t2di = data[..., gtab.b0s_mask].mean(-1)
    saveqform(nib.Nifti1Image(t2di.astype("float32"), affine), out['t2di'])

    if mask is None:
        out['mask_file'] = None
    elif mask.lower() == 'median_otsu':
        _, mask = median_otsu(t2di, 3, 2)
        saveqform(nib.Nifti1Image(mask.astype("uint8"), affine),
                  out['mask_file'])
    else:
        mask = nib.load(mask).get_data() > 0
        out['mask_file'] = None
    del t2di

    ten_model = TensorModel(gtab, min_signal=min_signal)
    ten = ten_model.fit(data, mask=mask)

    if save_tensor:
        lower_triangular = ten.lower_triangular()
        lower_triangular = lower_triangular.astype('float32')
        tensor_img = nifti1_symmat(lower_triangular, affine)
        saveqform(tensor_img, out['tensor'])
        del tensor_img, lower_triangular
    else:
        out['tensor'] = None

    saveqform(nib.Nifti1Image(ten.ad.astype("float32"), affine), out['ad_map'])
    saveqform(nib.Nifti1Image(ten.rd.astype("float32"), affine), out['rd_map'])
    saveqform(nib.Nifti1Image(ten.md.astype("float32"), affine), out['md_map'])
    saveqform(nib.Nifti1Image(ten.fa.astype("float32"), affine), out['fa_map'])

    dfa = np.abs(ten.fa[..., None] * ten.evecs[..., 0])
    dfa *= 256*(1.-np.finfo(float).eps)
    assert dfa.max() < 256
    assert dfa.min() >= 0
    dfa = dfa.astype('uint8')
    dtype = [('R', 'uint8'), ('G', 'uint8'), ('B', 'uint8')]
    dfa = dfa.view(dtype)
    dfa.shape = dfa.shape[:-1]
    saveqform(nib.Nifti1Image(dfa, affine), out['dirfa_map'])

    trk_hdr = empty_header()
    trk_hdr['voxel_order'] = orientation_to_string(nib.io_orientation(affine))
    trk_hdr['dim'] = ten.shape
    trk_hdr['voxel_size'] = voxel_size
    trk_hdr['vox_to_ras'] = affine
    # One streamline with two points at [0, 0, 0]
    dummy_track = [(np.zeros((2,3), dtype='float32'), None, None)]
    write(out['dummy_trk_file'], dummy_track, trk_hdr)

    return out

