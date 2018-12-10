#!/usr/bin/env python

import os
from os.path import join

import nibabel as nib
from nibabel.tmpdirs import TemporaryDirectory

import numpy as np

from nose.tools import assert_true

from dipy.data import get_data
from dipy.workflows.stats import SNRinCCFlow


def test_stats():
    with TemporaryDirectory() as out_dir:
        data_path, bval_path, bvec_path = get_data('small_101D')
        vol_img = nib.load(data_path)
        volume = vol_img.get_data()
        mask = np.ones_like(volume[:, :, :, 0])
        mask_img = nib.Nifti1Image(mask.astype(np.uint8), vol_img.affine)
        mask_path = join(out_dir, 'tmp_mask.nii.gz')
        nib.save(mask_img, mask_path)

        snr_flow = SNRinCCFlow(force=True)
        args = [data_path, bval_path, bvec_path, mask_path]

        snr_flow.run(*args, out_dir=out_dir)
        assert_true(os.path.exists(os.path.join(out_dir, 'product.json')))
        assert_true(os.stat(os.path.join(
            out_dir, 'product.json')).st_size != 0)
        assert_true(os.path.exists(os.path.join(out_dir, 'cc.nii.gz')))
        assert_true(os.stat(os.path.join(out_dir, 'cc.nii.gz')).st_size != 0)
        assert_true(os.path.exists(os.path.join(out_dir, 'mask_noise.nii.gz')))
        assert_true(os.stat(os.path.join(
            out_dir, 'mask_noise.nii.gz')).st_size != 0)

        snr_flow._force_overwrite = True
        snr_flow.run(*args, out_dir=out_dir)
        assert_true(os.path.exists(os.path.join(out_dir, 'product.json')))
        assert_true(os.stat(os.path.join(
            out_dir, 'product.json')).st_size != 0)
        assert_true(os.path.exists(os.path.join(out_dir, 'cc.nii.gz')))
        assert_true(os.stat(os.path.join(out_dir, 'cc.nii.gz')).st_size != 0)
        assert_true(os.path.exists(os.path.join(out_dir, 'mask_noise.nii.gz')))
        assert_true(os.stat(os.path.join(
            out_dir, 'mask_noise.nii.gz')).st_size != 0)

        snr_flow._force_overwrite = True
        snr_flow.run(*args, bbox_threshold=(0.5, 1, 0,
                                            0.15, 0, 0.2), out_dir=out_dir)
        assert_true(os.path.exists(os.path.join(out_dir, 'product.json')))
        assert_true(os.stat(os.path.join(
            out_dir, 'product.json')).st_size != 0)
        assert_true(os.path.exists(os.path.join(out_dir, 'cc.nii.gz')))
        assert_true(os.stat(os.path.join(out_dir, 'cc.nii.gz')).st_size != 0)
        assert_true(os.path.exists(os.path.join(out_dir, 'mask_noise.nii.gz')))
        assert_true(os.stat(os.path.join(
            out_dir, 'mask_noise.nii.gz')).st_size != 0)


if __name__ == '__main__':
    test_stats()
