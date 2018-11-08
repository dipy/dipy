#!/usr/bin/env python

import os
from os.path import join

import nibabel as nib
from nibabel.tmpdirs import TemporaryDirectory

import numpy as np

from nose.tools import assert_true, assert_equal

from dipy.data import get_data
from dipy.workflows.stats2 import SNRinCCFlow


def test_stats():
    with TemporaryDirectory() as out_dir:
        data_path, bval_path, bvec_path = get_data('small_101D')
        #data_path, bval_path, bvec_path = '/Users/davidhunt/Documents/5bb3f635cb555c003fa214c0/dwi.nii.gz', '/Users/davidhunt/Documents/5bb3f635cb555c003fa214c0/dwi.bvals', '/Users/davidhunt/Documents/5bb3f635cb555c003fa214c0/dwi.bvecs'
        vol_img = nib.load(data_path)
        volume = vol_img.get_data()
        mask = np.ones_like(volume[:, :, :, 0])
        mask_img = nib.Nifti1Image(mask.astype(np.uint8), vol_img.affine)
        mask_path = join(out_dir, 'tmp_mask.nii.gz')
        nib.save(mask_img, mask_path)


        snr_flow = SNRinCCFlow(force=True)

        args = [data_path, bval_path, bvec_path]
        snr_flow.run(*args, out_dir=out_dir)
        
        assert_true(os.path.exists(os.path.join(out_dir, 'product.json')))
        assert_true(os.stat(os.path.join(out_dir, 'product.json')).st_size != 0)

        file_obj = open(os.path.join(out_dir, 'product.json'), 'r')
        print file_obj.read()
        
        snr_flow.run(*args, mask=mask_path, out_dir=out_dir)
        
        assert_true(os.path.exists(os.path.join(out_dir,'product.json')))
        assert_true(os.stat(os.path.join(out_dir,'product.json')).st_size != 0)

        file_obj = open(os.path.join(out_dir, 'product.json'), 'r')
        print file_obj.read()
        
        snr_flow.run(*args, bbox_threshold=(0.3,1,0,1,0,0.5), out_dir=out_dir)

        assert_true(os.path.exists(os.path.join(out_dir, 'product.json')))
        assert_true(os.stat(os.path.join(out_dir, 'product.json')).st_size != 0)

        file_obj = open(os.path.join(out_dir, 'product.json'), 'r')
        print file_obj.read()

if __name__ == '__main__':
    test_stats()
