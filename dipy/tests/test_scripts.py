# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Test scripts

Run scripts and check outputs
"""
'''
from __future__ import division, print_function, absolute_import

import glob
import os
import shutil

from os.path import (dirname, join as pjoin, abspath)

from nose.tools import assert_true, assert_false, assert_equal
import numpy.testing as nt

import nibabel as nib
from nibabel.tmpdirs import InTemporaryDirectory

from dipy.data import get_fnames

# Quickbundles command-line requires matplotlib:
try:
    import matplotlib
    no_mpl = False
except ImportError:
    no_mpl = True

from dipy.tests.scriptrunner import ScriptRunner

runner = ScriptRunner(
    script_sdir='bin',
    debug_print_var='NIPY_DEBUG_PRINT')
run_command = runner.run_command

DATA_PATH = abspath(pjoin(dirname(__file__), 'data'))


def test_dipy_peak_extraction():
    # test dipy_peak_extraction script
    cmd = 'dipy_peak_extraction'
    code, stdout, stderr = run_command(cmd, check_code=False)
    assert_equal(code, 2)


def test_dipy_fit_tensor():
    # test dipy_fit_tensor script
    cmd = 'dipy_fit_tensor'
    code, stdout, stderr = run_command(cmd, check_code=False)
    assert_equal(code, 2)


def test_dipy_sh_estimate():
    # test dipy_sh_estimate script
    cmd = 'dipy_sh_estimate'
    code, stdout, stderr = run_command(cmd, check_code=False)
    assert_equal(code, 2)


def assert_image_shape_affine(filename, shape, affine):
    assert_true(os.path.isfile(filename))
    image = nib.load(filename)
    assert_equal(image.shape, shape)
    nt.assert_array_almost_equal(image.affine, affine)


def test_dipy_fit_tensor_again():
    with InTemporaryDirectory():
        dwi, bval, bvec = get_fnames("small_25")
        # Copy data to tmp directory
        shutil.copyfile(dwi, "small_25.nii.gz")
        shutil.copyfile(bval, "small_25.bval")
        shutil.copyfile(bvec, "small_25.bvec")
        # Call script
        cmd = ["dipy_fit_tensor", "--mask=none", "small_25.nii.gz"]
        out = run_command(cmd)
        assert_equal(out[0], 0)
        # Get expected values
        img = nib.load("small_25.nii.gz")
        affine = img.affine
        shape = img.shape[:-1]
        # Check expected outputs
        assert_image_shape_affine("small_25_fa.nii.gz", shape, affine)
        assert_image_shape_affine("small_25_t2di.nii.gz", shape, affine)
        assert_image_shape_affine("small_25_dirFA.nii.gz", shape, affine)
        assert_image_shape_affine("small_25_ad.nii.gz", shape, affine)
        assert_image_shape_affine("small_25_md.nii.gz", shape, affine)
        assert_image_shape_affine("small_25_rd.nii.gz", shape, affine)

    with InTemporaryDirectory():
        dwi, bval, bvec = get_fnames("small_25")
        # Copy data to tmp directory
        shutil.copyfile(dwi, "small_25.nii.gz")
        shutil.copyfile(bval, "small_25.bval")
        shutil.copyfile(bvec, "small_25.bvec")
        # Call script
        cmd = ["dipy_fit_tensor", "--save-tensor",
               "--mask=none", "small_25.nii.gz"]
        out = run_command(cmd)
        assert_equal(out[0], 0)
        # Get expected values
        img = nib.load("small_25.nii.gz")
        affine = img.affine
        shape = img.shape[:-1]
        # Check expected outputs
        assert_image_shape_affine("small_25_fa.nii.gz", shape, affine)
        assert_image_shape_affine("small_25_t2di.nii.gz", shape, affine)
        assert_image_shape_affine("small_25_dirFA.nii.gz", shape, affine)
        assert_image_shape_affine("small_25_ad.nii.gz", shape, affine)
        assert_image_shape_affine("small_25_md.nii.gz", shape, affine)
        assert_image_shape_affine("small_25_rd.nii.gz", shape, affine)
        # small_25_tensor saves the tensor as a symmetric matrix following
        # the nifti standard.
        ten_shape = shape + (1, 6)
        assert_image_shape_affine("small_25_tensor.nii.gz", ten_shape,
                                  affine)


@nt.dec.skipif(no_mpl)
def test_qb_commandline():
    with InTemporaryDirectory():
        tracks_file = get_fnames('fornix')
        cmd = ["dipy_quickbundles", tracks_file, '--pkl_file', 'mypickle.pkl',
               '--out_file', 'tracks300.trk']
        out = run_command(cmd)
        assert_equal(out[0], 0)

@nt.dec.skipif(no_mpl)
def test_qb_commandline_output_path_handling():
    with InTemporaryDirectory():
        # Create temporary subdirectory for input and for output
        os.mkdir('work')
        os.mkdir('output')

        os.chdir('work')
        tracks_file = get_fnames('fornix')

        # Need to specify an output directory with a "../" style path
        # to trigger old bug.
        cmd = ["dipy_quickbundles", tracks_file, '--pkl_file', 'mypickle.pkl',
               '--out_file', os.path.join('..', 'output', 'tracks300.trk')]
        out = run_command(cmd)
        assert_equal(out[0], 0)

        # Make sure the files were created in the output directory
        os.chdir('../')
        output_files_list = glob.glob('output/tracks300_*.trk')
        assert_true(output_files_list)
'''
