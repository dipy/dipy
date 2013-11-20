# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Test scripts

If we appear to be running from the development directory, use the scripts in
the top-level folder ``scripts``.  Otherwise try and get the scripts from the
path
"""
from __future__ import division, print_function, absolute_import

import sys
import os
import shutil

from os.path import dirname, join as pjoin, isfile, isdir, abspath, realpath

from subprocess import Popen, PIPE

from nose.tools import assert_true, assert_false, assert_equal
import numpy.testing as nt

import nibabel as nib
from nibabel.tmpdirs import InTemporaryDirectory

from dipy.data import get_data

# Quickbundles command-line requires matplotlib: 
try:
    import matplotlib
    no_mpl = False
except ImportError:
    no_mpl = True

# Need shell to get path to correct executables
USE_SHELL = True

DEBUG_PRINT = os.environ.get('NIPY_DEBUG_PRINT', False)

DATA_PATH = abspath(pjoin(dirname(__file__), 'data'))

def local_script_dir(script_sdir):
    # Check for presence of scripts in development directory.  ``realpath``
    # checks for the situation where the development directory has been linked
    # into the path.
    below_us_2 = realpath(pjoin(dirname(__file__), '..', '..'))
    devel_script_dir = pjoin(below_us_2, script_sdir)
    if isfile(pjoin(below_us_2, 'setup.py')) and isdir(devel_script_dir):
        return devel_script_dir
    return None

LOCAL_SCRIPT_DIR = local_script_dir('bin')

def run_command(cmd, check_code=True):
    if not LOCAL_SCRIPT_DIR is None:
        # Windows can't run script files without extensions natively so we need
        # to run local scripts (no extensions) via the Python interpreter.  On
        # Unix, we might have the wrong incantation for the Python interpreter
        # in the hash bang first line in the source file.  So, either way, run
        # the script through the Python interpreter
        cmd = "%s %s" % (sys.executable, pjoin(LOCAL_SCRIPT_DIR, cmd))
    if DEBUG_PRINT:
        print("Running command '%s'" % cmd)
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=USE_SHELL)
    stdout, stderr = proc.communicate()
    if proc.poll() == None:
        proc.terminate()
    if check_code and proc.returncode != 0:
        raise RuntimeError('Command "%s" failed with stdout\n%s\nstderr\n%s\n'
                           % (cmd, stdout, stderr))
    return proc.returncode, stdout, stderr


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
    nt.assert_array_almost_equal(image.get_affine(), affine)


def test_dipy_fit_tensor():
    with InTemporaryDirectory() as tmp:
        dwi, bval, bvec = get_data("small_25")

        # Copy data to tmp directory
        shutil.copyfile(dwi, "small_25.nii.gz")
        shutil.copyfile(bval, "small_25.bval")
        shutil.copyfile(bvec, "small_25.bvec")

        # Call script
        cmd = ["dipy_fit_tensor", "--mask=none", "small_25.nii.gz"]
        out = run_command(" ".join(cmd))
        assert_equal(out[0], 0)

        # Get expected values
        img = nib.load("small_25.nii.gz")
        affine = img.get_affine()
        shape = img.shape[:-1]

        # Check expected outputs
        assert_image_shape_affine("small_25_fa.nii.gz", shape, affine)
        assert_image_shape_affine("small_25_t2di.nii.gz", shape, affine)
        assert_image_shape_affine("small_25_dirFA.nii.gz", shape, affine)
        assert_image_shape_affine("small_25_ad.nii.gz", shape, affine)
        assert_image_shape_affine("small_25_md.nii.gz", shape, affine)
        assert_image_shape_affine("small_25_rd.nii.gz", shape, affine)

    with InTemporaryDirectory() as tmp:
        dwi, bval, bvec = get_data("small_25")

        # Copy data to tmp directory
        shutil.copyfile(dwi, "small_25.nii.gz")
        shutil.copyfile(bval, "small_25.bval")
        shutil.copyfile(bvec, "small_25.bvec")

        # Call script
        cmd = ["dipy_fit_tensor", "--save-tensor", "--mask=none", "small_25.nii.gz"]
        out = run_command(" ".join(cmd))
        assert_equal(out[0], 0)

        # Get expected values
        img = nib.load("small_25.nii.gz")
        affine = img.get_affine()
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
    with InTemporaryDirectory() as tmp:
        tracks_file = get_data('fornix')
        cmd = ["dipy_quickbundles", tracks_file, '--pkl_file ./mypickle.pkl',
               '--out_file ./tracks300.trk']
        out = run_command(" ".join(cmd))
        assert_equal(out[0], 0)
