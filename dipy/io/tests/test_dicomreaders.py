""" Testing reading DICOM files

"""

import os
from os.path import join as pjoin
import gzip
from glob import glob

import numpy as np

import dicom

import dipy.io.dicomreaders as didr
from dipy.io.vectors import vector_norm

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from dipy.testing import parametric

data_path = pjoin(os.path.dirname(__file__), 'data')

data_file = pjoin(data_path, 'siemens_dwi_1000.dcm.gz')

data = dicom.read_file(gzip.open(data_file))

# this affine from our converted image was shown to match our image
# spatially with an image from SPM DICOM conversion. We checked the
# matching with SPM check reg.
expected_affine = np.array(
    [[ -1.796875, 0, 0, 115],
     [0, -1.79684984, -0.01570896, 135.028779],
     [0, -0.00940843750, 2.99995887, -78.710481],
     [0, 0, 0, 1]])

# from Guys and Matthew's SPM code, with Y flip reversed
expected_params = [992.05050247, (0.99997450,
                                  0.00507649,
                                  -0.005023611)]


@parametric
def test_read_dwi():
    img = didr.mosaic_to_nii(data)
    arr = img.get_data()
    yield assert_equal(arr.shape, (128,128,48))
    yield assert_array_almost_equal(img.get_affine(), expected_affine)


@parametric
def test_read():
    yield assert_true(didr.has_csa(data))
    yield assert_equal(didr.get_csa_header(data,'image')['n_tags'],83)
    yield assert_equal(didr.get_csa_header(data,'series')['n_tags'],65)
    yield assert_raises(ValueError, didr.get_csa_header, data,'xxxx')
    yield assert_true(didr.is_mosaic(data))


@parametric
def test_dwi_params():
    b_matrix = didr.get_b_matrix(data)
    q = didr.get_q_vector(data)
    b = vector_norm(q)
    g = q / b
    yield assert_array_almost_equal(b, expected_params[0])
    yield assert_array_almost_equal(g, expected_params[1])


@parametric
def test_read_dwis():
    data_dir = os.path.expanduser(
        "~/data/20100114_195840/Series_012_CBU_DTI_64D_1A")
    data, aff, bs, gs = didr.read_mosaic_dwi_dir(data_dir)
    print data.shape
    print gs
