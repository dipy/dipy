""" Testing reading DICOM files

"""

import os
from os.path import join as pjoin
import gzip

import numpy as np

import dicom

import dipy.io.dicomreaders as didr

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from dipy.testing import parametric

data_path = pjoin(os.path.dirname(__file__), 'data')

data_file = pjoin(data_path, 'siemens_dwi_1000.dcm.gz')

data = dicom.read_file(gzip.open(data_file))

expected_affine = np.array(
    [[ -1.796875, 0, 0, 115],
     [0, -1.79684984, -0.01570896, 135.028779],
     [0, -0.00940843750, 2.99995887, -78.710481],
     [0, 0, 0, 1]])


@parametric
def test_read_dwi():
    img = didr.mosaic_to_nii(data)
    arr = img.get_data()
    yield assert_equal(arr.shape, (128,128,48))

@parametric
def test_read():
    yield assert_true(didr.has_csa(data))
    yield assert_equal(didr.get_csa_header(data,'image')['n_tags'],83)
    yield assert_equal(didr.get_csa_header(data,'image')['n_tags'],83)
    yield assert_raises(ValueError, didr.get_csa_header, data,'xxxx')
    yield assert_true(didr.is_mosaic(data))


@parametric
def test_affine():
    aff = didr.get_vox_to_dpcs(data)
    aff = np.dot(didr.DPCS_TO_TAL, aff)
    yield assert_array_almost_equal(aff, expected_affine)
