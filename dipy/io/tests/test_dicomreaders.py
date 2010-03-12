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


def test_affine():
    aff = didr.get_vox_to_dpcs(data)
    
