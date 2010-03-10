""" Testing reading DICOM files

"""

import os
from os.path import join as pjoin

import numpy as np

import dicom

import dipy.io.dicomreaders as didr

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from dipy.testing import parametric

data_file = os.path.expanduser('~/data/20100114_195840/'
                               'Series_012_CBU_DTI_64D_1A/'
                               '1.3.12.2.1107.5.2.32.35119.'
                               '2010011420300822323499745.dcm')

@parametric
def test_read():
    data = dicom.read_file(data_file)
    yield assert_true(didr.has_csa(data))
    yield assert_equal(didr.get_csa_header(data,'image')['n_tags'],83)
    yield assert_equal(didr.get_csa_header(data,'image')['n_tags'],83)
    yield assert_raises(ValueError, didr.get_csa_header, data,'xxxx')
    csa_image_header = didr.get_csa_header(data,'image')
    yield assert_true(didr.is_mosaic(data))
