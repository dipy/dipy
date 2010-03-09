""" Testing reading Siemens DWI

"""

import os
from os.path import join as pjoin

import numpy as np

import nibabel as nib

import dipy.io.siemens_dwi as sdwi

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from dipy.testing import parametric

data_file = os.path.expanduser('~/data/20100114_195840/'
                               'Series_012_CBU_DTI_64D_1A/'
                               '1.3.12.2.1107.5.2.32.35119.'
                               '2010011420300822323499745.dcm')

if os.path.isfile(data_file):

    @parametric
    def test_read():
        img = sdwi.read_dwi(data_file)
        data = img.get_data()
        yield assert_equal(data.shape, (128,128,48))
        nib.save(img, 'test.nii')
