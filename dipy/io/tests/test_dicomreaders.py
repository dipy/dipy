""" Testing reading DICOM files

"""

import numpy as np

from .. import dicomreaders as didr

from .test_dicomwrappers import (EXPECTED_AFFINE,
                                 DATA)

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from dipy.testing import parametric, IO_DATA_PATH


@parametric
def test_read_dwi():
    img = didr.mosaic_to_nii(DATA)
    arr = img.get_data()
    yield assert_equal(arr.shape, (128,128,48))
    yield assert_array_almost_equal(img.get_affine(), EXPECTED_AFFINE)


@parametric
def test_read_dwis():
    data, aff, bs, gs = didr.read_mosaic_dwi_dir(IO_DATA_PATH, '*.dcm.gz')
    yield assert_equal(data.ndim, 4)
    yield assert_equal(aff.shape, (4,4))
    yield assert_equal(bs.shape, (2,))
    yield assert_equal(gs.shape, (2,3))
    yield assert_raises(IOError, didr.read_mosaic_dwi_dir, 'improbable')
