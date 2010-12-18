""" Testing ModelArray

"""

import numpy as np
from numpy.ma import MaskedArray

from dipy.reconst.modelarray import ModelArray
#for reading in nifti test data

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_almost_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

import os


def test_model_shape():

    shape = (10,11,12)
    ma = ModelArray()
    ma.model_params = np.zeros(shape+(4,))
    assert_equal(ma[..., 0].shape, shape[:-1])
    assert_equal(ma[0].shape, shape[1:])
    ma.shape = (-1)
    assert(ma.ndim == 1)
    ma.shape = shape[::-1]
    assert(ma.ndim == len(shape))

def test_model_mask():

    shape = (10,11,12)
    data = np.zeros(shape+(4,))
    ma = ModelArray()
    ma.model_params = data
    assert(ma.mask.all())

    mask = np.random.random(shape+(4,)) > .3
    maskedarray = MaskedArray(data, mask)
    ma.model_params = maskedarray
    assert_array_equal(ma.mask, mask)

def test_indexing_and_setting():

    shape = (10,11,12)
    data = np.random.random(shape+(4,))
    ma = ModelArray()
    ma.model_params = data
    assert_array_equal(ma[0].model_params, data[0])
    assert_array_equal(ma[:, 1].model_params, data[:, 1])
    assert_array_equal(ma[:, :, 2].model_params, data[:, :, 2])
    assert_array_equal(ma[3:5].model_params, data[3:5])
    assert_array_equal(ma[..., 4:].model_params, data[..., 4:, :])

