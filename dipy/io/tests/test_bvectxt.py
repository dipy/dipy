import numpy as np
from numpy.testing import assert_raises

from dipy.io.bvectxt import (orientation_from_string, reorient_vectors,
                             orientation_to_string)
from dipy.utils.deprecator import ExpiredDeprecationError


def test_orientation_from_to_string():
    assert_raises(ExpiredDeprecationError, orientation_from_string, 'aasl')
    assert_raises(ExpiredDeprecationError, orientation_to_string, 'asl')


def test_reorient_vectors():
    bvec = np.arange(12).reshape((3, 4))
    assert_raises(ExpiredDeprecationError, reorient_vectors, bvec, 'ras',
                  'ras')
