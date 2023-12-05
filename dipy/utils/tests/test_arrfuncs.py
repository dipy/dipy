"""Testing array utilities."""

import sys

import numpy as np

from dipy.utils.arrfuncs import as_native_array, pinv

from numpy.testing import (assert_array_almost_equal, assert_equal,
                           assert_array_equal)
from dipy.testing import assert_true, assert_false
from dipy.testing.decorators import set_random_number_generator

NATIVE_ORDER = '<' if sys.byteorder == 'little' else '>'
SWAPPED_ORDER = '>' if sys.byteorder == 'little' else '<'


def test_as_native():
    arr = np.arange(5)  # native
    assert_equal(arr.dtype.byteorder, '=')
    narr = as_native_array(arr)
    assert_true(arr is narr)
    sdt = arr.view(arr.dtype.newbyteorder('s'))
    barr = arr.astype(sdt.dtype)
    assert_equal(barr.dtype.byteorder, SWAPPED_ORDER)
    narr = as_native_array(barr)
    assert_false(barr is narr)
    assert_array_equal(barr, narr)
    assert_equal(narr.dtype.byteorder, NATIVE_ORDER)


@set_random_number_generator()
def test_pinv(rng):
    arr = rng.standard_normal((4, 4, 4, 3, 7))
    _pinv = pinv(arr)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                assert_array_almost_equal(_pinv[i, j, k],
                                          np.linalg.pinv(arr[i, j, k]))
