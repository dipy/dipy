""" Testing array utilities
"""

import sys

import numpy as np

from ..arrfuncs import as_native_array, pinv_vec

from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal)

from nose.tools import assert_true, assert_false, assert_equal, assert_raises

NATIVE_ORDER = '<' if sys.byteorder == 'little' else '>'
SWAPPED_ORDER = '>' if sys.byteorder == 'little' else '<'

def test_as_native():
    arr = np.arange(5) # native
    assert_equal(arr.dtype.byteorder, '=')
    narr = as_native_array(arr)
    assert_true(arr is narr)
    sdt = arr.dtype.newbyteorder('s')
    barr = arr.astype(sdt)
    assert_equal(barr.dtype.byteorder, SWAPPED_ORDER)
    narr = as_native_array(barr)
    assert_false(barr is narr)
    assert_array_equal(barr, narr)
    assert_equal(narr.dtype.byteorder, NATIVE_ORDER)


def test_pinv_vec():
    arr = np.random.randn(4, 4, 4, 3, 7)
    pinv = pinv_vec(arr)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                assert_array_almost_equal(pinv[i, j, k],
                                          np.linalg.pinv(arr[i, j, k]))
