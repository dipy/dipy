""" Testing array utilities
"""

import sys

import numpy as np

from ..arrfuncs import as_native_array, pinv, eigh

from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal)

from nose.tools import assert_true, assert_false, assert_equal, assert_raises

NATIVE_ORDER = '<' if sys.byteorder == 'little' else '>'
SWAPPED_ORDER = '>' if sys.byteorder == 'little' else '<'


def test_as_native():
    arr = np.arange(5)  # native
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


def test_pinv():
    arr = np.random.randn(4, 4, 4, 3, 7)
    _pinv = pinv(arr)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                assert_array_almost_equal(_pinv[i, j, k],
                                          np.linalg.pinv(arr[i, j, k]))


def test_eigh():
    for i in range(10):
        arr = np.random.randn(7, 7)
        evals1, evecs1 = eigh(arr)
        evals2, evecs2 = np.linalg.eigh(arr)
        assert_array_almost_equal(evals1, evals2)
        assert_array_almost_equal(evecs1, evecs2)

    arr = np.random.randn(4, 4, 4, 7, 7)
    evals, evecs = eigh(arr)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                evals_vox, evecs_vox = np.linalg.eigh(arr[i, j, k])
                assert_array_almost_equal(evals[i, j, k], evals_vox)
                assert_array_almost_equal(evecs[i, j, k], evecs_vox)
