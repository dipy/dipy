
import numpy as np
import numpy.testing as npt
import ctypes
from multiprocessing import Array
from dipy.core.parallel import mparray_as_ndarray, ndarray_to_mparray


def test_mparray_as_ndarray():
    seq = range(10)
    mp_arr = Array('B', seq)
    res_ndarray = mparray_as_ndarray(mp_arr)

    npt.assert_(isinstance(res_ndarray, np.ndarray))
    npt.assert_array_equal(res_ndarray,  np.arange(10))
    npt.assert_equal(res_ndarray.shape, (10,))
    npt.assert_equal(res_ndarray.dtype, np.uint8)

    mp_arr = Array('f', [x / 10.0 for x in seq])
    res_ndarray = mparray_as_ndarray(mp_arr, (2, 5))

    npt.assert_(isinstance(res_ndarray, np.ndarray))
    npt.assert_array_equal(res_ndarray, np.arange(10, dtype=np.float32).reshape(2, 5) / 10)
    npt.assert_equal(res_ndarray.shape, (2, 5))
    npt.assert_equal(res_ndarray.dtype, np.float32)


def test_ndarray_to_mparray():
    arr = np.arange(10, dtype=np.float32)

    mp_arr = ndarray_to_mparray(arr)

    npt.assert_equal(len(arr), len(mp_arr))
    npt.assert_array_equal(arr, mp_arr)
    npt.assert_equal(mp_arr._type_, ctypes.c_float)

    arr = np.ones(10, dtype=np.bool)
    npt.assert_raises(KeyError, ndarray_to_mparray, arr)


if __name__ == "__main__":
    npt.run_module_suite()
