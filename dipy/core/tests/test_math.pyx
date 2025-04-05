

cimport numpy as cnp
import numpy as np
import numpy.testing as npt

from dipy.core.math cimport f_array_min, f_array_max, f_max, f_min


def test_f_array_max():
    cdef int size = 10
    cdef double[:] d_arr = np.arange(size, dtype=np.float64)
    cdef float[:] f_arr = np.arange(size, dtype=np.float32)

    npt.assert_equal(f_array_max(&d_arr[0], size), 9)
    npt.assert_equal(f_array_max(&f_arr[0], size), 9)


def test_f_array_min():
    cdef int size = 6
    cdef double[:] d_arr = np.arange(5, 5+size, dtype=np.float64)
    cdef float[:] f_arr = np.arange(5, 5+size, dtype=np.float32)

    npt.assert_almost_equal(f_array_min(&d_arr[0], size), 5)
    npt.assert_almost_equal(f_array_min(&f_arr[0], size), 5)


def test_f_max():
    npt.assert_equal(f_max[double](1, 2), 2)
    npt.assert_equal(f_max[float](2, 1), 2)
    npt.assert_equal(f_max(1., 1.), 1)


def test_f_min():
    npt.assert_equal(f_min[double](1, 2), 1)
    npt.assert_equal(f_min[float](2, 1), 1)
    npt.assert_equal(f_min(1.0, 1.0), 1)

