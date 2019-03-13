
cimport numpy as np

cpdef trilinear_interpolate4d(
    double[:, :, :, :] data,
    double[:] point,
    np.ndarray out=*)

cdef int trilinear_interpolate4d_c(
    double[:, :, :, :] data,
    double* point,
    double[:] result) nogil
