cdef int _trilinear_interpolate_c_4d(double[:, :, :, :] data, double[:] point,
                                     double[::1] result) nogil
cpdef trilinear_interpolate4d(double[:, :, :, :] data, double[:] point,
                              double[::1] out=*)

