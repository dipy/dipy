
cimport numpy as np

cdef class DirectionGetter:

    cpdef np.ndarray[np.float_t, ndim=2] initial_direction(
        self, double[::1] point)

    cpdef int get_direction(
        self,
        double[::1] point,
        double[::1] direction) except -1
    cdef int get_direction_c(
        self, double* point, double* direction)
