cimport numpy as np

cdef class DirectionGetter:
    cpdef int get_direction(self, double[::1] point, double[::1] direction) except -1
    cpdef np.ndarray[np.float_t, ndim=2] initial_direction(self, double[::1] point)

