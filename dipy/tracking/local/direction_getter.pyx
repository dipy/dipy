cimport numpy as np

"""
# DirectionGetter declaration:

cdef class DirectionGetter:
    cdef int get_direction(self, double *point, double *direction) except -1
    cdef np.ndarray[np.float_t, ndim=2] initial_direction(self, double *point)

"""

cdef class DirectionGetter:
    cpdef int get_direction(self,
                            double[::1] point,
                            double[::1] direction) except -1:
        pass
    cpdef np.ndarray[np.float_t, ndim=2] initial_direction(self,
                                                           double[::1] point):
        pass
