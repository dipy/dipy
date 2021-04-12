
from .direction_getter cimport DirectionGetter
from .stopping_criterion cimport (StreamlineStatus, StoppingCriterion)

cimport numpy as cnp

cdef class DirectionGetter:

    cpdef cnp.ndarray[cnp.float_t, ndim=2] initial_direction(
        self, double[::1] point)

    cpdef int generate_streamline(
        self,
        double[::1] seed,
        double[::1] dir,
        cnp.float_t[:, :] streamline,
        StreamlineStatus stream_status) except -1

    cdef int generate_streamline_c(
        self,
        double* seed,
        double* dir,
        cnp.float_t[:, :] streamline,
        StreamlineStatus* stream_status)


    cpdef int get_direction(
        self,
        double[::1] point,
        double[::1] direction) except -1

    cdef int get_direction_c(
        self, double* point, double* direction)
