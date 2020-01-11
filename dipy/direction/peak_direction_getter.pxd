cimport numpy as np

from dipy.tracking.direction_getter cimport DirectionGetter

cdef class PeakDirectionGetter(DirectionGetter):

    cdef:
        dict _pf_kwargs
        double[:, :, :, :, :] peaks
        double cos_similarity
        int nbr_peaks

    cpdef np.ndarray[np.float_t, ndim=2] initial_direction(
        self,
        double[::1] point)

    cpdef int get_direction(
        self,
        double[::1] point,
        double[::1] direction) except -1

    cdef int get_direction_c(
        self,
        double* point,
        double* direction)
