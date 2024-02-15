cimport numpy as cnp

from dipy.direction.pmf cimport PmfGen
from dipy.tracking.direction_getter cimport DirectionGetter


cdef int closest_peak(cnp.ndarray[cnp.float_t, ndim=2] peak_dirs,
                      double[::1] direction, double cos_similarity)


cdef class BasePmfDirectionGetter(DirectionGetter):

    cdef:
        dict _pf_kwargs
        double pmf_threshold
        double cos_similarity
        int len_pmf
        object sphere
        PmfGen pmf_gen

    cpdef cnp.ndarray[cnp.float_t, ndim=2] initial_direction(
        self,
        double[::1] point)

    cdef double[:] _get_pmf(
        self,
        double[::1] point) nogil

    cdef int get_direction_c(
        self,
        double[::1] point,
        double[::1] direction)


cdef class BaseDirectionGetter(BasePmfDirectionGetter):

    pass


cdef class PmfGenDirectionGetter(BasePmfDirectionGetter):

    pass


cdef class ClosestPeakDirectionGetter(PmfGenDirectionGetter):

    pass
