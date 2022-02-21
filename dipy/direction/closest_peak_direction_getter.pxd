cimport numpy as cnp

from dipy.direction.pmf cimport PmfGen
from dipy.tracking.direction_getter cimport DirectionGetter

cdef int closest_peak(cnp.ndarray[cnp.float_t, ndim=2] peak_dirs,
                      double* direction, double cos_similarity)

cdef class BasePmfDirectionGetter(DirectionGetter):

    cdef:
        object sphere
        dict _pf_kwargs
        PmfGen pmf_gen
        double pmf_threshold
        double cos_similarity

    cpdef cnp.ndarray[cnp.float_t, ndim=2] initial_direction(
        self,
        double[::1] point)

    cdef _get_pmf(
        self,
        double* point)

    cpdef int get_direction(
        self,
        double[::1] point,
        double[::1] direction) except -1

    cdef int get_direction_c(
        self,
        double* point,
        double* direction)

cdef class BaseDirectionGetter(BasePmfDirectionGetter):

    pass

cdef class PmfGenDirectionGetter(BasePmfDirectionGetter):

    pass


cdef class ClosestPeakDirectionGetter(PmfGenDirectionGetter):

    pass
