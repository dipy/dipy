from dipy.utils.fast_numpy cimport RNGState

cpdef enum StreamlineStatus:
    PYERROR = -2
    OUTSIDEIMAGE = -1
    INVALIDPOINT = 0
    TRACKPOINT = 1
    ENDPOINT = 2
    VALIDSTREAMLIME = 100
    INVALIDSTREAMLIME = -100


cdef class StoppingCriterion:

    cpdef StreamlineStatus check_point(self, double[::1] point)
    cdef StreamlineStatus check_point_c(self, double* point, RNGState* rng=*) noexcept nogil


cdef class BinaryStoppingCriterion(StoppingCriterion):
    cdef:
        unsigned char [:, :, :] mask
    pass


cdef class ThresholdStoppingCriterion(StoppingCriterion):
    cdef:
        double threshold
        double[:, :, :] metric_map
    pass


cdef class AnatomicalStoppingCriterion(StoppingCriterion):
    cdef:
        double[:, :, :] include_map, exclude_map
    cpdef double get_exclude(self, double[::1] point)
    cdef get_exclude_c(self, double* point)
    cpdef double get_include(self, double[::1] point)
    cdef get_include_c(self, double* point)
    pass


cdef class ActStoppingCriterion(AnatomicalStoppingCriterion):
    pass


cdef class CmcStoppingCriterion(AnatomicalStoppingCriterion):
    cdef:
        double step_size
        double average_voxel_size
        double correction_factor
    pass
