
cpdef enum StreamlineStatus:
    PYERROR = -2
    OUTSIDEIMAGE = -1
    INVALIDPOINT = 0
    TRACKPOINT = 1
    ENDPOINT = 2


cdef class StoppingCriterion:
    cdef:
        double interp_out_double[1]
        double[::1] interp_out_view
    cpdef StreamlineStatus check_point(self, double[::1] point)
    cdef StreamlineStatus check_point_c(self, double* point)


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
    cdef double get_exclude_c(self, double* point)
    cpdef double get_include(self, double[::1] point)
    cdef double get_include_c(self, double* point)
    pass


cdef class ActStoppingCriterion(AnatomicalStoppingCriterion):
    pass


cdef class CmcStoppingCriterion(AnatomicalStoppingCriterion):
    cdef:
        double step_size
        double average_voxel_size
        double correction_factor
    pass
