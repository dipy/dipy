

cdef enum TissueClass:
    PYERROR = -2
    OUTSIDEIMAGE = -1
    INVALIDPOINT = 0
    TRACKPOINT = 1
    ENDPOINT = 2


cdef class TissueClassifier:
    cdef:
        double interp_out_double[1]
        double[::1] interp_out_view
    cpdef TissueClass check_point(self, double[::1] point)
    cdef TissueClass check_point_c(self, double* point)


cdef class BinaryTissueClassifier(TissueClassifier):
    cdef:
        unsigned char [:, :, :] mask
    pass


cdef class ThresholdTissueClassifier(TissueClassifier):
    cdef:
        double threshold
        double[:, :, :] metric_map
    pass


cdef class ActTissueClassifier(TissueClassifier):
    cdef:
        double[:, :, :] include_map, exclude_map
