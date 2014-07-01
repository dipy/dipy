cdef enum TissueClass:
    PYERROR = -2
    OUTSIDEIMAGE = -1
    INVALIDPOINT = 0
    TRACKPOINT = 1
    ENDPOINT = 2

cdef class TissueClassifier:
    cdef:
        double threshold, interp_out_double[1]
        double[::1]  interp_out_view
        double[:, :, :] metric_map
    cpdef TissueClass check_point(self, double[::1] point) except PYERROR


cdef class ThresholdTissueClassifier(TissueClassifier):
    pass

