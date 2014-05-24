cdef enum TissueClass:
    PYERROR = -2
    OUTSIDEIMAGE = -1
    INVALIDPOINT = 0
    TRACKPOINT = 1
    ENDPOINT = 2

cdef class TissueClassifier:
    cdef:
        double threshold
        double[:, :, :] metric_map
    cpdef TissueClass check_point(self, double[::1] point) except PYERROR

