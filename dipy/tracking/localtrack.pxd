cimport numpy as np

cdef enum TissueClass:
    PYERROR = -2
    OUTSIDEIMAGE = -1
    INVALIDPOINT = 0
    TRACKPOINT = 1
    ENDPOINT = 2

cdef class TissueClassifier:
    cdef TissueClass check_point(self, double *point) except PYERROR

cdef class DirectionGetter:
    cdef int get_direction(self, double *point, double *direction) except -1
    cdef np.ndarray[np.float_t, ndim=2] initial_direction(self, double *point)

cdef class ThresholdTissueClassifier(TissueClassifier):
    cdef:
        double threshold
        double[:, :, :] metric_map
