cimport cython
cimport numpy as np

from libc.math cimport round

from .interpolation import trilinear_interpolate4d

cdef class TissueClassifier:
    cpdef TissueClass check_point(self, double[::1] point):
        pass


cdef class ThresholdTissueClassifier(TissueClassifier):
    """
    cdef:
        double threshold
        double[:, :, :] metric_map
    """

    def __init__(self, metric_map, threshold):
        self.metric_map = metric_map
        self.threshold = threshold

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef TissueClass check_point(self, double[::1] point):
        cdef:
            np.npy_intp ijk[3]
            double result
        try:
            result = trilinear_interpolate4d(self.metric_map[..., None], point)[0]
        except IndexError:
            return OUTSIDEIMAGE

        if result > self.threshold:
            return TRACKPOINT
        else:
            return ENDPOINT

