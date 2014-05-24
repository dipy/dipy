cimport cython
cimport numpy as np

from libc.math cimport round


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

        for i in range(3):
            # TODO: replace this with trilinear interpolation
            ijk[i] = <np.npy_intp> round(point[i])
            if ijk[i] < 0 or ijk[i] >= self.metric_map.shape[i]:
                return OUTSIDEIMAGE

        if self.metric_map[ijk[0], ijk[1], ijk[2]] > self.threshold:
            return TRACKPOINT
        else:
            return ENDPOINT

