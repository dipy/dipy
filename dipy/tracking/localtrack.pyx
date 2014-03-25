cimport cython

cimport numpy as np
from propspeed cimport _propagation_direction

import numpy as np

from libc.math cimport round
from libc.stdio cimport printf


cdef class TissueClassifier:
    cdef TissueClass check_point(self, double *point):
        pass


cdef class DirectionGetter:
    cdef int get_direction(self, double *point, double *direction):
        pass
    cdef np.ndarray[np.float_t, ndim=2] initial_direction(self, double *point):
        pass


cdef class PythonDirectionGetter(DirectionGetter):

    cdef:
        object point_array, direction_array
        double[::1] point_v, direction_v

    def __cinit__(self):
        self.point_array = np.empty(3)
        self.point_v = self.point_array
        self.direction_array = np.empty(3)
        self.direction_v = self.direction_array

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.profile(True)
    cdef int get_direction(self, double *point, double *direction) except -1:

        cdef np.ndarray[np.float_t, ndim=1] new_dir

        for i in range(3):
            self.point_v[i] = point[i]
            self.direction_v[i] = direction[i]
        new_dir = self._get_direction(self.point_array, self.direction_array)
        if new_dir is None:
            return 1
        for i in range(3):
            direction[i] = new_dir[i]
        return 0


def makeNd(array, N):
    """Makes an array that's less than then Nd - Nd

    We need this because numpy 1.6 does not return a "c contiguous array"
    when you call ``array(a, order='c', ndmin=N)``
    """
    if array.ndim > N:
        raise ValueError()
    new_shape = (1,) * (N - array.ndim) + array.shape
    return array.reshape(new_shape)


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
    cdef TissueClass check_point(self, double *point):
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


cdef void fixed_step(double *point, double *direction, double stepsize):
    for i in range(3):
        point[i] += direction[i] * stepsize


cdef inline void copypoint(double *a, double *b):
    for i in range(3):
        b[i] = a[i]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def local_tracker(DirectionGetter dg, TissueClassifier tc,
                  np.ndarray[np.float_t, ndim=1] seed,
                  np.ndarray[np.float_t, ndim=1] first_step,
                  np.ndarray[np.float_t, ndim=1] voxel_size,
                  np.ndarray[np.float_t, ndim=2, mode='c'] streamline,
                  double stepsize,
                  int fixedstep):

    if (seed.shape[0] != 3 or first_step.shape[0] != 3 or
        voxel_size.shape[0] != 3 or streamline.shape[1] != 3):
        raise ValueError()

    cdef:
        int i
        TissueClass tssuclass
        double point[3], dir[3], vs[3], voxdir[3]

    for i in range(3):
        point[i] = seed[i]
        dir[i] = first_step[i]
        vs[i] = voxel_size[i]

    for i in range(streamline.shape[0]):
        copypoint(point, &streamline[i, 0])
        if dg.get_direction(point, dir):
            i += 1
            break
        for j in range(3):
            voxdir[j] = dir[j] / vs[j]
        fixed_step(point, voxdir, stepsize)
        tssuclass = tc.check_point(point)
        if tssuclass == TRACKPOINT:
            continue
        elif tssuclass == ENDPOINT:
            i += 1
            break
        elif tssuclass == OUTSIDEIMAGE:
            break
        elif tssuclass == INVALIDPOINT:
            i = - (i + 1)
            break
    return i


def _testGetDirection(DirectionGetter dg,
                      double[::1] point not None,
                      double[::1] dir not None):

    cdef:
        double[::1] newdir = dir.copy()
        int state
    state = dg.get_direction(&point[0], &newdir[0])
    return (state, np.array(newdir))


def _testCheckPoint(TissueClassifier tc, double[::1] point not None):

    cdef:
        int tissue
    tissue = tc.check_point(&point[0])
    return tissue

