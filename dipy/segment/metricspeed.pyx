# distutils: language = c
# cython: wraparound=False, cdivision=True, boundscheck=False

import numpy as np
cimport numpy as np

from libc.math cimport sqrt

from cythonutils cimport tuple2shape, shape2tuple, same_shape
from featurespeed cimport IdentityFeature

DEF biggest_double = 1.7976931348623157e+308  #  np.finfo('f8').max


cdef class Metric(object):
    """ Provides functionalities to compute distance between sequences of
    N-dimensional features represented as 2D array of shape (points, coordinates).

    Parameters
    ----------
    feature : `Feature` object
        type of features that will be used for computing the distance between data

    Notes
    -----
    By default, when inheriting from `Metric`, Python methods will call their
    Python version (e.g. `Metric.c_dist` -> `self.dist`).
    """
    def __init__(Metric self, Feature feature=IdentityFeature()):
        # By default every metric will used data as its features representation.
        self.feature = feature

    property feature:
        """ type of features that will be used for computing distances between data """
        def __get__(Metric self):
            return self.feature

    property is_order_invariant:
        """ does the sequence's ordering matter for computing distances between data """
        def __get__(Metric self):
            return bool(self.is_order_invariant)

    cdef int c_compatible(Metric self, Shape shape1, Shape shape2) nogil:
        """ C version of `metric.compatible`. """
        with gil:
            return self.compatible(shape2tuple(shape1), shape2tuple(shape2))

    cdef double c_dist(Metric self, Data2D features1, Data2D features2) nogil:
        """ C version of `metric.dist`. """
        with gil:
            return self.dist(np.asarray(features1), np.asarray(features2))

    cpdef compatible(Metric self, shape1, shape2):
        """ Checks if features can be used by `metric.dist` based on their shape.

        Basically this method exists so we don't have to do this check
        inside the `metric.dist` function (speedup).

        Parameters
        ----------
        shape1 : tuple
            shape of the first data point's features
        shape2 : tuple
            shape of the second data point's features

        Returns
        -------
        is_compatible : bool
            features extracted from a data point.
        """
        raise NotImplementedError("Metric subclasses must implement method `compatible(self, shape1, shape2)`!")

    cpdef double dist(Metric self, features1, features2):
        """ Computes distance between two data points based on their features.

        Parameters
        ----------
        features1 : 2D array
            features of the first data point
        features2 : 2D array
            features of the second data point

        Returns
        -------
        dist : double
            distance measured between two data points using their features representation
        """
        raise NotImplementedError("Metric subclasses must implement method `dist(self, features1, features2)`!")


cdef class CythonMetric(Metric):
    """ Provides functionalities to compute distance between sequences
    of N-dimensional features represented as 2D array of shape (points, coordinates).

    Parameters
    ----------
    feature : `Feature` object
        type of features that will be used for computing the distance between data

    By default, when inheriting from `CythonMetric`, Python methods will call their
    C version (e.g. `CythonMetric.dist` -> `self.c_dist`).
    """
    cpdef compatible(CythonMetric self, shape1, shape2):
        """ Checks if features can be used by `metric.dist` based on their shape.

        Basically this method exists so we don't have to do this check
        inside the `metric.dist` function (speedup).

        *This method is calling the C version: `self.c_compatible`.*

        Parameters
        ----------
        shape1 : tuple
            shape of the first data point's features
        shape2 : tuple
            shape of the second data point's features

        Returns
        -------
        is_compatible : bool
            features extracted from a data point.
        """
        return self.c_compatible(tuple2shape(shape1), tuple2shape(shape2)) == 1

    cpdef double dist(CythonMetric self, features1, features2):
        """ Computes distance between two data points based on their features.

        *This method is calling the C version: `self.c_dist`.*

        Parameters
        ----------
        features1 : ndarray
            features of the first data point
        features2 : ndarray
            features of the second data point

        Returns
        -------
        is_compatible : bool
            features extracted from a data point.
        """
        if not self.compatible(features1.shape, features2.shape):
            raise ValueError("Features are not compatible to be used by this metric!")

        return self.c_dist(features1, features2)


cdef class PointwiseEuclideanMetric(CythonMetric):
    r""" Provides basic functionalities for subclasses working with pointwise
    euclidean distances between sequential data.

    A sequence of N-dimensional points is represented as 2D array of
    shape (points, coordinates).

    Parameters
    ----------
    feature : `Feature` object
        type of features that will be used for computing the distance between data
    """
    cdef double c_dist(PointwiseEuclideanMetric self, Data2D features1, Data2D features2) nogil:
        with gil:
            raise NotImplementedError("PointwiseEuclideanMetric subclasses must implement method `c_dist(self, Data2D features1, Data2D features2)`!")

    cdef int c_compatible(PointwiseEuclideanMetric self, Shape shape1, Shape shape2) nogil:
        return same_shape(shape1, shape2)


cdef class SumPointwiseEuclideanMetric(PointwiseEuclideanMetric):
    r""" Provides functionalities to compute the sum of pointwise euclidean
    distances between sequential data.

    A sequence of N-dimensional points is represented as 2D array of
    shape (points, coordinates).

    Parameters
    ----------
    feature : `Feature` object
        type of features that will be used for computing the distance between data

    Notes
    -----
    The distance calculated between two 2D sequences::

        s1       s2

        0*   a    *0
          \       |
           \      |
           1*     |
            |  b  *1
            |      \
            2*      \
                c    *2

    is equal to $a+b+c$ where $a$ is the euclidean distance between s1[0] and
    s2[0], $b$ between s1[1] and s2[1] and $c$ between s1[2] and s2[2].
    """
    cdef double c_dist(SumPointwiseEuclideanMetric self, Data2D features1, Data2D features2) nogil:
        cdef :
            int N = features1.shape[0], D = features1.shape[1]
            int n, d
            double dd, dist_n, dist = 0.0

        for n in range(N):
            dist_n = 0.0
            for d in range(D):
                dd = features1[n, d] - features2[n, d]
                dist_n += dd*dd

            dist += sqrt(dist_n)

        return dist


cdef class AveragePointwiseEuclideanMetric(SumPointwiseEuclideanMetric):
    r""" Provides functionalities to compute the average of pointwise euclidean
    distances between sequential data.

    A sequence of N-dimensional points is represented as 2D array of
    shape (points, coordinates).

    Parameters
    ----------
    feature : `Feature` object
        type of feature that will be used for computing the distance between data

    Notes
    -----
    The distance calculated between two 2D sequences::

        s1       s2

        0*   a    *0
          \       |
           \      |
           1*     |
            |  b  *1
            |      \
            2*      \
                c    *2

    is equal to $(a+b+c)/3$ where $a$ is the euclidean distance between s1[0] and
    s2[0], $b$ between s1[1] and s2[1] and $c$ between s1[2] and s2[2].
    """
    cdef double c_dist(AveragePointwiseEuclideanMetric self, Data2D features1, Data2D features2) nogil:
        cdef int N = features1.shape[0]
        cdef double dist = SumPointwiseEuclideanMetric.c_dist(self, features1, features2)
        return dist / N


cdef class MinimumAverageDirectFlipMetric(AveragePointwiseEuclideanMetric):
    r""" Provides functionalities to compute the sum of pointwise euclidean
    distances between sequential data.

    A sequence of N-dimensional points is represented as 2D array of
    shape (points, coordinates).

    Notes
    -----
    The distance calculated between two 2D sequences::

        s1       s2

        0*   a    *0
          \       |
           \      |
           1*     |
            |  b  *1
            |      \
            2*      \
                c    *2

    is equal to $\min((a+b+c)/3, (a'+b'+c')/3)$ where $a$ is the euclidean distance
    between s1[0] and s2[0], $b$ between s1[1] and s2[1], $c$ between s1[2]
    and s2[2], $a'$ between s1[0] and s2[2], $b'$ between s1[1] and s2[1]
    and $c'$ between s1[2] and s2[0].
    """
    def __init__(MinimumAverageDirectFlipMetric self):
        super(MinimumAverageDirectFlipMetric, self).__init__(feature=IdentityFeature())

    property is_order_invariant:
        """ does the sequence's ordering matter for computing distances between data """
        def __get__(MinimumAverageDirectFlipMetric self):
            return True  # Ordering is handled in the distance computation

    cdef double c_dist(MinimumAverageDirectFlipMetric self, Data2D features1, Data2D features2) nogil:
        cdef double dist_direct = AveragePointwiseEuclideanMetric.c_dist(self, features1, features2)
        cdef double dist_flipped = AveragePointwiseEuclideanMetric.c_dist(self, features1, features2[::-1])
        return min(dist_direct, dist_flipped)


cpdef distance_matrix(Metric metric, streamlines1, streamlines2):
    cdef int i, j
    shape = metric.feature.infer_shape(streamlines1[0])
    distance_matrix = np.zeros((len(streamlines1), len(streamlines2)), dtype=np.float64)
    cdef:
        Data2D features1 = np.empty(shape, np.float32)
        Data2D features2 = np.empty(shape, np.float32)

    for i in range(len(streamlines1)):
        streamline1 = streamlines1[i] if streamlines1[i].flags.writeable else streamlines1[i].astype(np.float32)
        metric.feature.c_extract(streamline1, features1)
        for j in range(len(streamlines2)):
            streamline2 = streamlines2[j] if streamlines2[j].flags.writeable else streamlines2[j].astype(np.float32)
            metric.feature.c_extract(streamline2, features2)
            distance_matrix[i, j] = metric.c_dist(features1, features2)

    return distance_matrix


cdef double c_dist(Metric metric, Data2D s1, Data2D s2) nogil except -1.0:
    cdef Data2D features1, features2
    cdef Shape shape1 = metric.feature.c_infer_shape(s1)
    cdef Shape shape2 = metric.feature.c_infer_shape(s2)

    with gil:
        if not metric.c_compatible(shape1, shape2):
            raise ValueError("Features' shapes must match!")

        features1 = np.empty(shape2tuple(shape1), s1.base.dtype)
        features2 = np.empty(shape2tuple(shape2), s2.base.dtype)

    metric.feature.c_extract(s1, features1)
    metric.feature.c_extract(s2, features2)
    return metric.c_dist(features1, features2)


cpdef double dist(Metric metric, Data2D s1, Data2D s2) except -1.0:
    return c_dist(metric, s1, s2)
