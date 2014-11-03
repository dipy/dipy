# distutils: language = c
# cython: wraparound=False, cdivision=True, boundscheck=False

import numpy as np
cimport numpy as np

from libc.math cimport sqrt

from cythonutils cimport tuple2shape, shape2tuple, same_shape
from featurespeed cimport IdentityFeature, ArcLengthFeature

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
        raise NotImplementedError("Subclass must implement this method!")

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
        raise NotImplementedError("Subclass must implement this method!")


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
            raise NotImplementedError("Subclass must implement this method!")

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


cdef class MinimumPointwiseEuclideanMetric(PointwiseEuclideanMetric):
    r""" Provides functionalities to compute the minimum of pointwise euclidean
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

    is equal to $\min(a, b, c)$ where $a$ is the euclidean distance between s1[0] and
    s2[0], $b$ between s1[1] and s2[1] and $c$ between s1[2] and s2[2].
    """
    cdef double c_dist(MinimumPointwiseEuclideanMetric self, Data2D features1, Data2D features2) nogil:
        cdef :
            int N = features1.shape[0], D = features1.shape[1]
            int n, d
            double dd, dist_n, dist = biggest_double

        for n in range(N):
            dist_n = 0.0
            for d in range(D):
                dd = features1[n, d] - features2[n, d]
                dist_n += dd*dd

            dist = min(dist, sqrt(dist_n))

        return dist


cdef class MaximumPointwiseEuclideanMetric(PointwiseEuclideanMetric):
    r""" Provides functionalities to compute the maximum of pointwise euclidean
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

    is equal to $\max(a, b, c)$ where $a$ is the euclidean distance between s1[0] and
    s2[0], $b$ between s1[1] and s2[1] and $c$ between s1[2] and s2[2].
    """
    cdef double c_dist(MaximumPointwiseEuclideanMetric self, Data2D features1, Data2D features2) nogil:
        cdef :
            int N = features1.shape[0], D = features1.shape[1]
            int n, d
            double dd, dist_n, dist = 0

        for n in range(N):
            dist_n = 0.0
            for d in range(D):
                dd = features1[n, d] - features2[n, d]
                dist_n += dd*dd

            dist = max(dist, sqrt(dist_n))

        return dist


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


cdef class HausdorffMetric(CythonMetric):
    r""" Provides basic functionalities to compute Hausdorff distance.

    A sequence of N-dimensional points is represented as 2D array of
    shape (points, coordinates).

    Parameters
    ----------
    feature : `Feature` object
        type of features that will be used for computing the distance between data
    """
    def __init__(self):
        super(HausdorffMetric, self).__init__(IdentityFeature())

    cdef double c_dist(HausdorffMetric self, Data2D features1, Data2D features2) nogil:
        cdef double min_d, max_d1, max_d2, dd, dist
        cdef int N1 = features1.shape[0], N2 = features2.shape[0]
        cdef int D = features1.shape[1]  # Assume features have the same number of dimensions
        cdef int n1, n2, d

        max_d1 = 0.0
        for n1 in range(N1):
            min_d = biggest_double
            for n2 in range(N2):
                dist = 0.0
                for d in range(D):
                    dd = features1[n1, d] - features2[n2, d]
                    dist += dd * dd

                min_d = min(min_d, sqrt(dist))
            max_d1 = max(max_d1, min_d)

        max_d2 = 0.0
        for n2 in range(N2):
            min_d = biggest_double
            for n1 in range(N1):
                dist = 0.0
                for d in range(D):
                    dd = features1[n1, d] - features2[n2, d]
                    dist += dd * dd

                min_d = min(min_d, sqrt(dist))
            max_d2 = max(max_d2, min_d)

        return max(max_d1, max_d2)

    cdef int c_compatible(HausdorffMetric self, Shape shape1, Shape shape2) nogil:
        return shape1.dims[1] == shape2.dims[1]


cdef class ArcLengthMetric(CythonMetric):
    r""" Provides functionalities to compute a distance between sequential
    data using their arc length.

    A sequence of N-dimensional points is represented as 2D array of
    shape (points, coordinates).

    Notes
    -----
    The distance calculated between two 2D sequences::

        s1       s2

        0*        *0
          \       |
         a \      | c
           1*     |
            |     *1
          b |      \ d
            2*      \
                     *2

    is equal to $|(a+b)-(c+d)|$ where $a$ is the euclidean distance
    between s1[0] and s1[1], $b$ between s1[1] and s1[2], $c$ between s2[0]
    and s2[1] and $d$ between s2[1] and s2[2].
    """
    def __init__(ArcLengthMetric self):
        super(ArcLengthMetric, self).__init__(feature=ArcLengthFeature())

    cdef int c_compatible(ArcLengthMetric self, Shape shape1, Shape shape2) nogil:
        return shape1.dims[0] == 1 & shape1.dims[1] == 1 & same_shape(shape1, shape2)

    cdef double c_dist(ArcLengthMetric self, Data2D features1, Data2D features2) nogil:
        cdef double dist = features1[0, 0] - features2[0, 0]
        return max(dist, -dist)  # Absolute value


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


#cdef double mdf(Data2D s1, Data2D s2) nogil except -1.0:
#    cdef:
#        Data2D features1, features2
#        Shape shape1 = AveragePointwiseEuclideanMetric.c_infer_features_shape(s1)
#        Shape shape2 = metric.c_infer_features_shape(s2)
#        double dist_direct, dist_flipped

#    with gil:
#        if not metric.c_compatible(shape1, shape2):
#            raise ValueError("Features' shape extracted from streamlines must match!")

#        features1 = np.empty(shape2tuple(shape1), s1.base.dtype)
#        features2 = np.empty(shape2tuple(shape2), s2.base.dtype)

#    metric.c_extract_features(s1, features1)
#    metric.c_extract_features(s2, features2)
#    dist_direct = metric.c_dist(features1, features2)

#    metric.c_extract_features(s2[::-1], features2)
#    dist_flipped = metric.c_dist(features1, features2)

#    return min(dist_direct, dist_flipped)
