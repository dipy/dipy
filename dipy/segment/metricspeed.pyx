# distutils: language = c
# cython: wraparound=False, cdivision=True, boundscheck=False

import numpy as np
cimport numpy as np

from libc.math cimport sqrt

cdef extern from "metricspeed.h":
    enum: MAX_NDIM

cdef extern from "stdlib.h" nogil:
    ctypedef unsigned long size_t
    void *calloc(size_t nelem, size_t elsize)

DEF biggest_double = 1.7976931348623157e+308  # np.finfo('f8').max


cdef Shape shape_from_memview(Data data) nogil:
    """ Retrieves shape from a memoryview """
    cdef Shape shape
    cdef int i
    shape.ndim = 0
    shape.size = 1
    for i in range(MAX_NDIM):
        shape.dims[i] = data.shape[i]
        if shape.dims[i] > 0:
            shape.size *= shape.dims[i]
            shape.ndim += 1

    return shape


cdef Shape tuple2shape(dims):
    """ Converts a tuple to a shape """
    cdef Shape shape
    cdef int i
    shape.ndim = len(dims)
    shape.size = np.prod(dims)
    for i in range(shape.ndim):
        shape.dims[i] = dims[i]

    return shape


cdef shape2tuple(Shape shape):
    """ Converts a shape to a tuple """
    cdef int i
    dims = []
    for i in range(shape.ndim):
        dims.append(shape.dims[i])

    return tuple(dims)


cdef int same_shape(Shape shape1, Shape shape2) nogil:
    """ Checks if shape1 and shape2 has the same ndim and same dims """
    cdef int i
    cdef int same_shape = True

    same_shape &= shape1.ndim == shape2.ndim

    for i in range(shape1.ndim):
        same_shape &= shape1.dims[i] == shape2.dims[i]

    return same_shape


cdef class Metric(object):
    """ Provides functionalities to compute distance between sequences
    of N-dimension features represented as 2D array (nb. points x nb. dimension).

    NB: When inheriting from `Metric` C methods will called accordingly
    their Python version (eg. `Metric.c_dist` -> `self.dist`).
    """
    def __init__(self, feature_type=IdentityFeature()):
        # By default every metric will used data as its features representation.
        self.feature_type = feature_type

    property feature_type:
        """ Property telling if the sequence's ordering matters """
        def __get__(self):
            return self.feature_type
        #def __set__(self, FeatureType value):
        #    self.feature_type = value

    cdef int c_compatible(Metric self, Shape shape1, Shape shape2) nogil:
        """ C version `metric.compatible`. """
        with gil:
            return self.compatible(shape2tuple(shape1), shape2tuple(shape2))

    cdef double c_dist(Metric self, Features features1, Features features2) nogil:
        """ C version `metric.dist`. """
        with gil:
            return self.dist(np.asarray(features1), np.asarray(features2))

    cpdef compatible(Metric self, shape1, shape2):
        """ Checks if features can be used by `metric.dist` based on their shape.

        Mostly this method exists so we don't have to do this check
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
    of N-dimension features represented as 2D array (nb. points x nb. dimension).

    NB: When inheriting from `CythonMetric` Python methods will called
    accordingly their C version (eg. `CythonMetric.dist` -> `self.c_dist`)
    unless they have been overridden.
    """
    cpdef compatible(CythonMetric self, shape1, shape2):
        """ Checks if features can be used by `metric.dist` based on their shape.

        Mostly this method exists so we don't have to do this check
        inside the `metric.dist` function (speedup).

        *This method will called the C version `self.c_compatible`.*

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

        *This method will called the C version `self.c_dist`.*

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
        return self.c_dist(features1, features2)


cdef class PointwiseEuclideanMetric(CythonMetric):
    cdef double c_dist(PointwiseEuclideanMetric self, Features features1, Features features2) nogil:
        with gil:
            raise NotImplementedError("Subclass must implement this method!")

    cdef int c_compatible(PointwiseEuclideanMetric self, Shape shape1, Shape shape2) nogil:
        return same_shape(shape1, shape2)

    cpdef double dist(PointwiseEuclideanMetric self, features1, features2):
        if not self.compatible(features1.shape, features2.shape):
            raise ValueError("PointwiseEuclideanMetric requires features' shape to match!")

        return self.c_dist(features1, features2)

cdef class SumPointwiseEuclideanMetric(PointwiseEuclideanMetric):
    """ Provides functionalities to compute the sum of pointwise euclidean
    distances between `Streamline` data.

    `Streamline` is a sequence of 3D points represented in a 2D array (points x coordinates).

    Parameters
    ----------
    feature_type : FeatureType
        type of feature that will be used for computing the distance between data.

    Notes
    -----
    The distance calculated between two 2D streamlines::

        s_1       s_2

        0*   a    *0
          \       |
           \      |
           1*     |
            |  b  *1
            |      \
            2*      \
                c    *2

    is equal to $a+b+c$ where $a$ the euclidean distance between s_1[0] and
    s_2[0], $b$ between s_1[1] and s_2[1] and $c$ between s_1[2] and s_2[2].
    """
    cdef double c_dist(SumPointwiseEuclideanMetric self, Features features1, Features features2) nogil:
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
    """ Provides functionalities to compute the average of pointwise euclidean
    distances between `Streamline` data.

    `Streamline` is a sequence of 3D points represented in a 2D array (points x coordinates).

    Parameters
    ----------
    feature_type : FeatureType
        type of feature that will be used for computing the distance between data.

    Notes
    -----
    The distance calculated between two 2D streamlines::

        s_1       s_2

        0*   a    *0
          \       |
           \      |
           1*     |
            |  b  *1
            |      \
            2*      \
                c    *2

    is equal to $(a+b+c)/3$ where $a$ the euclidean distance between s_1[0] and
    s_2[0], $b$ between s_1[1] and s_2[1] and $c$ between s_1[2] and s_2[2].
    """
    cdef double c_dist(AveragePointwiseEuclideanMetric self, Features features1, Features features2) nogil:
        cdef int N = features1.shape[0]
        cdef double dist = SumPointwiseEuclideanMetric.c_dist(self, features1, features2)
        return dist / N


cdef class MinimumPointwiseEuclideanMetric(PointwiseEuclideanMetric):
    """ Provides functionalities to compute the minimum of pointwise euclidean
    distances between `Streamline` data.

    `Streamline` is a sequence of 3D points represented in a 2D array (points x coordinates).

    Parameters
    ----------
    feature_type : FeatureType
        type of feature that will be used for computing the distance between data.

    Notes
    -----
    The distance calculated between two 2D streamlines::

        s_1       s_2

        0*   a    *0
          \       |
           \      |
           1*     |
            |  b  *1
            |      \
            2*      \
                c    *2

    is equal to $\min(a, b, c)$ where $a$ the euclidean distance between s_1[0] and
    s_2[0], $b$ between s_1[1] and s_2[1] and $c$ between s_1[2] and s_2[2].
    """
    cdef double c_dist(MinimumPointwiseEuclideanMetric self, Features features1, Features features2) nogil:
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
    """ Provides functionalities to compute the maximum of pointwise euclidean
    distances between `Streamline` data.

    `Streamline` is a sequence of 3D points represented in a 2D array (points x coordinates).

    Parameters
    ----------
    feature_type : FeatureType
        type of feature that will be used for computing the distance between data.

    Notes
    -----
    The distance calculated between two 2D streamlines::

        s_1       s_2

        0*   a    *0
          \       |
           \      |
           1*     |
            |  b  *1
            |      \
            2*      \
                c    *2

    is equal to $\max(a, b, c)$ where $a$ the euclidean distance between s_1[0] and
    s_2[0], $b$ between s_1[1] and s_2[1] and $c$ between s_1[2] and s_2[2].
    """
    cdef double c_dist(MaximumPointwiseEuclideanMetric self, Features features1, Features features2) nogil:
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


cdef class MDF(AveragePointwiseEuclideanMetric):
    def __init__(MDF self):
        AveragePointwiseEuclideanMetric.__init__(self, feature_type=IdentityFeature())
        self.feature_type.is_order_invariant = True

    cdef double c_dist(MDF self, Features features1, Features features2) nogil:
        cdef double dist_direct = AveragePointwiseEuclideanMetric.c_dist(self, features1, features2)
        cdef double dist_flipped = AveragePointwiseEuclideanMetric.c_dist(self, features1, features2[::-1])
        return min(dist_direct, dist_flipped)


cdef class ArcLengthMetric(CythonMetric):
    def __init__(self):
        CythonMetric.__init__(self, ArcLengthFeature())

    cdef int c_compatible(ArcLengthMetric self, Shape shape1, Shape shape2) nogil:
        return shape1.dims[0] == 1 & shape1.dims[1] == 1 & same_shape(shape1, shape2)

    cdef double c_dist(ArcLengthMetric self, Features features1, Features features2) nogil:
        cdef double dist = features1[0, 0] - features2[0, 0]
        return max(dist, -dist)  # Absolute value


cdef class FeatureType(object):
    def __cinit__(self):
        # By default every features are order invariant.
        self.is_order_invariant = True

    property is_order_invariant:
        """ Property telling if the sequence's ordering matters """
        def __get__(self):
            return bool(self.is_order_invariant)
        def __set__(self, int value):
            self.is_order_invariant = bool(value)

    cdef Shape c_infer_shape(FeatureType self, Streamline streamline) nogil:
        with gil:
            return tuple2shape(self.infer_shape(np.asarray(streamline)))

    cdef void c_extract(FeatureType self, Streamline streamline, Features out) nogil:
        cdef Features features
        with gil:
            features = self.extract(np.asarray(streamline))

        out[:] = features

    cpdef infer_shape(FeatureType self, streamline):
        """ From a data point infers what will be the shape of the extracted features.

        Parameters
        ----------
        datum : ndarray
            a data point

        Returns
        -------
        shape : tuple
            shape of the features extracted from a data point.
        """
        raise NotImplementedError("Subclass must implement this method!")

    cpdef extract(FeatureType self, streamline):
        """ Extracts features from a data point.

        Parameters
        ----------
        datum : ndarray
            a data point

        Returns
        -------
        features : ndarray
            features extracted from a data point.
        """
        raise NotImplementedError("Subclass must implement this method!")


cdef class CythonFeatureType(FeatureType):
    cpdef infer_shape(CythonFeatureType self, streamline):
        return shape2tuple(self.c_infer_shape(streamline))

    cpdef extract(CythonFeatureType self, streamline):
        shape = shape2tuple(self.c_infer_shape(streamline))
        cdef Features out = np.empty(shape, dtype=streamline.dtype)
        self.c_extract(streamline, out)
        return np.asarray(out)


cdef class IdentityFeature(CythonFeatureType):
    def __init__(self):
        self.is_order_invariant = False

    cdef Shape c_infer_shape(IdentityFeature self, Streamline streamline) nogil:
        return shape_from_memview(streamline)

    cdef void c_extract(IdentityFeature self, Streamline streamline, Features out) nogil:
        cdef:
            int N = streamline.shape[0], D = streamline.shape[1]
            int n, d

        for n in range(N):
            for d in range(D):
                out[n, d] = streamline[n, d]


cdef class CenterOfMassFeature(CythonFeatureType):
    cdef Shape c_infer_shape(CenterOfMassFeature self, Streamline streamline) nogil:
        cdef Shape shape = shape_from_memview(streamline)
        shape.size /= shape.dims[0]
        shape.dims[0] = 1  # Features boil down to only one point.
        return shape

    cdef void c_extract(CenterOfMassFeature self, Streamline streamline, Features out) nogil:
        cdef int N = streamline.shape[0], D = streamline.shape[1]
        cdef int i, d

        for d in range(D):
            out[0, d] = 0

        for i in range(N):
            for d in range(D):
                out[0, d] += streamline[i, d]

        for d in range(D):
            out[0, d] /= N


cdef class MidpointFeature(CythonFeatureType):
    def __init__(self):
        self.is_order_invariant = False

    cdef Shape c_infer_shape(MidpointFeature self, Streamline streamline) nogil:
        cdef Shape shape = shape_from_memview(streamline)
        shape.size /= shape.dims[0]
        shape.dims[0] = 1  # Features boil down to only one point.
        return shape

    cdef void c_extract(MidpointFeature self, Streamline streamline, Features out) nogil:
        cdef:
            int N = streamline.shape[0], D = streamline.shape[1]
            int mid = N/2
            int d

        for d in range(D):
            out[0, d] = streamline[mid, d]


cdef class ArcLengthFeature(CythonFeatureType):
    cdef Shape c_infer_shape(ArcLengthFeature self, Streamline streamline) nogil:
        cdef Shape shape
        shape.ndim = 2
        shape.dims[0] = 1
        shape.dims[1] = 1
        shape.size = 1
        return shape

    cdef void c_extract(ArcLengthFeature self, Streamline streamline, Features out) nogil:
        cdef:
            int N = streamline.shape[0], D = streamline.shape[1]
            int n, d
            double dn, sum_dn_sqr

        out[0, 0] = 0.
        for n in range(1, N):
            sum_dn_sqr = 0.0
            for d in range(D):
                dn = streamline[n, d] - streamline[n-1, d]
                sum_dn_sqr += dn * dn

            out[0, 0] += sqrt(sum_dn_sqr)


cdef double c_dist(Metric metric, Streamline s1, Streamline s2) nogil except -1.0:
    cdef Features features1, features2
    cdef Shape shape1 = metric.feature_type.c_infer_shape(s1)
    cdef Shape shape2 = metric.feature_type.c_infer_shape(s2)

    with gil:
        if not metric.c_compatible(shape1, shape2):
            raise ValueError("Features' shape extracted from streamlines must match!")

        features1 = np.empty(shape2tuple(shape1), s1.base.dtype)
        features2 = np.empty(shape2tuple(shape2), s2.base.dtype)

    metric.feature_type.c_extract(s1, features1)
    metric.feature_type.c_extract(s2, features2)
    return metric.c_dist(features1, features2)


cpdef double dist(Metric metric, Streamline s1, Streamline s2) except -1.0:
    return c_dist(metric, s1, s2)


cpdef distance_matrix(Metric metric, streamlines1, streamlines2):

    #TODO: check if all compatible
    #if not metric.c_compatible(shape1, shape2):
    #    raise ValueError("Features' shape extracted from streamlines must match!")
    shape = metric.feature_type.infer_shape(streamlines1[0])
    dtype = streamlines1[0].dtype
    cdef:
        double[:, :] distance_matrix = np.zeros((len(streamlines1), len(streamlines2)), dtype=np.float64)
        Features features1 = np.empty(shape, dtype)
        Features features2 = np.empty(shape, dtype)

    for i in range(len(streamlines1)):
        metric.feature_type.c_extract(streamlines1[i], features1)
        for j in range(len(streamlines2)):
            metric.feature_type.c_extract(streamlines2[j], features2)
            distance_matrix[i, j] = metric.c_dist(features1, features2)

    return np.asarray(distance_matrix)


#cdef double mdf(Streamline s1, Streamline s2) nogil except -1.0:
#    cdef:
#        Features features1, features2
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
