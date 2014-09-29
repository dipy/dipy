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
    cdef int i
    cdef int same_shape = True

    same_shape &= shape1.ndim == shape2.ndim

    for i in range(shape1.ndim):
        same_shape &= shape1.dims[i] == shape2.dims[i]

    return same_shape


cdef class Metric(object):
    def __cinit__(self):
        self.is_order_invariant = True

    property is_order_invariant:
        def __get__(self):
            return bool(self.is_order_invariant)
        def __set__(self, int value):
            self.is_order_invariant = bool(value)

    cdef Shape c_infer_features_shape(Metric self, Streamline streamline) nogil:
        with gil:
            return tuple2shape(self.infer_features_shape(np.asarray(streamline)))

    cdef void c_extract_features(Metric self, Streamline streamline, Features out) nogil:
        cdef Features features
        with gil:
            features = self.extract_features(np.asarray(streamline))

        out[:] = features

    cdef int c_compatible(Metric self, Shape shape1, Shape shape2) nogil:
        with gil:
            return self.compatible(shape2tuple(shape1), shape2tuple(shape2))

    cdef float c_dist(Metric self, Features features1, Features features2) nogil:
        with gil:
            return self.dist(np.asarray(features1), np.asarray(features2))

    cpdef infer_features_shape(Metric self, streamline):
        raise NotImplementedError("Subclass must implement this method!")

    cpdef extract_features(Metric self, streamline):
        raise NotImplementedError("Subclass must implement this method!")

    cpdef compatible(Metric self, shape1, shape2):
        raise NotImplementedError("Subclass must implement this method!")

    cpdef float dist(Metric self, features1, features2):
        raise NotImplementedError("Subclass must implement this method!")


cdef class CythonMetric(Metric):
    cpdef infer_features_shape(CythonMetric self, streamline):
        cdef Shape shape = self.c_infer_features_shape(streamline)
        return shape2tuple(shape)

    cpdef extract_features(CythonMetric self, streamline):
        shape = shape2tuple(self.c_infer_features_shape(streamline))
        cdef Features out = np.empty(shape, dtype=streamline.dtype)
        self.c_extract_features(streamline, out)
        return np.asarray(out)

    cpdef compatible(CythonMetric self, shape1, shape2):
        return self.c_compatible(tuple2shape(shape1), tuple2shape(shape2)) == 1

    cpdef float dist(CythonMetric self, features1, features2):
        return self.c_dist(features1, features2)


cdef class MDF(CythonMetric):
    def __init__(self):
        self.is_order_invariant = False

    cdef Shape c_infer_features_shape(MDF self, Streamline streamline) nogil:
        return shape_from_memview(streamline)

    cdef void c_extract_features(MDF self, Streamline streamline, Features out) nogil:
        cdef int n, d
        cdef int N = streamline.shape[0], D = streamline.shape[1]

        for n in range(N):
            for d in range(D):
                out[n, d] = streamline[n, d]

    cdef float c_dist(MDF self, Features features1, Features features2) nogil:
        cdef int N = features1.shape[0], D = features1.shape[1]
        cdef int n, d
        cdef float dist = 0.0
        cdef float dist_n, dd

        for n in range(N):
            dist_n = 0.0
            for d in range(D):
                dd = features1[n, d] - features2[n, d]
                dist_n += dd*dd

            dist += sqrt(dist_n)

        return dist / N

    cdef int c_compatible(MDF self, Shape shape1, Shape shape2) nogil:
        return same_shape(shape1, shape2)

    cpdef float dist(MDF self, features1, features2):
        if not self.compatible(features1.shape, features2.shape):
            raise ValueError("MDF requires features' shape to match!")

        return self.c_dist(features1, features2)

cdef class Euclidean(CythonMetric):
    cdef FeatureType feature_type

    def __init__(Euclidean self, FeatureType feature_type):
        self.feature_type = feature_type
        self.is_order_invariant = feature_type.is_order_invariant

    cdef Shape c_infer_features_shape(Euclidean self, Streamline streamline) nogil:
        return self.feature_type.c_infer_shape(streamline)

    cdef void c_extract_features(Euclidean self, Streamline streamline, Features out) nogil:
        self.feature_type.c_extract(streamline, out)

    cdef float c_dist(Euclidean self, Features features1, Features features2) nogil:
        cdef int D = features1.shape[1]
        cdef int d
        cdef float dd, dist = 0.0

        for d in range(D):
            dd = features1[0, d] - features2[0, d]
            dist += dd*dd

        return sqrt(dist)

    cdef int c_compatible(Euclidean self, Shape shape1, Shape shape2) nogil:
        return shape1.dims[0] == 1 & same_shape(shape1, shape2)

    cpdef float dist(Euclidean self, features1, features2):
        if not self.compatible(features1.shape, features2.shape):
            raise ValueError("Euclidean metric requires features' shape to match and first dimension to be 1!")

        return self.c_dist(features1, features2)


cdef class FeatureType(object):
    def __cinit__(self):
        self.is_order_invariant = True

    property is_order_invariant:
        def __get__(self):
            return bool(self.is_order_invariant)
        def __set__(self, int value):
            self.is_order_invariant = bool(value)

    cdef Shape c_infer_shape(FeatureType self, Streamline streamline) nogil:
        with gil:
            return tuple2shape(self.infer_shape(np.asarray(streamline)))

    cdef void c_extract(FeatureType self, Streamline streamline, Features out) nogil:
        with gil:
            out[:] = self.extract(np.asarray(streamline))

    cpdef infer_shape(FeatureType self, streamline):
        raise NotImplementedError("Subclass must implement this method!")

    cpdef extract(FeatureType self, streamline):
        raise NotImplementedError("Subclass must implement this method!")

cdef class CythonFeatureType(FeatureType):
    cpdef infer_shape(CythonFeatureType self, streamline):
        return shape2tuple(self.c_infer_shape(streamline))

    cpdef extract(CythonFeatureType self, streamline):
        shape = shape2tuple(self.c_infer_shape(streamline))
        cdef Features out = np.empty(shape, dtype=streamline.dtype)
        self.c_extract(streamline, out)
        return np.asarray(out)

cdef class CenterOfMass(CythonFeatureType):
    cdef Shape c_infer_shape(CenterOfMass self, Streamline streamline) nogil:
        cdef Shape shape = shape_from_memview(streamline)
        shape.size /= shape.dims[0]
        shape.dims[0] = 1  # Features boil down to only one point.
        return shape

    cdef void c_extract(CenterOfMass self, Streamline streamline, Features out) nogil:
        cdef int N = streamline.shape[0], D = streamline.shape[1]
        cdef int i, d

        for d in range(D):
            out[0, d] = 0

        for i in range(N):
            for d in range(D):
                out[0, d] += streamline[i, d]

        for d in range(D):
            out[0, d] /= N

cdef class Midpoint(CythonFeatureType):
    def __cinit__(self):
        self.is_order_invariant = False

    cdef Shape c_infer_shape(Midpoint self, Streamline streamline) nogil:
        cdef Shape shape = shape_from_memview(streamline)
        shape.size /= shape.dims[0]
        shape.dims[0] = 1  # Features boil down to only one point.
        return shape

    cdef void c_extract(Midpoint self, Streamline streamline, Features out) nogil:
        cdef int N = streamline.shape[0], D = streamline.shape[1]
        cdef int mid = N/2
        cdef int d

        for d in range(D):
            out[0, d] = streamline[mid, d]

cdef float c_dist(Metric metric, Streamline s1, Streamline s2) nogil except -1.0:
    cdef Features features1, features2
    cdef Shape shape1 = metric.c_infer_features_shape(s1)
    cdef Shape shape2 = metric.c_infer_features_shape(s2)

    with gil:
        if not metric.c_compatible(shape1, shape2):
            raise ValueError("Features' shape extracted from streamlines must match!")

        features1 = np.empty(shape2tuple(shape1), s1.base.dtype)
        features2 = np.empty(shape2tuple(shape2), s2.base.dtype)

    metric.c_extract_features(s1, features1)
    metric.c_extract_features(s2, features2)
    return metric.c_dist(features1, features2)

cpdef float dist(Metric metric, Streamline s1, Streamline s2) except -1.0:
    return c_dist(metric, s1, s2)

cpdef distance_matrix(Metric metric, streamlines1, streamlines2):

    #TODO: check if all compatible
    #if not metric.c_compatible(shape1, shape2):
    #    raise ValueError("Features' shape extracted from streamlines must match!")
    dtype = streamlines1[0].dtype
    cdef:
        float[:, :] distance_matrix = np.zeros((len(streamlines1), len(streamlines1)), dtype)
        Features features1 = np.empty(streamlines1[0].shape, dtype)
        Features features2 = np.empty(streamlines1[0].shape, dtype)

    for i in range(len(streamlines1)):
        metric.c_extract_features(streamlines1[i], features1)
        for j in range(len(streamlines2)):
            metric.c_extract_features(streamlines2[j], features2)
            distance_matrix[i, j] = metric.c_dist(features1, features2)

    return np.asarray(distance_matrix)