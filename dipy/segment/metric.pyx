# distutils: language = c
# cython: wraparound=False, cdivision=True, boundscheck=False

import numpy as np
cimport numpy as np

from libc.math cimport sqrt

cdef extern from "stdlib.h" nogil:
    ctypedef unsigned long size_t
    void *calloc(size_t nelem, size_t elsize)


##########
# Metric #
##########
cdef class Metric(object):
    def __cinit__(self):
        self.is_order_invariant = True

    property is_order_invariant:
        def __get__(self):
            return bool(self.is_order_invariant)
        def __set__(self, int value):
            self.is_order_invariant = bool(value)

    cdef int c_infer_features_shape(Metric self, Streamline streamline) nogil:
        with gil:
            return self.infer_features_shape(streamline)

    cdef void c_extract_features(Metric self, Streamline streamline, Features out) nogil:
        with gil:
            out[:] = self.extract_features(streamline)

    cdef float c_dist(Metric self, float* features1, Features features2) nogil:
        with gil:
            return self.dist(<float[:features2.shape[0]]>features1, features2)

    cpdef int infer_features_shape(Metric self, Streamline streamline):
        raise NotImplementedError("Subclass must implement this method!")

    cpdef Features extract_features(Metric self, Streamline streamline):
        raise NotImplementedError("Subclass must implement this method!")

    cpdef float dist(Metric self, Features features1, Features features2):
        raise NotImplementedError("Subclass must implement this method!")

cdef class CythonMetric(Metric):
    cpdef int infer_features_shape(CythonMetric self, Streamline streamline):
        return self.c_infer_features_shape(streamline)

    cpdef Features extract_features(CythonMetric self, Streamline streamline):
        cdef int shape = self.c_infer_features_shape(streamline)
        cdef Features out = <float[:shape]> calloc(shape, sizeof(float))
        self.c_extract_features(streamline, out)
        return out

    cpdef float dist(CythonMetric self, Features features1, Features features2):
        return self.c_dist(&features1[0], features2)

cdef class MDF(CythonMetric):
    def __cinit__(self):
        self.is_order_invariant = False

    cdef int c_infer_features_shape(MDF self, Streamline streamline) nogil:
        return streamline.shape[0] * streamline.shape[1]

    cdef void c_extract_features(MDF self, Streamline streamline, Features out) nogil:
        cdef int i, y
        cdef int N = streamline.shape[0], D = streamline.shape[1]

        for y in range(N):
            i = y*D
            out[i+0] = streamline[y, 0]
            out[i+1] = streamline[y, 1]
            out[i+2] = streamline[y, 2]

    cdef float c_dist(MDF self, float* features1, Features features2) nogil:
        cdef int i, y
        cdef int N = features2.shape[0]//3, D = 3
        cdef float d = 0.0
        cdef float dx, dy, dz

        for y in range(N):
            i = y*D
            dx = features1[i+0] - features2[i+0]
            dy = features1[i+1] - features2[i+1]
            dz = features1[i+2] - features2[i+2]
            d += sqrt(dx*dx + dy*dy + dz*dz)

        return d / N

    cpdef float dist(MDF self, Features features1, Features features2):
        if features1.shape[0] != features2.shape[0]:
            raise ValueError("MDF requires features to have the same shape!")

        return self.c_dist(&features1[0], features2)

cdef class Euclidean(CythonMetric):
    cdef FeatureType feature_type

    def __init__(Euclidean self, FeatureType feature_type):
        self.feature_type = feature_type
        self.is_order_invariant = feature_type.is_order_invariant

    cdef int c_infer_features_shape(Euclidean self, Streamline streamline) nogil:
        return self.feature_type.c_infer_shape(streamline)

    cdef void c_extract_features(Euclidean self, Streamline streamline, Features out) nogil:
        self.feature_type.c_extract(streamline, out)

    cdef float c_dist(Euclidean self, float* features1, Features features2) nogil:
        cdef int i
        cdef float dist=0.0, dn=0.0

        for i in range(features2.shape[0]):
            dn = features1[i] - features2[i]
            dist += dn*dn

        return sqrt(dist)

    cpdef float dist(Euclidean self, Features features1, Features features2):
        if features1.shape[0] != features2.shape[0]:
            raise ValueError("Euclidean metric requires features to have the same shape!")

        return self.c_dist(&features1[0], features2)


############
# Features #
############
cdef class FeatureType(object):
    def __cinit__(self):
        self.is_order_invariant = True

    property is_order_invariant:
        def __get__(self):
            return bool(self.is_order_invariant)
        def __set__(self, int value):
            self.is_order_invariant = bool(value)

    cdef int c_infer_shape(FeatureType self, Streamline streamline) nogil:
        with gil:
            return self.shape(streamline)

    cdef void c_extract(FeatureType self, Streamline streamline, Features out) nogil:
        with gil:
            out[:] = self.extract(streamline)

    cpdef int infer_shape(FeatureType self, Streamline streamline):
        raise NotImplementedError("Subclass must implement this method!")

    cpdef Features extract(FeatureType self, Streamline streamline):
        raise NotImplementedError("Subclass must implement this method!")

cdef class CythonFeatureType(FeatureType):
    cpdef int infer_shape(CythonFeatureType self, Streamline streamline):
        return self.c_infer_shape(streamline)

    cpdef Features extract(CythonFeatureType self, Streamline streamline):
        cdef int shape = self.c_infer_shape(streamline)
        cdef Features out = <float[:shape]> calloc(shape, sizeof(float))
        self.c_extract(streamline, out)
        return out

cdef class CenterOfMass(CythonFeatureType):
    cdef int c_infer_shape(CenterOfMass self, Streamline streamline) nogil:
        return streamline.shape[1]

    cdef void c_extract(CenterOfMass self, Streamline streamline, Features out) nogil:
        cdef int N = streamline.shape[0]
        cdef int i

        out[0] = out[1] = out[2] = 0

        for i in range(N):
            out[0] += streamline[i, 0]
            out[1] += streamline[i, 1]
            out[2] += streamline[i, 2]

        out[0] /= N
        out[1] /= N
        out[2] /= N

cdef class Midpoint(CythonFeatureType):
    def __cinit__(self):
        self.is_order_invariant = False

    cdef int c_infer_shape(Midpoint self, Streamline streamline) nogil:
        return streamline.shape[1]

    cdef void c_extract(Midpoint self, Streamline streamline, Features out) nogil:
        cdef int N = streamline.shape[0]
        cdef int mid = N/2

        out[0] = streamline[mid, 0]
        out[1] = streamline[mid, 1]
        out[2] = streamline[mid, 2]


####################
# Metric functions #
####################
cdef float dist(Metric metric, Streamline s1, Streamline s2) except -1.0:
    cdef Features features1 = metric.extract_features(s1)
    cdef Features features2 = metric.extract_features(s2)
    return metric.dist(features1, features2)

def mdf(s1, s2):
    return dist(MDF(), s1, s2)

def euclidean(s1, s2, feature_type="midpoint"):
    if feature_type == "midpoint":
        feature_type = Midpoint()
    elif feature_type == "center":
        feature_type = CenterOfMass()

    return dist(Euclidean(feature_type), s1, s2)
