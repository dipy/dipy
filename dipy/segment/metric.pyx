# distutils: language = c
# cython: wraparound=False, cdivision=True, boundscheck=False

import numpy as np
cimport numpy as np

from libc.math cimport sqrt

cdef extern from "stdlib.h" nogil:
    ctypedef unsigned long size_t
    void *calloc(size_t nelem, size_t elsize)


cdef class Metric(object):
    cdef int _nb_features(Metric self, Streamline streamline) nogil:
        with gil:
            return self.nb_features(streamline)

    cdef void _get_features(Metric self, Streamline streamline, Features out) nogil:
        with gil:
            out[:] = self.get_features(streamline)

    cdef float _dist(Metric self, float* features1, Features features2) nogil:
        with gil:
            return self.dist(<float[:features2.shape[0]]>features1, features2)

    cpdef int nb_features(Metric self, Streamline streamline):
        raise NotImplementedError("Subclass must implement this method!")

    cpdef Features get_features(Metric self, Streamline streamline):
        raise NotImplementedError("Subclass must implement this method!")

    cpdef float dist(Metric self, Features features1, Features features2):
        raise NotImplementedError("Subclass must implement this method!")


cdef class CythonMetric(Metric):
    cpdef int nb_features(CythonMetric self, Streamline streamline):
        return self._nb_features(streamline)

    cpdef Features get_features(CythonMetric self, Streamline streamline):
        cdef int nb_features = self._nb_features(streamline)
        cdef float[:] out = <float[:nb_features]> calloc(nb_features, sizeof(float))
        self._get_features(streamline, out)
        return out

    cpdef float dist(CythonMetric self, Features features1, Features features2):
        return self._dist(&features1[0], features2)


cdef class MDF(CythonMetric):
    cdef int _nb_features(MDF self, Streamline streamline) nogil:
        return streamline.shape[0] * streamline.shape[1]

    cdef void _get_features(MDF self, Streamline streamline, Features out) nogil:
        cdef int i, y
        cdef int N = streamline.shape[0], D = streamline.shape[1]

        for y in range(N):
            i = y*D
            out[i+0] = streamline[y, 0]
            out[i+1] = streamline[y, 1]
            out[i+2] = streamline[y, 2]

    cdef float _dist(MDF self, float* features1, Features features2) nogil:
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

cdef class Euclidean(CythonMetric):
    def __init__(Euclidean self, FeatureType feature_type):
        self.feature_type = feature_type

    cdef int _nb_features(Euclidean self, Streamline streamline) nogil:
        return self.feature_type._shape(streamline)

    cdef void _get_features(Euclidean self, Streamline streamline, Features out) nogil:
        self.feature_type._extract(streamline, out)

    cdef float _dist(Euclidean self, float* features1, Features features2) nogil:
        cdef int i
        cdef float dist=0.0, dn=0.0

        for i in range(features2.shape[0]):
            dn = features1[i] - features2[i]
            dist += dn*dn

        return sqrt(dist)


############
# Features #
############
cdef class FeatureType(object):
    cdef int _shape(FeatureType self, Streamline streamline) nogil:
        with gil:
            return self.shape(streamline)

    cdef void _extract(FeatureType self, Streamline streamline, Features out) nogil:
        with gil:
            out[:] = self.extract(streamline)

    cpdef int shape(FeatureType self, Streamline streamline):
        raise NotImplementedError("Subclass must implement this method!")

    cpdef Features extract(FeatureType self, Streamline streamline):
        raise NotImplementedError("Subclass must implement this method!")


cdef class CythonFeatureType(FeatureType):
    cpdef int shape(CythonFeatureType self, Streamline streamline):
        return self._shape(streamline)

    cpdef Features extract(CythonFeatureType self, Streamline streamline):
        cdef int shape = self._shape(streamline)
        cdef float[:] out = <float[:shape]> calloc(shape, sizeof(float))
        self._extract(streamline, out)
        return out

cdef class CenterOfMass(CythonFeatureType):
    cdef int _shape(CenterOfMass self, Streamline streamline) nogil:
        return streamline.shape[1]

    cdef void _extract(CenterOfMass self, Streamline streamline, Features out) nogil:
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
    cdef int _shape(Midpoint self, Streamline streamline) nogil:
        return streamline.shape[1]

    cdef void _extract(Midpoint self, Streamline streamline, Features out) nogil:
        cdef int N = streamline.shape[0]
        cdef int mid = N/2

        out[0] = streamline[mid, 0]
        out[1] = streamline[mid, 1]
        out[2] = streamline[mid, 2]
