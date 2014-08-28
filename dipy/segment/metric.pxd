## distutils: language = c

ctypedef float[:,:] float2d
ctypedef double[:,:] double2d

ctypedef float[:] Features
ctypedef float[:,:] Streamline

cdef class Metric(object):
    cdef int _nb_features(Metric self, Streamline streamline) nogil except -1
    cdef void _get_features(Metric self, Streamline streamline, Features out) nogil except *
    cdef float _dist(Metric self, float* features1, Features features2) nogil except -1.0

    cpdef int nb_features(Metric self, Streamline streamline) except -1
    cpdef Features get_features(Metric self, Streamline streamline) except *
    cpdef float dist(Metric self, Features features1, Features features2) except -1.0

cdef class CythonMetric(Metric):
    pass

cdef class MDF(CythonMetric):
    pass

cdef class Euclidean(CythonMetric):
    cdef FeatureType feature_type


cdef class FeatureType(object):
    cdef int _shape(FeatureType self, Streamline streamline) nogil except -1
    cdef void _extract(FeatureType self, Streamline streamline, Features out) nogil except *

    cpdef int shape(FeatureType self, Streamline streamline) except -1
    cpdef Features extract(FeatureType self, Streamline streamline) except *

cdef class CythonFeatureType(FeatureType):
    pass

cdef class CenterOfMass(CythonFeatureType):
    pass

cdef class Midpoint(CythonFeatureType):
    pass
