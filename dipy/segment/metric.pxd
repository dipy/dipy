ctypedef float[:,:] float2d
ctypedef double[:,:] double2d

ctypedef float[:] Features
ctypedef float[:,:] Streamline

cdef class Metric(object):
    cdef int is_order_invariant

    cdef int c_infer_features_shape(Metric self, Streamline streamline) nogil except -1
    cdef void c_extract_features(Metric self, Streamline streamline, Features out) nogil except *
    cdef float c_dist(Metric self, float* features1, Features features2) nogil except -1.0

    cpdef int infer_features_shape(Metric self, Streamline streamline) except -1
    cpdef Features extract_features(Metric self, Streamline streamline) except *
    cpdef float dist(Metric self, Features features1, Features features2) except -1.0


cdef class FeatureType(object):
    cdef int is_order_invariant

    cdef int c_infer_shape(FeatureType self, Streamline streamline) nogil except -1
    cdef void c_extract(FeatureType self, Streamline streamline, Features out) nogil except *

    cpdef int infer_shape(FeatureType self, Streamline streamline) except -1
    cpdef Features extract(FeatureType self, Streamline streamline) except *
