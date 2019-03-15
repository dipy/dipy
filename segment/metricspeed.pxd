from dipy.segment.cythonutils cimport Data2D, Shape
from dipy.segment.featurespeed cimport Feature


cdef class Metric(object):
    cdef Feature feature
    cdef int is_order_invariant

    cdef double c_dist(Metric self, Data2D features1, Data2D features2) nogil except -1
    cdef int c_are_compatible(Metric self, Shape shape1, Shape shape2) nogil except -1

    cpdef double dist(Metric self, features1, features2) except -1
    cpdef are_compatible(Metric self, shape1, shape2)
