# distutils: language = c
# cython: wraparound=False, cdivision=True, boundscheck=False

from cythonutils cimport Data2D, Shape
from featurespeed cimport Feature


cdef class Metric(object):
    cdef Feature feature

    cdef double c_dist(Metric self, Data2D features1, Data2D features2) nogil except -1.0
    cdef int c_compatible(Metric self, Shape shape1, Shape shape2) nogil except -1

    cpdef double dist(Metric self, features1, features2) except -1.0
    cpdef compatible(Metric self, shape1, shape2)
