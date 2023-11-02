# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False

cimport numpy as cnp

from libc.stdlib cimport rand, srand, RAND_MAX
from libc.math cimport sqrt


# Replaces a numpy.searchsorted(arr, number, 'right')
cdef int where_to_insert(
        cnp.float_t* arr,
        cnp.float_t number,
        int size) nogil

cdef void cumsum(
        cnp.float_t* arr_in,
        cnp.float_t* arr_out,
        int N) nogil

cdef void copy_point(
        double * a,
        double * b) nogil

cdef void scalar_muliplication_point(
        double * a,
        double scalar) nogil

cdef double norm(
        double * v) nogil

cdef double dot(
        double * v1,
        double * v2) nogil

cdef void normalize(
        double * v) nogil

cdef void cross(
        double * out,
        double * v1,
        double * v2) nogil

cdef void random_vector(
        double * out) nogil

cdef void random_perpendicular_vector(
        double * out,
        double * v) nogil

cpdef (double, double) random_point_within_circle(
        double r) nogil

cpdef double random() nogil

cpdef void seed(cnp.npy_uint32 s) nogil

