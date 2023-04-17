# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False

cimport numpy as np

from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport sqrt


# Replaces a numpy.searchsorted(arr, number, 'right')
cdef int where_to_insert(
        np.float_t* arr,
        np.float_t number,
        int size) nogil

cdef void cumsum(
        np.float_t* arr_in,
        np.float_t* arr_out,
        int N) nogil

cdef void copy_point(
        double * a,
        double * b) nogil

cdef void scalar_muliplication_point(
        double * a,
        double scalar) nogil

cpdef double random() nogil

cpdef double norm(
        double[:] v) nogil

cpdef double dot(
        double[:] v1,
        double[:] v2) nogil

cpdef void normalize(
        double[:] v) nogil

cpdef void cross(
        double[:] out,
        double[:] v1,
        double[:] v2) nogil

cpdef void random_vector(
        double[:] out)

cpdef void random_perpendicular_vector(
        double[:] out,
        double[:] v)

cpdef (double, double) random_point_within_circle(
        double r)

