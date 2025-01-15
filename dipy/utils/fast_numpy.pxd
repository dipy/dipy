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
        int size) noexcept nogil

cdef void cumsum(
        cnp.float_t* arr_in,
        cnp.float_t* arr_out,
        int N) noexcept nogil

cdef void copy_point(
        double * a,
        double * b) noexcept nogil

cdef void scalar_muliplication_point(
        double * a,
        double scalar) noexcept nogil

cdef double norm(
        double * v) noexcept nogil

cdef double dot(
        double * v1,
        double * v2) noexcept nogil

cdef void normalize(
        double * v) noexcept nogil

cdef void cross(
        double * out,
        double * v1,
        double * v2) noexcept nogil

cdef void random_vector(
        double * out) noexcept nogil

cdef void random_perpendicular_vector(
        double * out,
        double * v) noexcept nogil

cpdef (double, double) random_point_within_circle(
        double r) noexcept nogil

cpdef double random() noexcept nogil

cpdef void seed(cnp.npy_uint32 s) noexcept nogil

