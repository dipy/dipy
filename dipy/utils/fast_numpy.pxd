# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False

cimport numpy as cnp

from libc.stdlib cimport rand, srand, RAND_MAX
from libc.math cimport sqrt


cdef void take(
        double* odf,
        cnp.npy_intp* indices,
        cnp.npy_intp n_indices,
        double* values_out) noexcept nogil


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
        double * out,
        RNGState* rng=*) noexcept nogil

cdef void random_perpendicular_vector(
        double * out,
        double * v,
        RNGState* rng=*) noexcept nogil

cdef (double, double) random_point_within_circle(
        double r, RNGState* rng=*) noexcept nogil

cpdef double random() noexcept nogil

cpdef void seed(cnp.npy_uint32 s) noexcept nogil

cdef struct RNGState:
    cnp.npy_uint64 state
    cnp.npy_uint64 inc

cdef void seed_rng(RNGState* rng_state, cnp.npy_uint64 seed) noexcept nogil

cdef cnp.npy_uint32 next_rng(RNGState* rng_state) noexcept nogil

cdef double random_float(RNGState* rng_state) noexcept nogil
