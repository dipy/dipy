# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False

cimport numpy as np

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
