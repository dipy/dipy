# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False

cimport numpy as np

# Replaces a numpy.searchsorted(arr, number, 'right')
cdef inline int where_to_insert(
        np.float_t* arr,
        np.float_t number,
        int size) nogil;

cdef inline void cumsum(
        np.float_t* arr_in,
        np.float_t* arr_out,
        int N) nogil
