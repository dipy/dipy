cimport cython
ctypedef fused floating:
    cython.double

ctypedef fused number:
    cython.double
    cython.short
    cython.int
