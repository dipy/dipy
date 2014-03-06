cimport cython
ctypedef fused floating:
    cython.float

ctypedef fused number:
    cython.float
    cython.short
    cython.int
