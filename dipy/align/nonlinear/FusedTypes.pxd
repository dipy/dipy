cimport cython
ctypedef fused floating:
    cython.double
    cython.float

ctypedef fused integral:
    cython.short
    cython.int
    cython.longlong

ctypedef fused number:
    cython.double
    cython.float
    cython.short
    cython.int
    cython.longlong
