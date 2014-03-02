cimport cython

#ctypedef cython.float floating
ctypedef fused floating:
    cython.double
    cython.float

ctypedef fused number:
    cython.double
    cython.float
    cython.short
    cython.int
    cython.longlong
