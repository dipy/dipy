cimport cython

#ctypedef cython.float floating
ctypedef fused floating:
#    cython.float
    cython.double

ctypedef fused number:
    cython.double
    cython.float
    cython.short
    cython.int
