cimport numpy as np

cdef class PmfGen:
    cdef:
        double[:] pmf
        double[:, :, :, :] data
        double[:, :] vertices
        int nbr_vertices
        object sphere

    cdef double* get_pmf(self, double* point) nogil
    cdef int find_closest(self, double* xyz) nogil
    cdef double get_pmf_value(self, double* point, double* xyz) nogil
    cdef void __clear_pmf(self) nogil
    pass


cdef class SimplePmfGen(PmfGen):
    pass


cdef class SHCoeffPmfGen(PmfGen):
    cdef:
        double[:, :] B
        double[:] coeff
    pass
