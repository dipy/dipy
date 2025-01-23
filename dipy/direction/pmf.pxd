cimport numpy as np

cdef class PmfGen:
    cdef:
        double[:] pmf
        double[:, :, :, :] data
        double[:, :] vertices
        object sphere

    cdef double* get_pmf_c(self, double* point, double* out) noexcept nogil
    cdef int find_closest(self, double* xyz) noexcept nogil
    cdef double get_pmf_value_c(self, double* point, double* xyz) noexcept nogil
    pass


cdef class SimplePmfGen(PmfGen):
    pass


cdef class SHCoeffPmfGen(PmfGen):
    cdef:
        double[:, :] B
        double[:] coeff
    pass
