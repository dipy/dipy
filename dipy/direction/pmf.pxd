cimport numpy as np

cdef class PmfGen:
    cdef:
        double[:] pmf
        double[:, :, :, :] data
        double[:, :] vertices
        int nbr_vertices
        object sphere

    cpdef double[:] get_pmf(self, double[::1] point)
    cdef int find_closest(self, double[::1] xyz) nogil
    cpdef double get_pmf_value(self, double[::1] point, double[::1] xyz)
    cdef void __clear_pmf(self) nogil
    pass


cdef class SimplePmfGen(PmfGen):
    pass


cdef class SHCoeffPmfGen(PmfGen):
    cdef:
        double[:, :] B
        double[:] coeff
    pass
