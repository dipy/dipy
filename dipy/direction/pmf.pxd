

cdef class PmfGen:
    cpdef double[:] get_pmf(self, double[::1] point)
    cdef double[:] get_pmf_c(self, double* point) nogil


cdef class SimplePmfGen(PmfGen):
    cdef:
        double[:, :, :, :] pmf_array
        cdef double[:] out
    pass


cdef class SHCoeffPmfGen(PmfGen):
    cdef:
        double[:, :, :, :] shcoeff
        double[:, :] B
        object sphere

        double[:] coeff
        double[:] pmf
    pass
