cimport numpy as np

cdef class PmfGen:
    cdef:
        double[:] pmf
        double[:, :, :, :] data

    cpdef double[:] get_pmf(self, double[::1] point)
    cdef double[:] get_pmf_c(self, double* point)
    cdef void __clear_pmf(self)
    pass


cdef class SimplePmfGen(PmfGen):
    pass


cdef class SHCoeffPmfGen(PmfGen):
    cdef:
        double[:, :] B
        object sphere
        double[:] coeff
    pass


cdef class BootPmfGen(PmfGen):
    cdef:
        int sh_order
        double[:, :] R
        object sphere
        object model
        object H
        np.ndarray vox_data
        np.ndarray dwi_mask


    cpdef double[:] get_pmf_no_boot(self, double[::1] point)
    cdef double[:] get_pmf_no_boot_c(self, double* point)
    pass
