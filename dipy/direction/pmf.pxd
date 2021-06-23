cimport numpy as np

cdef class PmfGen:
    cdef:
        double[:] pmf
        double[:, :, :, :] data
        object sphere

    cpdef double[:] get_pmf(self, double[::1] point)
    cpdef double get_pmf_value(self, double[::1] point, double[::1] xyz)
    cdef void __clear_pmf(self)
    pass


cdef class SimplePmfGen(PmfGen):
    pass


cdef class SHCoeffPmfGen(PmfGen):
    cdef:
        double[:, :] B
        double[:] coeff
    pass


cdef class BootPmfGen(PmfGen):
    cdef:
        int sh_order
        double[:, :] R
        object model
        object H
        np.ndarray vox_data
        np.ndarray dwi_mask

    cpdef double[:] get_pmf_no_boot(self, double[::1] point)
    pass
