cimport numpy as np

cdef class PmfGen:
    cpdef double[:] get_pmf(self, double[::1] point)
    cdef double[:] get_pmf_c(self, double* point)


cdef class SimplePmfGen(PmfGen):
    cdef:
        double[:, :, :, :] pmf_array
        double[:] out
    pass


cdef class SHCoeffPmfGen(PmfGen):
    cdef:
        double[:, :, :, :] shcoeff
        double[:, :] B
        object sphere

        double[:] coeff
        double[:] pmf
    pass

cdef class BootPmfGen(SHCoeffPmfGen):
    cdef:
        int sh_order
        int nbr_b0s
        int nbr_dwi
        int nbr_data
        object model
        object H
        np.ndarray vox_data
        np.ndarray vox_b0s
        np.ndarray vox_dwi
        double[:, :] R
        double[:, :, :, :] b0s
        double[:, :, :, :] dwi

    cpdef double[:] get_pmf_no_boot(self, double[::1] point)
    cdef double[:] get_pmf_no_boot_c(self, double* point)
    cdef int _set_vox_data(self)

    pass
