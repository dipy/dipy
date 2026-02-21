cimport numpy as np
cimport numpy as cnp

cdef class PmfGen:
    cdef:
        double[:] pmf
        double[:, :, :, :] data
        double[:, :] vertices
        object sphere

    cdef double* get_pmf_c(self, double* point, double* out) noexcept nogil
    cdef int find_closest(self, double* xyz) noexcept nogil
    cdef double get_pmf_value_c(self, double* point, double* xyz) noexcept nogil
    cdef cnp.npy_intp get_peaks_c(self,
                                  double* point,
                                  double* out_values,
                                  double* out_indices,
                                  double* out_weights,
                                  cnp.npy_intp max_peaks,
                                  cnp.npy_intp* out_valid) noexcept nogil
    pass


cdef class SimplePmfGen(PmfGen):
    pass


cdef class SHCoeffPmfGen(PmfGen):
    cdef:
        double[:, :] B
        double[:] coeff
    pass


cdef class SimplePeakGen(PmfGen):
    cdef:
        double[:, :, :, :] peak_indices
        double[:, :, :, :] peak_values
        double[:, :] odf_vertices
        double* peak_indices_ptr
        double* peak_values_ptr
        double* odf_vertices_ptr
        int max_peaks
        cnp.npy_intp[4] peak_shape
        cnp.npy_intp[4] peak_strides
    cdef void _compute_trilinear(self,
                                 double* point,
                                 double* out_weights,
                                 cnp.npy_intp* out_index) noexcept nogil
    cdef cnp.npy_intp _inside_global_bounds(self,
                                            cnp.npy_intp* index) noexcept nogil
    pass
