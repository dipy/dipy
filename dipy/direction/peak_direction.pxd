cimport numpy as cnp

cdef class PeakDirectionGen:
    """Peak direction data container with trilinear interpolation.

    Stores sphere-based peak representation (indices + odf_vertices)
    for EUDX-style tracking with trilinear interpolation.
    """
    cdef:
        double[:, :, :, :] peak_indices   # (X, Y, Z, npeaks) - sphere indices
        double[:, :, :, :] peak_values    # (X, Y, Z, npeaks) - QA/GFA values
        double[:, :] odf_vertices         # (N_vertices, 3) - sphere directions
        int max_peaks                     # Maximum number of peaks per voxel
        cnp.npy_intp[4] shape             # Shape of the data volume
        cnp.npy_intp[4] strides           # Memory strides for indexing

    cdef void get_peak_data_c(self,
                              double* point,
                              double* indices_out,
                              double* values_out,
                              double* weights_out) noexcept nogil
