cimport numpy as cnp


cdef cnp.uint16_t peak_directions_c(
    double[:] odf,
    double[:, ::1] sphere_vertices,
    cnp.uint16_t[:, ::1] sphere_edges,
    double relative_peak_threshold,
    double min_separation_angle,
    bint is_symmetric,
    double[:, ::1] out_directions,
    double[::1] out_values,
    cnp.npy_intp[::1] out_indices,
    double[:, :] unique_vertices,
    cnp.uint16_t[:] mapping,
    cnp.uint16_t[:] index
) noexcept nogil
