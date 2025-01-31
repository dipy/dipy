cimport cython
cimport numpy as cnp


cdef void remove_similar_vertices_c(
    double[:, :] vertices,
    double theta,
    int remove_antipodal,
    int return_mapping,
    int return_index,
    double[:, :] unique_vertices,
    cnp.uint16_t[:] mapping,
    cnp.uint16_t[:] index,
    cnp.uint16_t* n_unique
) noexcept nogil


cdef cnp.npy_intp search_descending_c(
    cython.floating* arr,
    cnp.npy_intp size,
    double relative_threshold) noexcept nogil


cdef long local_maxima_c(
    double[:] odf, cnp.uint16_t[:, :] edges,
    double[::1] out_values,
    cnp.npy_intp[::1] out_indices) noexcept nogil