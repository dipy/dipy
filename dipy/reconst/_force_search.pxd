# Header for _force_search module
cimport numpy as cnp

cdef void knn_inner_product_streaming(
    const float[:, ::1] queries,
    const float[:, ::1] database,
    size_t k,
    float[:, ::1] distances_out,
    cnp.int64_t[:, ::1] indices_out
) noexcept nogil
