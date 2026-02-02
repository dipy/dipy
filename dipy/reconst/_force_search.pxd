# Header for _force_search module

cdef void knn_inner_product_streaming(
    const float[:, ::1] queries,
    const float[:, ::1] database,
    size_t k,
    float[:, ::1] distances_out,
    long[:, ::1] indices_out
) noexcept nogil
