cdef void select_top_k_parallel(
    const float* distances,
    size_t n_queries,
    size_t n_database,
    size_t k,
    float* out_distances,
    long* out_indices
) noexcept nogil

cdef void heap_init_parallel(
    size_t n_queries,
    size_t k,
    float* out_distances,
    long* out_indices
) noexcept nogil

cdef void heap_update_batch_parallel(
    const float* distances,
    size_t n_queries,
    size_t chunk_size,
    size_t chunk_offset,
    size_t k,
    float* out_distances,
    long* out_indices
) noexcept nogil

cdef void heap_finalize_parallel(
    size_t n_queries,
    size_t k,
    float* out_distances,
    long* out_indices
) noexcept nogil
