# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False
"""
High-performance min-heap implementation for k-NN search.
Uses Cython with OpenMP (prange) for parallel heap operations.
"""

cimport cython
from cython.parallel cimport prange
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void heap_swap(
    float* vals,
    long* ids,
    size_t i,
    size_t j
) noexcept nogil:
    """Swap two heap elements."""
    cdef float tmp_val = vals[i]
    cdef long tmp_id = ids[i]
    vals[i] = vals[j]
    ids[i] = ids[j]
    vals[j] = tmp_val
    ids[j] = tmp_id


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void heapify_down(
    float* vals,
    long* ids,
    size_t n,
    size_t i
) noexcept nogil:
    """Restore min-heap property downward from index i."""
    cdef size_t left, right, smallest

    while True:
        left = 2 * i + 1
        right = 2 * i + 2
        smallest = i

        if left < n and vals[left] < vals[smallest]:
            smallest = left
        if right < n and vals[right] < vals[smallest]:
            smallest = right

        if smallest == i:
            break

        heap_swap(vals, ids, i, smallest)
        i = smallest


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void heap_push(
    float* vals,
    long* ids,
    size_t k,
    float val,
    long idx
) noexcept nogil:
    """Push element into min-heap, replacing root if val is larger."""
    if val > vals[0]:
        vals[0] = val
        ids[0] = idx
        heapify_down(vals, ids, k, 0)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void heapify(
    float* vals,
    long* ids,
    size_t n
) noexcept nogil:
    """Build min-heap from unordered array."""
    cdef long i
    for i in range(<long>n // 2 - 1, -1, -1):
        heapify_down(vals, ids, n, i)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void heap_reorder(
    float* vals,
    long* ids,
    size_t n
) noexcept nogil:
    """Sort heap in descending order (largest first)."""
    cdef size_t i
    for i in range(n - 1, 0, -1):
        heap_swap(vals, ids, 0, i)
        heapify_down(vals, ids, i, 0)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void select_top_k_parallel(
    const float* distances,
    size_t n_queries,
    size_t n_database,
    size_t k,
    float* out_distances,
    long* out_indices
) noexcept nogil:
    """
    Select top-k from distance matrix in parallel.

    Uses min-heap for each query to find k largest distances.
    Parallelized with OpenMP (prange).
    """
    cdef Py_ssize_t i
    cdef size_t j
    cdef const float* dist_row
    cdef float* dist_out
    cdef long* idx_out

    for i in prange(<Py_ssize_t>n_queries, schedule='static', nogil=True):
        dist_row = distances + <size_t>i * n_database
        dist_out = out_distances + <size_t>i * k
        idx_out = out_indices + <size_t>i * k

        for j in range(k):
            dist_out[j] = dist_row[j]
            idx_out[j] = j

        heapify(dist_out, idx_out, k)

        for j in range(k, n_database):
            heap_push(dist_out, idx_out, k, dist_row[j], j)

        heap_reorder(dist_out, idx_out, k)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void heap_init_parallel(
    size_t n_queries,
    size_t k,
    float* out_distances,
    long* out_indices
) noexcept nogil:
    """
    Initialize heaps with -inf values for streaming top-k.
    """
    cdef Py_ssize_t i
    cdef size_t j
    cdef float* dist_out
    cdef long* idx_out
    cdef float neg_inf = -1e30

    for i in prange(<Py_ssize_t>n_queries, schedule='static', nogil=True):
        dist_out = out_distances + <size_t>i * k
        idx_out = out_indices + <size_t>i * k

        for j in range(k):
            dist_out[j] = neg_inf
            idx_out[j] = -1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void heap_update_batch_parallel(
    const float* distances,
    size_t n_queries,
    size_t chunk_size,
    size_t chunk_offset,
    size_t k,
    float* out_distances,
    long* out_indices
) noexcept nogil:
    """
    Update running heaps with a batch of distances (streaming approach).
    """
    cdef Py_ssize_t i
    cdef size_t j
    cdef const float* dist_row
    cdef float* dist_out
    cdef long* idx_out
    cdef long global_idx

    for i in prange(<Py_ssize_t>n_queries, schedule='static', nogil=True):
        dist_row = distances + <size_t>i * chunk_size
        dist_out = out_distances + <size_t>i * k
        idx_out = out_indices + <size_t>i * k

        for j in range(chunk_size):
            global_idx = <long>(chunk_offset + j)
            heap_push(dist_out, idx_out, k, dist_row[j], global_idx)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void heap_finalize_parallel(
    size_t n_queries,
    size_t k,
    float* out_distances,
    long* out_indices
) noexcept nogil:
    """
    Finalize heaps by sorting in descending order.
    """
    cdef Py_ssize_t i
    cdef float* dist_out
    cdef long* idx_out

    for i in prange(<Py_ssize_t>n_queries, schedule='static', nogil=True):
        dist_out = out_distances + <size_t>i * k
        idx_out = out_indices + <size_t>i * k
        heap_reorder(dist_out, idx_out, k)
