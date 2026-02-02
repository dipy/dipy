# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False
"""
High-performance k-NN search using SciPy BLAS.

Based on FORCE vec_search implementation:
- SciPy BLAS (sgemm) for fast matrix multiplication
- Parallel heap for memory-efficient top-k selection
"""

import numpy as np
cimport numpy as cnp
cimport cython
from cython.parallel cimport prange
from libc.stdlib cimport malloc, free

# Import SciPy BLAS
from scipy.linalg.cython_blas cimport sgemm

# Import heap functions from local module
from dipy.reconst._force_heap cimport (
    select_top_k_parallel,
    heap_init_parallel,
    heap_update_batch_parallel,
    heap_finalize_parallel
)


# Database chunk size for streaming
cdef size_t DB_CHUNK_SIZE = 8192


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void compute_distances_blas(
    const float[:, ::1] queries,
    const float[:, ::1] database,
    float* distances_out
) noexcept nogil:
    """
    Compute all inner products using SciPy BLAS sgemm.

    Computes: distances_out[i, j] = queries[i, :] . database[j, :]
    """
    cdef int nd = database.shape[0]
    cdef int nq = queries.shape[0]
    cdef int d = queries.shape[1]

    cdef float alpha = 1.0
    cdef float beta = 0.0
    cdef char trans_a = b'T'
    cdef char trans_b = b'N'

    cdef int m = nd
    cdef int n = nq
    cdef int kk = d
    cdef int lda = d
    cdef int ldb = d
    cdef int ldc = nd

    cdef float* db_ptr = <float*>&database[0, 0]
    cdef float* q_ptr = <float*>&queries[0, 0]

    sgemm(
        &trans_a,
        &trans_b,
        &m,
        &n,
        &kk,
        &alpha,
        db_ptr,
        &lda,
        q_ptr,
        &ldb,
        &beta,
        distances_out,
        &ldc
    )


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void knn_inner_product_streaming(
    const float[:, ::1] queries,
    const float[:, ::1] database,
    size_t k,
    float[:, ::1] distances_out,
    long[:, ::1] indices_out
) noexcept nogil:
    """
    Streaming k-NN search.

    Memory-efficient approach:
    1. Initialize top-k heaps for all queries
    2. Process database in chunks of DB_CHUNK_SIZE
    3. For each chunk: compute distances, update heaps
    4. Finalize heaps (sort in descending order)
    """
    cdef size_t n_queries = queries.shape[0]
    cdef size_t n_database = database.shape[0]
    cdef size_t d = queries.shape[1]

    cdef size_t chunk_start, chunk_end, chunk_size
    cdef float* distances = NULL

    # Step 1: Initialize heaps with -inf values
    heap_init_parallel(
        n_queries,
        k,
        &distances_out[0, 0],
        &indices_out[0, 0]
    )

    # Allocate buffer for one chunk worth of distances
    cdef size_t max_chunk = DB_CHUNK_SIZE
    if max_chunk > n_database:
        max_chunk = n_database

    distances = <float*>malloc(n_queries * max_chunk * sizeof(float))
    if distances == NULL:
        with gil:
            raise MemoryError("Cannot allocate chunk distance buffer")

    try:
        # Step 2: Process database in chunks
        chunk_start = 0
        while chunk_start < n_database:
            chunk_end = chunk_start + DB_CHUNK_SIZE
            if chunk_end > n_database:
                chunk_end = n_database
            chunk_size = chunk_end - chunk_start

            # Compute distances for this chunk using BLAS
            compute_distances_blas(
                queries,
                database[chunk_start:chunk_end],
                distances
            )

            # Update heaps with this chunk's distances
            heap_update_batch_parallel(
                distances,
                n_queries,
                chunk_size,
                chunk_start,
                k,
                &distances_out[0, 0],
                &indices_out[0, 0]
            )

            chunk_start = chunk_end

        # Step 3: Finalize heaps (sort in descending order)
        heap_finalize_parallel(
            n_queries,
            k,
            &distances_out[0, 0],
            &indices_out[0, 0]
        )

    finally:
        if distances != NULL:
            free(distances)


@cython.boundscheck(False)
@cython.wraparound(False)
def search_inner_product(
    const float[:, ::1] queries not None,
    const float[:, ::1] database not None,
    int k
):
    """
    Search for k-nearest neighbors using inner product.

    Uses SciPy BLAS for matrix multiplication and streaming
    heap for memory-efficient top-k selection.

    Parameters
    ----------
    queries : float32 memoryview (n_queries, d)
        Query vectors (C-contiguous)
    database : float32 memoryview (n_database, d)
        Database vectors (C-contiguous)
    k : int
        Number of nearest neighbors

    Returns
    -------
    distances : float32 array (n_queries, k)
        Inner products (descending order)
    indices : int64 array (n_queries, k)
        Indices of nearest neighbors
    """
    cdef int n_queries = queries.shape[0]
    cdef int n_database = database.shape[0]
    cdef int d = queries.shape[1]

    if queries.shape[1] != database.shape[1]:
        raise ValueError(
            f"Dimension mismatch: queries {queries.shape[1]} != database {database.shape[1]}"
        )

    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    if k > n_database:
        raise ValueError(f"k ({k}) cannot be larger than database size ({n_database})")

    # Allocate output arrays (C-contiguous)
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode='c'] distances = \
        np.empty((n_queries, k), dtype=np.float32, order='C')
    cdef cnp.ndarray[cnp.int64_t, ndim=2, mode='c'] indices = \
        np.empty((n_queries, k), dtype=np.int64, order='C')

    # Get memoryviews
    cdef float[:, ::1] distances_view = distances
    cdef long[:, ::1] indices_view = indices

    # Call streaming search
    with nogil:
        knn_inner_product_streaming(
            queries,
            database,
            k,
            distances_view,
            indices_view
        )

    return distances, indices
