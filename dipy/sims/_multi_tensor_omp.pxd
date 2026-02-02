# cython: language_level=3
"""
Header file for _multi_tensor_omp Cython module.

This exposes the cdef functions for use by other Cython modules.
"""

cdef inline void single_tensor(
    const double[::1] evals,
    const double[:, ::1] evec,
    const double[::1] bvals,
    const double[:, ::1] bvecs,
    double[::1] S
) noexcept nogil
