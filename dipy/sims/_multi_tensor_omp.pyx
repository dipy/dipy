# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
"""
OpenMP-accelerated Multi-Tensor Signal Generation

High-performance implementation of multi-tensor diffusion signal
computation with optional OpenMP parallelization.
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport exp


cdef inline void single_tensor(
    const double[::1] evals,
    const double[:, ::1] evec,
    const double[::1] bvals,
    const double[:, ::1] bvecs,
    double[::1] S
) noexcept nogil:
    """
    Compute signal for a single diffusion tensor.

    Parameters
    ----------
    evals : memoryview (3,)
        Eigenvalues of the diffusion tensor.
    evec : memoryview (3, 3)
        Eigenvectors of the diffusion tensor.
    bvals : memoryview (M,)
        B-values.
    bvecs : memoryview (M, 3)
        Gradient directions.
    S : memoryview (M,)
        Output signal array.
    """
    cdef:
        int i
        double D00, D01, D02, D11, D12, D22
        double bx, by, bz, val
        double S0 = 1.0

    # Compute diffusion tensor components from eigendecomposition
    D00 = (evec[0, 0] * evals[0] * evec[0, 0] +
           evec[0, 1] * evals[1] * evec[0, 1] +
           evec[0, 2] * evals[2] * evec[0, 2])
    D01 = (evec[0, 0] * evals[0] * evec[1, 0] +
           evec[0, 1] * evals[1] * evec[1, 1] +
           evec[0, 2] * evals[2] * evec[1, 2])
    D02 = (evec[0, 0] * evals[0] * evec[2, 0] +
           evec[0, 1] * evals[1] * evec[2, 1] +
           evec[0, 2] * evals[2] * evec[2, 2])
    D11 = (evec[1, 0] * evals[0] * evec[1, 0] +
           evec[1, 1] * evals[1] * evec[1, 1] +
           evec[1, 2] * evals[2] * evec[1, 2])
    D12 = (evec[1, 0] * evals[0] * evec[2, 0] +
           evec[1, 1] * evals[1] * evec[2, 1] +
           evec[1, 2] * evals[2] * evec[2, 2])
    D22 = (evec[2, 0] * evals[0] * evec[2, 0] +
           evec[2, 1] * evals[1] * evec[2, 1] +
           evec[2, 2] * evals[2] * evec[2, 2])

    # Compute signal attenuation for each gradient direction
    for i in range(bvals.shape[0]):
        bx = bvecs[i, 0]
        by = bvecs[i, 1]
        bz = bvecs[i, 2]

        val = (D00 * bx * bx + 2 * D01 * bx * by + 2 * D02 * bx * bz +
               D11 * by * by + 2 * D12 * by * bz + D22 * bz * bz)

        S[i] = S0 * exp(-bvals[i] * val)
