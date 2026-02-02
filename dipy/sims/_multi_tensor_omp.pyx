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


def multi_tensor(
    const double[:, ::1] mevals,
    const double[:, :, ::1] evecs,
    const double[::1] fractions,
    const double[::1] bvals,
    const double[:, ::1] bvecs
):
    """
    Compute multi-tensor diffusion signal.

    Accumulates signal contributions from multiple diffusion tensors
    weighted by their volume fractions.

    Parameters
    ----------
    mevals : ndarray (N, 3)
        Eigenvalues for each tensor.
    evecs : ndarray (N, 3, 3)
        Eigenvectors for each tensor.
    fractions : ndarray (N,)
        Volume fractions (should sum to 100).
    bvals : ndarray (M,)
        B-values.
    bvecs : ndarray (M, 3)
        Gradient directions.

    Returns
    -------
    signal : ndarray (M,)
        Combined multi-tensor signal.
    """
    cdef:
        int i, j
        int n = bvals.shape[0]
        int n_tensors = fractions.shape[0]
        double[::1] S = np.zeros(n, dtype=np.float64)
        double[::1] tmp_S_one = np.zeros(n, dtype=np.float64)
        double frac

    with nogil:
        for i in range(n_tensors):
            single_tensor(mevals[i], evecs[i], bvals, bvecs, tmp_S_one)

            frac = fractions[i] / 100.0

            if frac > 0:
                for j in range(n):
                    S[j] += tmp_S_one[j] * frac

    return np.asarray(S)
