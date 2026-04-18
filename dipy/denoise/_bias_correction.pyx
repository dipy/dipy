# cython: boundscheck=False, wraparound=False, cdivision=True
"""
Cython backend for bias field correction.

Provides accelerated B-spline design matrix construction, Gram matrix
accumulation, and Tukey biweight weight computation.
"""

cimport numpy as cnp
import numpy as np


cdef void _bspline_1d_values(
    double t,
    long n_ctrl,
    double* bvals,
    long* start_idx,
) noexcept nogil:
    """Evaluate cubic B-spline basis at a single parameter value.

    Parameters
    ----------
    t : double
        Parameter value in [0, n_ctrl - 1).  Clamped to this range if
        out of bounds.
    n_ctrl : long
        Number of control points along this axis.
    bvals : double*
        Output array of length 4.  Filled with the four non-zero cubic
        B-spline basis values at ``t``.
    start_idx : long*
        Output scalar.  Set to the control-point index corresponding to
        ``bvals[0]``; the active indices are ``start_idx[0] + 0..3``.
        Boundary control points (index < 0 or >= n_ctrl) must be
        discarded by the caller.

    Returns
    -------
    None
        Results are written into ``bvals`` and ``start_idx`` in place.
    """
    cdef:
        long k
        double u, u2, u3

    if t < 0.0:
        t = 0.0
    if t >= n_ctrl - 1:
        t = n_ctrl - 1 - 1e-10

    k = <long>t
    if k >= n_ctrl - 1:
        k = n_ctrl - 2

    u = t - k
    u2 = u * u
    u3 = u2 * u

    bvals[0] = (1.0 - u) * (1.0 - u) * (1.0 - u) / 6.0
    bvals[1] = (3.0 * u3 - 6.0 * u2 + 4.0) / 6.0
    bvals[2] = (-3.0 * u3 + 3.0 * u2 + 3.0 * u + 1.0) / 6.0
    bvals[3] = u3 / 6.0

    start_idx[0] = k - 1


def masked_voxel_coords(
    unsigned char[:, :, ::1] mask,
    cnp.int64_t[:, ::1] out,
):
    """Extract (i, j, k) voxel coordinates of all nonzero mask entries.

    Parameters
    ----------
    mask : ndarray
        3D uint8 mask, C-contiguous, shape (S, R, C).
    out : ndarray
        Pre-allocated output array, shape (N_max, 3), dtype int64.
        Filled with (i, j, k) coordinates of nonzero voxels in
        row-major order.

    Returns
    -------
    count : int
        Number of nonzero voxels found (actual rows written to ``out``).
    """
    cdef:
        long S = mask.shape[0]
        long R = mask.shape[1]
        long C = mask.shape[2]
        long i, j, k, count

    count = 0
    with nogil:
        for i in range(S):
            for j in range(R):
                for k in range(C):
                    if mask[i, j, k] != 0:
                        out[count, 0] = i
                        out[count, 1] = j
                        out[count, 2] = k
                        count += 1
    return count


def evaluate_bspline_rows(
    double[:, ::1] grid_coords,
    cnp.int64_t[::1] n_ctrl,
    cnp.int64_t[::1] row_ptr,
    cnp.int64_t[::1] col_idx,
    double[::1] values,
):
    """Evaluate cubic B-spline basis for N masked voxels, filling CSR arrays.

    Parameters
    ----------
    grid_coords : ndarray
        Voxel positions in control grid space, shape (N, 3), C-contiguous,
        dtype float64.  Axis order is (z, y, x).
    n_ctrl : ndarray
        Control grid dimensions (ns, nr, nc), shape (3,), dtype int64.
    row_ptr : ndarray
        Output CSR row pointers, shape (N+1,), dtype int64.
        Pre-allocated; written in-place.
    col_idx : ndarray
        Output CSR column indices, shape (N*64,), dtype int64.
        Pre-allocated; written in-place.
    values : ndarray
        Output CSR non-zero values, shape (N*64,), dtype float64.
        Pre-allocated; written in-place.

    Returns
    -------
    nnz : int
        Actual number of non-zero entries written to ``col_idx`` and
        ``values``.  Always <= N * 64.
    """
    cdef:
        long N = grid_coords.shape[0]
        long ns = n_ctrl[0]
        long nr = n_ctrl[1]
        long nc = n_ctrl[2]
        long row, nnz, iz, iy, ix, cz, cy, cx, col
        double bz_vals[4]
        double by_vals[4]
        double bx_vals[4]
        long start_z, start_y, start_x
        double vz, vy, vx

    nnz = 0
    row_ptr[0] = 0

    with nogil:
        for row in range(N):
            _bspline_1d_values(grid_coords[row, 0], ns, bz_vals, &start_z)
            _bspline_1d_values(grid_coords[row, 1], nr, by_vals, &start_y)
            _bspline_1d_values(grid_coords[row, 2], nc, bx_vals, &start_x)

            for iz in range(4):
                cz = start_z + iz
                if cz < 0 or cz >= ns:
                    continue
                vz = bz_vals[iz]

                for iy in range(4):
                    cy = start_y + iy
                    if cy < 0 or cy >= nr:
                        continue
                    vy = by_vals[iy]

                    for ix in range(4):
                        cx = start_x + ix
                        if cx < 0 or cx >= nc:
                            continue
                        vx = bx_vals[ix]

                        col = cz * nr * nc + cy * nc + cx
                        col_idx[nnz] = col
                        values[nnz] = vz * vy * vx
                        nnz += 1

            row_ptr[row + 1] = nnz

    return nnz


def gram_matrix_csr(
    double[::1] data,
    int[::1] indices,
    int[::1] indptr,
    double[::1] weights,
    double[::1] y,
    double[:, ::1] A,
    double[::1] b_vec,
):
    """Accumulate weighted Gram matrix and right-hand side from a CSR matrix.

    Computes in-place::

        A[j, k]   += sum_i  w_i * X[i,j] * X[i,k]
        b_vec[k]  += sum_i  w_i * y_i    * X[i,k]

    Rows with ``weights[i] == 0`` are skipped.

    Parameters
    ----------
    data : ndarray
        CSR non-zero values, shape (nnz,), dtype float64.
    indices : ndarray
        CSR column indices, shape (nnz,), dtype int32.
    indptr : ndarray
        CSR row pointers, shape (N+1,), dtype int32.
    weights : ndarray
        Per-row regression weights, shape (N,), dtype float64.
    y : ndarray
        Target values, shape (N,), dtype float64.
    A : ndarray
        K x K Gram matrix, pre-allocated zeros, shape (K, K), dtype
        float64.  Updated in-place.
    b_vec : ndarray
        Right-hand side vector, pre-allocated zeros, shape (K,), dtype
        float64.  Updated in-place.

    Returns
    -------
    None
        ``A`` and ``b_vec`` are modified in-place; nothing is returned.
    """
    cdef:
        long N = weights.shape[0]
        long i
        int p, q, jp, jq, start, end, nnz_i
        double w_i, wy_i, dp, dq

    with nogil:
        for i in range(N):
            w_i = weights[i]
            if w_i == 0.0:
                continue
            wy_i = w_i * y[i]
            start = indptr[i]
            end = indptr[i + 1]
            nnz_i = end - start

            for p in range(nnz_i):
                jp = indices[start + p]
                dp = data[start + p]
                b_vec[jp] += wy_i * dp

                for q in range(nnz_i):
                    jq = indices[start + q]
                    dq = data[start + q]
                    A[jp, jq] += w_i * dp * dq


def compute_tukey_weights(
    double[::1] residuals,
    double[::1] weights,
    *,
    double c=4.685,
):
    """Compute Tukey biweight weights and multiply into ``weights`` in-place.

    Estimates the residual scale via the Median Absolute Deviation (MAD),
    then applies the Tukey bisquare function::

        weights[i] *= (1 - (r_i / (c * MAD))^2)^2   if |r_i| < c * MAD
        weights[i]  = 0                               otherwise

    where ``MAD = median(|residuals|) / 0.6745``.

    Parameters
    ----------
    residuals : ndarray
        Regression residuals, C-contiguous float64, shape (N,).
    weights : ndarray
        Existing per-voxel weights to update in-place, C-contiguous
        float64, shape (N,).
    c : float
        Tukey breakdown constant.  The default (4.685) gives 95 %
        efficiency under Gaussian noise.

    Returns
    -------
    None
        ``weights`` is modified in-place; nothing is returned.
    """
    cdef:
        long N = residuals.shape[0]
        long i
        double mad, scale, u, w_val

    if N == 0:
        return

    abs_res_np = np.abs(np.asarray(residuals))
    mad = float(np.median(abs_res_np)) / 0.6745

    if mad < 1e-15:
        return

    scale = c * mad

    for i in range(N):
        u = residuals[i] / scale
        if u < -1.0 or u > 1.0:
            weights[i] = 0.0
        else:
            w_val = 1.0 - u * u
            weights[i] = weights[i] * w_val * w_val
