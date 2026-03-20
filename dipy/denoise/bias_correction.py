"""Bias field correction for diffusion MRI data.

Provides classical regression-based bias field correction via Legendre
polynomial regression and cubic B-spline regression.

The bias field is estimated exclusively from the mean b0 volume in the
log domain and applied uniformly to all DWI volumes.
"""

import numpy as np
from scipy import linalg as scipy_linalg, ndimage, sparse

try:
    from dipy.denoise._bias_correction import (
        compute_tukey_weights,
        evaluate_bspline_rows,
        gram_matrix_csr,
        masked_voxel_coords,
    )

    _HAVE_CYTHON = True
except ImportError:
    _HAVE_CYTHON = False

from dipy.core.gradients import extract_b0
from dipy.segment.mask import applymask, median_otsu
from dipy.utils.logging import logger

try:
    from dipy.align.vector_fields import gradient as _vf_gradient

    _HAVE_VF_GRADIENT = True
except ImportError:
    _HAVE_VF_GRADIENT = False


def _get_mean_b0(data, gtab):
    """Return mean b0 volume as float64.

    Parameters
    ----------
    data : ndarray
        4D DWI data (X, Y, Z, N).
    gtab : GradientTable
        Gradient table with b0s_mask attribute.

    Returns
    -------
    mean_b0 : ndarray
        3D mean b0 volume, dtype float64.
    """
    return extract_b0(data, gtab.b0s_mask, strategy="mean").astype(np.float64)


def _get_mask(mean_b0, mask):
    """Return binary brain mask, computing via median_otsu if not provided.

    Parameters
    ----------
    mean_b0 : ndarray
        3D mean b0 volume.
    mask : ndarray or None
        Existing 3D binary mask, or None to auto-compute.

    Returns
    -------
    mask : ndarray
        3D boolean brain mask.
    """
    if mask is None:
        _, mask = median_otsu(mean_b0, median_radius=4, numpass=4)
    else:
        mask = np.asarray(mask, dtype=bool)
    return mask


def _gradient_weights(*, log_b0, alpha=1.0):
    """Compute gradient-based edge suppression weight map.

    Parameters
    ----------
    log_b0 : ndarray
        3D log-domain b0 image, shape (S, R, C).
    alpha : float, optional
        Edge suppression strength.

    Returns
    -------
    weights : ndarray
        Float64 weight map, same shape as log_b0.
    """
    img = np.ascontiguousarray(log_b0, dtype=np.float64)
    if _HAVE_VF_GRADIENT:
        shape = np.array(img.shape, dtype=np.int32)
        eye4 = np.eye(4, dtype=np.float64)
        spacing = np.ones(3, dtype=np.float64)
        grad_out, _ = _vf_gradient(img, eye4, spacing, shape, eye4)
        grad_mag = np.sqrt(np.sum(grad_out**2, axis=-1))
    else:
        gx = ndimage.sobel(img, axis=0)
        gy = ndimage.sobel(img, axis=1)
        gz = ndimage.sobel(img, axis=2)
        grad_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    return np.exp(-alpha * grad_mag)


def _normalize_coords(*, shape, coords):
    """Normalize voxel coordinates to [-1, 1] along each axis.

    Parameters
    ----------
    shape : tuple of int
        Volume shape (S, R, C).
    coords : ndarray
        Integer coordinates, shape (N, 3).

    Returns
    -------
    coords_norm : ndarray
        Normalized float64 coordinates, shape (N, 3).
    """
    coords_norm = coords.astype(np.float64)
    for d, n in enumerate(shape):
        if n > 1:
            coords_norm[:, d] = 2.0 * coords_norm[:, d] / (n - 1) - 1.0
        else:
            coords_norm[:, d] = 0.0
    return coords_norm


def _legendre_basis(*, coords_flat, order):
    """Build Legendre polynomial design matrix.

    Parameters
    ----------
    coords_flat : ndarray
        Normalized coordinates in [-1, 1], shape (N, 3).
    order : int
        Maximum total polynomial degree (terms where i+j+k <= order).

    Returns
    -------
    X : ndarray
        Design matrix, shape (N, K) where K is the number of terms.
    """
    from numpy.polynomial.legendre import legval

    terms = [
        (i, j, k)
        for i in range(order + 1)
        for j in range(order + 1 - i)
        for k in range(order + 1 - i - j)
    ]
    N = coords_flat.shape[0]
    K = len(terms)
    X = np.zeros((N, K), dtype=np.float64)

    for col, (i, j, k) in enumerate(terms):
        ei = np.zeros(i + 1)
        ei[i] = 1.0
        ej = np.zeros(j + 1)
        ej[j] = 1.0
        ek = np.zeros(k + 1)
        ek[k] = 1.0
        X[:, col] = (
            legval(coords_flat[:, 0], ei)
            * legval(coords_flat[:, 1], ej)
            * legval(coords_flat[:, 2], ek)
        )
    return X


def _weighted_ridge_solve(*, X, y, weights, lambda_reg):
    """Solve weighted ridge regression min ||W^(1/2)(y - Xβ)||² + λ||β||².

    Parameters
    ----------
    X : ndarray
        Design matrix, shape (N, K).
    y : ndarray
        Target values, shape (N,).
    weights : ndarray
        Non-negative regression weights, shape (N,).
    lambda_reg : float
        Ridge regularization strength.

    Returns
    -------
    beta : ndarray
        Coefficient vector, shape (K,).
    """
    K = X.shape[1]
    WX = weights[:, None] * X
    A = X.T @ WX + lambda_reg * np.eye(K)
    b = X.T @ (weights * y)
    try:
        beta = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        beta, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return beta


def _tukey_weights_py(*, residuals, c):
    """Compute Tukey biweight weights (pure Python/NumPy).

    Parameters
    ----------
    residuals : ndarray
        Regression residuals, shape (N,).
    c : float
        Tukey breakdown constant.

    Returns
    -------
    weights : ndarray
        Tukey biweight weights in [0, 1], shape (N,).
    """
    mad = np.median(np.abs(residuals)) / 0.6745
    if mad < 1e-15:
        return np.ones(len(residuals), dtype=np.float64)
    u = residuals / (c * mad)
    w = np.where(np.abs(u) < 1.0, (1.0 - u**2) ** 2, 0.0)
    return w.astype(np.float64)


def _tukey_weights(*, residuals, c=4.685):
    """Compute Tukey biweight weights, using Cython backend if available.

    Parameters
    ----------
    residuals : ndarray
        Regression residuals, shape (N,).
    c : float, optional
        Tukey breakdown constant.

    Returns
    -------
    weights : ndarray
        Tukey biweight weights in [0, 1], shape (N,).
    """
    if _HAVE_CYTHON:
        w = np.ones(len(residuals), dtype=np.float64)
        compute_tukey_weights(np.ascontiguousarray(residuals, dtype=np.float64), w, c=c)
        return w
    return _tukey_weights_py(residuals=residuals, c=c)


def _polynomial_pyramid_fit(
    *,
    log_b0,
    mask,
    order,
    pyramid_levels,
    n_iter,
    lambda_reg,
    robust,
    gradient_weighting,
    sigma_factor=0.2,
):
    """Coarse-to-fine polynomial bias field regression.

    Parameters
    ----------
    log_b0 : ndarray
        3D log-domain b0 image, shape (S, R, C).
    mask : ndarray
        3D boolean brain mask.
    order : int
        Maximum Legendre polynomial order.
    pyramid_levels : tuple of int
        Downsampling factors, ordered coarse-first (e.g. (4, 2, 1)).
    n_iter : int
        Reweighting iterations per pyramid level.
    lambda_reg : float
        Ridge regularization strength.
    robust : bool
        Apply Tukey biweight robust reweighting.
    gradient_weighting : bool
        Apply gradient-based edge suppression weights.
    sigma_factor : float, optional
        Sigma = factor * sigma_factor for Gaussian smoothing.

    Returns
    -------
    log_bias : ndarray
        Estimated log-domain bias field, same shape as log_b0.
    """
    full_shape = log_b0.shape

    if gradient_weighting:
        grad_w_full = _gradient_weights(log_b0=log_b0)

    ii, jj, kk = np.meshgrid(
        np.arange(full_shape[0]),
        np.arange(full_shape[1]),
        np.arange(full_shape[2]),
        indexing="ij",
    )
    full_vox_coords = np.column_stack([ii.ravel(), jj.ravel(), kk.ravel()])
    full_coords_norm = _normalize_coords(shape=full_shape, coords=full_vox_coords)
    X_full = _legendre_basis(coords_flat=full_coords_norm, order=order)

    # Center log_b0 so the polynomial fits only spatial variation, not the
    # DC offset (overall intensity level).
    log_b0_dc = log_b0[mask].mean()
    residual = log_b0 - log_b0_dc

    log_bias = np.zeros(full_shape, dtype=np.float64)

    for factor in pyramid_levels:
        if factor == 1:
            level_residual = residual
            level_mask = mask
        else:
            sigma = factor * sigma_factor
            smoothed = ndimage.gaussian_filter(residual, sigma=sigma)
            level_residual = ndimage.zoom(smoothed, zoom=1.0 / factor, order=1)
            level_mask = (
                ndimage.zoom(mask.astype(np.float64), zoom=1.0 / factor, order=0) > 0.5
            )

        level_shape = level_residual.shape
        mask_flat = level_mask.ravel()
        n_masked = mask_flat.sum()

        # Need at least as many data points as parameters
        n_params = sum(
            1
            for i in range(order + 1)
            for j in range(order + 1 - i)
            for _ in range(order + 1 - i - j)
        )
        if n_masked < n_params:
            continue

        y = level_residual.ravel()[mask_flat]

        ii_l, jj_l, kk_l = np.meshgrid(
            np.arange(level_shape[0]),
            np.arange(level_shape[1]),
            np.arange(level_shape[2]),
            indexing="ij",
        )
        level_coords = np.column_stack(
            [
                ii_l.ravel()[mask_flat],
                jj_l.ravel()[mask_flat],
                kk_l.ravel()[mask_flat],
            ]
        )
        coords_norm = _normalize_coords(shape=level_shape, coords=level_coords)
        X = _legendre_basis(coords_flat=coords_norm, order=order)

        w = np.ones(n_masked, dtype=np.float64)
        if gradient_weighting:
            if factor == 1:
                gw = grad_w_full.ravel()[mask_flat]
            else:
                gw_down = ndimage.zoom(grad_w_full, zoom=1.0 / factor, order=1)
                gw = gw_down.ravel()[mask_flat]
            w = w * gw

        beta = None
        for _ in range(n_iter):
            beta = _weighted_ridge_solve(X=X, y=y, weights=w, lambda_reg=lambda_reg)
            residuals_iter = y - X @ beta
            if robust:
                w = w * _tukey_weights(residuals=residuals_iter)

        if beta is None:
            beta = _weighted_ridge_solve(X=X, y=y, weights=w, lambda_reg=lambda_reg)

        level_bias = (X_full @ beta).reshape(full_shape)
        log_bias += level_bias
        residual = residual - level_bias

    # Center: ensure bias_field has unit mean within mask
    log_bias -= log_bias[mask].mean()
    return log_bias


def polynomial_bias_field_dwi(
    data,
    gtab,
    *,
    mask=None,
    order=3,
    pyramid_levels=(4, 2, 1),
    n_iter=4,
    lambda_reg=1e-3,
    robust=True,
    gradient_weighting=True,
    zero_background=False,
):
    """DWI bias field correction via multi-resolution Legendre polynomial regression.

    Estimates the bias field from the mean b0 volume in log space using
    coarse-to-fine Legendre polynomial regression, then applies the estimated
    field to all DWI volumes.

    Parameters
    ----------
    data : ndarray
        4D DWI data (X, Y, Z, N).
    gtab : GradientTable
        Gradient table.
    mask : ndarray, optional
        3D binary brain mask. Auto-computed via median_otsu if None.
    order : int, optional
        Maximum Legendre polynomial order (terms where i+j+k <= order).
    pyramid_levels : tuple of int, optional
        Downsampling factors for coarse-to-fine pyramid (descending order).
    n_iter : int, optional
        Reweighting iterations per pyramid level.
    lambda_reg : float, optional
        Ridge regularization strength.
    robust : bool, optional
        Apply Tukey biweight robust reweighting.
    gradient_weighting : bool, optional
        Apply gradient-based edge suppression.
    zero_background : bool, optional
        If True, set the bias field to 1.0 (no correction) outside the brain
        mask. If False, the raw extrapolated field values are preserved in
        the returned bias_field array. Has no effect on the corrected DWI
        data (background voxels are always zeroed by the brain mask).

    Returns
    -------
    corrected : ndarray
        Bias-corrected 4D DWI data, same dtype as input.
    bias_field : ndarray
        Estimated 3D multiplicative bias field.
    """
    orig_dtype = data.dtype
    mean_b0 = _get_mean_b0(data, gtab)
    mask = _get_mask(mean_b0, mask)
    log_b0 = np.log(np.clip(mean_b0, 1e-10, None))

    log_bias = _polynomial_pyramid_fit(
        log_b0=log_b0,
        mask=mask,
        order=order,
        pyramid_levels=pyramid_levels,
        n_iter=n_iter,
        lambda_reg=lambda_reg,
        robust=robust,
        gradient_weighting=gradient_weighting,
    )

    if zero_background:
        log_bias[~mask] = 0.0
    bias_field = np.exp(log_bias)
    corrected = applymask(data.astype(np.float64) / bias_field[..., None], mask).astype(
        orig_dtype
    )
    return corrected, bias_field


def _build_bspline_design_matrix_py(*, log_b0_shape, n_control, mask_flat):
    """Build sparse B-spline design matrix (pure Python fallback).

    Parameters
    ----------
    log_b0_shape : tuple of int
        Shape of the 3D volume (S, R, C).
    n_control : tuple of int
        Control grid dimensions (ns, nr, nc).
    mask_flat : ndarray
        Flattened boolean mask, shape (S*R*C,).

    Returns
    -------
    X : scipy.sparse.csr_matrix
        Design matrix, shape (N_masked, K_ctrl_total).
    """
    S, R, C = log_b0_shape
    ns, nr, nc = n_control
    K = ns * nr * nc

    mask_3d = mask_flat.reshape(log_b0_shape)
    iz_all, iy_all, ix_all = np.where(mask_3d)
    N = len(iz_all)

    def _vox_to_ctrl_arr(vox, shape_d, n_ctrl_d):
        if shape_d <= 1 or n_ctrl_d <= 1:
            return np.zeros(len(vox), dtype=np.float64)
        return vox.astype(np.float64) * (n_ctrl_d - 1) / (shape_d - 1)

    tz = _vox_to_ctrl_arr(iz_all, S, ns)
    ty = _vox_to_ctrl_arr(iy_all, R, nr)
    tx = _vox_to_ctrl_arr(ix_all, C, nc)

    def _bspline_basis_batch(t, n_ctrl):
        """Vectorized cubic B-spline basis.

        Returns (N,4) basis values and (N,4) control indices.
        """
        t = np.clip(t, 0.0, n_ctrl - 1 - 1e-10)
        k = np.floor(t).astype(np.int64)
        k = np.minimum(k, n_ctrl - 2)
        u = t - k
        u2 = u * u
        u3 = u2 * u
        b = np.stack(
            [
                (1.0 - u) ** 3 / 6.0,
                (3.0 * u3 - 6.0 * u2 + 4.0) / 6.0,
                (-3.0 * u3 + 3.0 * u2 + 3.0 * u + 1.0) / 6.0,
                u3 / 6.0,
            ],
            axis=-1,
        )  # (N, 4)
        ctrl = np.stack([k - 1, k, k + 1, k + 2], axis=-1)  # (N, 4)
        return b, ctrl

    bz, cz = _bspline_basis_batch(tz, ns)  # (N, 4)
    by_, cy = _bspline_basis_batch(ty, nr)
    bx, cx = _bspline_basis_batch(tx, nc)

    # Tensor product: (N, 4, 4, 4) via broadcasting
    vals = (
        bz[:, :, np.newaxis, np.newaxis]
        * by_[:, np.newaxis, :, np.newaxis]
        * bx[:, np.newaxis, np.newaxis, :]
    )
    cols = (
        cz[:, :, np.newaxis, np.newaxis] * (nr * nc)
        + cy[:, np.newaxis, :, np.newaxis] * nc
        + cx[:, np.newaxis, np.newaxis, :]
    )
    rows = np.broadcast_to(
        np.arange(N, dtype=np.int64)[:, np.newaxis, np.newaxis, np.newaxis],
        (N, 4, 4, 4),
    )

    # Validity: all three ctrl indices must be in bounds
    valid = (
        (cz[:, :, np.newaxis, np.newaxis] >= 0)
        & (cz[:, :, np.newaxis, np.newaxis] < ns)
        & (cy[:, np.newaxis, :, np.newaxis] >= 0)
        & (cy[:, np.newaxis, :, np.newaxis] < nr)
        & (cx[:, np.newaxis, np.newaxis, :] >= 0)
        & (cx[:, np.newaxis, np.newaxis, :] < nc)
    )

    return sparse.csr_matrix(
        (vals[valid], (rows[valid], cols[valid])),
        shape=(N, K),
        dtype=np.float64,
    )


def _build_bspline_design_matrix(*, log_b0_shape, n_control, mask_flat):
    """Build sparse B-spline design matrix, using Cython backend if available.

    Parameters
    ----------
    log_b0_shape : tuple of int
        Shape of the 3D volume (S, R, C).
    n_control : tuple of int
        Control grid dimensions (ns, nr, nc).
    mask_flat : ndarray
        Flattened boolean mask, shape (S*R*C,).

    Returns
    -------
    X : scipy.sparse.csr_matrix
        Design matrix, shape (N_masked, K_ctrl_total).
    """
    if _HAVE_CYTHON:
        S, R, C = log_b0_shape
        ns, nr, nc = n_control
        K = ns * nr * nc
        mask_3d = mask_flat.reshape(log_b0_shape).astype(np.uint8)
        N_max = int(mask_flat.sum())

        out_coords = np.zeros((N_max, 3), dtype=np.int64)
        N_actual = int(masked_voxel_coords(np.ascontiguousarray(mask_3d), out_coords))
        out_coords = out_coords[:N_actual]

        def _scale(axis_coords, shape_d, n_ctrl_d):
            if shape_d <= 1 or n_ctrl_d <= 1:
                return np.zeros(len(axis_coords), dtype=np.float64)
            return axis_coords.astype(np.float64) * (n_ctrl_d - 1) / (shape_d - 1)

        grid_coords = np.column_stack(
            [
                _scale(out_coords[:, 0], S, ns),
                _scale(out_coords[:, 1], R, nr),
                _scale(out_coords[:, 2], C, nc),
            ]
        ).astype(np.float64)

        n_ctrl_arr = np.array([ns, nr, nc], dtype=np.int64)
        row_ptr = np.zeros(N_actual + 1, dtype=np.int64)
        col_idx = np.zeros(N_actual * 64, dtype=np.int64)
        values = np.zeros(N_actual * 64, dtype=np.float64)

        nnz = int(
            evaluate_bspline_rows(
                np.ascontiguousarray(grid_coords),
                n_ctrl_arr,
                row_ptr,
                col_idx,
                values,
            )
        )
        col_idx = col_idx[:nnz]
        values = values[:nnz]

        return sparse.csr_matrix(
            (values, col_idx, row_ptr),
            shape=(N_actual, K),
            dtype=np.float64,
        )

    return _build_bspline_design_matrix_py(
        log_b0_shape=log_b0_shape, n_control=n_control, mask_flat=mask_flat
    )


def _sparse_weighted_ridge_solve(*, X_sparse, y, weights, lambda_reg):
    """Solve sparse weighted ridge regression.

    The Gram matrix A = X^T W X is computed via chunked dense BLAS DGEMM,
    which is substantially faster than scipy sparse×sparse multiplication
    when the result is nearly dense (K < 1000).

    Parameters
    ----------
    X_sparse : scipy.sparse.csr_matrix
        Design matrix, shape (N, K).
    y : ndarray
        Target values, shape (N,).
    weights : ndarray
        Non-negative regression weights, shape (N,).
    lambda_reg : float
        Ridge regularization strength.

    Returns
    -------
    beta : ndarray
        Coefficient vector, shape (K,).
    """
    K = X_sparse.shape[1]
    N = X_sparse.shape[0]

    if _HAVE_CYTHON:
        # Fast path: Cython direct CSR accumulation
        A = np.zeros((K, K), dtype=np.float64)
        b_vec = np.zeros(K, dtype=np.float64)
        gram_matrix_csr(
            np.asarray(X_sparse.data, dtype=np.float64),
            np.asarray(X_sparse.indices, dtype=np.int32),
            np.asarray(X_sparse.indptr, dtype=np.int32),
            np.ascontiguousarray(weights, dtype=np.float64),
            np.ascontiguousarray(y, dtype=np.float64),
            A,
            b_vec,
        )
    else:
        # Chunked BLAS DGEMM: avoids sparse×sparse which is slow for dense-ish
        # results (K×K), converting sparse rows to dense in blocks and using
        # BLAS for the accumulation.
        chunk = min(4096, N)
        A = np.zeros((K, K), dtype=np.float64)
        b_vec = np.zeros(K, dtype=np.float64)
        for i in range(0, N, chunk):
            Xc = X_sparse[i : i + chunk].toarray()  # (chunk, K)
            wc = weights[i : i + chunk]
            A += Xc.T @ (wc[:, np.newaxis] * Xc)  # BLAS DGEMM
            b_vec += Xc.T.dot(wc * y[i : i + chunk])

    A += lambda_reg * np.eye(K)
    try:
        beta = scipy_linalg.solve(A, b_vec, assume_a="pos")
    except scipy_linalg.LinAlgError:
        beta, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
    return beta


def _refine_control_coeffs(*, coeffs, n_ctrl_coarse, n_ctrl_fine):
    """Trilinear interpolation of control grid coefficients.

    Parameters
    ----------
    coeffs : ndarray
        Flattened control point coefficients at coarse resolution.
    n_ctrl_coarse : tuple of int
        Coarse control grid dimensions.
    n_ctrl_fine : tuple of int
        Fine control grid dimensions.

    Returns
    -------
    fine_coeffs : ndarray
        Flattened coefficients at fine resolution.
    """
    coarse_grid = coeffs.reshape(n_ctrl_coarse)
    zoom_factors = tuple(f / c for f, c in zip(n_ctrl_fine, n_ctrl_coarse))
    fine_grid = ndimage.zoom(coarse_grid, zoom=zoom_factors, order=1)
    # Trim or pad to exactly match target shape
    slices = tuple(slice(0, n) for n in n_ctrl_fine)
    fine_grid = fine_grid[slices]
    if fine_grid.shape != tuple(n_ctrl_fine):
        padded = np.zeros(n_ctrl_fine, dtype=np.float64)
        src_slices = tuple(slice(0, s) for s in fine_grid.shape)
        padded[src_slices] = fine_grid
        fine_grid = padded
    return fine_grid.ravel()


def _eval_bspline_field(*, coeffs, n_control, out_shape):
    """Evaluate B-spline field at all voxel positions.

    Uses ``scipy.ndimage.map_coordinates`` with ``prefilter=False`` so that
    ``coeffs`` are treated directly as B-spline weights (not as values to
    interpolate through).  This avoids building a full-resolution sparse
    design matrix and is O(N) in C rather than O(N × 64) in Python.

    Parameters
    ----------
    coeffs : ndarray
        Flattened control point coefficients (B-spline weights).
    n_control : tuple of int
        Control grid dimensions (ns, nr, nc).
    out_shape : tuple of int
        Output volume shape (S, R, C).

    Returns
    -------
    field : ndarray
        Evaluated field, shape out_shape.
    """
    S, R, C = out_shape
    ns, nr, nc = n_control

    coeff_grid = np.ascontiguousarray(coeffs.reshape(n_control), dtype=np.float64)

    iz = np.linspace(0, ns - 1, S) if ns > 1 else np.zeros(S)
    iy = np.linspace(0, nr - 1, R) if nr > 1 else np.zeros(R)
    ix_ = np.linspace(0, nc - 1, C) if nc > 1 else np.zeros(C)

    II, JJ, KK = np.meshgrid(iz, iy, ix_, indexing="ij")
    coords = np.vstack([II.ravel(), JJ.ravel(), KK.ravel()])

    field = ndimage.map_coordinates(
        coeff_grid, coords, order=3, mode="nearest", prefilter=False
    )
    return field.reshape(out_shape)


def _bspline_pyramid_fit(
    *,
    log_b0,
    mask,
    n_control_points,
    pyramid_levels,
    n_iter,
    lambda_reg,
    robust,
    gradient_weighting,
    sigma_factor=0.2,
):
    """Coarse-to-fine B-spline bias field regression.

    Parameters
    ----------
    log_b0 : ndarray
        3D log-domain b0 image, shape (S, R, C).
    mask : ndarray
        3D boolean brain mask.
    n_control_points : tuple of int
        Control grid dimensions at finest level.
    pyramid_levels : tuple of int
        Downsampling factors, ordered coarse-first (e.g. (4, 2, 1)).
    n_iter : int
        Reweighting iterations per pyramid level.
    lambda_reg : float
        Ridge regularization strength.
    robust : bool
        Apply Tukey biweight robust reweighting.
    gradient_weighting : bool
        Apply gradient-based edge suppression weights.
    sigma_factor : float, optional
        Sigma = factor * sigma_factor for Gaussian smoothing.

    Returns
    -------
    log_bias : ndarray
        Estimated log-domain bias field, same shape as log_b0.
    """
    full_shape = log_b0.shape

    if gradient_weighting:
        grad_w_full = _gradient_weights(log_b0=log_b0)

    # Center log_b0 so the B-spline fits only spatial variation, not the
    # DC offset (overall intensity level).
    log_b0_dc = log_b0[mask].mean()
    residual = log_b0 - log_b0_dc

    log_bias = np.zeros(full_shape, dtype=np.float64)
    prev_coeffs = None
    prev_n_ctrl = None

    for factor in pyramid_levels:
        n_ctrl = tuple(max(2, int(np.round(n / factor))) for n in n_control_points)

        if factor == 1:
            level_residual = residual
            level_mask = mask
        else:
            sigma = factor * sigma_factor
            smoothed = ndimage.gaussian_filter(residual, sigma=sigma)
            level_residual = ndimage.zoom(smoothed, zoom=1.0 / factor, order=1)
            level_mask = (
                ndimage.zoom(mask.astype(np.float64), zoom=1.0 / factor, order=0) > 0.5
            )

        level_shape = level_residual.shape
        mask_flat_level = level_mask.ravel()
        n_masked = mask_flat_level.sum()
        K = n_ctrl[0] * n_ctrl[1] * n_ctrl[2]

        if n_masked < K:
            continue

        y = level_residual.ravel()[mask_flat_level]

        X = _build_bspline_design_matrix(
            log_b0_shape=level_shape,
            n_control=n_ctrl,
            mask_flat=mask_flat_level,
        )

        # Warm-start: refine coefficients from previous coarser level
        if prev_coeffs is not None and prev_n_ctrl is not None:
            coeffs = _refine_control_coeffs(
                coeffs=prev_coeffs,
                n_ctrl_coarse=prev_n_ctrl,
                n_ctrl_fine=n_ctrl,
            )
        else:
            coeffs = np.zeros(K, dtype=np.float64)

        w = np.ones(n_masked, dtype=np.float64)
        if gradient_weighting:
            if factor == 1:
                gw = grad_w_full.ravel()[mask_flat_level]
            else:
                gw_down = ndimage.zoom(grad_w_full, zoom=1.0 / factor, order=1)
                gw = gw_down.ravel()[mask_flat_level]
            w = w * gw

        for _ in range(n_iter):
            coeffs = _sparse_weighted_ridge_solve(
                X_sparse=X, y=y, weights=w, lambda_reg=lambda_reg
            )
            residuals_iter = y - X @ coeffs
            if robust:
                w = w * _tukey_weights(residuals=residuals_iter)

        prev_coeffs = coeffs
        prev_n_ctrl = n_ctrl

        level_bias = _eval_bspline_field(
            coeffs=coeffs, n_control=n_ctrl, out_shape=full_shape
        )
        log_bias += level_bias
        residual = residual - level_bias

    # Center: ensure bias_field has unit mean within mask
    log_bias -= log_bias[mask].mean()
    return log_bias


def _auto_select_fit(
    *,
    log_b0,
    mean_b0,
    mask,
    order,
    n_control_points,
    pyramid_levels,
    n_iter,
    lambda_reg,
    robust,
    gradient_weighting,
):
    """Run poly and bspline fits, return the log-bias with lower CoV.

    Parameters
    ----------
    log_b0 : ndarray
        Log-domain mean b0, shape (X, Y, Z), float64.
    mean_b0 : ndarray
        Mean b0 in signal domain, shape (X, Y, Z), float64.
    mask : ndarray
        3D boolean brain mask.
    order : int
        Legendre polynomial order for poly fit.
    n_control_points : tuple of int
        B-spline control grid dimensions for bspline fit.
    pyramid_levels : tuple of int
        Downsampling factors for coarse-to-fine pyramid.
    n_iter : int
        Reweighting iterations per pyramid level.
    lambda_reg : float
        Ridge regularization strength.
    robust : bool
        Apply Tukey biweight robust reweighting.
    gradient_weighting : bool
        Apply gradient-based edge suppression.

    Returns
    -------
    log_bias : ndarray
        Log-domain bias field from the winning method.
    """
    log_bias_poly = _polynomial_pyramid_fit(
        log_b0=log_b0,
        mask=mask,
        order=order,
        pyramid_levels=pyramid_levels,
        n_iter=n_iter,
        lambda_reg=lambda_reg,
        robust=robust,
        gradient_weighting=gradient_weighting,
    )
    log_bias_bspline = _bspline_pyramid_fit(
        log_b0=log_b0,
        mask=mask,
        n_control_points=n_control_points,
        pyramid_levels=pyramid_levels,
        n_iter=n_iter,
        lambda_reg=lambda_reg,
        robust=robust,
        gradient_weighting=gradient_weighting,
    )

    def _cov(log_bf):
        """CoV of mean b0 corrected by the given log bias field."""
        corrected_b0 = mean_b0 / np.where(np.exp(log_bf) > 1e-10, np.exp(log_bf), 1.0)
        vals = corrected_b0[mask]
        return vals.std() / (vals.mean() + 1e-12)

    cov_poly = _cov(log_bias_poly)
    cov_bspline = _cov(log_bias_bspline)

    if cov_poly <= cov_bspline:
        logger.info(
            "bias_field_correction auto: selected 'poly' " "(CoV %.4f vs bspline %.4f)",
            cov_poly,
            cov_bspline,
        )
        return log_bias_poly

    logger.info(
        "bias_field_correction auto: selected 'bspline' " "(CoV %.4f vs poly %.4f)",
        cov_bspline,
        cov_poly,
    )
    return log_bias_bspline


def bias_field_correction(
    data,
    gtab,
    *,
    mask=None,
    method="bspline",
    order=3,
    n_control_points=(8, 8, 8),
    pyramid_levels=(4, 2, 1),
    n_iter=4,
    lambda_reg=1e-3,
    robust=True,
    gradient_weighting=True,
    return_bias_field=False,
    zero_background=False,
):
    """Top-level DWI bias field correction via regression.

    Estimates a smooth multiplicative bias field from the mean b0 volume
    using polynomial or B-spline regression in log space, then applies the
    correction uniformly to all DWI volumes.

    Parameters
    ----------
    data : ndarray
        4D DWI data (X, Y, Z, N).
    gtab : GradientTable
        Gradient table.
    mask : ndarray, optional
        3D binary brain mask. If None, computed via median_otsu.
    method : str, optional
        Bias correction method:

        - ``"poly"``: Legendre polynomial regression — fast, low-parameter.
        - ``"bspline"``: Cubic B-spline regression — more flexible.
        - ``"auto"``: Run both methods and return the one with lower
          Coefficient of Variation within the brain mask. The chosen method
          is logged at INFO level.
    order : int, optional
        Maximum Legendre polynomial degree (used only for method="poly").
    n_control_points : tuple of int, optional
        Control grid dimensions at finest level (used only for
        method="bspline").
    pyramid_levels : tuple of int, optional
        Downsampling factors for coarse-to-fine pyramid (descending order).
    n_iter : int, optional
        Reweighting iterations per pyramid level.
    lambda_reg : float, optional
        Ridge regularization strength.
    robust : bool, optional
        Apply Tukey biweight robust reweighting at each level.
    gradient_weighting : bool, optional
        Weight regression by edge-suppression map derived from the
        image gradient.
    return_bias_field : bool, optional
        If True, return the bias field alongside the corrected data.
    zero_background : bool, optional
        If True, set the bias field to 1.0 (no correction) outside the brain
        mask. If False, the raw extrapolated field values are preserved in
        the returned bias_field array. Has no effect on the corrected DWI
        data (background voxels are always zeroed by the brain mask).

    Returns
    -------
    corrected : ndarray
        Bias-corrected DWI, same dtype as input.
    bias_field : ndarray
        3D multiplicative bias field (only returned if
        return_bias_field=True).
    """
    orig_dtype = data.dtype
    mean_b0 = _get_mean_b0(data, gtab)
    mask = _get_mask(mean_b0, mask)
    log_b0 = np.log(np.clip(mean_b0.astype(np.float64), 1e-10, None))

    if method == "poly":
        log_bias = _polynomial_pyramid_fit(
            log_b0=log_b0,
            mask=mask,
            order=order,
            pyramid_levels=pyramid_levels,
            n_iter=n_iter,
            lambda_reg=lambda_reg,
            robust=robust,
            gradient_weighting=gradient_weighting,
        )
    elif method == "bspline":
        log_bias = _bspline_pyramid_fit(
            log_b0=log_b0,
            mask=mask,
            n_control_points=n_control_points,
            pyramid_levels=pyramid_levels,
            n_iter=n_iter,
            lambda_reg=lambda_reg,
            robust=robust,
            gradient_weighting=gradient_weighting,
        )
    elif method == "auto":
        log_bias = _auto_select_fit(
            log_b0=log_b0,
            mean_b0=mean_b0,
            mask=mask,
            order=order,
            n_control_points=n_control_points,
            pyramid_levels=pyramid_levels,
            n_iter=n_iter,
            lambda_reg=lambda_reg,
            robust=robust,
            gradient_weighting=gradient_weighting,
        )
    else:
        raise ValueError(f"method must be 'poly', 'bspline', or 'auto', got '{method}'")

    if zero_background:
        log_bias[~mask] = 0.0
    bias_field = np.exp(log_bias)
    corrected = applymask(data.astype(np.float64) / bias_field[..., None], mask).astype(
        orig_dtype
    )

    if return_bias_field:
        return corrected, bias_field
    return corrected
