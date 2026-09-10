"""Bias field correction for diffusion MRI data.

Provides classical regression-based bias field correction via Legendre
polynomial regression and cubic B-spline regression.

The bias field is estimated exclusively from the mean b0 volume in the
log domain and applied uniformly to all DWI volumes. By default the
regression is wrapped in the iterative histogram-sharpening scheme of N4
:footcite:p:`Tustison2010`, which separates tissue contrast from the slowly
varying field.
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
from dipy.segment.mask import median_otsu
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


def _extrapolate_outside_mask(*, log_bias, mask, sigma=2.0):
    """Extend the log-domain bias field beyond the brain mask.

    Regression is only constrained inside the mask, so the raw field can
    diverge by orders of magnitude outside it. Each background voxel takes the
    value of its nearest in-mask voxel, and the result is Gaussian-smoothed
    so the field stays continuous across the mask boundary.

    Parameters
    ----------
    log_bias : ndarray
        3D log-domain bias field.
    mask : ndarray
        3D boolean brain mask.
    sigma : float, optional
        Gaussian smoothing sigma (voxels) applied to the extrapolated region.

    Returns
    -------
    log_bias : ndarray
        Log-domain bias field, unchanged inside the mask and extrapolated
        outside.
    """
    if mask.all():
        return log_bias
    _, nearest = ndimage.distance_transform_edt(~mask, return_indices=True)
    filled = log_bias[tuple(nearest)]
    smoothed = ndimage.gaussian_filter(filled, sigma=sigma)
    out = log_bias.copy()
    out[~mask] = smoothed[~mask]
    return out


def _apply_bias_field(*, data, log_bias, mask, zero_background):
    """Turn a log-domain field into a multiplicative field and apply it.

    Parameters
    ----------
    data : ndarray
        4D DWI data (X, Y, Z, N).
    log_bias : ndarray
        3D log-domain bias field, centered within the mask.
    mask : ndarray
        3D boolean brain mask used for the regression.
    zero_background : bool
        If True, the field is 1.0 outside the mask. If False, the in-mask
        field is extrapolated to the background.

    Returns
    -------
    corrected : ndarray
        Bias-corrected DWI, same dtype as ``data``. Integer dtypes are clipped
        to their representable range instead of wrapping.
    bias_field : ndarray
        3D multiplicative bias field.
    """
    if zero_background:
        log_bias = log_bias.copy()
        log_bias[~mask] = 0.0
    else:
        log_bias = _extrapolate_outside_mask(log_bias=log_bias, mask=mask)
    bias_field = np.exp(log_bias)
    corrected = data.astype(np.float64) / bias_field[..., None]
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        corrected = np.clip(corrected, info.min, info.max)
    return corrected.astype(data.dtype), bias_field


def _sharpen_log_intensities(*, values, n_bins=200, fwhm=0.15, wiener_noise=0.01):
    """Map log intensities to their expected tissue value (N4 sharpening).

    The histogram of ``values`` is deconvolved with a Gaussian via a Wiener
    filter, giving an estimate of the underlying tissue-class distribution.
    Each observed value is then replaced by the expectation of the true value
    given the observation, which pulls voxels toward their class mean and
    leaves the bias in the residual.

    Parameters
    ----------
    values : ndarray
        1D log-domain intensities inside the mask.
    n_bins : int, optional
        Histogram resolution.
    fwhm : float, optional
        Full width at half maximum of the deconvolution kernel, in log units.
    wiener_noise : float, optional
        Wiener filter noise term.

    Returns
    -------
    expected : ndarray
        Expected log intensity for each entry of ``values``.
    """
    lo, hi = values.min(), values.max()
    if hi - lo < 1e-12:
        return values.copy()
    width = (hi - lo) / (n_bins - 1)
    pos = (values - lo) / width
    idx = np.clip(np.floor(pos).astype(np.int64), 0, n_bins - 2)
    frac = pos - idx
    hist = np.bincount(idx, weights=1.0 - frac, minlength=n_bins) + np.bincount(
        idx + 1, weights=frac, minlength=n_bins
    )

    pad = 1 << (int(np.ceil(np.log2(n_bins))) + 1)
    v = np.zeros(pad)
    v[:n_bins] = hist
    scaled_fwhm = fwhm / width
    exp_factor = 4.0 * np.log(2.0) / scaled_fwhm**2
    scale = 2.0 * np.sqrt(np.log(2.0) / np.pi) / scaled_fwhm
    n = np.arange(pad)
    n = np.minimum(n, pad - n).astype(np.float64)
    kernel_f = np.fft.fft(scale * np.exp(-exp_factor * n**2))

    wiener_f = np.conj(kernel_f) / (np.abs(kernel_f) ** 2 + wiener_noise)
    sharpened = np.clip(np.real(np.fft.ifft(np.fft.fft(v) * wiener_f)), 0.0, None)

    centers = lo + np.arange(pad) * width
    num = np.real(np.fft.ifft(np.fft.fft(sharpened * centers) * kernel_f))
    den = np.real(np.fft.ifft(np.fft.fft(sharpened) * kernel_f))
    expected = np.divide(num, den, out=np.zeros_like(num), where=den != 0)[:n_bins]
    return (1.0 - frac) * expected[idx] + frac * expected[idx + 1]


def _shrink_volume(*, volume, mask, factor):
    """Decimate a volume and its mask by an integer factor.

    Plain strided decimation is used on purpose: smoothing before
    downsampling blurs tissue boundaries and fills the intensity histogram
    with partial-volume values, which defeats histogram sharpening.

    Parameters
    ----------
    volume : ndarray
        3D float volume.
    mask : ndarray
        3D boolean mask.
    factor : int
        Decimation factor. 1 returns the inputs unchanged.

    Returns
    -------
    small_volume : ndarray
        Decimated volume.
    small_mask : ndarray
        Decimated boolean mask.
    """
    if factor <= 1:
        return volume, mask
    strides = (slice(None, None, factor),) * 3
    return volume[strides], mask[strides]


def _sharpened_fit(
    *,
    log_b0,
    mask,
    smoother,
    max_iter,
    convergence_threshold,
    shrink_factor,
):
    """Iterate sharpening and smoothing until the field stops changing.

    This is the outer loop of N4 :footcite:p:`Tustison2010`: at each
    iteration the current corrected image is sharpened, the residual between
    image and sharpened image is smoothed by ``smoother`` and added to the
    running field estimate. Iterations run on a downsampled grid and the
    final field is interpolated back to full resolution.

    Parameters
    ----------
    log_b0 : ndarray
        3D log-domain b0 image.
    mask : ndarray
        3D boolean brain mask.
    smoother : callable
        ``smoother(image, mask)`` returning a smooth log-domain field of the
        same shape as ``image``, centered inside ``mask``.
    max_iter : int
        Maximum number of sharpening iterations.
    convergence_threshold : float
        Stop when the coefficient of variation of the multiplicative update
        inside the mask falls below this value.
    shrink_factor : int
        Downsampling factor applied before iterating.

    Returns
    -------
    log_bias : ndarray
        Estimated log-domain bias field at full resolution.
    """
    small_log_b0, small_mask = _shrink_volume(
        volume=log_b0, mask=mask, factor=shrink_factor
    )
    # Histogram sharpening needs a populated histogram
    if small_mask.sum() < 1000:
        small_log_b0, small_mask = log_b0, mask

    log_field = np.zeros_like(small_log_b0)
    current = small_log_b0.copy()
    residual = np.zeros_like(small_log_b0)
    for _ in range(max_iter):
        residual[small_mask] = current[small_mask] - _sharpen_log_intensities(
            values=current[small_mask]
        )
        update = smoother(residual, small_mask)
        log_field += update
        current = small_log_b0 - log_field
        ratio = np.exp(update[small_mask])
        if ratio.std() / ratio.mean() < convergence_threshold:
            break

    if log_field.shape != log_b0.shape:
        zoom = np.array(log_b0.shape) / np.array(log_field.shape)
        log_field = ndimage.zoom(log_field, zoom=zoom, order=3)
    log_field -= log_field[mask].mean()
    return log_field


def _bending_penalty(*, n_control):
    """Second-difference (bending energy) penalty on a control lattice.

    Parameters
    ----------
    n_control : tuple of int
        Control grid dimensions (ns, nr, nc).

    Returns
    -------
    penalty : ndarray
        Dense positive semi-definite matrix of shape (K, K), K the number of
        control points, such that ``beta @ penalty @ beta`` sums the squared
        second differences of the lattice along every axis.
    """
    K = int(np.prod(n_control))
    penalty = np.zeros((K, K), dtype=np.float64)
    eyes = [sparse.identity(n, format="csr") for n in n_control]
    for axis, n in enumerate(n_control):
        if n < 3:
            continue
        diff = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n))
        factors = [diff if i == axis else eyes[i] for i in range(3)]
        full = sparse.kron(sparse.kron(factors[0], factors[1]), factors[2])
        penalty += (full.T @ full).toarray()
    return penalty


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
    edge_weights=None,
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
    edge_weights : ndarray, optional
        Precomputed edge suppression weights, same shape as log_b0. Used
        instead of weights derived from log_b0 when gradient_weighting is
        True.

    Returns
    -------
    log_bias : ndarray
        Estimated log-domain bias field, same shape as log_b0.
    """
    full_shape = log_b0.shape

    if gradient_weighting:
        grad_w_full = (
            _gradient_weights(log_b0=log_b0) if edge_weights is None else edge_weights
        )

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
        mask, leaving background voxels untouched. If False, the field
        estimated inside the mask is extrapolated to the background (nearest
        in-mask value, smoothed) so the whole volume is corrected with a
        continuous field. The mask only restricts the regression; no voxel
        is ever zeroed in the corrected data.

    Returns
    -------
    corrected : ndarray
        Bias-corrected 4D DWI data, same dtype as input.
    bias_field : ndarray
        Estimated 3D multiplicative bias field.
    """
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

    return _apply_bias_field(
        data=data, log_bias=log_bias, mask=mask, zero_background=zero_background
    )


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


def _sparse_weighted_ridge_solve(
    *, X_sparse, y, weights, lambda_reg, penalty=None, smoothness=0.0
):
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
    penalty : ndarray, optional
        Quadratic penalty matrix of shape (K, K), typically from
        :func:`_bending_penalty`.
    smoothness : float, optional
        Weight of ``penalty`` relative to the data term. The penalty is
        scaled so that ``smoothness=1`` gives it the same trace as the
        Gram matrix, which makes the value independent of the number of
        voxels.

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

    if penalty is not None and smoothness > 0:
        trace_penalty = np.trace(penalty)
        if trace_penalty > 0:
            A += smoothness * (np.trace(A) / trace_penalty) * penalty
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
    smoothness=0.0,
    edge_weights=None,
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
    smoothness : float, optional
        Bending energy penalty weight on the control lattice, relative to
        the data term. 0 disables the penalty.
    edge_weights : ndarray, optional
        Precomputed edge suppression weights, same shape as log_b0. Used
        instead of weights derived from log_b0 when gradient_weighting is
        True.

    Returns
    -------
    log_bias : ndarray
        Estimated log-domain bias field, same shape as log_b0.
    """
    full_shape = log_b0.shape

    if gradient_weighting:
        grad_w_full = (
            _gradient_weights(log_b0=log_b0) if edge_weights is None else edge_weights
        )

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
        penalty = _bending_penalty(n_control=n_ctrl) if smoothness > 0 else None

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
                X_sparse=X,
                y=y,
                weights=w,
                lambda_reg=lambda_reg,
                penalty=penalty,
                smoothness=smoothness,
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


def _estimate_log_bias(
    *,
    log_b0,
    mask,
    method,
    order,
    n_control_points,
    pyramid_levels,
    n_iter,
    lambda_reg,
    robust,
    gradient_weighting,
    smoothness,
    sharpen,
    max_iter,
    convergence_threshold,
    shrink_factor,
):
    """Estimate the log-domain bias field with one regression method.

    Parameters
    ----------
    log_b0 : ndarray
        3D log-domain mean b0.
    mask : ndarray
        3D boolean brain mask.
    method : str
        ``"poly"`` or ``"bspline"``.
    order : int
        Legendre polynomial order (poly).
    n_control_points : tuple of int
        Control grid dimensions at the finest level (bspline).
    pyramid_levels : tuple of int
        Downsampling factors for the coarse-to-fine pyramid.
    n_iter : int
        Reweighting iterations per pyramid level.
    lambda_reg : float
        Ridge regularization strength.
    robust : bool
        Apply Tukey biweight robust reweighting.
    gradient_weighting : bool
        Apply gradient-based edge suppression.
    smoothness : float
        Bending energy penalty weight (bspline).
    sharpen : bool
        Wrap the regression in the N4 histogram-sharpening loop.
    max_iter : int
        Maximum sharpening iterations.
    convergence_threshold : float
        Sharpening convergence threshold.
    shrink_factor : int
        Downsampling factor for the sharpening iterations.

    Returns
    -------
    log_bias : ndarray
        Log-domain bias field, same shape as log_b0.
    """
    common = {
        "pyramid_levels": pyramid_levels,
        "n_iter": n_iter,
        "lambda_reg": lambda_reg,
        "robust": robust,
        "gradient_weighting": gradient_weighting,
    }

    def smoother(image, image_mask, *, edge_weights=None):
        if method == "poly":
            return _polynomial_pyramid_fit(
                log_b0=image,
                mask=image_mask,
                order=order,
                edge_weights=edge_weights,
                **common,
            )
        return _bspline_pyramid_fit(
            log_b0=image,
            mask=image_mask,
            n_control_points=n_control_points,
            smoothness=smoothness,
            edge_weights=edge_weights,
            **common,
        )

    if not sharpen:
        return smoother(log_b0, mask)

    small_log_b0, _ = _shrink_volume(volume=log_b0, mask=mask, factor=shrink_factor)
    edge_weights = (
        _gradient_weights(log_b0=small_log_b0) if gradient_weighting else None
    )

    def sharpened_smoother(image, image_mask):
        # _sharpened_fit falls back to full resolution on tiny masks
        weights = edge_weights
        if weights is not None and weights.shape != image.shape:
            weights = None
        return smoother(image, image_mask, edge_weights=weights)

    return _sharpened_fit(
        log_b0=log_b0,
        mask=mask,
        smoother=sharpened_smoother,
        max_iter=max_iter,
        convergence_threshold=convergence_threshold,
        shrink_factor=shrink_factor,
    )


def _auto_select_fit(*, mean_b0, mask, **fit_kwargs):
    """Run poly and bspline fits, return the log-bias with lower CoV.

    Parameters
    ----------
    mean_b0 : ndarray
        Mean b0 in signal domain, shape (X, Y, Z), float64.
    mask : ndarray
        3D boolean brain mask.
    fit_kwargs : dict
        Keyword arguments forwarded to :func:`_estimate_log_bias`, except
        ``method``.

    Returns
    -------
    log_bias : ndarray
        Log-domain bias field from the winning method.
    """
    log_bias_poly = _estimate_log_bias(mask=mask, method="poly", **fit_kwargs)
    log_bias_bspline = _estimate_log_bias(mask=mask, method="bspline", **fit_kwargs)

    def _cov(log_bf):
        """CoV of mean b0 corrected by the given log bias field."""
        corrected_b0 = mean_b0 / np.where(np.exp(log_bf) > 1e-10, np.exp(log_bf), 1.0)
        vals = corrected_b0[mask]
        return vals.std() / (vals.mean() + 1e-12)

    cov_poly = _cov(log_bias_poly)
    cov_bspline = _cov(log_bias_bspline)

    if cov_poly <= cov_bspline:
        logger.info(
            "bias_field_correction auto: selected 'poly' (CoV %.4f vs bspline %.4f)",
            cov_poly,
            cov_bspline,
        )
        return log_bias_poly

    logger.info(
        "bias_field_correction auto: selected 'bspline' (CoV %.4f vs poly %.4f)",
        cov_bspline,
        cov_poly,
    )
    return log_bias_bspline


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
    sharpen=True,
    max_iter=50,
    convergence_threshold=1e-3,
    shrink_factor=2,
    zero_background=False,
):
    """DWI bias field correction via multi-resolution Legendre polynomial regression.

    Estimates the bias field from the mean b0 volume in log space using
    coarse-to-fine Legendre polynomial regression, then applies the estimated
    field to all DWI volumes. See :func:`bias_field_correction` for the
    meaning of the parameters.

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
    sharpen : bool, optional
        Iterate the regression inside the N4 histogram-sharpening loop.
    max_iter : int, optional
        Maximum number of sharpening iterations.
    convergence_threshold : float, optional
        Sharpening stops when the coefficient of variation of the field
        update inside the mask falls below this value.
    shrink_factor : int, optional
        Downsampling factor used during the sharpening iterations.
    zero_background : bool, optional
        If True, set the bias field to 1.0 (no correction) outside the brain
        mask, leaving background voxels untouched. If False, the field
        estimated inside the mask is extrapolated to the background (nearest
        in-mask value, smoothed) so the whole volume is corrected with a
        continuous field. The mask only restricts the regression; no voxel
        is ever zeroed in the corrected data.

    Returns
    -------
    corrected : ndarray
        Bias-corrected 4D DWI data, same dtype as input.
    bias_field : ndarray
        Estimated 3D multiplicative bias field.

    References
    ----------
    .. footbibliography::
    """
    mean_b0 = _get_mean_b0(data, gtab)
    mask = _get_mask(mean_b0, mask)
    log_b0 = np.log(np.clip(mean_b0, 1e-10, None))

    log_bias = _estimate_log_bias(
        log_b0=log_b0,
        mask=mask,
        method="poly",
        order=order,
        n_control_points=None,
        pyramid_levels=pyramid_levels,
        n_iter=n_iter,
        lambda_reg=lambda_reg,
        robust=robust,
        gradient_weighting=gradient_weighting,
        smoothness=0.0,
        sharpen=sharpen,
        max_iter=max_iter,
        convergence_threshold=convergence_threshold,
        shrink_factor=shrink_factor,
    )
    return _apply_bias_field(
        data=data, log_bias=log_bias, mask=mask, zero_background=zero_background
    )


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
    smoothness=10.0,
    sharpen=True,
    max_iter=50,
    convergence_threshold=1e-3,
    shrink_factor=2,
    return_bias_field=False,
    zero_background=False,
):
    """Top-level DWI bias field correction via regression.

    Estimates a smooth multiplicative bias field from the mean b0 volume
    using polynomial or B-spline regression in log space, then applies the
    correction uniformly to all DWI volumes.

    A direct regression of the log b0 cannot tell tissue contrast from the
    bias field: white matter is darker than cortex on a b0 image, so the
    fit tilts the field toward the periphery. With ``sharpen=True`` the
    regression is wrapped in the iterative histogram-sharpening scheme of N4
    :footcite:p:`Tustison2010`. At each iteration the log intensities are
    pulled toward their tissue-class mean and only the residual is smoothed,
    so the field converges to the slowly varying component alone.

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
    smoothness : float, optional
        Bending energy penalty on the B-spline control lattice, relative to
        the data term (used only for method="bspline"). 0 disables it.
    sharpen : bool, optional
        Iterate the regression inside the N4 histogram-sharpening loop.
        If False, a single direct regression of the log b0 is used.
    max_iter : int, optional
        Maximum number of sharpening iterations.
    convergence_threshold : float, optional
        Sharpening stops when the coefficient of variation of the field
        update inside the mask falls below this value.
    shrink_factor : int, optional
        Decimation factor applied to the b0 during the sharpening
        iterations. The final field is interpolated back to full
        resolution. Full resolution is used when the decimated mask would
        hold fewer than 1000 voxels.
    return_bias_field : bool, optional
        If True, return the bias field alongside the corrected data.
    zero_background : bool, optional
        If True, set the bias field to 1.0 (no correction) outside the brain
        mask, leaving background voxels untouched. If False, the field
        estimated inside the mask is extrapolated to the background (nearest
        in-mask value, smoothed) so the whole volume is corrected with a
        continuous field. The mask only restricts the regression; no voxel
        is ever zeroed in the corrected data.

    Returns
    -------
    corrected : ndarray
        Bias-corrected DWI, same dtype as input.
    bias_field : ndarray
        3D multiplicative bias field (only returned if
        return_bias_field=True).

    References
    ----------
    .. footbibliography::
    """
    if method not in ("poly", "bspline", "auto"):
        raise ValueError(f"method must be 'poly', 'bspline', or 'auto', got '{method}'")

    mean_b0 = _get_mean_b0(data, gtab)
    mask = _get_mask(mean_b0, mask)
    log_b0 = np.log(np.clip(mean_b0.astype(np.float64), 1e-10, None))

    fit_kwargs = {
        "log_b0": log_b0,
        "order": order,
        "n_control_points": n_control_points,
        "pyramid_levels": pyramid_levels,
        "n_iter": n_iter,
        "lambda_reg": lambda_reg,
        "robust": robust,
        "gradient_weighting": gradient_weighting,
        "smoothness": smoothness,
        "sharpen": sharpen,
        "max_iter": max_iter,
        "convergence_threshold": convergence_threshold,
        "shrink_factor": shrink_factor,
    }
    if method == "auto":
        log_bias = _auto_select_fit(mean_b0=mean_b0, mask=mask, **fit_kwargs)
    else:
        log_bias = _estimate_log_bias(mask=mask, method=method, **fit_kwargs)

    corrected, bias_field = _apply_bias_field(
        data=data, log_bias=log_bias, mask=mask, zero_background=zero_background
    )

    if return_bias_field:
        return corrected, bias_field
    return corrected
