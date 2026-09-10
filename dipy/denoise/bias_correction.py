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


def _sharpened_fit(*, log_b0, mask, smoother, max_iter, convergence_threshold):
    """Iterate sharpening and smoothing until the field stops changing.

    This is the outer loop of N4 :footcite:p:`Tustison2010`: at each
    iteration the current corrected image is sharpened, the residual between
    image and sharpened image is smoothed by ``smoother`` and added to the
    running field estimate.

    Parameters
    ----------
    log_b0 : ndarray
        3D log-domain b0 image.
    mask : ndarray
        3D boolean brain mask.
    smoother : callable
        ``smoother(image)`` returning a smooth log-domain field of the same
        shape as ``image``, centered inside ``mask``.
    max_iter : int
        Maximum number of sharpening iterations.
    convergence_threshold : float
        Stop when the coefficient of variation of the multiplicative update
        inside the mask falls below this value.

    Returns
    -------
    log_bias : ndarray
        Estimated log-domain bias field, same shape as log_b0.
    """
    log_field = np.zeros_like(log_b0)
    current = log_b0.copy()
    residual = np.zeros_like(log_b0)
    for _ in range(max_iter):
        residual[mask] = current[mask] - _sharpen_log_intensities(values=current[mask])
        update = smoother(residual)
        log_field += update
        current = log_b0 - log_field
        ratio = np.exp(update[mask])
        if ratio.std() / ratio.mean() < convergence_threshold:
            break
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


def _bspline_axis_basis(*, n_vox, n_ctrl):
    """Dense 1-D cubic B-spline basis matrix for one axis.

    Uses the same parameterisation and clamping as
    :func:`_build_bspline_design_matrix`, so evaluating a field with the
    tensor product of these matrices agrees exactly with the design matrix
    used for fitting.

    Parameters
    ----------
    n_vox : int
        Number of voxels along the axis.
    n_ctrl : int
        Number of control points along the axis.

    Returns
    -------
    basis : ndarray
        Matrix of shape (n_vox, n_ctrl).
    """
    basis = np.zeros((n_vox, n_ctrl), dtype=np.float64)
    if n_vox <= 1 or n_ctrl <= 1:
        basis[:, 0] = 1.0
        return basis
    t = np.arange(n_vox, dtype=np.float64) * (n_ctrl - 1) / (n_vox - 1)
    t = np.clip(t, 0.0, n_ctrl - 1 - 1e-10)
    k = np.minimum(np.floor(t).astype(np.int64), n_ctrl - 2)
    u = t - k
    u2 = u * u
    u3 = u2 * u
    values = np.stack(
        [
            (1.0 - u) ** 3 / 6.0,
            (3.0 * u3 - 6.0 * u2 + 4.0) / 6.0,
            (-3.0 * u3 + 3.0 * u2 + 3.0 * u + 1.0) / 6.0,
            u3 / 6.0,
        ],
        axis=-1,
    )
    rows = np.repeat(np.arange(n_vox), 4)
    cols = (k[:, None] + np.arange(-1, 3)[None, :]).ravel()
    valid = (cols >= 0) & (cols < n_ctrl)
    basis[rows[valid], cols[valid]] = values.ravel()[valid]
    return basis


def _eval_bspline_field(*, coeffs, n_control, out_shape):
    """Evaluate a B-spline field on the full voxel grid.

    The tensor-product structure makes this three small matrix products
    instead of a per-voxel interpolation.

    Parameters
    ----------
    coeffs : ndarray
        Flattened control point coefficients.
    n_control : tuple of int
        Control grid dimensions (ns, nr, nc).
    out_shape : tuple of int
        Output volume shape (S, R, C).

    Returns
    -------
    field : ndarray
        Evaluated field, shape out_shape.
    """
    grid = coeffs.reshape(n_control)
    bases = [
        _bspline_axis_basis(n_vox=n, n_ctrl=k) for n, k in zip(out_shape, n_control)
    ]
    field = np.tensordot(bases[0], grid, axes=(1, 0))
    field = np.tensordot(bases[1], field, axes=(1, 1)).transpose(1, 0, 2)
    return np.tensordot(field, bases[2], axes=(2, 1))


def _gram_matrix(*, X, weights):
    """Compute the weighted Gram matrix X^T W X.

    Parameters
    ----------
    X : ndarray or scipy.sparse.csr_matrix
        Design matrix, shape (N, K).
    weights : ndarray
        Non-negative regression weights, shape (N,).

    Returns
    -------
    A : ndarray
        Dense matrix of shape (K, K).
    """
    K = X.shape[1]
    if not sparse.issparse(X):
        return X.T @ (weights[:, None] * X)
    A = np.zeros((K, K), dtype=np.float64)
    if _HAVE_CYTHON:
        gram_matrix_csr(
            np.asarray(X.data, dtype=np.float64),
            np.asarray(X.indices, dtype=np.int32),
            np.asarray(X.indptr, dtype=np.int32),
            np.ascontiguousarray(weights, dtype=np.float64),
            np.zeros(X.shape[0], dtype=np.float64),
            A,
            np.zeros(K, dtype=np.float64),
        )
        return A
    # Chunked dense products: sparse x sparse is slow when the result is dense
    chunk = min(4096, X.shape[0])
    for i in range(0, X.shape[0], chunk):
        Xc = X[i : i + chunk].toarray()
        A += Xc.T @ (weights[i : i + chunk, None] * Xc)
    return A


def _regularize(*, A, lambda_reg, penalty, smoothness):
    """Add ridge and bending penalties to a Gram matrix.

    Parameters
    ----------
    A : ndarray
        Gram matrix, shape (K, K).
    lambda_reg : float
        Ridge regularization strength.
    penalty : ndarray or None
        Quadratic penalty matrix of shape (K, K).
    smoothness : float
        Weight of ``penalty`` relative to the data term. The penalty is
        scaled so that ``smoothness=1`` gives it the same trace as ``A``,
        which makes the value independent of the number of voxels.

    Returns
    -------
    A_reg : ndarray
        Regularized system matrix.
    """
    A_reg = A.copy()
    if penalty is not None and smoothness > 0:
        trace_penalty = np.trace(penalty)
        if trace_penalty > 0:
            A_reg += smoothness * (np.trace(A) / trace_penalty) * penalty
    A_reg += lambda_reg * np.eye(A.shape[0])
    return A_reg


def _solve_normal_equations(*, A, b):
    """Solve A beta = b for a symmetric positive definite A.

    Parameters
    ----------
    A : ndarray
        System matrix, shape (K, K).
    b : ndarray
        Right-hand side, shape (K,).

    Returns
    -------
    beta : ndarray
        Solution, shape (K,).
    """
    try:
        return scipy_linalg.solve(A, b, assume_a="pos")
    except (scipy_linalg.LinAlgError, ValueError):
        return np.linalg.lstsq(A, b, rcond=None)[0]


def _downsample(*, volume, factor, sigma_factor=0.2):
    """Gaussian-smooth and downsample a volume for one pyramid level.

    Parameters
    ----------
    volume : ndarray
        3D float volume.
    factor : int
        Downsampling factor. 1 returns the input unchanged.
    sigma_factor : float, optional
        Sigma = factor * sigma_factor for the Gaussian smoothing.

    Returns
    -------
    small : ndarray
        Downsampled volume.
    """
    if factor == 1:
        return volume
    smoothed = ndimage.gaussian_filter(volume, sigma=factor * sigma_factor)
    return ndimage.zoom(smoothed, zoom=1.0 / factor, order=1)


def _plan_pyramid(
    *,
    shape,
    mask,
    method,
    pyramid_levels,
    order,
    n_control_points,
    lambda_reg,
    smoothness,
    edge_weights,
):
    """Precompute everything about the regression that does not depend on the image.

    Design matrices, Gram matrices, penalties and evaluation operators are
    functions of the mask and volume shape only. Building them once lets the
    sharpening loop solve dozens of systems at the cost of a right-hand side
    each.

    Parameters
    ----------
    shape : tuple of int
        Full volume shape (S, R, C).
    mask : ndarray
        3D boolean brain mask.
    method : str
        ``"poly"`` or ``"bspline"``.
    pyramid_levels : tuple of int
        Downsampling factors, coarse first.
    order : int
        Legendre polynomial order (poly).
    n_control_points : tuple of int
        Control grid dimensions at the finest level (bspline).
    lambda_reg : float
        Ridge regularization strength.
    smoothness : float
        Bending energy penalty weight (bspline).
    edge_weights : ndarray or None
        Edge suppression weights at full resolution.

    Returns
    -------
    levels : list of dict
        One entry per usable pyramid level with keys ``factor``,
        ``mask_flat``, ``weights``, ``X``, ``gram``, ``regularize`` and
        ``evaluate``.
    """
    if method == "poly":
        ii, jj, kk = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij")
        coords = np.column_stack([ii.ravel(), jj.ravel(), kk.ravel()])
        X_full = _legendre_basis(
            coords_flat=_normalize_coords(shape=shape, coords=coords), order=order
        )

    levels = []
    for factor in pyramid_levels:
        if factor == 1:
            level_mask = mask
        else:
            level_mask = (
                ndimage.zoom(mask.astype(np.float64), zoom=1.0 / factor, order=0) > 0.5
            )
        level_shape = level_mask.shape
        mask_flat = level_mask.ravel()
        n_masked = int(mask_flat.sum())

        if method == "poly":
            n_params = X_full.shape[1]
            penalty = None
        else:
            n_ctrl = tuple(max(2, int(np.round(n / factor))) for n in n_control_points)
            n_params = int(np.prod(n_ctrl))
            penalty = _bending_penalty(n_control=n_ctrl) if smoothness > 0 else None
        if n_masked < n_params:
            continue

        if method == "poly":
            ii, jj, kk = np.meshgrid(
                *(np.arange(n) for n in level_shape), indexing="ij"
            )
            coords = np.column_stack(
                [ii.ravel()[mask_flat], jj.ravel()[mask_flat], kk.ravel()[mask_flat]]
            )
            X = _legendre_basis(
                coords_flat=_normalize_coords(shape=level_shape, coords=coords),
                order=order,
            )

            def evaluate(beta, *, X_full=X_full):
                return (X_full @ beta).reshape(shape)

        else:
            X = _build_bspline_design_matrix(
                log_b0_shape=level_shape, n_control=n_ctrl, mask_flat=mask_flat
            )

            def evaluate(beta, *, n_ctrl=n_ctrl):
                return _eval_bspline_field(
                    coeffs=beta, n_control=n_ctrl, out_shape=shape
                )

        weights = np.ones(n_masked, dtype=np.float64)
        if edge_weights is not None:
            level_edge = (
                edge_weights
                if factor == 1
                else ndimage.zoom(edge_weights, zoom=1.0 / factor, order=1)
            )
            weights = weights * level_edge.ravel()[mask_flat]

        def regularize(A, *, penalty=penalty):
            return _regularize(
                A=A, lambda_reg=lambda_reg, penalty=penalty, smoothness=smoothness
            )

        levels.append(
            {
                "factor": factor,
                "mask_flat": mask_flat,
                "weights": weights,
                "X": X,
                "gram": regularize(_gram_matrix(X=X, weights=weights)),
                "regularize": regularize,
                "evaluate": evaluate,
            }
        )
    return levels


def _pyramid_fit(*, image, mask, levels, n_iter, robust):
    """Coarse-to-fine regression of a smooth field to ``image``.

    Parameters
    ----------
    image : ndarray
        3D log-domain image to smooth.
    mask : ndarray
        3D boolean brain mask.
    levels : list of dict
        Output of :func:`_plan_pyramid` for the same shape and mask.
    n_iter : int
        Reweighting iterations per level. Only the last solve is kept.
    robust : bool
        Multiply the weights by Tukey biweights of the residuals between
        iterations.

    Returns
    -------
    log_bias : ndarray
        Smooth field, same shape as image, with zero mean inside the mask.
    """
    # Remove the DC level so the basis only explains spatial variation
    residual = image - image[mask].mean()
    log_bias = np.zeros(image.shape, dtype=np.float64)

    for level in levels:
        X = level["X"]
        y = _downsample(volume=residual, factor=level["factor"]).ravel()[
            level["mask_flat"]
        ]
        weights = level["weights"]
        A = level["gram"]
        beta = None
        for it in range(max(n_iter, 1)):
            beta = _solve_normal_equations(A=A, b=X.T @ (weights * y))
            if robust and it < n_iter - 1:
                weights = weights * _tukey_weights(residuals=y - X @ beta)
                A = level["regularize"](_gram_matrix(X=X, weights=weights))
        field = level["evaluate"](beta)
        log_bias += field
        residual = residual - field

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
        Reweighting iterations per pyramid level (direct fit only).
    lambda_reg : float
        Ridge regularization strength.
    robust : bool
        Apply Tukey biweight robust reweighting (direct fit only).
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
        Decimation factor for the sharpening iterations.

    Returns
    -------
    log_bias : ndarray
        Log-domain bias field, same shape as log_b0.
    """
    if sharpen:
        image, image_mask = _shrink_volume(
            volume=log_b0, mask=mask, factor=shrink_factor
        )
        # Histogram sharpening needs a populated histogram
        if image_mask.sum() < 1000:
            image, image_mask = log_b0, mask
    else:
        image, image_mask = log_b0, mask

    levels = _plan_pyramid(
        shape=image.shape,
        mask=image_mask,
        method=method,
        pyramid_levels=pyramid_levels,
        order=order,
        n_control_points=n_control_points,
        lambda_reg=lambda_reg,
        smoothness=smoothness,
        edge_weights=_gradient_weights(log_b0=image) if gradient_weighting else None,
    )

    if not sharpen:
        return _pyramid_fit(
            image=image, mask=image_mask, levels=levels, n_iter=n_iter, robust=robust
        )

    def smoother(residual):
        return _pyramid_fit(
            image=residual, mask=image_mask, levels=levels, n_iter=1, robust=False
        )

    log_bias = _sharpened_fit(
        log_b0=image,
        mask=image_mask,
        smoother=smoother,
        max_iter=max_iter,
        convergence_threshold=convergence_threshold,
    )
    if log_bias.shape != log_b0.shape:
        zoom = np.array(log_b0.shape) / np.array(log_bias.shape)
        log_bias = ndimage.zoom(log_bias, zoom=zoom, order=3)
        log_bias -= log_bias[mask].mean()
    return log_bias


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
        Reweighting iterations per pyramid level. Only used when
        ``sharpen=False``; the sharpening loop solves each level once per
        iteration.
    lambda_reg : float, optional
        Ridge regularization strength.
    robust : bool, optional
        Apply Tukey biweight robust reweighting. Only used when
        ``sharpen=False``; inside the sharpening loop the histogram model
        already accounts for tissue outliers and reweighting the small
        residuals degrades the fit.
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
        Reweighting iterations per pyramid level. Only used when
        ``sharpen=False``; the sharpening loop solves each level once per
        iteration.
    lambda_reg : float, optional
        Ridge regularization strength.
    robust : bool, optional
        Apply Tukey biweight robust reweighting at each level. Only used
        when ``sharpen=False``; inside the sharpening loop the histogram
        model already accounts for tissue outliers and reweighting the
        small residuals degrades the fit.
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
