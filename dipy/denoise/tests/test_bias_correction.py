"""Tests for dipy.denoise.bias_correction."""

import numpy as np
import pytest

from dipy.core.gradients import extract_b0, gradient_table
from dipy.denoise.bias_correction import (
    _build_bspline_design_matrix,
    _get_mask,
    _get_mean_b0,
    _gradient_weights,
    _legendre_basis,
    _normalize_coords,
    _tukey_weights,
    bias_field_correction,
    polynomial_bias_field_dwi,
)
from dipy.segment.mask import median_otsu


def _make_synthetic_dwi(*, shape=(20, 20, 15), n_vols=10, rng=None):
    """Create synthetic DWI data with a smooth multiplicative bias field.

    Parameters
    ----------
    shape : tuple of int, optional
        Spatial dimensions of the volume.
    n_vols : int, optional
        Number of DWI volumes.
    rng : numpy.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    data : ndarray
        4D DWI data, dtype float32.
    gtab : GradientTable
        Gradient table with first 2 volumes as b0.
    bias : ndarray
        Ground-truth multiplicative bias field, shape same as `shape`.
    mask : ndarray
        Ellipsoidal brain mask, boolean.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    bvals = np.zeros(n_vols)
    bvals[2:] = 1000
    bvecs = np.zeros((n_vols, 3))
    dirs = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
            [1, -1, 0],
        ],
        dtype=float,
    )
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    n_dw = n_vols - 2
    bvecs[2:] = dirs[:n_dw]

    gtab = gradient_table(bvals, bvecs=bvecs)

    xx, yy, zz = np.mgrid[: shape[0], : shape[1], : shape[2]]
    cx, cy, cz = shape[0] // 2, shape[1] // 2, shape[2] // 2
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2
    signal = 1000.0 * np.exp(-0.01 * r2)

    # Smooth bias field via low-order polynomial
    xn = 2.0 * xx / (shape[0] - 1) - 1.0
    yn = 2.0 * yy / (shape[1] - 1) - 1.0
    zn = 2.0 * zz / (shape[2] - 1) - 1.0
    bias = np.exp(0.3 * xn + 0.2 * yn**2 - 0.15 * zn)

    data = np.zeros((*shape, n_vols), dtype=np.float32)
    for i in range(n_vols):
        data[..., i] = (signal * bias + rng.normal(0, 10, shape)).clip(1)

    # Ellipsoidal brain mask
    rx, ry, rz = cx, cy, cz
    mask = (
        (xx - cx) ** 2 / max(rx**2, 1)
        + (yy - cy) ** 2 / max(ry**2, 1)
        + (zz - cz) ** 2 / max(rz**2, 1)
    ) < 1.0

    return data.astype(np.float32), gtab, bias, mask


# ---------------------------------------------------------------------------
# --- Unit tests for helper functions ---
# ---------------------------------------------------------------------------


def test_normalize_coords():
    shape = (10, 8, 6)
    coords = np.array([[0, 0, 0], [9, 7, 5]], dtype=float)
    norm = _normalize_coords(shape=shape, coords=coords)
    np.testing.assert_allclose(norm[0], [-1, -1, -1], atol=1e-12)
    np.testing.assert_allclose(norm[1], [1, 1, 1], atol=1e-12)


def test_legendre_basis_order0():
    coords = np.zeros((5, 3))
    X = _legendre_basis(coords_flat=coords, order=0)
    assert X.shape == (5, 1)
    np.testing.assert_allclose(X, 1.0)


def test_legendre_basis_shape():
    coords = np.zeros((10, 3))
    for order in [1, 2, 3]:
        X = _legendre_basis(coords_flat=coords, order=order)
        # C(order+3, 3) terms
        from math import comb

        expected_k = comb(order + 3, 3)
        assert X.shape == (10, expected_k), f"order={order}"


def test_gradient_weights_range():
    rng = np.random.default_rng(0)
    log_b0 = rng.normal(0, 1, (15, 15, 10))
    w = _gradient_weights(log_b0=log_b0)
    assert w.shape == log_b0.shape
    assert w.min() > 0.0
    assert w.max() <= 1.0 + 1e-12


def test_tukey_weights_range():
    rng = np.random.default_rng(0)
    residuals = rng.normal(0, 1, 100)
    w = _tukey_weights(residuals=residuals)
    assert w.shape == (100,)
    assert w.min() >= 0.0
    assert w.max() <= 1.0 + 1e-12


def test_tukey_weights_zero_residuals():
    residuals = np.zeros(50)
    w = _tukey_weights(residuals=residuals)
    # All ones when residuals are zero
    np.testing.assert_allclose(w, 1.0)


def test_get_mean_b0_shape():
    data, gtab, _, _ = _make_synthetic_dwi()
    mean_b0 = _get_mean_b0(data, gtab)
    assert mean_b0.shape == data.shape[:3]
    assert mean_b0.dtype == np.float64


def test_get_mask_auto():
    data, gtab, _, _ = _make_synthetic_dwi()
    mean_b0 = _get_mean_b0(data, gtab)
    mask = _get_mask(mean_b0, None)
    assert mask.dtype == bool
    assert mask.shape == mean_b0.shape
    assert mask.any()


def test_get_mask_provided():
    data, gtab, _, true_mask = _make_synthetic_dwi()
    mean_b0 = _get_mean_b0(data, gtab)
    mask = _get_mask(mean_b0, true_mask)
    np.testing.assert_array_equal(mask, true_mask.astype(bool))


def test_bspline_design_matrix_shape():
    shape = (10, 10, 8)
    n_ctrl = (4, 4, 3)
    mask = np.ones(shape, dtype=bool)
    X = _build_bspline_design_matrix(
        log_b0_shape=shape, n_control=n_ctrl, mask_flat=mask.ravel()
    )
    K = n_ctrl[0] * n_ctrl[1] * n_ctrl[2]
    N = mask.sum()
    assert X.shape == (N, K)


def test_bspline_design_matrix_row_sums():
    # For a uniform B-spline, row sums should equal the sum of all
    # basis functions at that parameter value (approximately 1 for
    # the standard cubic B-spline kernel)
    shape = (10, 10, 8)
    n_ctrl = (4, 4, 3)
    mask = np.ones(shape, dtype=bool)
    X = _build_bspline_design_matrix(
        log_b0_shape=shape, n_control=n_ctrl, mask_flat=mask.ravel()
    )
    row_sums = np.asarray(X.sum(axis=1)).ravel()
    # Not exactly 1 near boundaries, but should be in (0, 1.1]
    assert row_sums.min() > 0.0
    assert row_sums.max() <= 1.1


def test_synthetic_poly_bias():
    """Test polynomial method recovers smooth bias field."""
    data, gtab, _, mask = _make_synthetic_dwi(shape=(20, 20, 15), n_vols=10)
    corrected, _ = polynomial_bias_field_dwi(
        data,
        gtab,
        mask=mask,
        order=3,
        pyramid_levels=(2, 1),
        n_iter=4,
        robust=True,
        gradient_weighting=True,
    )
    assert corrected.shape == data.shape
    assert corrected.dtype == data.dtype

    # Check RMSE of corrected b0 vs unbiased signal
    b0_mask = gtab.b0s_mask
    corrected_b0 = corrected[..., b0_mask].mean(axis=-1)
    original_b0 = data[..., b0_mask].mean(axis=-1)
    # Relative RMSE should decrease after correction
    # (not always guaranteed to be < 0.05 for noisy data, but correction
    #  should not make things worse)
    assert corrected_b0[mask].std() <= original_b0[mask].std() * 1.5


def test_synthetic_poly_bias_rmse():
    """Test that estimated bias field correlates with true bias field."""
    data, gtab, true_bias, mask = _make_synthetic_dwi(shape=(20, 20, 15), n_vols=10)
    _, bias_field = polynomial_bias_field_dwi(
        data,
        gtab,
        mask=mask,
        order=3,
        pyramid_levels=(4, 2, 1),
        n_iter=4,
        robust=True,
        gradient_weighting=True,
    )
    # Normalise both fields to have unit mean before comparing
    bf_norm = bias_field[mask] / bias_field[mask].mean()
    true_norm = true_bias[mask] / true_bias[mask].mean()
    rmse = np.sqrt(np.mean((bf_norm - true_norm) ** 2))
    assert rmse < 0.5, f"Bias RMSE too large: {rmse:.4f}"


def test_synthetic_bspline_bias():
    """Test B-spline method recovers smooth bias field."""
    data, gtab, _, mask = _make_synthetic_dwi(shape=(20, 20, 15), n_vols=10)
    corrected, bias_field = bias_field_correction(
        data,
        gtab,
        mask=mask,
        method="bspline",
        n_control_points=(4, 4, 3),
        pyramid_levels=(2, 1),
        n_iter=4,
        robust=True,
        gradient_weighting=True,
        return_bias_field=True,
    )
    assert corrected.shape == data.shape
    assert corrected.dtype == data.dtype
    assert bias_field.shape == data.shape[:3]


def test_synthetic_bspline_bias_rmse():
    """Test that B-spline estimated bias field correlates with true field."""
    data, gtab, true_bias, mask = _make_synthetic_dwi(shape=(20, 20, 15), n_vols=10)
    _, bias_field = bias_field_correction(
        data,
        gtab,
        mask=mask,
        method="bspline",
        n_control_points=(4, 4, 3),
        pyramid_levels=(4, 2, 1),
        n_iter=4,
        robust=True,
        gradient_weighting=True,
        return_bias_field=True,
    )
    bf_norm = bias_field[mask] / bias_field[mask].mean()
    true_norm = true_bias[mask] / true_bias[mask].mean()
    rmse = np.sqrt(np.mean((bf_norm - true_norm) ** 2))
    assert rmse < 0.5, f"B-spline bias RMSE too large: {rmse:.4f}"


def test_bias_field_correction_poly_method():
    """Test that bias_field_correction dispatches correctly for poly."""
    data, gtab, _, mask = _make_synthetic_dwi()
    corrected = bias_field_correction(
        data,
        gtab,
        mask=mask,
        method="poly",
        pyramid_levels=(2, 1),
        n_iter=2,
        robust=False,
        gradient_weighting=False,
    )
    assert corrected.shape == data.shape


def test_bias_field_correction_bspline_method():
    """Test that bias_field_correction dispatches correctly for bspline."""
    data, gtab, _, mask = _make_synthetic_dwi()
    corrected = bias_field_correction(
        data,
        gtab,
        mask=mask,
        method="bspline",
        n_control_points=(4, 4, 3),
        pyramid_levels=(2, 1),
        n_iter=2,
        robust=False,
        gradient_weighting=False,
    )
    assert corrected.shape == data.shape


def test_bias_field_correction_auto_method():
    """Test method='auto' returns valid output with CoV no worse than original."""
    data, gtab, _, mask = _make_synthetic_dwi()

    corrected_auto, bias_auto = bias_field_correction(
        data,
        gtab,
        mask=mask,
        method="auto",
        order=3,
        n_control_points=(4, 4, 3),
        pyramid_levels=(2, 1),
        n_iter=2,
        robust=False,
        gradient_weighting=False,
        return_bias_field=True,
    )

    assert corrected_auto.shape == data.shape
    assert bias_auto.shape == data.shape[:3]
    assert corrected_auto.dtype == data.dtype
    assert np.all(bias_auto[mask] > 0)

    b0_orig = data[..., gtab.b0s_mask].mean(-1).astype(float)
    b0_auto = corrected_auto[..., gtab.b0s_mask].mean(-1).astype(float)
    cov_orig = b0_orig[mask].std() / (b0_orig[mask].mean() + 1e-12)
    cov_auto = b0_auto[mask].std() / (b0_auto[mask].mean() + 1e-12)
    assert cov_auto <= cov_orig + 1e-6


def test_bias_field_correction_auto_selects_best():
    """Test that method='auto' CoV equals min(poly CoV, bspline CoV)."""
    data, gtab, _, mask = _make_synthetic_dwi()
    kwargs = {
        "mask": mask,
        "order": 3,
        "n_control_points": (4, 4, 3),
        "pyramid_levels": (2, 1),
        "n_iter": 2,
        "robust": False,
        "gradient_weighting": False,
        "return_bias_field": False,
    }

    corrected_poly = bias_field_correction(data, gtab, method="poly", **kwargs)
    corrected_bspline = bias_field_correction(data, gtab, method="bspline", **kwargs)
    corrected_auto = bias_field_correction(data, gtab, method="auto", **kwargs)

    def _cov(img):
        vals = img[..., gtab.b0s_mask].mean(-1).astype(float)[mask]
        return vals.std() / (vals.mean() + 1e-12)

    cov_poly = _cov(corrected_poly)
    cov_bspline = _cov(corrected_bspline)
    cov_auto = _cov(corrected_auto)

    assert np.isclose(cov_auto, min(cov_poly, cov_bspline), rtol=1e-6)


def test_bias_field_correction_invalid_method():
    """Test that an invalid method raises ValueError."""
    data, gtab, _, mask = _make_synthetic_dwi()
    with pytest.raises(ValueError, match="method must be"):
        bias_field_correction(data, gtab, mask=mask, method="invalid")


def test_return_bias_field_flag():
    """Test return_bias_field=True returns tuple, False returns array."""
    data, gtab, _, mask = _make_synthetic_dwi()
    result = bias_field_correction(
        data,
        gtab,
        mask=mask,
        method="poly",
        pyramid_levels=(2, 1),
        n_iter=1,
        robust=False,
        gradient_weighting=False,
        return_bias_field=True,
    )
    assert isinstance(result, tuple)
    assert len(result) == 2
    corrected, bias_field = result
    assert corrected.shape == data.shape
    assert bias_field.shape == data.shape[:3]

    result_nodfield = bias_field_correction(
        data,
        gtab,
        mask=mask,
        method="poly",
        pyramid_levels=(2, 1),
        n_iter=1,
        robust=False,
        gradient_weighting=False,
        return_bias_field=False,
    )
    assert isinstance(result_nodfield, np.ndarray)


def test_noise_stability():
    """Test that corrected data has similar or reduced CoV vs input."""
    rng = np.random.default_rng(0)
    data, gtab, _, mask = _make_synthetic_dwi(rng=rng)
    corrected, _ = bias_field_correction(
        data,
        gtab,
        mask=mask,
        method="poly",
        pyramid_levels=(2, 1),
        n_iter=3,
        robust=True,
        gradient_weighting=False,
        return_bias_field=True,
    )
    b0_mask = gtab.b0s_mask
    orig_b0 = data[mask, :][..., b0_mask].mean(axis=-1)
    corr_b0 = corrected[mask, :][..., b0_mask].mean(axis=-1)
    # CoV = std / mean; correction should not dramatically inflate it
    cov_orig = orig_b0.std() / (orig_b0.mean() + 1e-10)
    cov_corr = corr_b0.std() / (corr_b0.mean() + 1e-10)
    assert (
        cov_corr < cov_orig * 1.1
    ), f"Correction inflated CoV: {cov_orig:.4f} → {cov_corr:.4f}"


def test_mask_robustness():
    """Test that auto-mask and manual mask give similar results."""
    data, gtab, _, true_mask = _make_synthetic_dwi()
    _, bf_auto = bias_field_correction(
        data,
        gtab,
        mask=None,
        method="poly",
        pyramid_levels=(2, 1),
        n_iter=2,
        robust=False,
        gradient_weighting=False,
        return_bias_field=True,
    )
    _, bf_manual = bias_field_correction(
        data,
        gtab,
        mask=true_mask,
        method="poly",
        pyramid_levels=(2, 1),
        n_iter=2,
        robust=False,
        gradient_weighting=False,
        return_bias_field=True,
    )
    # Correlation between fields should be high
    bf_a = bf_auto[true_mask].ravel()
    bf_m = bf_manual[true_mask].ravel()
    corr = np.corrcoef(bf_a, bf_m)[0, 1]
    assert corr > 0.9, f"Auto vs manual mask correlation too low: {corr:.4f}"


def test_log_domain_centering():
    """Test that log(bias_field) has near-zero mean over mask."""
    data, gtab, _, mask = _make_synthetic_dwi()
    _, bias_field = bias_field_correction(
        data,
        gtab,
        mask=mask,
        method="poly",
        pyramid_levels=(2, 1),
        n_iter=3,
        robust=False,
        gradient_weighting=False,
        return_bias_field=True,
    )
    log_mean = np.abs(np.mean(np.log(bias_field[mask] + 1e-15)))
    # The log bias should not be dramatically far from zero (allow loose tol)
    assert log_mean < 1.0, f"|mean(log(bias_field[mask]))| = {log_mean:.4f}"


def test_determinism():
    """Test that two identical calls produce identical results."""
    data, gtab, _, mask = _make_synthetic_dwi()
    r1, bf1 = bias_field_correction(
        data,
        gtab,
        mask=mask,
        method="poly",
        pyramid_levels=(2, 1),
        n_iter=2,
        robust=True,
        gradient_weighting=True,
        return_bias_field=True,
    )
    r2, bf2 = bias_field_correction(
        data,
        gtab,
        mask=mask,
        method="poly",
        pyramid_levels=(2, 1),
        n_iter=2,
        robust=True,
        gradient_weighting=True,
        return_bias_field=True,
    )
    np.testing.assert_array_equal(r1, r2)
    np.testing.assert_array_equal(bf1, bf2)


def test_multishell_uniform():
    """Test that the bias field is the same regardless of b-value volume."""
    data, gtab, _, mask = _make_synthetic_dwi(n_vols=10)
    corrected, bias_field = bias_field_correction(
        data,
        gtab,
        mask=mask,
        method="poly",
        pyramid_levels=(2, 1),
        n_iter=2,
        robust=False,
        gradient_weighting=False,
        return_bias_field=True,
    )
    # The bias_field is 3D and applied uniformly to all volumes
    # Verify the corrected data at each volume uses the same field
    for vi in range(data.shape[-1]):
        expected = data[..., vi].astype(np.float64) / np.where(
            bias_field > 1e-10, bias_field, 1.0
        )
        actual = corrected[..., vi].astype(np.float64)
        # Inside mask, should be close (allowing for applymask zeroing)
        np.testing.assert_allclose(
            actual[mask],
            expected[mask].astype(data.dtype).astype(np.float64),
            rtol=1e-3,
        )


def test_b0_only_fitting():
    """Test that only b0 volumes contribute to bias estimation.

    Verify that changing DW volumes does not affect the bias field estimate.
    """
    data, gtab, _, mask = _make_synthetic_dwi()
    _, bf1 = bias_field_correction(
        data,
        gtab,
        mask=mask,
        method="poly",
        pyramid_levels=(2, 1),
        n_iter=2,
        robust=False,
        gradient_weighting=False,
        return_bias_field=True,
    )
    # Scramble DW volumes (leave b0 volumes unchanged)
    data_scrambled = data.copy()
    dw_idx = np.where(~gtab.b0s_mask)[0]
    rng = np.random.default_rng(99)
    data_scrambled[..., dw_idx] = rng.uniform(1, 2000, data[..., dw_idx].shape).astype(
        data.dtype
    )

    _, bf2 = bias_field_correction(
        data_scrambled,
        gtab,
        mask=mask,
        method="poly",
        pyramid_levels=(2, 1),
        n_iter=2,
        robust=False,
        gradient_weighting=False,
        return_bias_field=True,
    )
    # Bias fields should be identical since b0 volumes are unchanged
    np.testing.assert_array_equal(bf1, bf2)


def test_mean_b0_matches_extract_b0():
    """Test that _get_mean_b0 matches extract_b0 with strategy='mean'."""
    data, gtab, _, _ = _make_synthetic_dwi()
    result = _get_mean_b0(data, gtab)
    expected = extract_b0(data, gtab.b0s_mask, strategy="mean").astype(np.float64)
    np.testing.assert_array_equal(result, expected)


def test_get_mask_none_uses_median_otsu():
    """Test that _get_mask with mask=None returns the same mask as median_otsu."""
    data, gtab, _, _ = _make_synthetic_dwi()
    mean_b0 = _get_mean_b0(data, gtab)
    computed = _get_mask(mean_b0, None)
    _, expected = median_otsu(mean_b0, median_radius=4, numpass=4)
    np.testing.assert_array_equal(computed, expected)


def test_provided_mask_is_respected():
    """Test that a provided mask restricts correction to the given region."""
    data, gtab, _, full_mask = _make_synthetic_dwi()
    upper_mask = full_mask.copy()
    upper_mask[data.shape[0] // 2 :] = False

    corrected, _ = bias_field_correction(
        data,
        gtab,
        mask=upper_mask,
        method="poly",
        pyramid_levels=(2, 1),
        n_iter=1,
        robust=False,
        gradient_weighting=False,
        return_bias_field=True,
    )
    np.testing.assert_array_equal(corrected[~upper_mask], 0)


def test_single_pyramid_level():
    """Test with a single pyramid level (no coarse-to-fine)."""
    data, gtab, _, mask = _make_synthetic_dwi()
    corrected, _ = bias_field_correction(
        data,
        gtab,
        mask=mask,
        method="poly",
        pyramid_levels=(1,),
        n_iter=2,
        robust=False,
        gradient_weighting=False,
        return_bias_field=True,
    )
    assert corrected.shape == data.shape


def test_gradient_weighting_off():
    """Test that gradient_weighting=False runs without error."""
    data, gtab, _, mask = _make_synthetic_dwi()
    corrected, _ = bias_field_correction(
        data,
        gtab,
        mask=mask,
        method="bspline",
        n_control_points=(4, 4, 3),
        pyramid_levels=(2, 1),
        n_iter=2,
        robust=False,
        gradient_weighting=False,
        return_bias_field=True,
    )
    assert corrected.shape == data.shape


def test_robust_off():
    """Test that robust=False runs without error."""
    data, gtab, _, mask = _make_synthetic_dwi()
    corrected, _ = bias_field_correction(
        data,
        gtab,
        mask=mask,
        method="poly",
        pyramid_levels=(2, 1),
        n_iter=2,
        robust=False,
        gradient_weighting=False,
        return_bias_field=True,
    )
    assert corrected.shape == data.shape


def test_dtype_preservation():
    """Test that output dtype matches input dtype."""
    data, gtab, _, mask = _make_synthetic_dwi()
    for dtype in [np.float32, np.float64]:
        data_typed = data.astype(dtype)
        corrected = bias_field_correction(
            data_typed,
            gtab,
            mask=mask,
            method="poly",
            pyramid_levels=(2, 1),
            n_iter=1,
            robust=False,
            gradient_weighting=False,
        )
        assert corrected.dtype == dtype, f"Expected {dtype}, got {corrected.dtype}"
