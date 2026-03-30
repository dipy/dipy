"""Tests for FORCE reconstruction module."""

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_less

from dipy.reconst.force import (
    SignalIndex,
    _fwhm_kde_batch,
    _weighted_percentile,
    compute_microstructure_uncertainty_ambiguity,
    compute_uncertainty_ambiguity,
    create_signal_index,
    normalize_signals,
    softmax_stable,
)


def test_normalize_signals():
    """Test signal normalization."""
    signals = np.array([[3, 4], [0, 0], [1, 0]], dtype=np.float32)
    normalized = normalize_signals(signals)

    # First row should have unit norm
    assert_almost_equal(np.linalg.norm(normalized[0]), 1.0)

    # Zero row should remain zero (handled by eps)
    assert normalized[1, 0] == 0.0

    # Third row should have unit norm
    assert_almost_equal(np.linalg.norm(normalized[2]), 1.0)


def test_softmax_stable():
    """Test numerically stable softmax."""
    x = np.array([[1000, 1001, 1002], [0, 0, 0]], dtype=np.float32)
    result = softmax_stable(x, axis=1)

    # Should sum to 1 along axis
    assert_almost_equal(np.sum(result, axis=1), [1.0, 1.0])

    # Should not have NaN or Inf
    assert np.all(np.isfinite(result))


def test_compute_uncertainty_ambiguity():
    """Test uncertainty and ambiguity metrics."""
    scores = np.array(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.5, 0.5, 0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )

    uncertainty, ambiguity = compute_uncertainty_ambiguity(scores)

    assert uncertainty.shape == (2,)
    assert ambiguity.shape == (2,)

    # First row has spread, second is uniform
    assert uncertainty[0] > uncertainty[1]


def test_signal_index():
    """Test SignalIndex inner product search."""
    index = SignalIndex(10)

    # Add some vectors
    vectors = np.random.randn(100, 10).astype(np.float32)
    index.add(vectors)

    assert index.ntotal == 100

    # Search
    query = np.random.randn(5, 10).astype(np.float32)
    D, neighbors = index.search(query, k=10)

    assert D.shape == (5, 10)
    assert neighbors.shape == (5, 10)

    # Distances should be in descending order
    for i in range(5):
        assert np.all(D[i, :-1] >= D[i, 1:])


def test_create_signal_index():
    """Test signal index creation."""
    signals = np.random.randn(100, 50).astype(np.float32)
    signals_norm = signals / np.linalg.norm(signals, axis=1, keepdims=True)

    index = create_signal_index(signals_norm)

    assert index.ntotal == 100
    assert index.d == 50


def test_signal_search():
    """Test signal matching search."""
    # Create mock index
    signals = np.random.randn(100, 50).astype(np.float32)
    signals_norm = signals / np.linalg.norm(signals, axis=1, keepdims=True)
    index = create_signal_index(signals_norm)

    # Query signals
    query = np.random.randn(10, 50).astype(np.float32)
    query_norm = query / np.linalg.norm(query, axis=1, keepdims=True)

    D, neighbors = index.search(query_norm, k=20)

    assert D.shape == (10, 20)
    assert neighbors.shape == (10, 20)


def test_weighted_percentile():
    """Test weighted percentile computation."""
    # Simple case: uniform weights should give standard percentile
    vals = np.array([[1, 2, 3, 4, 5]], dtype=np.float32)
    weights = np.array([[0.2, 0.2, 0.2, 0.2, 0.2]], dtype=np.float32)

    q50 = _weighted_percentile(vals, weights, 0.5)
    assert q50.shape == (1,)
    # Median of [1,2,3,4,5] with uniform weights should be around 3
    assert 2.0 <= q50[0] <= 4.0

    # Concentrated weights: all weight on first value
    weights_conc = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    q50_conc = _weighted_percentile(vals, weights_conc, 0.5)
    assert_almost_equal(q50_conc[0], 1.0)

    # Test batch processing
    vals_batch = np.array([[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]], dtype=np.float32)
    weights_batch = np.array(
        [[0.2, 0.2, 0.2, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2, 0.2]], dtype=np.float32
    )
    q75 = _weighted_percentile(vals_batch, weights_batch, 0.75)
    assert q75.shape == (2,)
    # Second row values are 10x larger
    assert q75[1] > q75[0]


def test_fwhm_kde_batch():
    """Test FWHM via weighted KDE."""
    # Concentrated distribution should have small FWHM
    vals_narrow = np.array([[5.0, 5.1, 5.0, 4.9, 5.0]], dtype=np.float32)
    weights_uniform = np.array([[0.2, 0.2, 0.2, 0.2, 0.2]], dtype=np.float32)

    fwhm_narrow = _fwhm_kde_batch(vals_narrow, weights_uniform)
    assert fwhm_narrow.shape == (1,)
    assert fwhm_narrow[0] >= 0

    # Spread distribution should have larger FWHM
    vals_wide = np.array([[0.0, 2.5, 5.0, 7.5, 10.0]], dtype=np.float32)
    fwhm_wide = _fwhm_kde_batch(vals_wide, weights_uniform)

    assert fwhm_wide[0] > fwhm_narrow[0]

    # Test batch processing
    vals_batch = np.vstack([vals_narrow, vals_wide])
    weights_batch = np.vstack([weights_uniform, weights_uniform])
    fwhm_batch = _fwhm_kde_batch(vals_batch, weights_batch)

    assert fwhm_batch.shape == (2,)
    assert fwhm_batch[1] > fwhm_batch[0]


def test_compute_microstructure_uncertainty_ambiguity():
    """Test microstructure uncertainty and ambiguity metrics."""
    # Case 1: Concentrated values - low uncertainty and ambiguity
    vals_narrow = np.array(
        [[0.5, 0.51, 0.49, 0.5, 0.5], [0.5, 0.51, 0.49, 0.5, 0.5]],
        dtype=np.float32,
    )
    weights_uniform = np.array(
        [[0.2, 0.2, 0.2, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2, 0.2]],
        dtype=np.float32,
    )
    prior_range = 1.0

    unc_narrow, amb_narrow = compute_microstructure_uncertainty_ambiguity(
        vals_narrow, weights_uniform, prior_range
    )

    assert unc_narrow.shape == (2,)
    assert amb_narrow.shape == (2,)
    # Should be low values since distribution is concentrated
    assert_array_less(unc_narrow, 0.5)
    assert_array_less(amb_narrow, 0.5)

    # Case 2: Spread values - higher uncertainty and ambiguity
    vals_wide = np.array(
        [[0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.25, 0.5, 0.75, 1.0]],
        dtype=np.float32,
    )

    unc_wide, amb_wide = compute_microstructure_uncertainty_ambiguity(
        vals_wide, weights_uniform, prior_range
    )

    # Wide distribution should have higher uncertainty than narrow
    assert unc_wide[0] > unc_narrow[0]
    assert amb_wide[0] > amb_narrow[0]

    # Case 3: Values are in [0, 1], results normalized by prior_range
    assert np.all(unc_wide >= 0)
    assert np.all(amb_wide >= 0)

    # Case 4: Test with concentrated weights
    weights_conc = np.array(
        [[0.9, 0.025, 0.025, 0.025, 0.025], [0.9, 0.025, 0.025, 0.025, 0.025]],
        dtype=np.float32,
    )
    unc_conc, amb_conc = compute_microstructure_uncertainty_ambiguity(
        vals_wide, weights_conc, prior_range
    )

    # Concentrated weights should reduce uncertainty even with spread values
    assert unc_conc[0] < unc_wide[0]


def test_compute_microstructure_uncertainty_ambiguity_different_ranges():
    """Test that prior_range correctly normalizes uncertainty/ambiguity."""
    vals = np.array([[0.0, 0.25, 0.5, 0.75, 1.0]], dtype=np.float32)
    weights = np.array([[0.2, 0.2, 0.2, 0.2, 0.2]], dtype=np.float32)

    # Same values but different prior ranges
    unc_small, amb_small = compute_microstructure_uncertainty_ambiguity(
        vals, weights, prior_range=1.0
    )
    unc_large, amb_large = compute_microstructure_uncertainty_ambiguity(
        vals, weights, prior_range=10.0
    )

    # Larger prior range should give smaller normalized values
    assert unc_large[0] < unc_small[0]
    assert amb_large[0] < amb_small[0]
