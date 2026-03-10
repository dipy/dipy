"""Tests for FORCE reconstruction module."""

import numpy as np
from numpy.testing import assert_almost_equal

from dipy.reconst.force import (
    SignalIndex,
    compute_uncertainty_ambiguity,
    create_signal_index,
    labels_to_peak_indices,
    normalize_signals,
    pick_top_peaks_from_weights,
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


def test_labels_to_peak_indices():
    """Test conversion of binary labels to peak indices."""
    labels = np.array(
        [
            [0, 1, 0, 1, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    peak_idx = labels_to_peak_indices(labels, max_peaks=3)

    assert peak_idx.shape == (3, 3)
    assert peak_idx[0, 0] == 1  # First peak at index 1
    assert peak_idx[0, 1] == 3  # Second peak at index 3
    assert peak_idx[1, 0] == 0  # Single peak at index 0
    assert peak_idx[2, 0] == -1  # No peaks


def test_pick_top_peaks_from_weights():
    """Test peak extraction from directional weights."""
    weights = np.zeros(100, dtype=np.float32)
    weights[10] = 1.0
    weights[50] = 0.8
    weights[90] = 0.6

    sphere_dirs = np.random.randn(100, 3).astype(np.float32)
    sphere_dirs = sphere_dirs / np.linalg.norm(sphere_dirs, axis=1, keepdims=True)

    peak_dirs, peak_inds, peak_vals = pick_top_peaks_from_weights(
        weights, sphere_dirs, n_peaks=5
    )

    assert peak_dirs.shape == (5, 3)
    assert peak_inds.shape == (5,)
    assert peak_vals.shape == (5,)

    # First peak should be at index 10 with value 1.0
    assert peak_inds[0] == 10
    assert peak_vals[0] == 1.0


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
