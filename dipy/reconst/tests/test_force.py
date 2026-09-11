"""Tests for FORCE reconstruction module."""

from concurrent.futures import ThreadPoolExecutor
import json
import types

import numpy as np
import numpy.testing as npt
from numpy.testing import assert_almost_equal, assert_array_less

from dipy.core.gradients import gradient_table
from dipy.data import default_sphere
from dipy.reconst.force import (
    DEFAULT_FORCE_SEED,
    DEFAULT_NUM_ODI_VALUES,
    DEFAULT_ODI_RANGE,
    FORCEModel,
    SignalIndex,
    _fwhm_kde_batch,
    _odi_grid_matches,
    _seed_matches,
    _weighted_percentile,
    compute_microstructure_uncertainty_ambiguity,
    compute_uncertainty_ambiguity,
    create_signal_index,
    normalize_signals,
    softmax_stable,
)


def _make_gtab(shells):
    """Minimal GradientTable: two b0s + 6 directions per non-zero shell."""
    from dipy.core.gradients import gradient_table

    dirs = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=float,
    )
    bvals = [0, 0]
    bvecs = [[0, 0, 0], [0, 0, 0]]
    for b in shells:
        bvals.extend([b] * 6)
        bvecs.extend(dirs.tolist())
    return gradient_table(np.array(bvals, dtype=float), bvecs=np.array(bvecs))


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


def _tiny_model(n_sims=64, n_grad=12, seed=0):
    """A FORCEModel backed by a small hand-made library (no generation)."""
    rng = np.random.default_rng(seed)
    bvals = np.concatenate(([0.0], np.full(n_grad - 1, 1000.0)))
    bvecs = np.zeros((n_grad, 3))
    bvecs[1:, 0] = 1.0

    sims = {
        "signals": rng.random((n_sims, n_grad)).astype(np.float32) + 0.5,
        "labels": np.zeros((n_sims, len(default_sphere.vertices)), dtype=np.uint8),
        "num_fibers": np.ones(n_sims, dtype=np.float32),
        "fraction_array": rng.random((n_sims, 3)).astype(np.float32),
    }
    for key in (
        "fa",
        "md",
        "rd",
        "wm_fraction",
        "gm_fraction",
        "csf_fraction",
        "dispersion",
        "nd",
    ):
        sims[key] = rng.random(n_sims).astype(np.float32)

    gtab = gradient_table(bvals, bvecs=bvecs)
    return FORCEModel(gtab, simulations=sims, n_neighbors=8), sims


def test_micro_postprocessing_serial_inside_ray_worker(monkeypatch):
    """The per-parameter loop uses threads normally, but not inside a Ray worker.

    Also checks both paths produce the same uncertainty and ambiguity maps.
    """
    model, sims = _tiny_model()
    query = sims["signals"][:12]

    pool_uses = []

    def spy(*args, **kwargs):
        pool_uses.append(kwargs.get("max_workers"))
        return ThreadPoolExecutor(*args, **kwargs)

    monkeypatch.setattr("dipy.reconst.force.ThreadPoolExecutor", spy)
    # Pin the core count so the threaded path does not depend on runner size.
    monkeypatch.setattr(
        "dipy.reconst.force.get_usable_cpu_affinity", lambda: {0, 1, 2, 3}
    )

    # Outside a Ray worker the loop is threaded. Force the flag rather than
    # relying on ambient state: another test may have left Ray initialised,
    # since paramap never shuts it down.
    monkeypatch.setattr("dipy.reconst.force.has_ray", False)
    threaded = model.fit(query)
    assert pool_uses, "expected the thread pool to be used outside a Ray worker"

    # Inside a live Ray worker it stays serial.
    pool_uses.clear()
    monkeypatch.setattr("dipy.reconst.force.has_ray", True)
    monkeypatch.setattr(
        "dipy.reconst.force.ray", types.SimpleNamespace(is_initialized=lambda: True)
    )
    serial = model.fit(query)
    assert not pool_uses, "expected no thread pool inside a Ray worker"

    for param in ("fa", "nd"):
        for metric in ("uncertainty", "ambiguity"):
            npt.assert_array_equal(
                np.asarray(getattr(threaded, f"{metric}_{param}")),
                np.asarray(getattr(serial, f"{metric}_{param}")),
            )


def test_odi_grid_matches_legacy_and_exact():
    """Cache matching on the ODI grid, incl. legacy (keyless) entries."""
    # A legacy entry (written before the ODI grid was part of the key) is
    # treated as the historical default grid.
    legacy = {"num_simulations": 100}
    assert _odi_grid_matches(legacy, DEFAULT_ODI_RANGE, DEFAULT_NUM_ODI_VALUES)
    assert not _odi_grid_matches(legacy, (0.01, 0.6), 19)
    assert not _odi_grid_matches(legacy, DEFAULT_ODI_RANGE, 5)

    # A modern entry matches only its exact range and grid.
    entry = {"odi_range": [0.01, 0.6], "num_odi_values": 19}
    assert _odi_grid_matches(entry, (0.01, 0.6), 19)
    assert not _odi_grid_matches(entry, (0.01, 0.6), 10)  # same range, diff grid
    assert not _odi_grid_matches(entry, (0.01, 0.3), 19)  # diff range, same grid


def test_seed_matches_legacy_and_exact():
    """Cache matching on the generation seed, incl. legacy (keyless) entries."""
    # A legacy entry (written before the seed was part of the key) was not
    # generated reproducibly, so it only satisfies seed=None requests.
    legacy = {"num_simulations": 100}
    assert _seed_matches(legacy, None)
    assert not _seed_matches(legacy, DEFAULT_FORCE_SEED)

    # A modern entry matches only its exact seed.
    entry = {"seed": DEFAULT_FORCE_SEED}
    assert _seed_matches(entry, DEFAULT_FORCE_SEED)
    assert not _seed_matches(entry, 42)
    assert not _seed_matches(entry, None)


def _registry(dipy_home):
    path = dipy_home / "force_simulations" / "cache_registry.json"
    return json.load(open(path)) if path.exists() else []


def test_force_cache_keys_on_odi_grid(tmp_path, monkeypatch):
    """The simulation cache is keyed on odi_range and the resolved grid."""
    monkeypatch.setenv("DIPY_HOME", str(tmp_path))
    gtab = _make_gtab([1000])

    def gen(**kwargs):
        FORCEModel(gtab).generate(
            num_simulations=60, num_cpus=1, verbose=False, **kwargs
        )

    def entries():
        return [
            (tuple(e["odi_range"]), e["num_odi_values"]) for e in _registry(tmp_path)
        ]

    # 1. default range -> one entry recorded at the autoscaled grid (10).
    gen()
    assert entries() == [((0.01, 0.3), DEFAULT_NUM_ODI_VALUES)]

    # 2. a wider range is a distinct library (autoscaled grid 19), not a hit.
    gen(odi_range=(0.01, 0.6))
    assert len(entries()) == 2
    assert ((0.01, 0.6), 19) in entries()

    # 3. re-running the default params hits the cache (no new entry).
    n_before = len(entries())
    gen()
    assert len(entries()) == n_before

    # 4. an explicit grid at the default range is again distinct.
    gen(num_odi_values=5)
    assert ((0.01, 0.3), 5) in entries()
    assert len(entries()) == 3


def test_force_cache_keys_on_seed(tmp_path, monkeypatch):
    """The simulation cache is keyed on the generation seed."""
    monkeypatch.setenv("DIPY_HOME", str(tmp_path))
    gtab = _make_gtab([1000])

    def gen(**kwargs):
        model = FORCEModel(gtab)
        model.generate(num_simulations=60, num_cpus=1, verbose=False, **kwargs)
        return model

    # 1. the default seed is recorded in the registry.
    first = gen()
    assert [e["seed"] for e in _registry(tmp_path)] == [DEFAULT_FORCE_SEED]

    # 2. re-running with the default seed hits the cache (no new entry) and
    # serves the identical library.
    again = gen()
    assert len(_registry(tmp_path)) == 1
    npt.assert_array_equal(again.simulations["signals"], first.simulations["signals"])

    # 3. a different seed is a distinct library, not a cache hit.
    other = gen(seed=42)
    assert len(_registry(tmp_path)) == 2
    assert {e["seed"] for e in _registry(tmp_path)} == {DEFAULT_FORCE_SEED, 42}
    assert not np.array_equal(
        other.simulations["signals"], first.simulations["signals"]
    )


def test_force_cache_reports_seedless_legacy_entry(tmp_path, monkeypatch):
    """A legacy cache miss explains why a seeded library is regenerated."""
    monkeypatch.setenv("DIPY_HOME", str(tmp_path))
    gtab = _make_gtab([1000])

    FORCEModel(gtab).generate(
        num_simulations=10,
        num_cpus=1,
        seed=None,
        compute_dti=False,
        verbose=False,
    )

    registry_path = tmp_path / "force_simulations" / "cache_registry.json"
    registry = _registry(tmp_path)
    registry[0].pop("seed")
    with open(registry_path, "w") as f:
        json.dump(registry, f)

    messages = []
    monkeypatch.setattr("dipy.reconst.force.logger.info", messages.append)
    FORCEModel(gtab).generate(
        num_simulations=10,
        num_cpus=1,
        compute_dti=False,
        verbose=False,
    )

    assert any(
        "predate seed tracking" in message
        and "new seeded library will be generated" in message
        for message in messages
    )
    assert _registry(tmp_path)[-1]["seed"] == DEFAULT_FORCE_SEED


def test_cache_registry_separates_min_crossing_angles(tmp_path, monkeypatch):
    """Libraries built with different crossing-angle limits get separate cache slots."""
    from dipy.core.gradients import gradient_table

    monkeypatch.setenv("DIPY_HOME", str(tmp_path))

    dirs = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=float,
    )
    bvals = np.array([0.0] + [1000.0] * 6 + [2000.0] * 6)
    bvecs = np.vstack(([[0.0, 0.0, 0.0]], dirs, dirs))
    gtab = gradient_table(bvals, bvecs=bvecs)

    strict = FORCEModel(gtab)
    strict.generate(num_simulations=60, num_cpus=1, verbose=False)

    relaxed = FORCEModel(gtab)
    relaxed.generate(
        num_simulations=60, num_cpus=1, two_fiber_min_angle=0.0, verbose=False
    )

    cache_dir = tmp_path / "force_simulations"
    assert len(list(cache_dir.glob("force_sim_*.npz"))) == 2

    # The relaxed request must not be served the strict library, and a repeat
    # of the relaxed request must reuse the one just written.
    again = FORCEModel(gtab)
    again.generate(
        num_simulations=60, num_cpus=1, two_fiber_min_angle=0.0, verbose=False
    )
    assert len(list(cache_dir.glob("force_sim_*.npz"))) == 2
    npt.assert_array_equal(again.simulations["signals"], relaxed.simulations["signals"])
