"""Tests for FORCE reconstruction module."""

from concurrent.futures import ThreadPoolExecutor
import json
import types
import warnings

import numpy as np
import numpy.testing as npt
from numpy.testing import assert_almost_equal, assert_array_less
import pytest

from dipy.core.gradients import gradient_table
from dipy.data import default_sphere, get_sphere
from dipy.reconst._force_align import (
    alignment_system as _alignment_system,
    kernel_signal as _kernel_signal,
)
from dipy.reconst.force import (
    DEFAULT_FORCE_SEED,
    DEFAULT_NUM_ODI_VALUES,
    DEFAULT_ODI_RANGE,
    FORCEFit,
    FORCEModel,
    SignalIndex,
    _alignment_operators,
    _fiber_kernels,
    _fwhm_kde_batch,
    _kernel_prior_table,
    _odi_grid_matches,
    _seed_matches,
    _weighted_percentile,
    compute_microstructure_uncertainty_ambiguity,
    compute_uncertainty_ambiguity,
    create_signal_index,
    force_peaks,
    normalize_signals,
    softmax_stable,
)
from dipy.reconst.multi_voxel import MultiVoxelFit
from dipy.reconst.shm import descoteaux07_legacy_msg, sf_to_sh


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


# Fit parameters of a voxel whose matched entry carries no fiber.
_NO_FIBERS = {
    "fiber_dirs": np.zeros((3, 3)),
    "fiber_ref": np.zeros((3, 3)),
    "fiber_keep": np.zeros(3, dtype=bool),
    "fiber_fractions": np.zeros(3),
    "fiber_hess": np.zeros((6, 6)),
    "fiber_grad": np.zeros(6),
    "fiber_aligned": True,
    "wm_fraction": 0.0,
}


@pytest.mark.parametrize("with_odf", [False, True])
@pytest.mark.parametrize("use_mask", [False, True])
def test_force_peaks_volume(with_odf, use_mask):
    """Preserve SH across sparse, non-contiguous volumes; no fiber, no peak."""
    shape = (3, 67, 101)
    fits = np.full((67, 101, 3), None, dtype=object).transpose(2, 0, 1)
    fit_mask = np.zeros(shape, dtype=bool)
    peak_mask = np.zeros(shape, dtype=bool)
    expected_dirs = np.zeros((*shape, 5, 3), dtype=np.float32)
    expected_inds = np.full((*shape, 5), -1, dtype=np.int32)
    expected_vals = np.zeros((*shape, 5), dtype=np.float32)
    expected_sh = np.zeros((*shape, 28)) if with_odf else None
    normalized_odfs = []
    original_odfs = []

    # Leave the beginning empty and include separated voxels near the end.
    for i, flat_index in enumerate((10_000, 10_001, 19_999, 20_000, 20_300)):
        voxel = np.unravel_index(flat_index, shape)
        odf = None
        if with_odf:
            odf = np.linspace(i, i + 2, len(default_sphere.vertices), dtype=np.float32)
            if i == 0:
                odf *= 0.4
            elif i == 2:
                odf.fill(2)
            original_odfs.append((odf, odf.copy()))
            normalized_odfs.append(
                (odf - odf.min()) / (odf.max() - odf.min() + 1e-12)
                if odf.max() > 1
                else odf
            )
        fits[voxel] = FORCEFit(None, {"odf": odf, **_NO_FIBERS})
        fit_mask[voxel] = True
        peak_mask[voxel] = i != 1

    fit = MultiVoxelFit(None, fits, fit_mask)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning,
        )
        if with_odf:
            expected_sh[fit_mask] = sf_to_sh(
                np.asarray(normalized_odfs), default_sphere, sh_order_max=6
            )
        pam = force_peaks(fit, mask=peak_mask if use_mask else None, sh_order=6)

    npt.assert_array_equal(pam.peak_dirs, expected_dirs)
    npt.assert_array_equal(pam.peak_indices, expected_inds)
    npt.assert_array_equal(pam.peak_values, expected_vals)
    if with_odf:
        npt.assert_allclose(pam.shm_coeff, expected_sh, rtol=1e-12, atol=1e-12)
        for odf, original in original_odfs:
            npt.assert_array_equal(odf, original)
    else:
        assert pam.shm_coeff is None


def test_force_peaks_single_voxel():
    """Single-voxel conversion retains its original ODF scaling."""
    odf = np.linspace(1, 3, len(default_sphere.vertices), dtype=np.float32)
    fit = FORCEFit(None, {"odf": odf, **_NO_FIBERS})

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning,
        )
        expected_sh = sf_to_sh(odf, default_sphere, sh_order_max=6)
        pam = force_peaks(fit, sh_order=6)

    npt.assert_array_equal(pam.peak_dirs, np.zeros((5, 3)))
    npt.assert_array_equal(pam.peak_indices, np.full(5, -1))
    npt.assert_array_equal(pam.peak_values, np.zeros(5))
    npt.assert_allclose(pam.shm_coeff, expected_sh, rtol=1e-12, atol=1e-12)


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


_VERTICES = default_sphere.vertices
# A fiber vertex, one about 35 degrees from it and one about 70 degrees away.
_V0 = 10
_V35 = 67
_V70 = 163
# The ODF conversion still uses the legacy spherical harmonic basis.
_legacy_sh_basis = pytest.mark.filterwarnings(
    "ignore:The legacy descoteaux07 SH basis:PendingDeprecationWarning"
)


def _shell_gtab(*shells):
    """GradientTable with one b0 and the given ``(b-value, n_directions)`` shells.

    Without arguments: one shell of 64 directions at b = 2000.
    """
    dirs = get_sphere(name="repulsion100").vertices
    bvals, bvecs, first = [0.0], [np.zeros((1, 3))], 0
    for bval, n_dirs in shells or [(2000.0, 64)]:
        bvals += [bval] * n_dirs
        bvecs.append(dirs[first : first + n_dirs])
        first += n_dirs
    return gradient_table(np.array(bvals), bvecs=np.concatenate(bvecs))


def _stick_signal(gtab, axes, weights):
    """Signal of sticks along ``axes`` on top of an isotropic ball."""
    signal = (1.0 - np.sum(weights)) * np.exp(-gtab.bvals * 1.0e-3)
    for axis, weight in zip(axes, weights):
        signal = signal + weight * np.exp(
            -gtab.bvals * 2.0e-3 * (gtab.bvecs @ axis) ** 2
        )
    return 100.0 * signal


def _stick_library(gtab, entries, *, wm=0.8):
    """Hand-made FORCE library: sticks on sphere vertices plus a ball.

    ``entries`` lists, for each entry, the sphere vertices of its fibers and
    their fractions. As in the simulator, fractions are stored in an order
    unrelated to the vertices and the ODF marks which fiber is which.
    """
    n_entries = len(entries)
    sims = {
        "signals": np.zeros((n_entries, len(gtab.bvals)), dtype=np.float32),
        "labels": np.zeros((n_entries, len(_VERTICES)), dtype=np.uint8),
        "odfs": np.zeros((n_entries, len(_VERTICES)), dtype=np.float16),
        "fraction_array": np.zeros((n_entries, 3), dtype=np.float32),
        "num_fibers": np.zeros(n_entries, dtype=np.float32),
    }
    for i, (vertex, fractions) in enumerate(entries):
        vertex, fractions = list(vertex), np.asarray(fractions, dtype=float)
        sims["signals"][i] = _stick_signal(gtab, _VERTICES[vertex], wm * fractions)
        sims["labels"][i, vertex] = 1
        sims["odfs"][i, vertex] = fractions
        sims["fraction_array"][i, : len(vertex)] = fractions[::-1]
        sims["num_fibers"][i] = len(vertex)
    for key, value in (
        ("wm_fraction", wm),
        ("gm_fraction", 1.0 - wm),
        ("csf_fraction", 0.0),
        ("dispersion", 1.0 - wm),
        ("nd", 0.5),
        ("fa", 0.5),
        ("md", 1.0e-3),
        ("rd", 5.0e-4),
    ):
        sims[key] = np.full(n_entries, value, dtype=np.float32)
    return sims


def _tilt(axis, toward, degrees):
    """Unit vector ``degrees`` away from ``axis``, in the plane of ``toward``."""
    perp = toward - (toward @ axis) * axis
    perp = perp / np.linalg.norm(perp)
    angle = np.deg2rad(degrees)
    return np.cos(angle) * axis + np.sin(angle) * perp


def _axial_angle(a, b):
    """Angle in degrees between axes, ignoring their sign."""
    return np.degrees(np.arccos(np.clip(np.abs(np.sum(a * b, axis=-1)), 0.0, 1.0)))


def _n_peaks(pam):
    return np.count_nonzero(np.linalg.norm(pam.peak_dirs, axis=-1) > 0, axis=-1)


@_legacy_sh_basis
@pytest.mark.parametrize(
    "shells, tol", [((), 0.1), (((1000.0, 30), (2500.0, 64)), 0.5)]
)
def test_force_peaks_leave_the_sphere_grid(shells, tol):
    """Peaks follow fibers that lie between library axes, each fiber on its own."""
    gtab = _shell_gtab(*shells)
    sims = _stick_library(
        gtab,
        [((_V0,), (1.0,)), ((_V35,), (1.0,)), ((_V0, _V70), (0.7, 0.3))],
    )
    model = FORCEModel(gtab, simulations=sims, n_neighbors=3)

    one = _tilt(_VERTICES[_V0], _VERTICES[_V70], 3.0)
    data = np.tile(_stick_signal(gtab, [one], [0.8]), (3, 3, 3, 1))
    fit = model.fit(data)
    for pam in (force_peaks(fit), force_peaks(fit.fit_array[0, 0, 0])):
        assert _axial_angle(pam.peak_dirs[..., 0, :], one).max() < tol
        npt.assert_array_equal(_n_peaks(pam), 1)
        npt.assert_array_equal(pam.peak_indices[..., 0], _V0)
        npt.assert_array_equal(pam.peak_indices[..., 1:], -1)

    # Two fibers tilted by different amounts in different directions.
    major = _tilt(_VERTICES[_V0], _VERTICES[_V35], 4.0)
    minor = _tilt(_VERTICES[_V70], _VERTICES[_V35], -5.0)
    data = np.tile(_stick_signal(gtab, [major, minor], [0.56, 0.24]), (3, 3, 3, 1))
    pam = force_peaks(model.fit(data))
    assert _axial_angle(pam.peak_dirs[..., 0, :], major).max() < tol
    assert _axial_angle(pam.peak_dirs[..., 1, :], minor).max() < tol
    # Values are the fractions of the fibers the peaks belong to, largest first.
    expected = np.broadcast_to([0.7, 0.3], pam.peak_values[..., :2].shape)
    npt.assert_allclose(pam.peak_values[..., :2], expected, rtol=1e-6)
    npt.assert_array_equal(pam.peak_values[..., 2:], 0.0)


@_legacy_sh_basis
def test_force_peaks_stay_within_max_angle():
    """A peak stops 25 degrees from the library axis, on the way to the fiber."""
    gtab = _shell_gtab()
    sims = _stick_library(gtab, [((_V0,), (1.0,)), ((_V70,), (1.0,))])
    model = FORCEModel(gtab, simulations=sims, n_neighbors=2)
    fiber = _tilt(_VERTICES[_V0], _VERTICES[_V70], 30.0)
    data = np.tile(_stick_signal(gtab, [fiber], [0.8]), (2, 2, 2, 1))
    peak = force_peaks(model.fit(data)).peak_dirs[..., 0, :]
    npt.assert_allclose(_axial_angle(peak, _VERTICES[_V0]), 25.0, atol=1e-3)
    npt.assert_allclose(_axial_angle(peak, fiber), 5.0, atol=1e-3)


@_legacy_sh_basis
def test_force_peaks_drop_undetermined_fibers():
    """A fiber the data cannot orient is not reported."""
    gtab = _shell_gtab()
    sims = _stick_library(gtab, [((_V0, _V70), (0.99, 0.01))])
    model = FORCEModel(gtab, simulations=sims, n_neighbors=1)
    rng = np.random.default_rng(0)
    clean = _stick_signal(gtab, _VERTICES[[_V0, _V70]], [0.792, 0.008])
    data = clean + 5.0 * rng.normal(size=(4, 4, 4, clean.size))

    pam = force_peaks(model.fit(data))
    npt.assert_array_equal(_n_peaks(pam), 1)
    assert _axial_angle(pam.peak_dirs[..., 0, :], _VERTICES[_V0]).max() < 5.0
    npt.assert_allclose(pam.peak_values[..., 0], 0.99, rtol=1e-6)
    npt.assert_array_equal(pam.peak_indices[..., 1:], -1)


@_legacy_sh_basis
def test_force_peaks_fill_white_matter_from_posterior_odf():
    """A white-matter voxel whose entry has no fiber takes its ODF peaks."""
    gtab = _shell_gtab()
    sims = _stick_library(gtab, [((_V0,), (1.0,)), ((_V0,), (0.6,)), ((_V0,), (0.3,))])
    # Entries 1 and 2: fibers not labelled, in white matter (fraction 0.6)
    # and outside it (0.4).
    sims["labels"][1:] = 0
    sims["wm_fraction"][1:] = [0.6, 0.4]
    model = FORCEModel(gtab, simulations=sims, n_neighbors=3)
    fiber = _tilt(_VERTICES[_V0], _VERTICES[_V70], 3.0)
    data = np.tile(_stick_signal(gtab, [fiber], [0.8]), (3, 3, 3, 1))
    data[1, 1, 1] = _stick_signal(gtab, [fiber], [0.48])
    data[0, 0, 0] = _stick_signal(gtab, [fiber], [0.24])
    fit = model.fit(data)
    assert not fit.fit_array[1, 1, 1]._params["fiber_keep"].any()

    pam = force_peaks(fit)
    assert _n_peaks(pam)[1, 1, 1] == 1
    assert _n_peaks(pam)[0, 0, 0] == 0
    # The ODF peak took part in the spatial stage: it left the grid with its
    # neighbors instead of staying on the vertex.
    assert _axial_angle(pam.peak_dirs[1, 1, 1, 0], fiber) < 1.0
    npt.assert_allclose(pam.peak_values[1, 1, 1, 0], 1.0)
    assert _n_peaks(force_peaks(fit, wm_threshold=0.3))[0, 0, 0] == 1


@_legacy_sh_basis
def test_force_peaks_coherence():
    """Continuity averages out noise along a bundle, not across two bundles."""
    gtab = _shell_gtab((2000.0, 30))
    sims = _stick_library(
        gtab, [((_V0,), (1.0,)), ((_V35,), (1.0,)), ((_V70,), (1.0,))]
    )
    model = FORCEModel(gtab, simulations=sims, n_neighbors=3)
    rng = np.random.default_rng(1)

    fiber = _tilt(_VERTICES[_V0], _VERTICES[_V70], 3.0)
    clean = _stick_signal(gtab, [fiber], [0.8])
    data = clean + 8.0 * rng.normal(size=(6, 6, 6, clean.size))
    fit = model.fit(data)
    coherent = force_peaks(fit).peak_dirs[..., 0, :]
    alone = np.array(
        [force_peaks(voxel).peak_dirs[0] for voxel in fit.fit_array.ravel()]
    ).reshape(coherent.shape)
    error = [_axial_angle(dirs, fiber).mean() for dirs in (coherent, alone)]
    assert error[0] < 0.7 * error[1]

    # Two bundles 35 degrees apart meeting at a plane keep their own axes.
    axes = _VERTICES[[_V0, _V35]]
    clean = np.empty((8, 4, 4, clean.size))
    clean[:4] = _stick_signal(gtab, axes[:1], [0.8])
    clean[4:] = _stick_signal(gtab, axes[1:], [0.8])
    data = clean + 5.0 * rng.normal(size=clean.shape)
    coherent = force_peaks(model.fit(data)).peak_dirs[..., 0, :]
    at_interface = np.concatenate(
        [
            _axial_angle(coherent[3], axes[0]).ravel(),
            _axial_angle(coherent[4], axes[1]).ravel(),
        ]
    )
    assert at_interface.mean() < 3.0


@_legacy_sh_basis
def test_force_peaks_need_a_usable_shell():
    """Without a shell that resolves anisotropy the library axes are kept."""
    gtab = _make_gtab([1000])  # six gradients along three axes only
    sims = _stick_library(gtab, [((_V0,), (1.0,)), ((_V70,), (1.0,))])
    model = FORCEModel(gtab, simulations=sims, n_neighbors=2)
    data = np.tile(sims["signals"][0], (2, 2, 2, 1))
    pam = force_peaks(model.fit(data))
    npt.assert_array_equal(_n_peaks(pam), 1)
    npt.assert_array_equal(pam.peak_indices[..., 0], _V0)
    assert _axial_angle(pam.peak_dirs[..., 0, :], _VERTICES[_V0]).max() < 1e-3


def test_fiber_kernels_predict_moved_fibers():
    """Kernels taken from an entry's signal give its signal for any fiber axes."""
    gtab = _shell_gtab()
    ops = _alignment_operators(gtab, default_sphere)
    # Three mutually (almost) orthogonal fibers: their second-degree parts
    # cancel in the signal, so that part of each kernel must come from what
    # single-fiber entries of the library look like.
    second = int(np.argmin(np.abs(_VERTICES @ _VERTICES[_V0])))
    third = int(np.argmin(np.abs(_VERTICES @ _VERTICES[[_V0, second]].T).sum(axis=1)))
    triple = [_V0, second, third]
    assert (
        _axial_angle(_VERTICES[triple][[0, 0, 1]], _VERTICES[triple][[1, 2, 2]]).min()
        > 85.0
    )
    fractions = np.array([0.4, 0.35, 0.25])
    sims = _stick_library(gtab, [((_V35,), (1.0,)), (triple, fractions)])
    table = _kernel_prior_table(sims, default_sphere, ops)

    vertex = np.array([triple])
    valid = np.ones((1, 3), dtype=bool)
    prior = 0.8 * fractions[None, :, None, None] * table["kappa"][[0]][:, None]
    kernels = {
        "blind": _fiber_kernels(
            sims["signals"][1:], vertex, valid, default_sphere, ops
        ),
        "informed": _fiber_kernels(
            sims["signals"][1:], vertex, valid, default_sphere, ops, prior=prior
        ),
    }

    def predicted(poly, axes):
        return _kernel_signal(
            np.ascontiguousarray(axes[None], dtype=np.float64),
            valid.astype(np.uint8),
            np.ascontiguousarray(poly),
            ops["shell"],
            ops["gdirs"],
        )[0]

    moved = _VERTICES[triple].copy()
    moved[1] = _tilt(moved[1], moved[0], 15.0)
    change = _stick_signal(gtab, moved, 0.8 * fractions) - sims["signals"][1]
    error = {}
    for name, (poly, _) in kernels.items():
        model_change = predicted(poly, moved) - predicted(poly, _VERTICES[triple])
        error[name] = np.linalg.norm(model_change - change) / np.linalg.norm(change)
    assert error["informed"] < 0.05
    assert error["blind"] > 3 * error["informed"]


def test_alignment_system_matches_finite_differences():
    """The descent direction is the slope of the cost along the tangent frame."""
    gtab = _shell_gtab()
    ops = _alignment_operators(gtab, default_sphere)
    sims = _stick_library(gtab, [((_V0, _V35, _V70), (0.5, 0.3, 0.2))])
    vertex = np.array([[_V0, _V35, _V70]])
    valid = np.ones((1, 3), dtype=bool)
    poly, _ = _fiber_kernels(sims["signals"], vertex, valid, default_sphere, ops)
    poly = np.ascontiguousarray(poly)
    valid8 = valid.astype(np.uint8)
    rng = np.random.default_rng(3)
    signal = (sims["signals"] + 3.0 * rng.normal(size=sims["signals"].shape)).astype(
        np.float32
    )
    rest = sims["signals"] - _kernel_signal(
        np.ascontiguousarray(_VERTICES[vertex]),
        valid8,
        poly,
        ops["shell"],
        ops["gdirs"],
    )
    dirs = _VERTICES[vertex] + 0.05 * rng.normal(size=(1, 3, 3))
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

    def system(axes):
        return _alignment_system(
            signal,
            rest,
            np.ascontiguousarray(axes),
            valid8,
            poly,
            ops["shell"],
            ops["gdirs"],
        )

    hess, grad, _, frames = system(dirs)
    npt.assert_allclose(hess, np.swapaxes(hess, 1, 2), atol=1e-12)
    assert np.linalg.eigvalsh(hess[0]).min() > -1e-9
    step = 1e-4
    slope = np.zeros(6)
    for k in range(3):
        for c in range(2):
            costs = []
            for sign in (1.0, -1.0):
                moved = dirs.copy()
                moved[0, k] += sign * step * frames[0, k, c]
                moved[0, k] /= np.linalg.norm(moved[0, k])
                costs.append(system(moved)[2][0])
            slope[2 * k + c] = (costs[0] - costs[1]) / (2 * step)
    npt.assert_allclose(grad[0], -0.5 * slope, rtol=2e-3, atol=1e-7)
