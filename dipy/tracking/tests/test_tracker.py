import warnings

import nibabel as nib
import numpy as np
import numpy.testing as npt
import pytest

from dipy.core.geometry import cart2sphere
from dipy.core.sphere import HemiSphere, unit_octahedron
from dipy.data import get_fnames, get_sphere
from dipy.direction import peak_directions
from dipy.direction.peaks import PeaksAndMetrics
from dipy.direction.pmf import SimplePeakGen, SimplePmfGen
from dipy.reconst.shm import descoteaux07_legacy_msg, sh_to_sf
from dipy.sims.voxel import multi_tensor_odf
from dipy.tracking import tracker
from dipy.tracking.accel import (
    ACCEL_BACKENDS,
    accel_backend_available,
    select_accel_backend,
    webgpu,
)
from dipy.tracking.stopping_criterion import (
    BinaryStoppingCriterion,
    ThresholdStoppingCriterion,
)
from dipy.tracking.streamline import Streamlines
from dipy.tracking.utils import random_seeds_from_mask


def track(method, **kwargs):
    """This tests that the number of streamlines equals the number of seeds
    when return_all=True.
    """
    fnames = get_fnames(name="disco1", include_optional=True)
    sphere = HemiSphere.from_sphere(get_sphere(name="repulsion724"))
    sh = nib.load(fnames[20]).get_fdata()
    fODFs = sh_to_sf(sh, sphere, sh_order_max=12, basis_type="tournier07", legacy=False)
    fODFs[fODFs < 0] = 0

    # seeds position and initial directions
    mask = nib.load(fnames[25]).get_fdata()
    affine = nib.load(fnames[25]).affine
    seed_mask = np.ones(mask.shape)
    seeds = random_seeds_from_mask(
        seed_mask, affine, seeds_count=100, seed_count_per_voxel=False
    )
    directions = np.random.random(seeds.shape)
    directions = np.array([v / np.linalg.norm(v) for v in directions])

    backend = kwargs.get("backend", "cpu")
    use_sf = kwargs.get("use_sf", False)
    use_directions = kwargs.get("use_dirs", False)

    if backend == "cpu":
        sc = BinaryStoppingCriterion(mask)
    else:
        sc = ThresholdStoppingCriterion(mask, 0.5)

    # test return_all=True
    params = {
        "max_len": 500,
        "min_len": 0,
        "step_size": 0.5,
        "voxel_size": np.ones(3),
        "max_angle": 20,
        "random_seed": 0,
        "return_all": True,
        "sf": fODFs if use_sf else None,
        "sh": sh if not use_sf else None,
        "seed_directions": directions if use_directions else None,
        "sphere": sphere,
    }
    if "backend" in kwargs:
        params["backend"] = backend
    stream_gen = method(seeds, sc, affine, **params)

    streamlines = Streamlines(stream_gen)
    if use_directions:
        npt.assert_equal(len(streamlines), len(seeds))
    else:
        npt.assert_array_less(len(streamlines), len(seeds))

    # test return_all=False
    params = {
        "max_len": 500,
        "min_len": 10,
        "step_size": 0.5,
        "voxel_size": np.ones(3),
        "max_angle": 20,
        "random_seed": 0,
        "return_all": False,
        "sf": fODFs if use_sf else None,
        "sh": sh if not use_sf else None,
        "seed_directions": directions if use_directions else None,
        "sphere": sphere,
    }
    if "backend" in kwargs:
        params["backend"] = backend

    stream_gen = method(seeds, sc, affine, **params)

    streamlines = Streamlines(stream_gen)
    npt.assert_array_less(len(streamlines), len(seeds))


def test_deterministic_tracking():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning,
        )
        track(tracker.deterministic_tracking, use_dirs=True)
        track(tracker.deterministic_tracking, use_sf=True, use_dirs=True)


def test_probabilistic_tracking():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning,
        )
        track(tracker.probabilistic_tracking, use_dirs=True, backend="cpu")
        track(tracker.probabilistic_tracking, use_sf=True, use_dirs=True, backend="cpu")


@pytest.mark.parametrize("backend", list(ACCEL_BACKENDS))
def test_accel_probabilistic_tracking(backend):
    if not accel_backend_available(backend):
        pytest.skip(f"{backend} GPU tracker backend not available")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning,
        )
        track(tracker.probabilistic_tracking, backend=backend)
        track(tracker.probabilistic_tracking, use_dirs=True, backend=backend)
        track(tracker.probabilistic_tracking, use_sf=True, backend=backend)


def _synthetic_pmf(shape=(16, 16, 8)):
    sphere = HemiSphere.from_sphere(get_sphere(name="repulsion724"))
    mevals = np.array([[0.0015, 0.0003, 0.0003]] * 2)
    pmf = np.empty(shape + (len(sphere.vertices),))
    for i, j, k in np.ndindex(shape):
        x, y = i - (shape[0] - 1) / 2, j - (shape[1] - 1) / 2
        _, theta, phi = cart2sphere(-y, x, 0.3)
        angles = [(np.rad2deg(theta), np.rad2deg(phi)), (0, 0)]
        pmf[i, j, k] = multi_tensor_odf(sphere.vertices, mevals, angles, [70, 30])
    radius = np.hypot(
        *np.meshgrid(*[np.arange(n) - (n - 1) / 2 for n in shape[:2]], indexing="ij")
    )[..., None] * np.ones(shape[2])
    stop_map = np.clip(1.2 - np.abs(radius - shape[0] / 4) / 3, 0, 1)
    seeds = np.ascontiguousarray(np.argwhere(stop_map > 0.7)[::3] + 0.25)
    return pmf, stop_map, sphere, seeds


@pytest.mark.parametrize("return_all", [True, False])
@pytest.mark.parametrize("backend", list(ACCEL_BACKENDS))
def test_accel_matches_cpu(backend, return_all):
    if not accel_backend_available(backend):
        pytest.skip(f"{backend} GPU tracker backend not available")
    pmf, stop_map, sphere, seeds = _synthetic_pmf()

    params = {
        "sf": pmf,
        "sphere": sphere,
        "step_size": 0.5,
        "max_angle": 30,
        "max_len": 60,
        "random_seed": 3,
        "return_all": return_all,
        "save_seeds": True,
    }
    sc = ThresholdStoppingCriterion(stop_map, 0.5)

    cpu = list(
        tracker.probabilistic_tracking(seeds, sc, np.eye(4), backend="cpu", **params)
    )
    gpu = list(
        tracker.probabilistic_tracking(seeds, sc, np.eye(4), backend=backend, **params)
    )
    npt.assert_(len(cpu) > 0)
    npt.assert_(abs(len(gpu) - len(cpu)) <= 0.05 * len(cpu))

    # Same random numbers, but float32 fast math on the device and float64 on
    # the CPU: a step that lands on a threshold can flip and send a streamline
    # elsewhere, so only most streamlines are expected to match.
    gpu_by_seed = {tuple(seed): sl for sl, seed in gpu}
    same = [
        tuple(seed) in gpu_by_seed
        and len(sl) == len(gpu_by_seed[tuple(seed)])
        and np.allclose(sl, gpu_by_seed[tuple(seed)], atol=1e-4)
        for sl, seed in cpu
    ]
    npt.assert_(np.mean(same) >= 0.9, f"only {np.mean(same):.0%} of streamlines match")


def _synthetic_pam(npeaks=3):
    pmf, stop_map, sphere, seeds = _synthetic_pmf()
    pam = PeaksAndMetrics()
    pam.sphere = sphere
    pam.odf_vertices = sphere.vertices
    pam.peak_indices = np.full(pmf.shape[:3] + (npeaks,), -1, dtype=np.int32)
    pam.peak_values = np.zeros(pmf.shape[:3] + (npeaks,))
    for idx in np.ndindex(pmf.shape[:3]):
        _, values, indices = peak_directions(pmf[idx], sphere)
        n = min(npeaks, len(indices))
        pam.peak_indices[idx][:n] = indices[:n]
        pam.peak_values[idx][:n] = values[:n]
    # GPU backends must track the peaks, not this ODF
    pam.odf = np.ones_like(pmf)
    # SimplePeakGen has no pmf within one voxel of the border, where the
    # interpolation of the sparse SF of the GPU backends still has one: stop
    # the streamlines before they reach it.
    stop_map = stop_map.copy()
    stop_map[..., [0, -1]] = 0
    seeds = np.ascontiguousarray(np.argwhere(stop_map > 0.7)[::3] + 0.25)
    return pam, stop_map, sphere, seeds


def test_peaks_to_sf_matches_simple_peak_gen():
    pam, _, sphere, _ = _synthetic_pam()
    peak_gen = SimplePeakGen(
        pam.peak_indices, pam.peak_values, pam.odf_vertices, sphere
    )
    sf = tracker._peaks_to_sf(pam.peak_indices, pam.peak_values, len(sphere.vertices))
    sf_gen = SimplePmfGen(sf.astype(float), sphere)
    rng = np.random.default_rng(0)
    # interior points: SimplePeakGen returns 0 when a neighbour is outside
    points = rng.uniform(0, np.array(sf.shape[:3]) - 1.001, size=(50, 3))
    for point in points:
        point = np.ascontiguousarray(point)
        npt.assert_allclose(
            np.asarray(sf_gen.get_pmf(point)).copy(),
            np.asarray(peak_gen.get_pmf(point)).copy(),
            rtol=1e-6,
            atol=1e-7,
        )


@pytest.mark.parametrize("backend", list(ACCEL_BACKENDS))
def test_accel_pam_matches_cpu(backend):
    if not accel_backend_available(backend):
        pytest.skip(f"{backend} GPU tracker backend not available")
    pam, stop_map, sphere, seeds = _synthetic_pam()
    sc = ThresholdStoppingCriterion(stop_map, 0.5)
    params = {
        "pam": pam,
        "sphere": sphere,
        "step_size": 0.5,
        "max_angle": 30,
        "max_len": 60,
        "random_seed": 3,
        "save_seeds": True,
    }
    cpu = list(
        tracker.probabilistic_tracking(seeds, sc, np.eye(4), backend="cpu", **params)
    )
    gpu = list(
        tracker.probabilistic_tracking(seeds, sc, np.eye(4), backend=backend, **params)
    )
    npt.assert_(len(cpu) > 0)
    npt.assert_(abs(len(gpu) - len(cpu)) <= 0.05 * len(cpu))

    # one streamline per peak: key by seed and rank among its streamlines
    def by_seed_and_peak(result):
        keyed, rank = {}, {}
        for sl, seed in result:
            seed = tuple(seed)
            rank[seed] = rank.get(seed, -1) + 1
            keyed[seed, rank[seed]] = sl
        return keyed

    cpu_by_key, gpu_by_key = by_seed_and_peak(cpu), by_seed_and_peak(gpu)
    same = [
        key in gpu_by_key
        and len(sl) == len(gpu_by_key[key])
        and np.allclose(sl, gpu_by_key[key], atol=1e-4)
        for key, sl in cpu_by_key.items()
    ]
    npt.assert_(np.mean(same) >= 0.9, f"only {np.mean(same):.0%} of streamlines match")


def test_webgpu_requires_kernel_subgroup_size():
    if not webgpu.have_wgpu:
        pytest.skip("wgpu not installed")
    adapter = webgpu.wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if adapter is None or "subgroup" not in adapter.features:
        pytest.skip("no WebGPU adapter with subgroup support")
    device = adapter.request_device_sync(required_features=["subgroup"])
    size = webgpu._subgroup_size(device)
    npt.assert_equal(webgpu.webgpu_available(), size == webgpu.THREADS_PER_STREAMLINE)


def test_webgpu_rejects_software_adapter():
    if not webgpu.have_wgpu:
        pytest.skip("wgpu not installed")
    try:
        fallback = webgpu.wgpu.gpu.request_adapter_sync(force_fallback_adapter=True)
    except RuntimeError:
        fallback = None
    if fallback is not None:
        npt.assert_(webgpu._is_software_adapter(fallback))
    adapter = webgpu.wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if adapter is None:
        pytest.skip("no WebGPU adapter")
    if webgpu._is_software_adapter(adapter):
        npt.assert_(not webgpu.webgpu_available())


@pytest.mark.parametrize("backend", ["cpu", *ACCEL_BACKENDS])
def test_seed_order_independent(backend):
    if backend != "cpu" and not accel_backend_available(backend):
        pytest.skip(f"{backend} GPU tracker backend not available")
    pmf, stop_map, sphere, seeds = _synthetic_pmf()
    sc = ThresholdStoppingCriterion(stop_map, 0.5)
    params = {
        "sf": pmf,
        "sphere": sphere,
        "step_size": 0.5,
        "max_angle": 30,
        "max_len": 60,
        "random_seed": 5,
        "save_seeds": True,
        "backend": backend,
    }

    def by_seed(seed_positions, **kwargs):
        out = tracker.probabilistic_tracking(
            seed_positions, sc, np.eye(4), **params, **kwargs
        )
        return {tuple(seed): sl for sl, seed in out}

    ref = by_seed(seeds, chunk_size=1000, nbr_threads=1)
    other = by_seed(seeds[::-1].copy(), chunk_size=7, nbr_threads=2)
    npt.assert_equal(sorted(ref), sorted(other))
    for key, sl in ref.items():
        npt.assert_array_equal(sl, other[key])


def test_select_backend():
    stop_map = np.ones((4, 4, 4))
    threshold = ThresholdStoppingCriterion(stop_map, 0.5)
    binary = BinaryStoppingCriterion(stop_map)
    inputs = {"sf": np.ones((4, 4, 4, 5)), "sh": None, "pam": None}
    iso, aniso = np.ones(3), np.array([1.0, 1.0, 2.0])

    # explicit choices are kept
    for backend in ["cpu", *ACCEL_BACKENDS]:
        npt.assert_equal(
            tracker._select_backend(backend, sc=binary, voxel_size=iso, **inputs),
            backend,
        )
    # auto: GPU when available and the inputs allow it, else CPU
    expected = select_accel_backend() or "cpu"
    npt.assert_equal(
        tracker._select_backend("auto", sc=threshold, voxel_size=iso, **inputs),
        expected,
    )
    # a pam is tracked from its peaks: they are enough for the GPU
    peaks = PeaksAndMetrics()
    peaks.peak_indices = np.zeros((4, 4, 4, 1), dtype=np.int32)
    peaks.peak_values = np.ones((4, 4, 4, 1))
    npt.assert_equal(
        tracker._select_backend(
            "auto", sc=threshold, voxel_size=iso, sf=None, sh=None, pam=peaks
        ),
        expected,
    )
    odf_only = PeaksAndMetrics()
    odf_only.odf = np.ones((4, 4, 4, 5))
    for sc, voxel_size, kw in [
        (binary, iso, inputs),
        (threshold, aniso, inputs),
        (threshold, iso, {"sf": None, "sh": None, "pam": None}),
        (threshold, iso, {"sf": None, "sh": None, "pam": odf_only}),
    ]:
        npt.assert_equal(
            tracker._select_backend("auto", sc=sc, voxel_size=voxel_size, **kw),
            "cpu",
        )


@pytest.mark.parametrize("backend", list(ACCEL_BACKENDS))
def test_explicit_gpu_backend_rejects_unsupported_inputs(backend):
    pmf, stop_map, sphere, seeds = _synthetic_pmf()
    sc = BinaryStoppingCriterion(stop_map > 0.5)
    with pytest.raises(ValueError, match="GPU backends cannot be used"):
        tracker.probabilistic_tracking(
            seeds, sc, np.eye(4), sf=pmf, sphere=sphere, backend=backend
        )


def test_resolve_is_symmetric():
    npt.assert_equal(
        tracker._resolve_is_symmetric(None, sh=None, full_basis=None), True
    )
    npt.assert_equal(
        tracker._resolve_is_symmetric(
            None, sh=np.zeros((1, 1, 1, 15)), full_basis=None
        ),
        True,
    )
    npt.assert_equal(
        tracker._resolve_is_symmetric(
            None, sh=np.zeros((1, 1, 1, 25)), full_basis=None
        ),
        False,
    )
    npt.assert_equal(
        tracker._resolve_is_symmetric(
            True, sh=np.zeros((1, 1, 1, 25)), full_basis=None
        ),
        True,
    )


def test_ptt_tracking():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning,
        )
        track(tracker.ptt_tracking)
        track(tracker.ptt_tracking, use_sf=True, use_dirs=True)


def test_tracking_error():
    sh = np.array([(64, 61, 57)])
    seeds = [np.array([0.0, 0.0, 0.0], "float"), np.array([1.0, 2.0, 3.0], "float")]
    mask = np.ones((10, 10, 10))
    sc = BinaryStoppingCriterion(mask)

    npt.assert_raises(
        ValueError, tracker.deterministic_tracking, seeds, sc, np.eye(4), sf=sh, sh=sh
    )
    npt.assert_raises(ValueError, tracker.deterministic_tracking, seeds, sc, np.eye(4))
    npt.assert_raises(
        ValueError,
        tracker.deterministic_tracking,
        seeds,
        sc,
        np.eye(4),
        pam=sh,
    )
    npt.assert_raises(
        ValueError, tracker.deterministic_tracking, seeds, sc, np.eye(4), sf=sh
    )
    npt.assert_raises(
        ValueError,
        tracker.deterministic_tracking,
        seeds,
        sc,
        np.eye(4),
        sh=sh,
        seed_directions=1,
    )
    npt.assert_raises(
        ValueError,
        tracker.deterministic_tracking,
        seeds,
        sc,
        np.eye(4),
        sh=sh,
        seed_directions=[1],
    )
    npt.assert_raises(
        ValueError,
        tracker.deterministic_tracking,
        seeds,
        sc,
        np.eye(4),
        sf=sh,
        seed_directions=[1],
    )


def test_eudx_tracking():
    """Test the eudx_tracking function with PeaksAndMetrics."""
    sphere = HemiSphere.from_sphere(unit_octahedron)

    # A simple image with three possible configurations, a vertical tract,
    # a horizontal tract and a crossing
    peaks_values_lookup = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
    peaks_indices_lookup = np.array([[-1, -1], [0, -1], [1, -1], [0, 1]])
    # EuDX needs at least 3 slices on each axis to work
    simple_image = np.zeros([5, 6, 3], dtype=int)
    simple_image[:, :, 1] = np.array(
        [
            [0, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 3, 2, 2, 2, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
        ]
    )

    pam = PeaksAndMetrics()
    pam.sphere = sphere
    pam.peak_values = peaks_values_lookup[simple_image]
    pam.peak_indices = peaks_indices_lookup[simple_image]
    pam.odf_vertices = sphere.vertices
    pam.ang_thr = 90

    mask = (simple_image >= 0).astype(float)
    sc = ThresholdStoppingCriterion(mask, 0.5)
    seeds = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, 4.0, 1.0],
            [1.0, 3.0, 1.0],
        ]
    )

    # Test default parallel tracking
    streamlines = list(
        tracker.eudx_tracking(
            seeds,
            sc,
            np.eye(4),
            pam=pam,
            sphere=sphere,
            step_size=1.0,
            max_angle=90,
            pmf_threshold=0.01,
            min_len=0,
            return_all=True,
        )
    )
    npt.assert_equal(len(streamlines), len(seeds))

    # Test with explicit thread count
    streamlines_explicit = list(
        tracker.eudx_tracking(
            seeds,
            sc,
            np.eye(4),
            pam=pam,
            sphere=sphere,
            step_size=1.0,
            max_angle=90,
            pmf_threshold=0.01,
            min_len=0,
            return_all=True,
            nbr_threads=2,
        )
    )
    npt.assert_equal(len(streamlines_explicit), len(seeds))
