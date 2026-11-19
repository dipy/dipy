import warnings

import nibabel as nib
import numpy as np
import numpy.testing as npt
import pytest

from dipy.core.geometry import cart2sphere
from dipy.core.sphere import HemiSphere, unit_octahedron
from dipy.data import get_fnames, get_sphere
from dipy.direction.peaks import PeaksAndMetrics
from dipy.reconst.shm import descoteaux07_legacy_msg, sh_to_sf
from dipy.sims.voxel import multi_tensor_odf
from dipy.tracking import tracker
from dipy.tracking.accel import (
    ACCEL_BACKENDS,
    accel_backend_available,
    select_accel_backend,
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
def test_accel_matches_cpu_philox(backend, return_all):
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
        "rng": "philox",
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


@pytest.mark.parametrize("backend", list(ACCEL_BACKENDS))
def test_accel_pam_odf_sphere(backend):
    if not accel_backend_available(backend):
        pytest.skip(f"{backend} GPU tracker backend not available")
    pmf, stop_map, sphere, seeds = _synthetic_pmf()
    pam = PeaksAndMetrics()
    pam.odf = pmf
    pam.sphere = sphere
    sc = ThresholdStoppingCriterion(stop_map, 0.5)
    # the default sphere has another number of vertices: pam.sphere is used
    streamlines = list(
        tracker.probabilistic_tracking(
            seeds, sc, np.eye(4), pam=pam, random_seed=1, backend=backend
        )
    )
    npt.assert_(len(streamlines) > 0)


def test_cpu_pcg_seed_order_independent():
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
        "backend": "cpu",
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
            tracker._select_backend(
                backend, sc=binary, voxel_size=iso, rng=None, **inputs
            ),
            backend,
        )
    # auto: GPU when available and the inputs allow it, else CPU
    expected = select_accel_backend() or "cpu"
    npt.assert_equal(
        tracker._select_backend(
            "auto", sc=threshold, voxel_size=iso, rng=None, **inputs
        ),
        expected,
    )
    for sc, voxel_size, rng, kw in [
        (binary, iso, None, inputs),
        (threshold, aniso, None, inputs),
        (threshold, iso, "pcg", inputs),
        (threshold, iso, None, {"sf": None, "sh": None, "pam": None}),
    ]:
        npt.assert_equal(
            tracker._select_backend(
                "auto", sc=sc, voxel_size=voxel_size, rng=rng, **kw
            ),
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
