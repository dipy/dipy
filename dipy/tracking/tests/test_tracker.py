import warnings

import nibabel as nib
import numpy as np
import numpy.testing as npt

from dipy.core.sphere import HemiSphere, unit_octahedron
from dipy.data import get_fnames, get_sphere
from dipy.direction.peaks import PeaksAndMetrics
from dipy.reconst.shm import descoteaux07_legacy_msg, sh_to_sf
from dipy.tracking import tracker
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
    sc = BinaryStoppingCriterion(mask)
    affine = nib.load(fnames[25]).affine
    seed_mask = np.ones(mask.shape)
    seeds = random_seeds_from_mask(
        seed_mask, affine, seeds_count=100, seed_count_per_voxel=False
    )
    directions = np.random.random(seeds.shape)
    directions = np.array([v / np.linalg.norm(v) for v in directions])

    use_sf = kwargs.get("use_sf", False)
    use_directions = kwargs.get("use_dirs", False)

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
        track(tracker.probabilistic_tracking, use_dirs=True)
        track(tracker.probabilistic_tracking, use_sf=True, use_dirs=True)


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
    # peaks now supported but requires a valid PeaksAndMetrics object
    npt.assert_raises(
        ValueError,
        tracker.deterministic_tracking,
        seeds,
        sc,
        np.eye(4),
        peaks=sh,
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
