import warnings

import nibabel as nib
import numpy as np
import numpy.testing as npt

from dipy.core.gradients import gradient_table
from dipy.core.sphere import HemiSphere, unit_octahedron
from dipy.data import get_fnames, get_sphere
from dipy.direction import (BootDirectionGetter,
                            ClosestPeakDirectionGetter,
                            DeterministicMaximumDirectionGetter,
                            PeaksAndMetrics,
                            ProbabilisticDirectionGetter)
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.reconst.shm import descoteaux07_legacy_msg
from dipy.tracking.local_tracking import (LocalTracking,
                                          ParticleFilteringTracking)
from dipy.tracking.streamline import Streamlines
from dipy.tracking.stopping_criterion import (ActStoppingCriterion,
                                              BinaryStoppingCriterion,
                                              ThresholdStoppingCriterion,
                                              StreamlineStatus)
from dipy.tracking.utils import seeds_from_mask
from dipy.sims.voxel import single_tensor, multi_tensor


def test_stop_conditions():
    """This tests that the Local Tracker behaves as expected for the
    following tissue types.
    """
    # StreamlineStatus.TRACKPOINT = 1
    # StreamlineStatus.ENDPOINT = 2
    # StreamlineStatus.INVALIDPOINT = 0
    tissue = np.array([[2, 1, 1, 2, 1],
                       [2, 2, 1, 1, 2],
                       [1, 1, 1, 1, 1],
                       [1, 1, 1, 2, 2],
                       [0, 1, 1, 1, 2],
                       [0, 1, 1, 0, 2],
                       [1, 0, 1, 1, 1],
                       [2, 1, 2, 0, 0]])
    tissue = tissue[None]

    sphere = HemiSphere.from_sphere(unit_octahedron)
    pmf_lookup = np.array([[0., 0., 0., ],
                           [0., 0., 1.]])
    pmf = pmf_lookup[(tissue > 0).astype("int")]

    # Create a seeds along
    x = np.array([0., 0, 0, 0, 0, 0, 0, 0])
    y = np.array([0., 1, 2, 3, 4, 5, 6, 7])
    z = np.array([1., 1, 1, 0, 1, 1, 1, 1])
    seeds = np.column_stack([x, y, z])

    # Set up tracking
    endpoint_mask = tissue == StreamlineStatus.ENDPOINT
    invalidpoint_mask = tissue == StreamlineStatus.INVALIDPOINT
    sc = ActStoppingCriterion(endpoint_mask, invalidpoint_mask)
    dg = ProbabilisticDirectionGetter.from_pmf(pmf, 60, sphere)

    # valid streamlines only
    streamlines_generator = LocalTracking(direction_getter=dg,
                                          stopping_criterion=sc,
                                          seeds=seeds,
                                          affine=np.eye(4),
                                          step_size=1.,
                                          return_all=False)
    streamlines_not_all = iter(streamlines_generator)

    # all streamlines
    streamlines_all_generator = LocalTracking(direction_getter=dg,
                                              stopping_criterion=sc,
                                              seeds=seeds,
                                              affine=np.eye(4),
                                              step_size=1.,
                                              return_all=True)
    streamlines_all = iter(streamlines_all_generator)

    # Check that the first streamline stops at 1 and 2 (ENDPOINT)
    y = 0
    sl = next(streamlines_not_all)
    npt.assert_equal(sl[0], [0, y, 1])
    npt.assert_equal(sl[-1], [0, y, 2])
    npt.assert_equal(len(sl), 2)

    sl = next(streamlines_all)
    npt.assert_equal(sl[0], [0, y, 1])
    npt.assert_equal(sl[-1], [0, y, 2])
    npt.assert_equal(len(sl), 2)

    # Check that the next streamline stops at 1 and 3 (ENDPOINT)
    y = 1
    sl = next(streamlines_not_all)
    npt.assert_equal(sl[0], [0, y, 1])
    npt.assert_equal(sl[-1], [0, y, 3])
    npt.assert_equal(len(sl), 3)

    sl = next(streamlines_all)
    npt.assert_equal(sl[0], [0, y, 1])
    npt.assert_equal(sl[-1], [0, y, 3])
    npt.assert_equal(len(sl), 3)

    # This streamline should be the same as above. This row does not have
    # ENDPOINTs, but the streamline should stop at the edge and not include
    # OUTSIDEIMAGE points.
    y = 2
    sl = next(streamlines_not_all)
    npt.assert_equal(sl[0], [0, y, 0])
    npt.assert_equal(sl[-1], [0, y, 4])
    npt.assert_equal(len(sl), 5)

    sl = next(streamlines_all)
    npt.assert_equal(sl[0], [0, y, 0])
    npt.assert_equal(sl[-1], [0, y, 4])
    npt.assert_equal(len(sl), 5)

    # If we seed on the edge, the first (or last) point in the streamline
    # should be the seed.
    y = 3
    sl = next(streamlines_not_all)
    npt.assert_equal(sl[0], seeds[y])

    sl = next(streamlines_all)
    npt.assert_equal(sl[0], seeds[y])

    # The last 3 seeds should not produce streamlines,
    # INVALIDPOINT streamlines are rejected (return_all=False).
    npt.assert_equal(len(list(streamlines_not_all)), 0)

    # The last 3 seeds should produce invalid streamlines,
    # INVALIDPOINT streamlines are kept (return_all=True).
    # The streamline stops at 1 (INVALIDPOINT) and 3 (ENDPOINT)
    y = 4
    sl = next(streamlines_all)
    npt.assert_equal(sl[0], [0, y, 1])
    npt.assert_equal(sl[-1], [0, y, 3])
    npt.assert_equal(len(sl), 3)

    # The streamline stops at 0 (INVALIDPOINT) and 2 (INVALIDPOINT)
    y = 5
    sl = next(streamlines_all)
    npt.assert_equal(sl[0], [0, y, 1])
    npt.assert_equal(sl[-1], [0, y, 2])
    npt.assert_equal(len(sl), 2)

    # The streamline should contain only one point, the seed point,
    # because no valid inital direction was returned.
    y = 6
    sl = next(streamlines_all)
    npt.assert_equal(sl[0], seeds[y])
    npt.assert_equal(sl[-1], seeds[y])
    npt.assert_equal(len(sl), 1)

    # The streamline should contain only one point, the seed point,
    # because no valid neighboring voxel (ENDPOINT)
    y = 7
    sl = next(streamlines_all)
    npt.assert_equal(sl[0], seeds[y])
    npt.assert_equal(sl[-1], seeds[y])
    npt.assert_equal(len(sl), 1)


def test_save_seeds():
    tissue = np.array([[2, 1, 1, 2, 1],
                       [2, 2, 1, 1, 2],
                       [1, 1, 1, 1, 1],
                       [1, 1, 1, 2, 2],
                       [0, 1, 1, 1, 2],
                       [0, 1, 1, 0, 2],
                       [1, 0, 1, 1, 1]])
    tissue = tissue[None]

    sphere = HemiSphere.from_sphere(unit_octahedron)
    pmf_lookup = np.array([[0., 0., 0., ],
                           [0., 0., 1.]])
    pmf = pmf_lookup[(tissue > 0).astype("int")]

    # Create a seeds along
    x = np.array([0., 0, 0, 0, 0, 0, 0])
    y = np.array([0., 1, 2, 3, 4, 5, 6])
    z = np.array([1., 1, 1, 0, 1, 1, 1])
    seeds = np.column_stack([x, y, z])

    # Set up tracking
    endpoint_mask = tissue == StreamlineStatus.ENDPOINT
    invalidpoint_mask = tissue == StreamlineStatus.INVALIDPOINT
    sc = ActStoppingCriterion(endpoint_mask, invalidpoint_mask)
    dg = ProbabilisticDirectionGetter.from_pmf(pmf, 60, sphere)

    # valid streamlines only
    streamlines_generator = LocalTracking(direction_getter=dg,
                                          stopping_criterion=sc,
                                          seeds=seeds,
                                          affine=np.eye(4),
                                          step_size=1.,
                                          return_all=False,
                                          save_seeds=True)

    streamlines_not_all = iter(streamlines_generator)
    # Verifiy that seeds are returned by the LocalTracker
    _, seed = next(streamlines_not_all)
    npt.assert_equal(seed, seeds[0])
    _, seed = next(streamlines_not_all)
    npt.assert_equal(seed, seeds[1])
    # Verifiy that seeds are returned by the PFTTracker also
    pft_streamlines = ParticleFilteringTracking(direction_getter=dg,
                                                stopping_criterion=sc,
                                                seeds=seeds,
                                                affine=np.eye(4),
                                                step_size=1.,
                                                max_cross=1,
                                                return_all=False,
                                                save_seeds=True)
    streamlines = iter(pft_streamlines)
    _, seed = next(streamlines)
    npt.assert_equal(seed, seeds[0])
    _, seed = next(streamlines)
    npt.assert_equal(seed, seeds[1])


def test_probabilistic_odf_weighted_tracker():
    """This tests that the Probabalistic Direction Getter plays nice
    LocalTracking and produces reasonable streamlines in a simple example.
    """
    sphere = HemiSphere.from_sphere(unit_octahedron)

    # A simple image with three possible configurations, a vertical tract,
    # a horizontal tract and a crossing
    pmf_lookup = np.array([[0., 0., 1.],
                           [1., 0., 0.],
                           [0., 1., 0.],
                           [.6, .4, 0.]])
    simple_image = np.array([[0, 1, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 3, 2, 2, 2, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             ])

    simple_image = simple_image[..., None]
    pmf = pmf_lookup[simple_image]

    seeds = [np.array([1., 1., 0.])] * 30

    mask = (simple_image > 0).astype(float)
    sc = ThresholdStoppingCriterion(mask, .5)

    dg = ProbabilisticDirectionGetter.from_pmf(pmf, 90, sphere,
                                               pmf_threshold=0.1)
    streamlines = LocalTracking(dg, sc, seeds, np.eye(4), 1.)

    expected = [np.array([[0., 1., 0.],
                          [1., 1., 0.],
                          [2., 1., 0.],
                          [2., 2., 0.],
                          [2., 3., 0.],
                          [2., 4., 0.]]),
                np.array([[0., 1., 0.],
                          [1., 1., 0.],
                          [2., 1., 0.],
                          [3., 1., 0.],
                          [4., 1., 0.]])]

    def allclose(x, y):
        return x.shape == y.shape and np.allclose(x, y)

    path = [False, False]
    for sl in streamlines:
        if allclose(sl, expected[0]):
            path[0] = True
        elif allclose(sl, expected[1]):
            path[1] = True
        else:
            raise AssertionError()
    npt.assert_(all(path))

    # The first path is not possible if 90 degree turns are excluded
    dg = ProbabilisticDirectionGetter.from_pmf(pmf, 80, sphere,
                                               pmf_threshold=0.1)
    streamlines = LocalTracking(dg, sc, seeds, np.eye(4), 1.)

    for sl in streamlines:
        npt.assert_(np.allclose(sl, expected[1]))

    # The first path is not possible if pmf_threshold > 0.67
    # 0.4/0.6 < 2/3, multiplying the pmf should not change the ratio
    dg = ProbabilisticDirectionGetter.from_pmf(10 * pmf, 90, sphere,
                                               pmf_threshold=0.67)
    streamlines = LocalTracking(dg, sc, seeds, np.eye(4), 1.)

    for sl in streamlines:
        npt.assert_(np.allclose(sl, expected[1]))

    # Test non WM seed position
    seeds = [[0, 0, 0], [5, 5, 5]]
    streamlines = LocalTracking(dg, sc, seeds, np.eye(4), 0.2, max_cross=1,
                                return_all=True)
    streamlines = Streamlines(streamlines)
    npt.assert_(len(streamlines[0]) == 1)  # INVALIDPOINT
    npt.assert_(len(streamlines[1]) == 1)  # OUTSIDEIMAGE

    # Test that all points are within the image volume
    seeds = seeds_from_mask(np.ones(mask.shape), np.eye(4), density=2)
    streamline_generator = LocalTracking(dg, sc, seeds, np.eye(4), 0.5,
                                         return_all=True)
    streamlines = Streamlines(streamline_generator)
    for s in streamlines:
        npt.assert_(np.all((s + 0.5).astype(int) >= 0))
        npt.assert_(np.all((s + 0.5).astype(int) < mask.shape))
    # Test that the number of streamline return with return_all=True equal the
    # number of seeds places

    npt.assert_(np.array([len(streamlines) == len(seeds)]))

    # Test reproducibility
    tracking_1 = Streamlines(LocalTracking(dg, sc, seeds, np.eye(4),
                                           0.5,
                                           random_seed=0))._data
    tracking_2 = Streamlines(LocalTracking(dg, sc, seeds, np.eye(4),
                                           0.5,
                                           random_seed=0))._data
    npt.assert_equal(tracking_1, tracking_2)


def test_particle_filtering_tractography():
    """This tests that the ParticleFilteringTracking produces
    more streamlines connecting the gray matter than LocalTracking.
    """
    sphere = get_sphere('repulsion100')
    step_size = 0.2

    # Simple tissue masks
    simple_wm = np.array([[0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0],
                          [0, 1, 1, 1, 0, 0],
                          [0, 1, 1, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0]])
    simple_wm = np.dstack([np.zeros(simple_wm.shape),
                           simple_wm,
                           simple_wm,
                           simple_wm,
                           np.zeros(simple_wm.shape)])
    simple_gm = np.array([[1, 1, 0, 0, 0, 0],
                          [1, 1, 0, 0, 0, 0],
                          [0, 1, 0, 0, 1, 0],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0]])
    simple_gm = np.dstack([np.zeros(simple_gm.shape),
                           simple_gm,
                           simple_gm,
                           simple_gm,
                           np.zeros(simple_gm.shape)])
    simple_csf = np.ones(simple_wm.shape) - simple_wm - simple_gm

    sc = ActStoppingCriterion.from_pve(simple_wm, simple_gm, simple_csf)
    seeds = seeds_from_mask(simple_wm, np.eye(4), density=2)

    # Random pmf in every voxel
    shape_img = list(simple_wm.shape)
    shape_img.extend([sphere.vertices.shape[0]])
    np.random.seed(0)  # Random number generator initialization
    pmf = np.random.random(shape_img)

    # Test that PFT recover equal or more streamlines than localTracking
    dg = ProbabilisticDirectionGetter.from_pmf(pmf, 60, sphere)
    local_streamlines_generator = LocalTracking(dg, sc, seeds, np.eye(4),
                                                step_size, max_cross=1,
                                                return_all=False)
    local_streamlines = Streamlines(local_streamlines_generator)

    pft_streamlines_generator = ParticleFilteringTracking(
        dg, sc, seeds, np.eye(4), step_size, max_cross=1, return_all=False,
        pft_back_tracking_dist=1, pft_front_tracking_dist=0.5)
    pft_streamlines = Streamlines(pft_streamlines_generator)

    npt.assert_(np.array([len(pft_streamlines) > 0]))
    npt.assert_(np.array([len(pft_streamlines) >= len(local_streamlines)]))

    # Test that all points are equally spaced
    for l in [1, 2, 5, 10, 100]:
        pft_streamlines = ParticleFilteringTracking(dg, sc, seeds, np.eye(4),
                                                    step_size, max_cross=1,
                                                    return_all=True, maxlen=l)
        for s in pft_streamlines:
            for i in range(len(s) - 1):
                npt.assert_almost_equal(np.linalg.norm(s[i] - s[i + 1]),
                                        step_size)
    # Test that all points are within the image volume
    seeds = seeds_from_mask(np.ones(simple_wm.shape), np.eye(4), density=1)
    pft_streamlines_generator = ParticleFilteringTracking(
        dg, sc, seeds, np.eye(4), step_size, max_cross=1, return_all=True)
    pft_streamlines = Streamlines(pft_streamlines_generator)

    for s in pft_streamlines:
        npt.assert_(np.all((s + 0.5).astype(int) >= 0))
        npt.assert_(np.all((s + 0.5).astype(int) < simple_wm.shape))

    # Test that the number of streamline return with return_all=True equal the
    # number of seeds places
    npt.assert_(np.array([len(pft_streamlines) == len(seeds)]))

    # Test non WM seed position
    seeds = [[0, 5, 4], [0, 0, 1], [50, 50, 50]]
    pft_streamlines_generator = ParticleFilteringTracking(
        dg, sc, seeds, np.eye(4), step_size, max_cross=1, return_all=True)
    pft_streamlines = Streamlines(pft_streamlines_generator)

    npt.assert_equal(len(pft_streamlines[0]), 3)  # INVALIDPOINT
    npt.assert_equal(len(pft_streamlines[1]), 3)  # ENDPOINT
    npt.assert_equal(len(pft_streamlines[2]), 1)  # OUTSIDEIMAGE

    # Test with wrong StoppingCriterion type
    sc_bin = BinaryStoppingCriterion(simple_wm)
    npt.assert_raises(ValueError,
                      lambda: ParticleFilteringTracking(dg, sc_bin, seeds,
                                                        np.eye(4), step_size))
    # Test with invalid back/front tracking distances
    npt.assert_raises(
        ValueError,
        lambda: ParticleFilteringTracking(dg, sc, seeds, np.eye(4), step_size,
                                          pft_back_tracking_dist=0,
                                          pft_front_tracking_dist=0))
    npt.assert_raises(
        ValueError,
        lambda: ParticleFilteringTracking(dg, sc, seeds, np.eye(4), step_size,
                                          pft_back_tracking_dist=-1))
    npt.assert_raises(
        ValueError,
        lambda: ParticleFilteringTracking(dg, sc, seeds, np.eye(4), step_size,
                                          pft_back_tracking_dist=0,
                                          pft_front_tracking_dist=-2))

    # Test with invalid affine shape
    npt.assert_raises(
        ValueError,
        lambda: ParticleFilteringTracking(dg, sc, seeds, np.eye(3), step_size))

    # Test with invalid maxlen
    npt.assert_raises(
        ValueError,
        lambda: ParticleFilteringTracking(dg, sc, seeds, np.eye(4), step_size,
                                          maxlen=0))
    npt.assert_raises(
        ValueError,
        lambda: ParticleFilteringTracking(dg, sc, seeds, np.eye(4), step_size,
                                          maxlen=-1))

    # Test with invalid particle count
    npt.assert_raises(
        ValueError,
        lambda: ParticleFilteringTracking(dg, sc, seeds, np.eye(4), step_size,
                                          particle_count=0))
    npt.assert_raises(
        ValueError,
        lambda: ParticleFilteringTracking(dg, sc, seeds, np.eye(4), step_size,
                                          particle_count=-1))

    # Test reproducibility
    tracking1 = Streamlines(ParticleFilteringTracking(dg, sc, seeds, np.eye(4),
                                                      step_size,
                                                      random_seed=0))._data
    tracking2 = Streamlines(ParticleFilteringTracking(dg, sc, seeds, np.eye(4),
                                                      step_size,
                                                      random_seed=0))._data
    npt.assert_equal(tracking1, tracking2)


def test_maximum_deterministic_tracker():
    """This tests that the Maximum Deterministic Direction Getter plays nice
    LocalTracking and produces reasonable streamlines in a simple example.
    """
    sphere = HemiSphere.from_sphere(unit_octahedron)

    # A simple image with three possible configurations, a vertical tract,
    # a horizontal tract and a crossing
    pmf_lookup = np.array([[0., 0., 1.],
                           [1., 0., 0.],
                           [0., 1., 0.],
                           [.4, .6, 0.]])
    simple_image = np.array([[0, 1, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 3, 2, 2, 2, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             ])

    simple_image = simple_image[..., None]
    pmf = pmf_lookup[simple_image]

    seeds = [np.array([1., 1., 0.])] * 30

    mask = (simple_image > 0).astype(float)
    sc = ThresholdStoppingCriterion(mask, .5)

    dg = DeterministicMaximumDirectionGetter.from_pmf(pmf, 90, sphere,
                                                      pmf_threshold=0.1)
    streamlines = LocalTracking(dg, sc, seeds, np.eye(4), 1.)

    expected = [np.array([[0., 1., 0.],
                          [1., 1., 0.],
                          [2., 1., 0.],
                          [2., 2., 0.],
                          [2., 3., 0.],
                          [2., 4., 0.]]),
                np.array([[0., 1., 0.],
                          [1., 1., 0.],
                          [2., 1., 0.],
                          [3., 1., 0.],
                          [4., 1., 0.]]),
                np.array([[0., 1., 0.],
                          [1., 1., 0.],
                          [2., 1., 0.]])]

    def allclose(x, y):
        return x.shape == y.shape and np.allclose(x, y)

    for sl in streamlines:
        if not allclose(sl, expected[0]):
            raise AssertionError()

    # The first path is not possible if 90 degree turns are excluded
    dg = DeterministicMaximumDirectionGetter.from_pmf(pmf, 80, sphere,
                                                      pmf_threshold=0.1)
    streamlines = LocalTracking(dg, sc, seeds, np.eye(4), 1.)

    for sl in streamlines:
        npt.assert_(np.allclose(sl, expected[1]))

    # Both path are not possible if 90 degree turns are exclude and
    # if pmf_threshold is larger than 0.67. Streamlines should stop at
    # the crossing.
    # 0.4/0.6 < 2/3, multiplying the pmf should not change the ratio
    dg = DeterministicMaximumDirectionGetter.from_pmf(10 * pmf, 80, sphere,
                                                      pmf_threshold=0.67)
    streamlines = LocalTracking(dg, sc, seeds, np.eye(4), 1.)

    for sl in streamlines:
        npt.assert_(np.allclose(sl, expected[2]))


def test_bootstap_peak_tracker():
    """This tests that the Bootstrat Peak Direction Getter plays nice
    LocalTracking and produces reasonable streamlines in a simple example.
    """
    sphere = get_sphere('repulsion100')

    # A simple image with three possible configurations, a vertical tract,
    # a horizontal tract and a crossing
    simple_image = np.array([[0, 1, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             [2, 3, 2, 2, 2, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             ])
    simple_image = simple_image[..., None]

    bvecs = sphere.vertices
    bvals = np.ones(len(bvecs)) * 1000
    bvecs = np.insert(bvecs, 0, np.array([0, 0, 0]), axis=0)
    bvals = np.insert(bvals, 0, 0)
    gtab = gradient_table(bvals, bvecs)
    angles = [(90, 90), (90, 0)]
    fracs = [50, 50]
    mevals = np.array([[1.5, 0.4, 0.4], [1.5, 0.4, 0.4]]) * 1e-3
    mevecs = [np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
              np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])]
    voxel1 = single_tensor(gtab, 1, mevals[0], mevecs[0], snr=None)
    voxel2 = single_tensor(gtab, 1, mevals[0], mevecs[1], snr=None)
    voxel3, _ = multi_tensor(gtab, mevals, fractions=fracs, angles=angles,
                             snr=None)
    data = np.tile(voxel3, [5, 6, 1, 1])
    data[simple_image == 1] = voxel1
    data[simple_image == 2] = voxel2

    response = (np.array(mevals[1]), 1)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=6)

    seeds = [np.array([0., 1., 0.]), np.array([2., 4., 0.])]

    sc = BinaryStoppingCriterion((simple_image > 0).astype(float))
    sphere = HemiSphere.from_sphere(get_sphere('symmetric724'))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        boot_dg = BootDirectionGetter.from_data(data, csd_model, 60,
                                                sphere=sphere)

    streamlines_generator = LocalTracking(boot_dg, sc, seeds, np.eye(4), 1.)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        streamlines = Streamlines(streamlines_generator)
    expected = [np.array([[0., 1., 0.],
                          [1., 1., 0.],
                          [2., 1., 0.],
                          [3., 1., 0.],
                          [4., 1., 0.]]),
                np.array([[2., 4., 0.],
                          [2., 3., 0.],
                          [2., 2., 0.],
                          [2., 1., 0.],
                          [2., 0., 0.],
                          ])]

    def allclose(x, y):
        return x.shape == y.shape and np.allclose(x, y, atol=0.5)

    if not allclose(streamlines[0], expected[0]):
        raise AssertionError()
    if not allclose(streamlines[1], expected[1]):
        raise AssertionError()


def test_closest_peak_tracker():
    """This tests that the Closest Peak Direction Getter plays nice
    LocalTracking and produces reasonable streamlines in a simple example.
    """
    sphere = HemiSphere.from_sphere(unit_octahedron)

    # A simple image with three possible configurations, a vertical tract,
    # a horizontal tract and a crossing
    pmf_lookup = np.array([[0., 0., 1.],
                           [1., 0., 0.],
                           [0., 1., 0.],
                           [.5, .5, 0.]])
    simple_image = np.array([[0, 1, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             [2, 3, 2, 2, 2, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             ])

    simple_image = simple_image[..., None]
    pmf = pmf_lookup[simple_image]

    seeds = [np.array([1., 1., 0.]), np.array([2., 4., 0.])]

    mask = (simple_image > 0).astype(float)
    sc = BinaryStoppingCriterion(mask)

    dg = ClosestPeakDirectionGetter.from_pmf(pmf, 90, sphere,
                                             pmf_threshold=0.1)
    streamlines = Streamlines(LocalTracking(dg, sc, seeds, np.eye(4), 1.))

    expected = [np.array([[0., 1., 0.],
                          [1., 1., 0.],
                          [2., 1., 0.],
                          [3., 1., 0.],
                          [4., 1., 0.]]),
                np.array([[2., 0., 0.],
                          [2., 1., 0.],
                          [2., 2., 0.],
                          [2., 3., 0.],
                          [2., 4., 0.]])]

    def allclose(x, y):
        return x.shape == y.shape and np.allclose(x, y)

    if not allclose(streamlines[0], expected[0]):
        raise AssertionError()
    if not allclose(streamlines[1], expected[1]):
        raise AssertionError()


def test_eudx_tracker():
    """This tests that the Peaks And Metrics Direction Getter plays nice
    LocalTracking and produces reasonable streamlines in a simple example.
    """
    sphere = HemiSphere.from_sphere(unit_octahedron)

    # A simple image with three possible configurations, a vertical tract,
    # a horizontal tract and a crossing
    peaks_values_lookup = np.array([[0., 0.],
                                    [1., 0.],
                                    [1., 0.],
                                    [0.5, 0.5]])
    peaks_indices_lookup = np.array([[-1, -1],
                                     [0, -1],
                                     [1, -1],
                                     [0,  1]])
    # EuDXDirectionGetter needs at 3 slices on each axis to work
    simple_image = np.zeros([5, 6, 3], dtype=int)
    simple_image[:, :, 1] = np.array([[0, 1, 0, 1, 0, 0],
                                      [0, 1, 0, 1, 0, 0],
                                      [0, 3, 2, 2, 2, 0],
                                      [0, 1, 0, 0, 0, 0],
                                      [0, 1, 0, 0, 0, 0],
                                      ])

    dg = PeaksAndMetrics()
    dg.sphere = sphere
    dg.peak_values = peaks_values_lookup[simple_image]
    dg.peak_indices = peaks_indices_lookup[simple_image]
    dg.ang_thr = 90

    mask = (simple_image >= 0).astype(float)
    sc = ThresholdStoppingCriterion(mask, 0.5)
    seeds = [np.array([1., 1., 1.]),
             np.array([2., 4., 1.]),
             np.array([1., 3., 1.]),
             np.array([4., 4., 1.])]

    streamlines = LocalTracking(dg, sc, seeds, np.eye(4), 1.)

    expected = [np.array([[0., 1., 1.],
                          [1., 1., 1.],
                          [2., 1., 1.],
                          [3., 1., 1.],
                          [4., 1., 1.]]),
                np.array([[2., 0., 1.],
                          [2., 1., 1.],
                          [2., 2., 1.],
                          [2., 3., 1.],
                          [2., 4., 1.],
                          [2., 5., 1.]]),
                np.array([[0., 3., 1.],
                          [1., 3., 1.],
                          [2., 3., 1.],
                          [2., 4., 1.],
                          [2., 5., 1.]]),
                np.array([[4., 4., 1.]])]

    for i, sl in enumerate(streamlines):
        npt.assert_(np.allclose(sl, expected[i]))


def test_affine_transformations():
    """This tests that the input affine is properly handled by
    LocalTracking and produces reasonable streamlines in a simple example.
    """
    sphere = HemiSphere.from_sphere(unit_octahedron)

    # A simple image with three possible configurations, a vertical tract,
    # a horizontal tract and a crossing
    pmf_lookup = np.array([[0., 0., 1.],
                           [1., 0., 0.],
                           [0., 1., 0.],
                           [.4, .6, 0.]])
    simple_image = np.array([[0, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 3, 2, 2, 2, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             ])

    simple_image = simple_image[..., None]
    pmf = pmf_lookup[simple_image]

    seeds = [np.array([1., 1., 0.]),
             np.array([2., 4., 0.])]

    expected = [np.array([[1., 1., 0.],
                          [2., 1., 0.],
                          [3., 1., 0.]]),
                np.array([[2., 1., 0.],
                          [2., 2., 0.],
                          [2., 3., 0.],
                          [2., 4., 0.]])]

    mask = (simple_image > 0).astype(float)
    sc = BinaryStoppingCriterion(mask)

    dg = DeterministicMaximumDirectionGetter.from_pmf(pmf, 60, sphere,
                                                      pmf_threshold=0.1)

    # TST- bad affine wrong shape
    bad_affine = np.eye(3)
    npt.assert_raises(ValueError, LocalTracking, dg, sc, seeds, bad_affine, 1.)

    # TST - bad affine with shearing
    bad_affine = np.eye(4)
    bad_affine[0, 1] = 1.
    npt.assert_raises(ValueError, LocalTracking, dg, sc, seeds, bad_affine, 1.)

    # TST - bad seeds
    bad_seeds = 1000
    npt.assert_raises(ValueError, LocalTracking, dg, sc, bad_seeds,
                      np.eye(4), 1.)

    # TST - identity
    a0 = np.eye(4)
    # TST - affines with positive/negative offsets
    a1 = np.eye(4)
    a1[:3, 3] = [1, 2, 3]
    a2 = np.eye(4)
    a2[:3, 3] = [-2, 0, -1]
    # TST - affine with scaling
    a3 = np.eye(4)
    a3[0, 0] = a3[1, 1] = a3[2, 2] = 8
    # TST - affine with axes inverting (negative value)
    a4 = np.eye(4)
    a4[1, 1] = a4[2, 2] = -1
    # TST - combined affines
    a5 = a1 + a2 + a3
    a5[3, 3] = 1
    # TST - in vivo affine example
    # Sometimes data have affines with tiny shear components.
    # For example, the small_101D data-set has some of that:
    fdata, _, _ = get_fnames('small_101D')
    a6 = nib.load(fdata).affine

    for affine in [a0, a1, a2, a3, a4, a5, a6]:
        lin = affine[:3, :3]
        offset = affine[:3, 3]
        seeds_trans = [np.dot(lin, s) + offset for s in seeds]

        # We compute the voxel size to adjust the step size to one voxel
        voxel_size = np.mean(np.sqrt(np.dot(lin, lin).diagonal()))

        streamlines = LocalTracking(direction_getter=dg,
                                    stopping_criterion=sc,
                                    seeds=seeds_trans,
                                    affine=affine,
                                    step_size=voxel_size,
                                    return_all=True)

        # We apply the inverse affine transformation to the generated
        # streamlines. It should be equals to the expected streamlines
        # (generated with the identity affine matrix).
        affine_inv = np.linalg.inv(affine)
        lin = affine_inv[:3, :3]
        offset = affine_inv[:3, 3]
        streamlines_inv = []
        for line in streamlines:
            streamlines_inv.append([np.dot(pts, lin) + offset for pts in line])

        npt.assert_equal(len(streamlines_inv[0]), len(expected[0]))
        npt.assert_(np.allclose(streamlines_inv[0], expected[0], atol=0.3))
        npt.assert_equal(len(streamlines_inv[1]), len(expected[1]))
        npt.assert_(np.allclose(streamlines_inv[1], expected[1], atol=0.3))
