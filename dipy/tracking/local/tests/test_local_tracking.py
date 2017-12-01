
import nibabel as nib
import numpy as np
import numpy.testing as npt

from dipy.core.sphere import HemiSphere, unit_octahedron
from dipy.data import get_data
from dipy.direction import (DeterministicMaximumDirectionGetter,
                            PeaksAndMetrics,
                            ProbabilisticDirectionGetter)
from dipy.tracking.local import (ActTissueClassifier,
                                 BinaryTissueClassifier,
                                 DirectionGetter,
                                 LocalTracking,
                                 ThresholdTissueClassifier,
                                 TissueClassifier)
from dipy.tracking.local.interpolation import trilinear_interpolate4d
from dipy.tracking.local.localtracking import TissueTypes


def test_stop_conditions():
    """This tests that the Local Tracker behaves as expected for the
    following tissue types.
    """
    # TissueTypes.TRACKPOINT = 1
    # TissueTypes.ENDPOINT = 2
    # TissueTypes.INVALIDPOINT = 0
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
    endpoint_mask = tissue == TissueTypes.ENDPOINT
    invalidpoint_mask = tissue == TissueTypes.INVALIDPOINT
    tc = ActTissueClassifier(endpoint_mask, invalidpoint_mask)
    dg = ProbabilisticDirectionGetter.from_pmf(pmf, 60, sphere)

    streamlines_not_all = LocalTracking(direction_getter=dg,
                                        tissue_classifier=tc,
                                        seeds=seeds,
                                        affine=np.eye(4),
                                        step_size=1.,
                                        return_all=False)
    streamlines_all = LocalTracking(direction_getter=dg,
                                    tissue_classifier=tc,
                                    seeds=seeds,
                                    affine=np.eye(4),
                                    step_size=1.,
                                    return_all=True)

    streamlines_not_all = iter(streamlines_not_all)  # valid streamlines only
    streamlines_all = iter(streamlines_all)  # all streamlines

    # Check that the first streamline stops at 0 and 3 (ENDPOINT)
    y = 0
    sl = next(streamlines_not_all)
    npt.assert_equal(sl[0], [0, y, 0])
    npt.assert_equal(sl[-1], [0, y, 3])
    npt.assert_equal(len(sl), 4)

    sl = next(streamlines_all)
    npt.assert_equal(sl[0], [0, y, 0])
    npt.assert_equal(sl[-1], [0, y, 3])
    npt.assert_equal(len(sl), 4)

    # Check that the first streamline stops at 0 and 4 (ENDPOINT)
    y = 1
    sl = next(streamlines_not_all)
    npt.assert_equal(sl[0], [0, y, 0])
    npt.assert_equal(sl[-1], [0, y, 4])
    npt.assert_equal(len(sl), 5)

    sl = next(streamlines_all)
    npt.assert_equal(sl[0], [0, y, 0])
    npt.assert_equal(sl[-1], [0, y, 4])
    npt.assert_equal(len(sl), 5)

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
    # The streamline stops at 0 (INVALIDPOINT) and 4 (ENDPOINT)
    y = 4
    sl = next(streamlines_all)
    npt.assert_equal(sl[0], [0, y, 0])
    npt.assert_equal(sl[-1], [0, y, 4])
    npt.assert_equal(len(sl), 5)

    # The streamline stops at 0 (INVALIDPOINT) and 4 (INVALIDPOINT)
    y = 5
    sl = next(streamlines_all)
    npt.assert_equal(sl[0], [0, y, 0])
    npt.assert_equal(sl[-1], [0, y, 3])
    npt.assert_equal(len(sl), 4)

    # The last streamline should contain only one point, the seed point,
    # because no valid inital direction was returned.
    y = 6
    sl = next(streamlines_all)
    npt.assert_equal(sl[0], seeds[y])
    npt.assert_equal(sl[-1], seeds[y])
    npt.assert_equal(len(sl), 1)


def test_trilinear_interpolate():

    a, b, c = np.random.random(3)

    def linear_function(x, y, z):
        return a * x + b * y + c * z

    N = 6
    x, y, z = np.mgrid[:N, :N, :N]
    data = np.empty((N, N, N, 2))
    data[..., 0] = linear_function(x, y, z)
    data[..., 1] = 99.

    # Use a point not near the edges
    point = np.array([2.1, 4.8, 3.3])
    out = trilinear_interpolate4d(data, point)
    expected = [linear_function(*point), 99.]
    npt.assert_array_almost_equal(out, expected)

    # Pass in out ourselves
    out[:] = -1
    trilinear_interpolate4d(data, point, out)
    npt.assert_array_almost_equal(out, expected)

    # use a point close to an edge
    point = np.array([-.1, -.1, -.1])
    expected = [0., 99.]
    out = trilinear_interpolate4d(data, point)
    npt.assert_array_almost_equal(out, expected)

    # different edge
    point = np.array([2.4, 5.4, 3.3])
    # On the edge 5.4 get treated as the max y value, 5.
    expected = [linear_function(point[0], 5., point[2]), 99.]
    out = trilinear_interpolate4d(data, point)
    npt.assert_array_almost_equal(out, expected)

    # Test index errors
    point = np.array([2.4, 5.5, 3.3])
    npt.assert_raises(IndexError, trilinear_interpolate4d, data, point)
    point = np.array([2.4, -1., 3.3])
    npt.assert_raises(IndexError, trilinear_interpolate4d, data, point)


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
    tc = ThresholdTissueClassifier(mask, .5)

    dg = ProbabilisticDirectionGetter.from_pmf(pmf, 90, sphere, pmf_threshold=0.1)
    streamlines = LocalTracking(dg, tc, seeds, np.eye(4), 1.)

    expected = [np.array([[0., 1., 0.],
                          [1., 1., 0.],
                          [2., 1., 0.],
                          [2., 2., 0.],
                          [2., 3., 0.],
                          [2., 4., 0.],
                          [2., 5., 0.]]),
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
    streamlines = LocalTracking(dg, tc, seeds, np.eye(4), 1.)

    for sl in streamlines:
        npt.assert_(np.allclose(sl, expected[1]))

    # The first path is not possible if pmf_threshold > 0.4
    dg = ProbabilisticDirectionGetter.from_pmf(pmf, 90, sphere,
                                               pmf_threshold=0.5)
    streamlines = LocalTracking(dg, tc, seeds, np.eye(4), 1.)

    for sl in streamlines:
        npt.assert_(np.allclose(sl, expected[1]))


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
    tc = ThresholdTissueClassifier(mask, .5)

    dg = DeterministicMaximumDirectionGetter.from_pmf(pmf, 90, sphere,
                                                      pmf_threshold=0.1)
    streamlines = LocalTracking(dg, tc, seeds, np.eye(4), 1.)

    expected = [np.array([[0., 1., 0.],
                          [1., 1., 0.],
                          [2., 1., 0.],
                          [2., 2., 0.],
                          [2., 3., 0.],
                          [2., 4., 0.],
                          [2., 5., 0.]]),
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
    streamlines = LocalTracking(dg, tc, seeds, np.eye(4), 1.)

    for sl in streamlines:
        npt.assert_(np.allclose(sl, expected[1]))

    # Both path are not possible if 90 degree turns are exclude and
    # if pmf_threhold is larger than 0.4. Streamlines should stop at
    # the crossing

    dg = DeterministicMaximumDirectionGetter.from_pmf(pmf, 80, sphere,
                                                      pmf_threshold=0.5)
    streamlines = LocalTracking(dg, tc, seeds, np.eye(4), 1.)

    for sl in streamlines:
        npt.assert_(np.allclose(sl, expected[2]))


def test_peak_direction_tracker():
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
    # PeaksAndMetricsDirectionGetter needs at 3 slices on each axis to work
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
    tc = ThresholdTissueClassifier(mask, 0.5)
    seeds = [np.array([1., 1., 1.]),
             np.array([2., 4., 1.]),
             np.array([1., 3., 1.]),
             np.array([4., 4., 1.])]

    streamlines = LocalTracking(dg, tc, seeds, np.eye(4), 1.)

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

    expected = [np.array([[0., 1., 0.],
                          [1., 1., 0.],
                          [2., 1., 0.],
                          [3., 1., 0.],
                          [4., 1., 0.]]),
                np.array([[2., 0., 0.],
                          [2., 1., 0.],
                          [2., 2., 0.],
                          [2., 3., 0.],
                          [2., 4., 0.],
                          [2., 5., 0.]])]

    mask = (simple_image > 0).astype(float)
    tc = BinaryTissueClassifier(mask)

    dg = DeterministicMaximumDirectionGetter.from_pmf(pmf, 60, sphere,
                                                      pmf_threshold=0.1)
    streamlines = LocalTracking(dg, tc, seeds, np.eye(4), 1.)

    # TST- bad affine wrong shape
    bad_affine = np.eye(3)
    npt.assert_raises(ValueError, LocalTracking, dg, tc, seeds, bad_affine, 1.)

    # TST - bad affine with shearing
    bad_affine = np.eye(4)
    bad_affine[0, 1] = 1.
    npt.assert_raises(ValueError, LocalTracking, dg, tc, seeds, bad_affine, 1.)

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
    # TST - in vivo affine exemple
    # Sometimes data have affines with tiny shear components.
    # For example, the small_101D data-set has some of that:
    fdata, _, _ = get_data('small_101D')
    a6 = nib.load(fdata).affine

    for affine in [a0, a1, a2, a3, a4, a5, a6]:
        lin = affine[:3, :3]
        offset = affine[:3, 3]
        seeds_trans = [np.dot(lin, s) + offset for s in seeds]

        # We compute the voxel size to ajust the step size to one voxel
        voxel_size = np.mean(np.sqrt(np.dot(lin, lin).diagonal()))

        streamlines = LocalTracking(direction_getter=dg,
                                    tissue_classifier=tc,
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


if __name__ == "__main__":
    npt.run_module_suite()
