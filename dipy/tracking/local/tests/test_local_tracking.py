import numpy as np
import numpy.testing as npt

from dipy.core.sphere import HemiSphere, unit_octahedron
from dipy.core.gradients import gradient_table
from dipy.tracking.local import (LocalTracking, ThresholdTissueClassifier,
                                 DirectionGetter, TissueClassifier)
from dipy.direction import (ProbabilisticDirectionGetter,
                            DeterministicMaximumDirectionGetter)
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

    class SimpleTissueClassifier(TissueClassifier):
        def check_point(self, point):
            p = np.round(point).astype(int)
            if any(p < 0) or any(p >= tissue.shape):
                return TissueTypes.OUTSIDEIMAGE
            return tissue[p[0], p[1], p[2]]

    class SimpleDirectionGetter(DirectionGetter):
        def initial_direction(self, point):
            # Test tracking along the rows (z direction)
            # of the tissue array above
            p = np.round(point).astype(int)
            if (any(p < 0) or
                any(p >= tissue.shape) or
                tissue[p[0], p[1], p[2]] == TissueTypes.INVALIDPOINT):
                return np.array([])
            return np.array([[0., 0., 1.]])

        def get_direction(self, p, d):
            # Always keep previous direction
            return 0

    # Create a seeds along
    x = np.array([0., 0, 0, 0, 0, 0, 0])
    y = np.array([0., 1, 2, 3, 4, 5, 6])
    z = np.array([1., 1, 1, 0, 1, 1, 1])
    seeds = np.column_stack([x, y, z])

    # Set up tracking
    dg = SimpleDirectionGetter()
    tc = SimpleTissueClassifier()

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

    streamlines_not_all = iter(streamlines_not_all) # valid streamlines only
    streamlines_all = iter(streamlines_all) # all streamlines

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
    y=4
    sl = next(streamlines_all)
    npt.assert_equal(sl[0], [0, y, 0])
    npt.assert_equal(sl[-1], [0, y, 4])
    npt.assert_equal(len(sl), 5)

    # The streamline stops at 0 (INVALIDPOINT) and 4 (INVALIDPOINT)
    y=5
    sl = next(streamlines_all)
    npt.assert_equal(sl[0], [0, y, 0])
    npt.assert_equal(sl[-1], [0, y, 3])
    npt.assert_equal(len(sl), 4)

    # The last streamline should contain only one point, the seed point,
    # because no valid inital direction was returned.
    y=6
    sl = next(streamlines_all)
    npt.assert_equal(sl[0], seeds[y])
    npt.assert_equal(sl[-1], seeds[y])
    npt.assert_equal(len(sl), 1)

    bad_affine = np.eye(3)
    npt.assert_raises(ValueError, LocalTracking, dg, tc, seeds, bad_affine, 1.)

    bad_affine = np.eye(4)
    bad_affine[0, 1] = 1.
    npt.assert_raises(ValueError, LocalTracking, dg, tc, seeds, bad_affine, 1.)


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


def test_ProbabilisticOdfWeightedTracker():
    """This tests that the Probabalistic Direction Getter plays nice
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
                             [0, 3, 2, 2, 2, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             ])

    simple_image = simple_image[..., None]
    pmf = pmf_lookup[simple_image]

    seeds = [np.array([1., 1., 0.])] * 30

    mask = (simple_image > 0).astype(float)
    tc = ThresholdTissueClassifier(mask, .5)

    dg = ProbabilisticDirectionGetter.from_pmf(pmf, 90, sphere)
    streamlines = LocalTracking(dg, tc, seeds, np.eye(4), 1.)

    expected = [np.array([[0.,  1.,  0.],
                          [1.,  1.,  0.],
                          [2.,  1.,  0.],
                          [2.,  2.,  0.],
                          [2.,  3.,  0.],
                          [2.,  4.,  0.],
                          [2.,  5.,  0.]]),
                np.array([[0.,  1.,  0.],
                          [1.,  1.,  0.],
                          [2.,  1.,  0.],
                          [3.,  1.,  0.],
                          [4.,  1.,  0.]])]

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
    dg = ProbabilisticDirectionGetter.from_pmf(pmf, 80, sphere)
    streamlines = LocalTracking(dg, tc, seeds, np.eye(4), 1.)

    for sl in streamlines:
        npt.assert_(np.allclose(sl, expected[1]))


def test_MaximumDeterministicTracker():
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

    dg = DeterministicMaximumDirectionGetter.from_pmf(pmf, 90, sphere)
    streamlines = LocalTracking(dg, tc, seeds, np.eye(4), 1.)

    expected = [np.array([[0.,  1.,  0.],
                          [1.,  1.,  0.],
                          [2.,  1.,  0.],
                          [2.,  2.,  0.],
                          [2.,  3.,  0.],
                          [2.,  4.,  0.],
                          [2.,  5.,  0.]]),
                np.array([[0.,  1.,  0.],
                          [1.,  1.,  0.],
                          [2.,  1.,  0.],
                          [3.,  1.,  0.],
                          [4.,  1.,  0.]])]

    def allclose(x, y):
        return x.shape == y.shape and np.allclose(x, y)

    for sl in streamlines:
        if not allclose(sl, expected[0]):
            raise AssertionError()

    # The first path is not possible if 90 degree turns are excluded
    dg = DeterministicMaximumDirectionGetter.from_pmf(pmf, 80, sphere)
    streamlines = LocalTracking(dg, tc, seeds, np.eye(4), 1.)

    for sl in streamlines:
        npt.assert_(np.allclose(sl, expected[1]))

if __name__ == "__main__":
    npt.run_module_suite()
