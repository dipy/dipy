import numpy as np
import numpy.testing as npt

from dipy.core.sphere import HemiSphere, unit_octahedron
from dipy.core.gradients import gradient_table
from dipy.data import get_sim_voxels
from dipy.reconst.shm import SphHarmFit, SphHarmModel
from dipy.tracking.local import (ProbabilisticDirectionGetter, LocalTracking,
                                 ThresholdTissueClassifier, DirectionGetter,
                                 TissueClassifier)

def test_ProbabilisticDirectionGetter():
    # Test the constructors and errors of the ProbabilisticDirectionGetter

    class SillyModel(SphHarmModel):

        sh_order = 4

        def fit(self, data, mask=None):
            coeff = np.zeros(data.shape[:-1] + (15,))
            return SphHarmFit(self, coeff, mask=None)

    model = SillyModel(gtab=None)
    data = np.zeros((3, 3, 3, 7))
    fit = model.fit(data)

    # Sample point and direction
    point = np.zeros(3)
    dir = unit_octahedron.vertices[0].copy()

    # make a dg from a fit
    dg = ProbabilisticDirectionGetter.from_shcoeff(fit.shm_coeff, 90,
                                                   unit_octahedron)
    state = dg.get_direction(point, dir)
    npt.assert_equal(state, 1)

    # Make a dg from a pmf
    N = unit_octahedron.theta.shape[0]
    pmf = np.zeros((3, 3, 3, N))
    dg = ProbabilisticDirectionGetter.from_pmf(pmf, 90, unit_octahedron)
    state = dg.get_direction(point, dir)
    npt.assert_equal(state, 1)

    # pmf shape must match sphere
    bad_pmf = pmf[..., 1:]
    npt.assert_raises(ValueError, ProbabilisticDirectionGetter.from_pmf,
                      bad_pmf, 90, unit_octahedron)

    # pmf must have 4 dimensions
    bad_pmf = pmf[0, ...]
    npt.assert_raises(ValueError, ProbabilisticDirectionGetter.from_pmf,
                      bad_pmf, 90, unit_octahedron)
    # pmf cannot have negative values
    pmf[0, 0, 0, 0] = -1
    npt.assert_raises(ValueError, ProbabilisticDirectionGetter.from_pmf, pmf,
                      90, unit_octahedron)


def test_stop_conditions():
    """This tests that the Local Tracker behaves as expected for the
    following tissue types:

        OUTSIDEIMAGE = -1
        INVALIDPOINT = 0
        TRACKPOINT = 1
        ENDPOINT = 2
    """
    tissue = np.array([[2, 1, 1, 2, 1],
                       [2, 2, 1, 1, 2],
                       [1, 1, 1, 1, 1],
                       [1, 1, 1, 2, 2],
                       [0, 1, 1, 1, 2],
                       [2, 1, 1, 0, 2]])
    tissue = tissue[None]

    class SimpleTissueClassifier(TissueClassifier):
        def check_point(self, point):
            p = np.round(point)
            if any(p < 0) or any(p >= tissue.shape):
                return -1 # OUTSIDEIMAGE
            return tissue[p[0], p[1], p[2]]

    class SimpleDirectionGetter(DirectionGetter):
        def initial_direction(self, point):
            # Test tracking along the rows (z direction)
            # of the tissue array above
            return np.array([[0., 0., 1.]])
        def get_direction(self, p, d):
            # Always keep previous direction
            return 0

    # Create a seeds along the second column of every row, ie along [:, 1]
    x = np.array([0., 0, 0, 0, 0, 0])
    y = np.array([0., 1, 2, 3, 4, 5])
    z = np.array([1., 1, 1, 0, 1, 1])
    seeds = np.column_stack([x, y, z])

    # Set up tracking
    dg = SimpleDirectionGetter()
    tc = SimpleTissueClassifier()
    streamlines = LocalTracking(dg, tc, seeds, np.eye(4), 1.)
    streamlines = iter(streamlines)

    # Check that the first streamline stops at 0 and 3 (ENDPOINT)
    sl = next(streamlines)
    y = 0
    npt.assert_equal(sl[0], [0, y, 0])
    npt.assert_equal(sl[-1], [0, y, 3])
    npt.assert_equal(len(sl), 4)

    # Check that the first streamline stops at 0 and 4 (ENDPOINT)
    sl = next(streamlines)
    y = 1
    npt.assert_equal(sl[0], [0, y, 0])
    npt.assert_equal(sl[-1], [0, y, 4])
    npt.assert_equal(len(sl), 5)

    # This streamline should be the same as above. This row does not have
    # ENDPOINTs, but the streamline should stop at the edge and not include
    # OUTSIDEIMAGE points.
    sl = next(streamlines)
    y = 2
    npt.assert_equal(sl[0], [0, y, 0])
    npt.assert_equal(sl[-1], [0, y, 4])
    npt.assert_equal(len(sl), 5)

    # If we seed on the edge, the first (or last) point in the streamline
    # should be the seed.
    sl = next(streamlines)
    y = 3
    npt.assert_equal(sl[0], seeds[y])

    # The last 2 seeds should not produce streamlines, INVALIDPOINT streamlines
    # are rejected.
    npt.assert_equal(len(list(streamlines)), 0)

    bad_affine = np.eye(3.)
    npt.assert_raises(ValueError, LocalTracking, dg, tc, seeds, bad_affine, 1.)

    bad_affine = np.eye(4.)
    bad_affine[0, 1] = 1.
    npt.assert_raises(ValueError, LocalTracking, dg, tc, seeds, bad_affine, 1.)

    lin = bad_affine[:3, :3]
    dotlin = np.dot(lin.T, lin)


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

    expected = [np.array([[ 0.,  1.,  0.],
                          [ 1.,  1.,  0.],
                          [ 2.,  1.,  0.],
                          [ 2.,  2.,  0.],
                          [ 2.,  3.,  0.],
                          [ 2.,  4.,  0.],
                          [ 2.,  5.,  0.]]),
                np.array([[ 0.,  1.,  0.],
                          [ 1.,  1.,  0.],
                          [ 2.,  1.,  0.],
                          [ 3.,  1.,  0.],
                          [ 4.,  1.,  0.]])
               ]

    def allclose(x, y):
        return x.shape == y.shape and np.allclose(x, y)

    path = [False, False]
    for sl in streamlines:
        print(sl)
        dir = ( -sphere.vertices[0] ).copy()
        print dg.get_direction(sl[0], dir)
        print dir
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

if __name__ == "__main__":
    npt.run_module_suite()

