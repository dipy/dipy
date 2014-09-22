import numpy as np
import numpy.testing as npt

from dipy.core.sphere import HemiSphere, unit_octahedron
from dipy.core.gradients import gradient_table
from dipy.data import get_sim_voxels
from dipy.reconst.shm import SphHarmFit, SphHarmModel
from dipy.tracking.local import (ProbabilisticDirectionGetter, LocalTracking,
                                 ThresholdTissueClassifier)

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
    dg = ProbabilisticDirectionGetter.fromShmFit(fit, 90, unit_octahedron)
    state = dg.get_direction(point, dir)
    npt.assert_equal(state, 1)

    # Make a dg from a pmf
    N = unit_octahedron.theta.shape[0]
    pmf = np.zeros((3, 3, 3, N))
    dg = ProbabilisticDirectionGetter.fromPmf(pmf, 90, unit_octahedron)
    state = dg.get_direction(point, dir)
    npt.assert_equal(state, 1)

    # pmf shape must match sphere
    bad_pmf = pmf[..., 1:]
    npt.assert_raises(ValueError, ProbabilisticDirectionGetter.fromPmf,
                      bad_pmf, 90, unit_octahedron)

    # pmf must have 4 dimensions
    bad_pmf = pmf[0, ...]
    npt.assert_raises(ValueError, ProbabilisticDirectionGetter.fromPmf,
                      bad_pmf, 90, unit_octahedron)
    # pmf cannot have negative values
    pmf[0, 0, 0, 0] = -1
    npt.assert_raises(ValueError, ProbabilisticDirectionGetter.fromPmf, pmf,
                      90, unit_octahedron)

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

    dg = ProbabilisticDirectionGetter.fromPmf(pmf, 90, sphere)
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
    dg = ProbabilisticDirectionGetter.fromPmf(pmf, 80, sphere)
    streamlines = LocalTracking(dg, tc, seeds, np.eye(4), 1.)

    for sl in streamlines:
        npt.assert_(np.allclose(sl, expected[1]))

if __name__ == "__main__":
    npt.run_module_suite()

