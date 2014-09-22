import numpy as np
import numpy.testing as npt

from dipy.core.sphere import HemiSphere, unit_octahedron
from dipy.tracking.local import (ProbabilisticDirectionGetter, LocalTracking,
                                 ThresholdTissueClassifier)

def test_ProbabilisticOdfWeightedTracker():
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

