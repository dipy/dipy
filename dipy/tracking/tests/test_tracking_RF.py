
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
from dipy.tracking.local_tracking import (LocalTracking,
                                          ParticleFilteringTracking)
from dipy.tracking.streamline import Streamlines
from dipy.tracking.stopping_criterion import (ActStoppingCriterion,
                                              BinaryStoppingCriterion,
                                              ThresholdStoppingCriterion,
                                              StreamlineStatus)
from dipy.tracking.utils import seeds_from_mask
from dipy.sims.voxel import single_tensor, multi_tensor


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

    dg = DeterministicMaximumDirectionGetter.from_pmf(pmf, max_angle=90, sphere=sphere, sc=sc,
                                                      voxel_size=1, step_size=1,
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
    dg = DeterministicMaximumDirectionGetter.from_pmf(pmf, 80, sphere, sc,
                                                      voxel_size=1, step_size=1,
                                                      pmf_threshold=0.1)
    streamlines = LocalTracking(dg, sc, seeds, np.eye(4), 1.)

    for sl in streamlines:
        npt.assert_(np.allclose(sl, expected[1]))

    # Both path are not possible if 90 degree turns are exclude and
    # if pmf_threshold is larger than 0.67. Streamlines should stop at
    # the crossing.
    # 0.4/0.6 < 2/3, multiplying the pmf should not change the ratio
    dg = DeterministicMaximumDirectionGetter.from_pmf(10 * pmf, 80, sphere, sc,
                                                      voxel_size=1, step_size=1,
                                                      pmf_threshold=0.67)
    streamlines = LocalTracking(dg, sc, seeds, np.eye(4), 1.)

    for sl in streamlines:
        npt.assert_(np.allclose(sl, expected[2]))
