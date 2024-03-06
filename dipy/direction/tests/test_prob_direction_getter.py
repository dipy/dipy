import warnings

import numpy as np
import numpy.testing as npt

from dipy.core.sphere import unit_octahedron
from dipy.reconst.shm import (
    descoteaux07_legacy_msg, tournier07_legacy_msg, SphHarmFit, SphHarmModel)
from dipy.direction import (DeterministicMaximumDirectionGetter,
                            ProbabilisticDirectionGetter)


def test_ProbabilisticDirectionGetter():
    # Test the constructors and errors of the ProbabilisticDirectionGetter

    class SillyModel(SphHarmModel):

        sh_order = 4

        def fit(self, data, mask=None):
            coeff = np.zeros(data.shape[:-1] + (15,))
            return SphHarmFit(self, coeff, mask=None)

    model = SillyModel(gtab=None)
    data = np.zeros((3, 3, 3, 7))

    # Test if the tracking works on different dtype of the same data.
    for dtype in [np.float32, np.float64]:
        fit = model.fit(data.astype(dtype))

        # Sample point and direction
        point = np.zeros(3)
        direction = unit_octahedron.vertices[0].copy()

        # make a dg from a fit
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=descoteaux07_legacy_msg,
                category=PendingDeprecationWarning)
            dg = ProbabilisticDirectionGetter.from_shcoeff(
                fit.shm_coeff, 90, unit_octahedron)

        state = dg.get_direction(point, direction)
        npt.assert_equal(state, 1)

        # make a dg from a fit (using sh_to_pmf=True)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=descoteaux07_legacy_msg,
                category=PendingDeprecationWarning)
            dg = ProbabilisticDirectionGetter.from_shcoeff(
                fit.shm_coeff, 90, unit_octahedron, sh_to_pmf=True)

        state = dg.get_direction(point, direction)
        npt.assert_equal(state, 1)

        # Make a dg from a pmf
        N = unit_octahedron.theta.shape[0]
        pmf = np.zeros((3, 3, 3, N))
        dg = ProbabilisticDirectionGetter.from_pmf(pmf, 90, unit_octahedron)
        state = dg.get_direction(point, direction)
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
        npt.assert_raises(ValueError, ProbabilisticDirectionGetter.from_pmf,
                          pmf, 90, unit_octahedron)

        # Check basis_type keyword
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=tournier07_legacy_msg,
                category=PendingDeprecationWarning)

            dg = ProbabilisticDirectionGetter.from_shcoeff(
                fit.shm_coeff, 90, unit_octahedron, basis_type="tournier07")

        npt.assert_raises(ValueError,
                          ProbabilisticDirectionGetter.from_shcoeff,
                          fit.shm_coeff, 90, unit_octahedron,
                          basis_type="not a basis")


def test_DeterministicMaximumDirectionGetter():
    # Test the DeterministicMaximumDirectionGetter

    direction = unit_octahedron.vertices[-1].copy()
    point = np.zeros(3)
    N = unit_octahedron.theta.shape[0]

    # No valid direction
    pmf = np.zeros((3, 3, 3, N))
    dg = DeterministicMaximumDirectionGetter.from_pmf(pmf, 90,
                                                      unit_octahedron)
    state = dg.get_direction(point, direction)
    npt.assert_equal(state, 1)

    # Test BF #1566 - bad condition in DeterministicMaximumDirectionGetter
    pmf = np.zeros((3, 3, 3, N))
    pmf[0, 0, 0, 0] = 1
    dg = DeterministicMaximumDirectionGetter.from_pmf(pmf, 0,
                                                      unit_octahedron)
    state = dg.get_direction(point, direction)
    npt.assert_equal(state, 1)
