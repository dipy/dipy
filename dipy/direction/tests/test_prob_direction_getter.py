import numpy as np
import numpy.testing as npt

from dipy.core.sphere import unit_octahedron
from dipy.reconst.shm import SphHarmFit, SphHarmModel
from dipy.direction import ProbabilisticDirectionGetter


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

    # Check basis_type keyword
    ProbabilisticDirectionGetter.from_shcoeff(fit.shm_coeff, 90,
                                              unit_octahedron,
                                              pmf_threshold=0.1,
                                              basis_type="mrtrix")

    npt.assert_raises(ValueError, ProbabilisticDirectionGetter.from_shcoeff,
                      fit.shm_coeff, 90, unit_octahedron,
                      pmf_threshold=0.1,
                      basis_type="not a basis")
