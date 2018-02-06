import numpy as np
import numpy.testing as npt

from dipy.core.sphere import HemiSphere, unit_octahedron
from dipy.direction.pmf import SimplePmfGen, SHCoeffPmfGen


def test_pmf_from_sh():
    sphere = HemiSphere.from_sphere(unit_octahedron)
    pmfgen = SHCoeffPmfGen(np.ones([2, 2, 2, 28]), sphere, None)

    # Test that the pmf is greater than 0 for a valid point
    pmf = pmfgen.get_pmf(np.array([0, 0, 0], dtype='float'))
    npt.assert_equal(np.sum(pmf) > 0, True)

    # Test that the pmf is 0 for invalid Points
    npt.assert_array_equal(pmfgen.get_pmf(np.array([-1, 0, 0], dtype='float')),
                           np.zeros(len(sphere.vertices)))
    npt.assert_array_equal(pmfgen.get_pmf(np.array([0, 0, 10], dtype='float')),
                           np.zeros(len(sphere.vertices)))


def test_pmf_from_array():
    sphere = HemiSphere.from_sphere(unit_octahedron)
    pmfgen = SimplePmfGen(np.ones([2, 2, 2, len(sphere.vertices)]))

    # Test that the pmf is greater than 0 for a valid point
    pmf = pmfgen.get_pmf(np.array([0, 0, 0], dtype='float'))
    npt.assert_equal(np.sum(pmf) > 0, True)

    # Test that the pmf is 0 for invalid Points
    npt.assert_array_equal(pmfgen.get_pmf(np.array([-1, 0, 0], dtype='float')),
                           np.zeros(len(sphere.vertices)))
    npt.assert_array_equal(pmfgen.get_pmf(np.array([0, 0, 10], dtype='float')),
                           np.zeros(len(sphere.vertices)))

    npt.assert_raises(
        ValueError,
        lambda: SimplePmfGen(np.ones([2, 2, 2, len(sphere.vertices)])*-1))


if __name__ == '__main__':
    npt.run_module_suite()
