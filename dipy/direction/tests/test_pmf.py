import warnings

import numpy as np
import numpy.testing as npt

from dipy.core.sphere import HemiSphere, unit_octahedron
from dipy.data import default_sphere, get_sphere
from dipy.direction.pmf import SHCoeffPmfGen, SimplePmfGen
from dipy.reconst import shm
from dipy.testing.decorators import set_random_number_generator

response = (np.array([1.5e3, 0.3e3, 0.3e3]), 1)


@set_random_number_generator()
def test_pmf_val(rng):
    sphere = get_sphere(name="symmetric724")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=shm.descoteaux07_legacy_msg,
            category=PendingDeprecationWarning,
        )
        pmfgen = SHCoeffPmfGen(rng.random([2, 2, 2, 28]), sphere, None)
    point = np.array([1, 1, 1], dtype="float")

    out = np.ones(len(sphere.vertices))
    for idx in [0, 5, 15, -1]:
        pmf = pmfgen.get_pmf(point)
        pmf_2 = pmfgen.get_pmf(point, out)

        npt.assert_array_almost_equal(pmf, out)
        npt.assert_array_almost_equal(pmf, pmf_2)
        # Create a direction vector close to the vertex idx
        xyz = sphere.vertices[idx] + rng.random([3]) / 100
        pmf_idx = pmfgen.get_pmf_value(point, xyz)
        # Test that the pmf sampled for the direction xyz is correct
        npt.assert_array_almost_equal(pmf[idx], pmf_idx)


def test_pmf_from_sh():
    sphere = HemiSphere.from_sphere(unit_octahedron)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=shm.descoteaux07_legacy_msg,
            category=PendingDeprecationWarning,
        )
        pmfgen = SHCoeffPmfGen(np.ones([2, 2, 2, 28]), sphere, None)

    out = np.zeros(len(sphere.vertices))
    # Test that the pmf is greater than 0 for a valid point
    pmf = pmfgen.get_pmf(np.array([0, 0, 0], dtype="float"))
    out = pmfgen.get_pmf(np.array([0, 0, 0], dtype="float"), out)
    npt.assert_equal(np.sum(pmf) > 0, True)
    npt.assert_array_almost_equal(pmf, out)

    # Test that the pmf is 0 for invalid Points
    npt.assert_array_equal(
        pmfgen.get_pmf(np.array([-1, 0, 0], dtype="float")),
        np.zeros(len(sphere.vertices)),
    )
    npt.assert_array_equal(
        pmfgen.get_pmf(np.array([0, 0, 10], dtype="float")),
        np.zeros(len(sphere.vertices)),
    )


def test_pmf_from_array():
    sphere = HemiSphere.from_sphere(unit_octahedron)
    pmfgen = SimplePmfGen(np.ones([2, 2, 2, len(sphere.vertices)]), sphere)

    out = np.zeros(len(sphere.vertices))
    # Test that the pmf is greater than 0 for a valid point
    pmf = pmfgen.get_pmf(np.array([0, 0, 0], dtype="float"))
    out = pmfgen.get_pmf(np.array([0, 0, 0], dtype="float"), out)
    npt.assert_equal(np.sum(pmf) > 0, True)
    npt.assert_array_almost_equal(pmf, out)

    # Test that the pmf is 0 for invalid Points
    npt.assert_array_equal(
        pmfgen.get_pmf(np.array([-1, 0, 0], dtype=float)),
        np.zeros(len(sphere.vertices)),
    )
    npt.assert_array_equal(
        pmfgen.get_pmf(np.array([0, 0, 10], dtype=float)),
        np.zeros(len(sphere.vertices)),
    )

    # Test ValueError for non matching pmf and sphere
    npt.assert_raises(
        ValueError,
        lambda: SimplePmfGen(np.ones([2, 2, 2, len(sphere.vertices)]), default_sphere),
    )
