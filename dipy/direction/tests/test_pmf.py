import warnings

import numpy as np
import numpy.testing as npt

from dipy.core.sphere import HemiSphere, unit_octahedron
from dipy.data import default_sphere, get_sphere
from dipy.direction.pmf import SHCoeffPmfGen, SimplePeakGen, SimplePmfGen
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


def test_pmf_from_peaks():
    sphere = HemiSphere.from_sphere(unit_octahedron)
    n_vertices = len(sphere.vertices)

    peak_indices = np.full((2, 2, 2, 2), -1.0, dtype=float)
    peak_values = np.zeros((2, 2, 2, 2), dtype=float)

    for x in range(2):
        for y in range(2):
            for z in range(2):
                peak_indices[x, y, z, 0] = 0
                peak_values[x, y, z, 0] = x + y + z + 1
                peak_indices[x, y, z, 1] = 1
                peak_values[x, y, z, 1] = 2.0

    pmfgen = SimplePeakGen(peak_indices, peak_values, sphere.vertices, sphere)
    point = np.array([0.5, 0.5, 0.5], dtype=float)
    pmf = pmfgen.get_pmf(point)

    npt.assert_allclose(pmf[0], 2.5)
    npt.assert_allclose(pmf[1], 2.0)
    npt.assert_array_equal(pmf[2:], np.zeros(n_vertices - 2))

    pmf_value_0 = pmfgen.get_pmf_value(point, sphere.vertices[0])
    pmf_value_1 = pmfgen.get_pmf_value(point, sphere.vertices[1])
    pmf_value_2 = pmfgen.get_pmf_value(point, sphere.vertices[2])
    npt.assert_allclose(pmf_value_0, 2.5)
    npt.assert_allclose(pmf_value_1, 2.0)
    npt.assert_allclose(pmf_value_2, 0.0)


def test_pmf_from_peaks_invalid_point():
    sphere = HemiSphere.from_sphere(unit_octahedron)

    peak_indices = np.zeros((2, 2, 2, 1), dtype=float)
    peak_values = np.ones((2, 2, 2, 1), dtype=float)
    pmfgen = SimplePeakGen(peak_indices, peak_values, sphere.vertices, sphere)

    invalid_point = np.array([-0.1, 0.5, 0.5], dtype=float)
    npt.assert_array_equal(
        pmfgen.get_pmf(invalid_point), np.zeros(len(sphere.vertices))
    )
    npt.assert_allclose(
        pmfgen.get_pmf_value(invalid_point, sphere.vertices[0]), 0.0
    )
