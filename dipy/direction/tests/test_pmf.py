import warnings

import numpy as np
import numpy.testing as npt

from dipy.core.sphere import HemiSphere, unit_octahedron
from dipy.data import default_sphere, get_sphere
from dipy.direction.pmf import SHCoeffPmfGen, SimplePmfGen, PeakPmfGen
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
    shape = (2, 2, 2)
    n_peaks = 1


    peak_dirs = np.zeros(shape + (n_peaks, 3))
    peak_values = np.zeros(shape + (n_peaks,))


    peak_dirs[0, 0, 0, 0, :] = np.array([1.0, 0.0, 0.0])
    peak_values[0, 0, 0, 0] = 5.0


    pmfgen = PeakPmfGen(peak_dirs, peak_values, sphere)

    # Test get_pmf() inside the bounds for voxel (0, 0, 0)
    point = np.array([0.0, 0.0, 0.0])
    pmf_array = pmfgen.get_pmf(point)

    # Assert mass is conserved (0.0 everywhere, except 5.0 on the correct vertex)
    npt.assert_equal(np.sum(pmf_array), 5.0)

    # Find the sphere vertex closest to [1, 0, 0] to check if mass was grouped there
    target_dir = np.array([1.0, 0.0, 0.0])
    closest_idx = np.argmax(np.dot(sphere.vertices, target_dir))
    
    # Assert that this EXACT vertex received the probability mass 5.0
    npt.assert_equal(pmf_array[closest_idx], 5.0)

    # Test get_pmf_value() directly against our target direction
    pmf_val = pmfgen.get_pmf_value(point, target_dir)
    npt.assert_equal(pmf_val, 5.0)

    # Test out-of-bounds queries gracefully return an array of zeros
    out_of_bounds_point = np.array([-1.0, -1.0, -1.0])
    pmf_array_out = pmfgen.get_pmf(out_of_bounds_point)
    npt.assert_array_equal(pmf_array_out, np.zeros(len(sphere.vertices)))

