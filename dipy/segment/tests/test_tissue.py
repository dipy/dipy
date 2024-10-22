import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises
import pytest

from dipy.segment.tissue import compute_directional_average, dam_classifier
from dipy.utils.optpkg import optional_package

sklearn, has_sklearn, _ = optional_package("sklearn")
needs_sklearn = pytest.mark.skipif(not has_sklearn, reason="Requires sklearn")


@needs_sklearn
def test_compute_directional_average_valid():
    data = np.array([100, 80, 60, 50, 40, 20, 10])
    bvals = np.array([0, 100, 500, 1000, 1500, 2000, 3000])
    P, V = compute_directional_average(data, bvals)

    assert_(isinstance(P, float), "P should be a float")
    assert_(isinstance(V, float), "V should be a float")
    assert_(P != 0, "P should not be zero")
    assert_(V != 0, "V should not be zero")


@needs_sklearn
def test_compute_directional_average_low_signal():
    data = np.array([20, 10, 5, 3, 2, 1, 1])  # Very low signal
    bvals = np.array([0, 100, 500, 1000, 1500, 2000, 3000])

    P, V = compute_directional_average(data, bvals, low_signal_threshold=50)

    assert_equal(P, 0, "P should be 0 when low signal")
    assert_equal(V, 0, "V should be 0 when low signal")


@needs_sklearn
def test_compute_directional_average_div_by_zero():
    data = np.array([100, 100, 100, 100, 100, 100, 100])
    bvals = np.array([0, 100, 500, 1000, 1500, 2000, 3000])

    P, V = compute_directional_average(data, bvals)

    assert_(np.allclose(P, 0))


@needs_sklearn
def test_dam_classifier_valid():
    data = np.random.rand(3, 3, 3, 7) * 100  # Simulated random data
    bvals = np.array([0, 100, 500, 1000, 1500, 2000, 3000])

    assert_equal(
        data.shape[-1],
        bvals.shape[0],
        "The number of bvals must match the last dimension of data",
    )

    wm_mask, gm_mask = dam_classifier(data, bvals, wm_threshold=0.5)

    assert_equal(wm_mask.shape, (3, 3, 3), "Shape of wm_mask should be (3, 3, 3)")
    assert_equal(gm_mask.shape, (3, 3, 3), "Shape of gm_mask should be (3, 3, 3)")

    data = np.array([100, 80, 60, 50])
    bvals = np.array([0, 0, 100, 100])

    assert_raises(ValueError, dam_classifier, data, bvals, wm_threshold=0.5)
