import numpy as np
from numpy.testing import assert_equal, assert_
from dipy.segment.tissue import dam_classifier, compute_directional_average


def test_compute_directional_average_valid():

    data = np.array([100, 80, 60, 50, 40, 20, 10])
    bvals = np.array([0, 100, 500, 1000, 1500, 2000, 3000])
    P, V = compute_directional_average(data, bvals)

    assert_(isinstance(P, float), 'P should be a float')
    assert_(isinstance(V, float), 'V should be a float')
    assert_(P != 0, "P should not be zero")
    assert_(V != 0, "V should not be zero")


def test_compute_directional_average_insufficient_bvals():

    data = np.array([100, 80, 60, 50])
    bvals = np.array([0, 0, 100, 100])
    wm_threshold = 0.5

    assert_equal(data.shape[0], bvals.shape[0],
                 "The length of bvals must match the last dimension of data")

    try:
        compute_directional_average(data, bvals)
    except ValueError as e:
        assert_equal(str(e), "Insufficient unique b-values for fitting.")
    else:
        raise AssertionError(
            "ValueError not raised for insufficient unique b-values")


def test_compute_directional_average_low_signal():
    data = np.array([20, 10, 5, 3, 2, 1, 1])  # Very low signal
    bvals = np.array([0, 100, 500, 1000, 1500, 2000, 3000])

    P, V = compute_directional_average(data, bvals, low_signal_threshold=50)

    assert_equal(P, 0, "P should be 0 when low signal")
    assert_equal(V, 0, "V should be 0 when low signal")


def test_compute_directional_average_div_by_zero():
    data = np.array([100, 100, 100, 100, 100, 100, 100])
    bvals = np.array([0, 100, 500, 1000, 1500, 2000, 3000])

    P, V = compute_directional_average(data, bvals)

    assert_(np.allclose(P, 0))


def test_dam_classifier_valid():
    data = np.random.rand(3, 3, 3, 7) * 100  # Simulated random data
    bvals = np.array([0, 100, 500, 1000, 1500, 2000, 3000])

    assert_equal(data.shape[-1], bvals.shape[0],
                 "The number of bvals must match the last dimension of data")

    wm_mask, gm_mask = dam_classifier(data, bvals, wm_threshold=0.5)

    assert_equal(wm_mask.shape, (3, 3, 3),
                 "Shape of wm_mask should be (3, 3, 3)")
    assert_equal(gm_mask.shape, (3, 3, 3),
                 "Shape of gm_mask should be (3, 3, 3)")
