import numpy as np
from numpy.testing import (assert_,
                           assert_equal,
                           assert_array_almost_equal,
                           assert_raises)
from dipy.denoise.non_local_means import non_local_means
from dipy.testing.decorators import set_random_number_generator


def test_nlmeans_static():
    S0 = 100 * np.ones((20, 20, 20), dtype='f8')
    S0nb = non_local_means(S0, sigma=1.0, rician=False)
    assert_array_almost_equal(S0, S0nb)


@set_random_number_generator()
def test_nlmeans_random_noise(rng):
    S0 = 100 + 2 * rng.standard_normal((22, 23, 30))

    masker = np.zeros(S0.shape[:3]).astype(bool)
    masker[8:15, 8:15, 8:15] = 1
    for mask in [None, masker]:
        S0nb = non_local_means(S0, sigma=np.std(S0), rician=False, mask=mask)

        assert_(S0nb[mask].min() > S0[mask].min())
        assert_(S0nb[mask].max() < S0[mask].max())
        assert_equal(np.round(S0nb[mask].mean()), 100)

        S0nb = non_local_means(S0, sigma=np.std(S0), rician=False, mask=mask)

        assert_(S0nb[mask].min() > S0[mask].min())
        assert_(S0nb[mask].max() < S0[mask].max())
        assert_equal(np.round(S0nb[mask].mean()), 100)


@set_random_number_generator()
def test_scalar_sigma(rng):
    S0 = 100 + np.zeros((20, 20, 20))
    noise = 2 * rng.standard_normal((20, 20, 20))
    S0 += noise
    S0[:10, :10, :10] = 300 + noise[:10, :10, :10]

    assert_raises(
        ValueError, non_local_means, S0, sigma=noise, rician=False)


@set_random_number_generator()
def test_nlmeans_boundary(rng):
    # nlmeans preserves boundaries
    S0 = 100 + np.zeros((20, 20, 20))
    noise = 2 * rng.standard_normal((20, 20, 20))
    S0 += noise
    S0[:10, :10, :10] = 300 + noise[:10, :10, :10]
    non_local_means(S0, sigma=np.std(noise), rician=False)
    assert_(S0[9, 9, 9] > 290)
    assert_(S0[10, 10, 10] < 110)


def test_nlmeans_wrong():
    S0 = 100 + np.zeros((10, 10, 10, 10, 10))
    assert_raises(ValueError, non_local_means, S0, 1.0)
    S0 = 100 + np.zeros((20, 20, 20))
    mask = np.ones((10, 10))
    assert_raises(ValueError, non_local_means, S0, 1.0, mask)


def test_nlmeans_4D_and_mask():
    S0 = 200 * np.ones((20, 20, 20, 3), dtype='f8')
    mask = np.zeros((20, 20, 20))
    mask[10, 10, 10] = 1
    S0n = non_local_means(S0, sigma=1, mask=mask, rician=True)
    assert_equal(S0.shape, S0n.shape)
    assert_equal(np.round(S0n[10, 10, 10]), 200)
    assert_equal(S0n[8, 8, 8], 0)


def test_nlmeans_dtype():

    S0 = 200 * np.ones((20, 20, 20, 3), dtype='f4')
    mask = np.zeros((20, 20, 20))
    mask[10:14, 10:14, 10:14] = 1
    S0n = non_local_means(S0, sigma=1, mask=mask, rician=True)
    assert_equal(S0.dtype, S0n.dtype)
    S0 = 200 * np.ones((20, 20, 20), dtype=np.uint16)
    mask = np.zeros((20, 20, 20))
    mask[10:14, 10:14, 10:14] = 1
    S0n = non_local_means(S0, sigma=1, mask=mask, rician=True)
    assert_equal(S0.dtype, S0n.dtype)
