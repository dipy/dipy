import numpy as np
from numpy.testing import (run_module_suite,
                           assert_,
                           assert_equal,
                           assert_raises,
                           assert_array_almost_equal)
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.denspeed import add_padding_reflection, remove_padding


def test_nlmeans_padding():
    S0 = 100 + 2 * np.random.standard_normal((50, 50, 50))
    S0 = S0.astype('f8')
    S0n = add_padding_reflection(S0, 5)
    S0n2 = remove_padding(S0n, 5)
    assert_equal(S0.shape, S0n2.shape)


def test_nlmeans_static():
    S0 = 100 * np.ones((20, 20, 20), dtype='f8')
    S0n = nlmeans(S0, sigma=1, rician=False)
    assert_array_almost_equal(S0, S0n)


def test_nlmeans_random_noise():
    S0 = 100 + 2 * np.random.standard_normal((22, 23, 30))

    S0n = nlmeans(S0, sigma=np.std(S0), rician=False)

    print(S0.mean(), S0.min(), S0.max())
    print(S0n.mean(), S0n.min(), S0n.max())

    assert_(S0n.min() > S0.min())
    assert_(S0n.max() < S0.max())
    assert_equal(np.round(S0n.mean()), 100)


def test_nlmeans_boundary():
    # nlmeans preserves boundaries

    S0 = 100 + np.zeros((20, 20, 20))

    noise = 2 * np.random.standard_normal((20, 20, 20))

    S0 += noise

    S0[:10, :10, :10] = 300 + noise[:10, :10, :10]

    S0n = nlmeans(S0, sigma=np.std(noise), rician=False)

    print(S0[9, 9, 9])
    print(S0[10, 10, 10])

    assert_(S0[9, 9, 9] > 290)
    assert_(S0[10, 10, 10] < 110)


def test_nlmeans_4D_and_mask():
    S0 = 200 * np.ones((20, 20, 20, 3), dtype='f8')

    mask = np.zeros((20, 20, 20))
    mask[10, 10, 10] = 1

    S0n = nlmeans(S0, sigma=1, mask=mask, rician=True)
    assert_equal(S0.shape, S0n.shape)
    assert_equal(np.round(S0n[10, 10, 10]), 200)
    assert_equal(S0n[8, 8, 8], 0)


def test_nlmeans_dtype():

    S0 = 200 * np.ones((20, 20, 20, 3), dtype='f4')
    mask = np.zeros((20, 20, 20))
    mask[10:14, 10:14, 10:14] = 1
    S0n = nlmeans(S0, sigma=1, mask=mask, rician=True)
    assert_equal(S0.dtype, S0n.dtype)

    S0 = 200 * np.ones((20, 20, 20), dtype=np.uint16)
    mask = np.zeros((20, 20, 20))
    mask[10:14, 10:14, 10:14] = 1
    S0n = nlmeans(S0, sigma=1, mask=mask, rician=True)
    assert_equal(S0.dtype, S0n.dtype)


if __name__ == '__main__':
    run_module_suite()
