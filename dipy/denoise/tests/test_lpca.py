import numpy as np
from numpy.testing import (run_module_suite,
                           assert_,
                           assert_equal,
                           assert_array_almost_equal)
from dipy.denoise.noise_estimate_localpca import estimate_sigma_localpca
from dipy.denoise.localpca import localpca

from time import time


def test_lpca_static():
    S0 = 100 * np.ones((20, 20, 20, 20), dtype='f8')
    S0n = localpca(S0, sigma=np.ones((20, 20, 20)))
    assert_array_almost_equal(S0, S0n)


def test_lpca_random_noise():
    S0 = 100 + 2 * np.random.standard_normal((22, 23, 30, 20))

    S0n = localpca(S0, sigma=np.ones((22, 23, 30, 20)) * np.std(S0))

    print(S0.mean(), S0.min(), S0.max())
    print(S0n.mean(), S0n.min(), S0n.max())

    assert_(S0n.min() > S0.min())
    assert_(S0n.max() < S0.max())
    assert_equal(np.round(S0n.mean()), 100)


def test_lpca_boundary_behaviour():
    # check is first slice is getting denoised or not ?
    S0 = 100 * np.ones((20, 20, 20, 20), dtype='f8')
    S0[:, :, 0, :] = S0[:, :, 0, :] + 2 * \
        np.random.standard_normal((20, 20, 20))
    S0n = localpca(S0, sigma=np.ones((20, 20, 20, 20)) * np.std(S0))
    S0_first = S0[:, :, 0, :]
    S0n_first = S0n[:, :, 0, :]
    rmse = np.sum(np.abs(S0n_first - S0_first)) / (100.0 * 20.0 * 20.0 * 20.0)

    print(rmse)
    print(S0_first.mean())
    print(S0n_first.mean())

    # shows that S0n_first is not very close to S0_first
    assert_(rmse > 0.0001)
    assert_equal(np.round(S0n_first.mean()), 100)


def test_lpca_rmse():
    S0 = 100 + 2 * np.random.standard_normal((22, 23, 30, 20))

    S0n = localpca(S0, sigma=np.ones((22, 23, 30, 20)) * np.std(S0))
    rmse = np.sum(np.abs(S0n - 100) / np.sum(100 * np.ones(S0.shape)))

    print(rmse)

    # error should be less than 5%
    assert_(rmse < 0.05)


def test_lpca_sharpness():
    S0 = np.ones((30, 30, 30, 20)) * 100
    S0[10:20, 10:20, 10:20, :] = 50
    S0[20:30, 20:30, 20:30, :] = 0
    S0 = S0 + 20 * np.random.standard_normal((30, 30, 30, 20))
    S0n = localpca(S0, sigma=400)

    # check the edge gradient
    edg = np.abs(np.mean(S0n[8, 10:20, 10:20] - S0n[12, 10:20, 10:20]) - 50)

    print(edg)

    assert_(edg < 2)

if __name__ == '__main__':

    run_module_suite()
