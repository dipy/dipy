import numpy as np
import scipy as sp
import scipy.special as sps
from numpy.testing import (run_module_suite,
                           assert_,
                           assert_equal,
                           assert_raises,
                           assert_array_almost_equal)
from dipy.denoise.randomlpca_denoise import randomlpca_denoise as randommatrix_lpca


def test_lpca_static():
    S0 = 100 * np.ones((20, 20, 20, 20), dtype='f8')
    S0ns, _, _ = randommatrix_lpca(S0)
    assert_array_almost_equal(S0, S0ns)


def test_lpca_random_noise():
    S0 = 100 + 2 * np.random.standard_normal((22, 23, 30, 20))
    S0ns, _, _ = randommatrix_lpca(S0)
    assert_(S0ns.min() > S0.min())
    assert_(S0ns.max() < S0.max())
    assert_equal(np.round(S0ns.mean()), 100)

def test_lpca_boundary_behaviour():
    # check is first slice is getting denoised or not ?
    S0 = 100 * np.ones((20, 20, 20, 20), dtype='f8')
    S0[:, :, 0, :] = S0[:, :, 0, :] + 2 * \
        np.random.standard_normal((20, 20, 20))
    S0_first = S0[:, :, 0, :]
    S0ns, _, _ = randommatrix_lpca(S0)
    S0ns_first = S0ns[:, :, 0, :]
    rmses = np.sum(np.abs(S0ns_first - S0_first)) / \
        (100.0 * 20.0 * 20.0 * 20.0)

    # shows that S0n_first is not very close to S0_first
    assert_(rmses > 0.0001)
    assert_equal(np.round(S0ns_first.mean()), 100)

    rmses = np.sum(np.abs(S0ns_first - S0_first)) / \
        (100.0 * 20.0 * 20.0 * 20.0)

    # shows that S0n_first is not very close to S0_first
    assert_(rmses > 0.0001)
    assert_equal(np.round(S0ns_first.mean()), 100)


def test_lpca_rmse():
    S0_w_noise = 100 + 2 * np.random.standard_normal((22, 23, 30, 20))
    rmse_w_noise = np.sqrt(np.mean((S0_w_noise - 100) ** 2))
    S0_denoised,_,_ = randommatrix_lpca(S0_w_noise)
    rmse_denoised = np.sqrt(np.mean((S0_denoised - 100) ** 2))
    # Denoising should always improve the RMSE:
    assert_(rmse_denoised < rmse_w_noise)


def test_lpca_sharpness():
    S0 = np.ones((30, 30, 30, 20), dtype=np.float64) * 100
    S0[10:20, 10:20, 10:20, :] = 50
    S0[20:30, 20:30, 20:30, :] = 0
    S0 = S0 + 20 * np.random.standard_normal((30, 30, 30, 20))
    S0ns, _, _ = randommatrix_lpca(S0)
    # check the edge gradient
    edgs = np.abs(np.mean(S0ns[8, 10:20, 10:20,:] -
                          S0ns[12, 10:20, 10:20,:]) - 50)
    assert_(edgs < 2)


def test_lpca_dtype():
    # If out_dtype is not specified, we retain the original precision:
    S0 = 200 * np.ones((20, 20, 20, 3), dtype=np.float64)
    S0ns, _, _ = randommatrix_lpca(S0)
    assert_equal(S0.dtype, S0ns.dtype)

    S0 = 200 * np.ones((20, 20, 20, 20), dtype=np.uint16)
    S0ns, _, _ = randommatrix_lpca(S0)
    assert_equal(S0.dtype, S0ns.dtype)

    # If we set out_dtype, we get what we asked for:
    S0 = 200 * np.ones((20, 20, 20, 20), dtype=np.uint16)
    S0ns, _, _ = randommatrix_lpca(S0, out_dtype= np.float32)

    assert_equal(np.float32, S0ns.dtype)

    # If we set a few entries to zero, this induces negative entries in the
    # Resulting denoised array:
    S0[5:8, 5:8, 5:8] = 0
    # But if we should always get all non-negative results:
    S0ns, _, _ = randommatrix_lpca(S0)
    assert_(np.all(S0ns >= 0))
    # And no wrap-around to crazy high values:
    assert_(np.all(S0ns <= 200))


def test_lpca_wrong():
    S0 = np.ones((20, 20))
    assert_raises(ValueError, randommatrix_lpca, S0)


if __name__ == '__main__':
    run_module_suite()
