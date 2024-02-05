import numpy as np
import dipy.data as dpd
import nibabel as nib
from numpy.testing import (assert_,
                           assert_equal,
                           assert_array_almost_equal)
from dipy.denoise.non_local_means import non_local_means
from dipy.denoise.noise_estimate import estimate_sigma
from dipy.denoise.adaptive_soft_matching import adaptive_soft_matching
from dipy.testing.decorators import set_random_number_generator


def test_ascm_static():
    S0 = 100 * np.ones((20, 20, 20), dtype='f8')
    S0n1 = non_local_means(S0, sigma=0, rician=False,
                           patch_radius=1, block_radius=1)
    S0n2 = non_local_means(S0, sigma=0, rician=False,
                           patch_radius=2, block_radius=1)
    S0n = adaptive_soft_matching(S0, S0n1, S0n2, 0)
    assert_array_almost_equal(S0, S0n)


@set_random_number_generator()
def test_ascm_random_noise(rng):
    S0 = 100 + 2 * rng.standard_normal((22, 23, 30))
    S0n1 = non_local_means(S0, sigma=1, rician=False,
                           patch_radius=1, block_radius=1)
    S0n2 = non_local_means(S0, sigma=1, rician=False,
                           patch_radius=2, block_radius=1)
    S0n = adaptive_soft_matching(S0, S0n1, S0n2, 1)

    print(S0.mean(), S0.min(), S0.max())
    print(S0n.mean(), S0n.min(), S0n.max())

    assert_(S0n.min() > S0.min())
    assert_(S0n.max() < S0.max())
    assert_equal(np.round(S0n.mean()), 100)


@set_random_number_generator()
def test_ascm_rmse_with_nlmeans(rng):
    # checks the smoothness
    S0 = np.ones((30, 30, 30)) * 100
    S0[10:20, 10:20, 10:20] = 50
    S0[20:30, 20:30, 20:30] = 0
    S0_noise = S0 + 20 * rng.standard_normal((30, 30, 30))
    print("Original RMSE", np.sum(np.abs(S0 - S0_noise)) / np.sum(S0))

    S0n1 = non_local_means(
        S0_noise,
        sigma=400,
        rician=False,
        patch_radius=1,
        block_radius=1)
    print("Smaller patch RMSE", np.sum(np.abs(S0 - S0n1)) / np.sum(S0))
    S0n2 = non_local_means(
        S0_noise,
        sigma=400,
        rician=False,
        patch_radius=2,
        block_radius=2)
    print("Larger patch RMSE", np.sum(np.abs(S0 - S0n2)) / np.sum(S0))
    S0n = adaptive_soft_matching(S0, S0n1, S0n2, 400)
    print("ASCM RMSE", np.sum(np.abs(S0 - S0n)) / np.sum(S0))

    assert_(np.sum(np.abs(S0 - S0n)) / np.sum(S0) <
            np.sum(np.abs(S0 - S0n1)) / np.sum(S0))
    assert_(np.sum(np.abs(S0 - S0n)) / np.sum(S0) <
            np.sum(np.abs(S0 - S0_noise)) / np.sum(S0))
    assert_(90 < np.mean(S0n) < 110)


@set_random_number_generator()
def test_sharpness(rng):
    # check the edge-preserving nature
    S0 = np.ones((30, 30, 30)) * 100
    S0[10:20, 10:20, 10:20] = 50
    S0[20:30, 20:30, 20:30] = 0
    S0_noise = S0 + 20 * rng.standard_normal((30, 30, 30))
    S0n1 = non_local_means(
        S0_noise,
        sigma=400,
        rician=False,
        patch_radius=1,
        block_radius=1)
    edg1 = np.abs(np.mean(S0n1[8, 10:20, 10:20] - S0n1[12, 10:20, 10:20]) - 50)
    print("Edge gradient smaller patch", edg1)
    S0n2 = non_local_means(
        S0_noise,
        sigma=400,
        rician=False,
        patch_radius=2,
        block_radius=2)
    edg2 = np.abs(np.mean(S0n2[8, 10:20, 10:20] - S0n2[12, 10:20, 10:20]) - 50)
    print("Edge gradient larger patch", edg2)
    S0n = adaptive_soft_matching(S0, S0n1, S0n2, 400)
    edg = np.abs(np.mean(S0n[8, 10:20, 10:20] - S0n[12, 10:20, 10:20]) - 50)
    print("Edge gradient ASCM", edg)

    assert_(edg2 > edg1)
    assert_(edg2 > edg)
    assert_(np.abs(edg1 - edg) < 1.5)


def test_ascm_accuracy():
    f_name = dpd.get_fnames("ascm_test")
    test_ascm_data_ref = np.asanyarray(nib.load(f_name).dataobj)
    test_data = np.asanyarray(nib.load(dpd.get_fnames("aniso_vox")).dataobj)

    # the test data was constructed in this manner
    mask = test_data > 50
    sigma = estimate_sigma(test_data, N=4).item()

    den_small = non_local_means(
        test_data,
        sigma=sigma,
        mask=mask,
        patch_radius=1,
        block_radius=1,
        rician=True)

    den_large = non_local_means(
        test_data,
        sigma=sigma,
        mask=mask,
        patch_radius=2,
        block_radius=1,
        rician=True)

    S0n = np.array(adaptive_soft_matching(test_data,
                                          den_small, den_large, sigma))

    assert_array_almost_equal(S0n, test_ascm_data_ref, decimal=4)
