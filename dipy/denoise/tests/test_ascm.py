import nibabel as nib
import numpy as np
from numpy.testing import assert_, assert_equal

import dipy.data as dpd
from dipy.denoise.adaptive_soft_matching import adaptive_soft_matching
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma
from dipy.testing.decorators import set_random_number_generator


def test_ascm_static():
    S0 = 100 * np.ones((20, 20, 20), dtype="f8")
    S0n1 = nlmeans(S0, sigma=0, rician=False, patch_radius=1, block_radius=1)
    S0n2 = nlmeans(S0, sigma=0, rician=False, patch_radius=2, block_radius=1)
    S0n = adaptive_soft_matching(S0, S0n1, S0n2, 0)

    assert_equal(np.round(S0n.mean()), 100)
    assert_(np.abs(S0n.mean() - S0.mean()) < 1.0)

    close_values = np.abs(S0n - S0) < 10.0
    assert_(np.sum(close_values) / S0.size > 0.95)


@set_random_number_generator()
def test_ascm_random_noise(rng):
    S0 = 100 + 2 * rng.standard_normal((22, 23, 30))
    S0n1 = nlmeans(S0, sigma=1, rician=False, patch_radius=1, block_radius=1)
    S0n2 = nlmeans(S0, sigma=1, rician=False, patch_radius=2, block_radius=1)
    S0n = adaptive_soft_matching(S0, S0n1, S0n2, 1)

    assert_(np.abs(S0n.mean() - S0.mean()) < 3.0)
    assert_(np.isfinite(S0n).all())
    assert_(not np.isnan(S0n).any())
    assert_(S0n.std() < S0.std() * 1.5)
    assert_(S0n.min() > 0)
    assert_(S0n.max() < 200)

    reasonable_values = np.abs(S0n - S0n.mean()) < 3 * S0n.std()
    assert_(np.mean(reasonable_values) > 0.95)


@set_random_number_generator()
def test_ascm_rmse_with_nlmeans(rng):
    # checks the smoothness
    S0 = np.ones((30, 30, 30)) * 100
    S0[10:20, 10:20, 10:20] = 50
    S0[20:30, 20:30, 20:30] = 0
    S0_noise = S0 + 20 * rng.standard_normal((30, 30, 30))
    print("Original RMSE", np.sum(np.abs(S0 - S0_noise)) / np.sum(S0))

    S0n1 = nlmeans(S0_noise, sigma=400, rician=False, patch_radius=1, block_radius=1)
    print("Smaller patch RMSE", np.sum(np.abs(S0 - S0n1)) / np.sum(S0))
    S0n2 = nlmeans(S0_noise, sigma=400, rician=False, patch_radius=2, block_radius=2)
    print("Larger patch RMSE", np.sum(np.abs(S0 - S0n2)) / np.sum(S0))
    S0n = adaptive_soft_matching(S0, S0n1, S0n2, 400)
    print("ASCM RMSE", np.sum(np.abs(S0 - S0n)) / np.sum(S0))

    assert_(
        np.sum(np.abs(S0 - S0n)) / np.sum(S0) < np.sum(np.abs(S0 - S0n1)) / np.sum(S0)
    )
    assert_(
        np.sum(np.abs(S0 - S0n)) / np.sum(S0)
        < np.sum(np.abs(S0 - S0_noise)) / np.sum(S0)
    )
    assert_(90 < np.mean(S0n) < 110)


@set_random_number_generator()
def test_sharpness(rng):
    # check the edge-preserving nature
    S0 = np.ones((30, 30, 30)) * 100
    S0[10:20, 10:20, 10:20] = 50
    S0[20:30, 20:30, 20:30] = 0
    S0_noise = S0 + 20 * rng.standard_normal((30, 30, 30))
    S0n1 = nlmeans(S0_noise, sigma=400, rician=False, patch_radius=1, block_radius=1)
    edg1 = np.abs(np.mean(S0n1[8, 10:20, 10:20] - S0n1[12, 10:20, 10:20]) - 50)
    print("Edge gradient smaller patch", edg1)
    S0n2 = nlmeans(S0_noise, sigma=400, rician=False, patch_radius=2, block_radius=2)
    edg2 = np.abs(np.mean(S0n2[8, 10:20, 10:20] - S0n2[12, 10:20, 10:20]) - 50)
    print("Edge gradient larger patch", edg2)
    S0n = adaptive_soft_matching(S0, S0n1, S0n2, 400)
    edg = np.abs(np.mean(S0n[8, 10:20, 10:20] - S0n[12, 10:20, 10:20]) - 50)
    print("Edge gradient ASCM", edg)

    assert_(edg2 > edg1)
    assert_(edg2 > edg)
    assert_(np.abs(edg1 - edg) < 1.5)


def test_ascm_accuracy():
    f_name = dpd.get_fnames(name="ascm_test")
    test_ascm_data_ref = np.asanyarray(nib.load(f_name).dataobj)
    test_data = np.asanyarray(nib.load(dpd.get_fnames(name="aniso_vox")).dataobj)

    # the test data was constructed in this manner
    mask = test_data > 50
    sigma = estimate_sigma(test_data, N=4).item()

    den_small = nlmeans(
        test_data, sigma=sigma, mask=mask, patch_radius=1, block_radius=1, rician=True
    )

    den_large = nlmeans(
        test_data, sigma=sigma, mask=mask, patch_radius=2, block_radius=1, rician=True
    )

    S0n = np.array(adaptive_soft_matching(test_data, den_small, den_large, sigma))

    S0n_masked = S0n[mask]
    ref_masked = test_ascm_data_ref[mask]
    orig_masked = test_data[mask]

    correlation = np.corrcoef(S0n_masked.flatten(), ref_masked.flatten())[0, 1]
    assert_(correlation > 0.9)

    mean_diff = np.abs(np.mean(S0n_masked) - np.mean(ref_masked))
    assert_(mean_diff < 10.0)

    assert_(np.var(S0n_masked) < np.var(orig_masked))
