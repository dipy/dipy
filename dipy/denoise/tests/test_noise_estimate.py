import numpy as np

from numpy.testing import (assert_almost_equal, assert_equal, assert_,
                           assert_array_almost_equal, assert_warns)
from dipy.denoise.noise_estimate import _inv_nchi_cdf, piesno, estimate_sigma
from dipy.denoise.noise_estimate import _piesno_3D
from dipy.denoise.pca_noise_estimate import pca_noise_estimate
import dipy.data as dpd
import dipy.core.gradients as dpg
from dipy.io.image import load_nifti_data
from dipy.testing.decorators import set_random_number_generator


def test_inv_nchi():
    # See page 5 of the reference paper for tested values
    # Values taken from hispeed.MedianPIESNO.lambdaPlus
    # and hispeed.MedianPIESNO.lambdaMinus
    N = 8
    K = 20
    alpha = 0.01

    lambdaMinus = _inv_nchi_cdf(N, K, alpha/2)
    lambdaPlus = _inv_nchi_cdf(N, K, 1 - alpha/2)

    assert_almost_equal(lambdaMinus, 6.464855180579397)
    assert_almost_equal(lambdaPlus, 9.722849086419043)


@set_random_number_generator()
def test_piesno(rng):
    # Values taken from hispeed.OptimalPIESNO with the test data
    # in the package computed in matlab
    test_piesno_data = load_nifti_data(dpd.get_fnames("test_piesno"))
    sigma = piesno(test_piesno_data, N=8, alpha=0.01, l=1, eps=1e-10,
                   return_mask=False)
    assert_almost_equal(sigma, 0.010749458025559)

    noise1 = (rng.standard_normal((100, 100, 100)) * 50) + 10
    noise2 = (rng.standard_normal((100, 100, 100)) * 50) + 10
    rician_noise = np.sqrt(noise1**2 + noise2**2)
    sigma, mask = piesno(rician_noise, N=1, alpha=0.01, l=1, eps=1e-10,
                         return_mask=True)

    # less than 3% of error?
    assert_(np.abs(sigma - 50) / sigma < 0.03)

    # Test using the median as the initial estimation
    initial_estimation = (np.median(sigma) /
                          np.sqrt(2 * _inv_nchi_cdf(1, 1, 0.5)))

    sigma, mask = _piesno_3D(rician_noise, N=1, alpha=0.01, l=1, eps=1e-10,
                             return_mask=True,
                             initial_estimation=initial_estimation)

    assert_(np.abs(sigma - 50) / sigma < 0.03)

    sigma = _piesno_3D(rician_noise, N=1, alpha=0.01, l=1, eps=1e-10,
                       return_mask=False,
                       initial_estimation=initial_estimation)
    assert_(np.abs(sigma - 50) / sigma < 0.03)

    sigma = _piesno_3D(np.zeros_like(rician_noise), N=1, alpha=0.01, l=1,
                       eps=1e-10, return_mask=False,
                       initial_estimation=initial_estimation)

    assert_(np.all(sigma == 0))

    sigma, mask = _piesno_3D(np.zeros_like(rician_noise), N=1, alpha=0.01, l=1,
                             eps=1e-10, return_mask=True,
                             initial_estimation=initial_estimation)

    assert_(np.all(sigma == 0))
    assert_(np.all(mask == 0))

    # Check if no noise points found in array it exits
    sigma = _piesno_3D(1000*np.ones_like(rician_noise), N=1, alpha=0.01, l=1,
                       eps=1e-10, return_mask=False, initial_estimation=10)
    assert_(np.all(sigma == 10))


def test_piesno_type():
    # This is testing if the `sum_m2` cast is overflowing
    data = np.ones((10, 10, 10), dtype=np.int16)
    for i in range(10):
        data[:, i, :] = i * 26

    sigma = piesno(data, N=2, alpha=0.01, l=1, eps=1e-10, return_mask=False)
    assert_almost_equal(sigma, 79.970003117424739)


def test_estimate_sigma():
    sigma = estimate_sigma(np.ones((7, 7, 7)), disable_background_masking=True)
    assert_equal(sigma, 0.)

    sigma = estimate_sigma(np.ones((7, 7, 7, 3)),
                           disable_background_masking=True)
    assert_equal(sigma, np.array([0., 0., 0.]))

    sigma = estimate_sigma(5 * np.ones((7, 7, 7)),
                           disable_background_masking=False)
    assert_equal(sigma, 0.)

    sigma = estimate_sigma(5 * np.ones((7, 7, 7, 3)),
                           disable_background_masking=False)
    assert_equal(sigma, np.array([0., 0., 0.]))

    arr = np.zeros((3, 3, 3))
    arr[0, 0, 0] = 1
    sigma = estimate_sigma(arr, disable_background_masking=False, N=1)
    assert_array_almost_equal(sigma,
                              (0.10286889997472792 /
                               np.sqrt(0.42920367320510366)))

    arr = np.zeros((3, 3, 3, 3))
    arr[0, 0, 0] = 1
    sigma = estimate_sigma(arr, disable_background_masking=False, N=1)
    assert_array_almost_equal(sigma,
                              np.array([0.10286889997472792 /
                                        np.sqrt(0.42920367320510366),
                                        0.10286889997472792 /
                                        np.sqrt(0.42920367320510366),
                                        0.10286889997472792 /
                                        np.sqrt(0.42920367320510366)]))

    arr = np.zeros((3, 3, 3))
    arr[0, 0, 0] = 1
    sigma = estimate_sigma(arr, disable_background_masking=True, N=4)
    assert_array_almost_equal(sigma, 0.46291005 / np.sqrt(0.4834941393603609))

    arr = np.zeros((3, 3, 3))
    arr[0, 0, 0] = 1
    sigma = estimate_sigma(arr, disable_background_masking=True, N=0)
    assert_array_almost_equal(sigma, 0.46291005 / np.sqrt(1))
    arr = np.zeros((3, 3, 3, 3))

    arr[0, 0, 0] = 1
    sigma = estimate_sigma(arr, disable_background_masking=True, N=12)
    assert_array_almost_equal(sigma, np.array([0.46291005 /
                                               np.sqrt(0.4946862482541263),
                                               0.46291005 /
                                               np.sqrt(0.4946862482541263),
                                               0.46291005 /
                                               np.sqrt(0.4946862482541263)]))


@set_random_number_generator(1984)
def test_pca_noise_estimate(rng):
    # MUBE:
    bvals1 = np.concatenate([np.zeros(17), np.ones(3) * 1000])
    bvecs1 = np.concatenate([np.zeros((17, 3)), np.eye(3)])
    gtab1 = dpg.gradient_table(bvals1, bvecs1)
    # SIBE:
    bvals2 = np.concatenate([np.zeros(1), np.ones(3) * 1000])
    bvecs2 = np.concatenate([np.zeros((1, 3)), np.eye(3)])
    gtab2 = dpg.gradient_table(bvals2, bvecs2)

    for images_as_samples in [True, False]:

        for patch_radius in [1, 2]:
            for gtab in [gtab1, gtab2]:
                for dtype in [np.int16, np.float64]:
                    signal = np.ones((20, 20, 20, gtab.bvals.shape[0]))
                    for correct_bias in [True, False]:
                        if not correct_bias:
                            # High signal for no bias correction
                            signal = signal * 100

                        sigma = 1
                        noise1 = rng.normal(0, sigma, size=signal.shape)
                        noise2 = rng.normal(0, sigma, size=signal.shape)

                        # Rician noise:
                        data = np.sqrt((signal + noise1) ** 2 + noise2 ** 2)

                        sigma_est = pca_noise_estimate(data.astype(dtype), gtab,
                                                       correct_bias=correct_bias,
                                                       patch_radius=patch_radius,
                                                       images_as_samples=images_as_samples)
                        #print("sigma_est:", sigma_est)
                        assert_array_almost_equal(np.mean(sigma_est), sigma,
                                                  decimal=1)

        # check that Rician corrects produces larger noise estimate
        assert_(np.mean(pca_noise_estimate(data, gtab, correct_bias=True,
                          images_as_samples=images_as_samples)) >
                np.mean(pca_noise_estimate(data, gtab, correct_bias=False,
                          images_as_samples=images_as_samples)))

        assert_warns(UserWarning, pca_noise_estimate, data, gtab, patch_radius=0)
