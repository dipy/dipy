from __future__ import division, print_function

import numpy as np
import nibabel as nib

from numpy.testing import (assert_almost_equal, assert_equal, assert_,
                           assert_array_almost_equal)
from dipy.denoise.noise_estimate import _inv_nchi_cdf, piesno, estimate_sigma, _piesno_3D
import dipy.data

# See page 5 of the reference paper for tested values
def test_inv_nchi():
    # Values taken from hispeed.MedianPIESNO.lambdaPlus
    # and hispeed.MedianPIESNO.lambdaMinus
    N = 8
    K = 20
    alpha = 0.01

    lambdaMinus = _inv_nchi_cdf(N, K, alpha/2)
    lambdaPlus = _inv_nchi_cdf(N, K, 1 - alpha/2)

    assert_almost_equal(lambdaMinus, 6.464855180579397)
    assert_almost_equal(lambdaPlus, 9.722849086419043)


def test_piesno():
    # Values taken from hispeed.OptimalPIESNO with the test data
    # in the package computed in matlab
    test_piesno_data = nib.load(dipy.data.get_data("test_piesno")).get_data()
    sigma = piesno(test_piesno_data, N=8, alpha=0.01, l=1, eps=1e-10, return_mask=False)
    assert_almost_equal(sigma, 0.010749458025559)

    noise1 = (np.random.randn(100, 100, 100) * 50) + 10
    noise2 = (np.random.randn(100, 100, 100) * 50) + 10
    rician_noise = np.sqrt(noise1**2 + noise2**2)
    sigma, mask = piesno(rician_noise, N=1, alpha=0.01, l=1, eps=1e-10, return_mask=True)

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

    sigma = _piesno_3D(np.zeros_like(rician_noise), N=1, alpha=0.01, l=1, eps=1e-10,
                     return_mask=False,
                     initial_estimation=initial_estimation)

    assert_(np.all(sigma == 0))

    sigma, mask = _piesno_3D(np.zeros_like(rician_noise), N=1, alpha=0.01, l=1, eps=1e-10,
                 return_mask=True,
                 initial_estimation=initial_estimation)

    assert_(np.all(sigma == 0))
    assert_(np.all(mask == 0))

    # Check if no noise points found in array it exits
    sigma = _piesno_3D(1000*np.ones_like(rician_noise), N=1, alpha=0.01, l=1, eps=1e-10,
             return_mask=False, initial_estimation=10)
    assert_(np.all(sigma == 10))

def test_estimate_sigma():

    sigma = estimate_sigma(np.ones((7, 7, 7)), disable_background_masking=True)
    assert_equal(sigma, 0.)

    sigma = estimate_sigma(np.ones((7, 7, 7, 3)), disable_background_masking=True)
    assert_equal(sigma, np.array([0., 0., 0.]))

    sigma = estimate_sigma(5 * np.ones((7, 7, 7)), disable_background_masking=False)
    assert_equal(sigma, 0.)

    sigma = estimate_sigma(5 * np.ones((7, 7, 7, 3)), disable_background_masking=False)
    assert_equal(sigma, np.array([0., 0., 0.]))

    arr = np.zeros((3, 3, 3))
    arr[0, 0, 0] = 1
    sigma = estimate_sigma(arr, disable_background_masking=False, N=1)
    assert_array_almost_equal(sigma, 0.10286889997472792 / np.sqrt(0.42920367320510366))

    arr = np.zeros((3, 3, 3, 3))
    arr[0, 0, 0] = 1
    sigma = estimate_sigma(arr, disable_background_masking=False, N=1)
    assert_array_almost_equal(sigma, np.array([0.10286889997472792 / np.sqrt(0.42920367320510366),
                                               0.10286889997472792 / np.sqrt(0.42920367320510366),
                                               0.10286889997472792 / np.sqrt(0.42920367320510366)]))

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
    assert_array_almost_equal(sigma, np.array([0.46291005 / np.sqrt(0.4946862482541263),
                                               0.46291005 / np.sqrt(0.4946862482541263),
                                               0.46291005 / np.sqrt(0.4946862482541263)]))
