from __future__ import division, print_function

import numpy as np
from numpy.testing import assert_almost_equal
from dipy.denoise.signal_transformation_framework import _inv_nchi_cdf, piesno


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
    # Values taken from hispeed.MedianPIESNO with the test data
    # in the package computed in matlab

    sigma = piesno(test_piesno, N=8, alpha=0.01, l=1)
    assert_almost_equal(sigma, 0.010635911195599)
