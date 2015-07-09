#! /usr/bin/env python

from __future__ import division, print_function

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from scipy.special import factorialk
from scipy.stats import norm

from scilpy.denoising.stabilizer import (_test_marcumq_cython, _test_beta,
    _test_fixed_point_k, _test_xi, fixed_point_finder, chi_to_gauss,
    _test_inv_cdf_gauss, _test_multifactorial)

# hispeed is the closed source java reference implementation,
# from which most values are taken from.


def test_inv_cdf_gauss():
    loc = np.random.randint(-10, 10)
    scale = np.random.rand()
    y = np.random.rand() * scale + loc
    assert_almost_equal(_test_inv_cdf_gauss(norm.cdf(y, loc=loc, scale=scale), loc, scale), y, decimal=10)


def test_beta():
    # Values taken from hispeed.SignalFixedPointFinder.beta
    assert_almost_equal(_test_beta(3), 2.349964007466563, decimal=10)
    assert_almost_equal(_test_beta(7), 3.675490580428171, decimal=10)
    assert_almost_equal(_test_beta(12), 4.848227898082543, decimal=10)


def test_xi():
    # Values taken from hispeed.SignalFixedPointFinder.xi
    assert_almost_equal(_test_xi(50, 2, 2), 0.9976038446303619)
    assert_almost_equal(_test_xi(100, 25, 12), 0.697674262651006)
    assert_almost_equal(_test_xi(4, 1, 12), 0.697674262651006)


def test_fixed_point_finder():
    # Values taken from hispeed.SignalFixedPointFinder.fixedPointFinder
    assert_almost_equal(fixed_point_finder(50, 30, 12), -192.78288201533618)
    assert_almost_equal(fixed_point_finder(650, 45, 1), 648.4366584016703)


def test_chi_to_gauss():
    # Values taken from hispeed.DistributionalMapping.nonCentralChiToGaussian
    assert_almost_equal(chi_to_gauss(470, 600, 80, 12), 331.2511087335721, decimal=3)
    assert_almost_equal(chi_to_gauss(700, 600, 80, 12), 586.5304199340127, decimal=3)
    assert_almost_equal(chi_to_gauss(700, 600, 80, 1), 695.0548001366581, decimal=3)
    assert_almost_equal(chi_to_gauss(470, 600, 80, 1), 463.965319619292, decimal=3)


def test_marcumq():
    # Values taken from octave's marcumq function
    assert_almost_equal(_test_marcumq_cython(2, 1, 0),  0.730987939964090, decimal=6)
    assert_almost_equal(_test_marcumq_cython(7, 5, 0),  0.972285213704037, decimal=6)
    assert_almost_equal(_test_marcumq_cython(3, 7, 5),  0.00115139503866225, decimal=6)
    assert_almost_equal(_test_marcumq_cython(0, 7, 5),  4.07324330517049e-07, decimal=6)
    assert_almost_equal(_test_marcumq_cython(7, 0, 5),  1., decimal=6)


def test_factorial():
    assert_almost_equal(_test_multifactorial(10, 1), factorialk(10, 1), decimal=0)
    assert_almost_equal(_test_multifactorial(20, 2), factorialk(20, 2), decimal=0)
    assert_almost_equal(_test_multifactorial(20, 3), factorialk(20, 3), decimal=0)
    assert_almost_equal(_test_multifactorial(0, 3), factorialk(0, 3), decimal=0)


test_chi_to_gauss()
test_fixed_point_finder()
test_xi()
test_beta()
test_inv_cdf_gauss()
test_factorial()
test_marcumq()
