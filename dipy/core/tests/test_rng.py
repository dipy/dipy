"""File dedicated to test ``dipy.core.rng`` module."""

from scipy.stats import chisquare
from dipy.core import rng
import numpy.testing as npt


def test_wichmann_hill2006():
    n_generated = [rng.WichmannHill2006() for i in range(10000)]
    # The chi-squared test statistic as result and The p-value of the test
    chisq, pvalue = chisquare(n_generated)
    # P-values equal 1 show evidence of the null hypothesis which indicates
    # that it is uniformly distributed. This is what we want to check here
    npt.assert_almost_equal(pvalue, 1.0)
    npt.assert_raises(ValueError, rng.WichmannHill2006, ix=0)


def test_wichmann_hill1982():
    n_generated = [rng.WichmannHill1982() for i in range(10000)]
    chisq, pvalue = chisquare(n_generated)
    # P-values equal 1 show evidence of the null hypothesis which indicates
    # that it is uniformly distributed. This is what we want to check here
    npt.assert_almost_equal(pvalue, 1.0)
    npt.assert_raises(ValueError, rng.WichmannHill1982, iz=0)


def test_LEcuyer():
    n_generated = [rng.LEcuyer() for i in range(10000)]
    chisq, pvalue = chisquare(n_generated)
    # P-values equal 1 show evidence of the null hypothesis which indicates
    # that it is uniformly distributed. This is what we want to check here
    npt.assert_almost_equal(pvalue, 1.0)
    npt.assert_raises(ValueError, rng.LEcuyer, s2=0)
