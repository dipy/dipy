"""File dedicated to test ``dipy.core.rng`` module."""

from scipy.stats import chisquare
from dipy.core import rng
import numpy.testing as npt


def test_wichmann_hill2006():
    rng.ix, rng.iy, rng.iz, rng.it = 100001, 200002, 300003, 400004
    n_generated = [rng.WichmannHill2006() for i in range(10000)]
    # The chi-squared test statistic as result and The p-value of the test
    chisq, pvalue = chisquare(n_generated)
    npt.assert_almost_equal(pvalue, 1.0)


def test_wichmann_hill1982():
    rng.ix, rng.iy, rng.iz, rng.it = 100001, 200002, 300003, 400004
    n_generated = [rng.WichmannHill1982() for i in range(10000)]
    chisq, pvalue = chisquare(n_generated)
    npt.assert_almost_equal(pvalue, 1.0)


def test_LEcuyer():
    rng.s1, rng.s2 = 100001, 200002
    n_generated = [rng.LEcuyer() for i in range(10000)]
    chisq, pvalue = chisquare(n_generated)
    npt.assert_almost_equal(pvalue, 1.0)


if __name__ == "__main__":
    test_LEcuyer()
    test_wichmann_hill2006()
    test_wichmann_hill1982()
