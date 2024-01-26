import numpy as np
import pytest

from dipy.stats.resampling import bootstrap, jackknife


# Test bootstrap function
def test_bootstrap():
    # Generate a sample data
    rng = np.random.default_rng()
    x = rng.standard_normal(size=100)

    # Test bootstrap function with default parameters
    bs_pdf, se, ci = bootstrap(x)

    assert len(bs_pdf) == 1000
    assert se > 0

    # Test bootstrap function with custom parameters
    bs_pdf, se, ci = bootstrap(x, statistic=np.mean, B=500, alpha=0.90)

    assert len(bs_pdf) == 500
    assert se > 0


# Test jackknife function
def test_jackknife():
    # Generate a sample data
    rng = np.random.default_rng()
    pdf = rng.standard_normal(size=100)

    # Test jackknife function with default parameters
    jk_pdf, bias, se = jackknife(pdf)

    assert len(jk_pdf) == 99
    assert bias == pytest.approx(0, abs=1e-1)
    assert se > 0

    # Test jackknife function with custom parameters
    jk_pdf, bias, se = jackknife(pdf, statistic=np.median, M=50)
    assert len(jk_pdf) == 50
    assert bias == pytest.approx(0, abs=1e-1)
    assert se > 0
