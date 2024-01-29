from numpy.testing import assert_almost_equal
import numpy as np
import pytest

from dipy.stats.resampling import bootstrap, jackknife
from dipy.testing.decorators import set_random_number_generator


@set_random_number_generator(1741332)
def test_bootstrap(rng):
    # Generate a sample data
    x = rng.standard_normal(size=100)

    # Test bootstrap function with default parameters
    bs_pdf, se, ci = bootstrap(x, rng=rng)

    assert len(bs_pdf) == 1000
    assert se > 0

    # Test bootstrap function with custom parameters
    bs_pdf, se, ci = bootstrap(x, statistic=np.mean, B=500, alpha=0.90)

    assert len(bs_pdf) == 500
    assert se > 0


@set_random_number_generator(1741333)
def test_jackknife(rng):
    # Generate a sample data
    pdf = rng.standard_normal(size=100)

    # Test jackknife function with default parameters
    jk_pdf, bias, se = jackknife(pdf, rng=rng)

    assert len(jk_pdf) == 99
    assert bias == pytest.approx(0, abs=1e-1)
    assert se > 0

    # Test jackknife function with custom parameters
    jk_pdf, bias, se = jackknife(pdf, statistic=np.median, M=50)
    assert len(jk_pdf) == 50
    assert se > 0
