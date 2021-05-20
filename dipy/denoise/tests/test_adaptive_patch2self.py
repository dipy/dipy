import numpy as np
from dipy.denoise import adaptive_patch2self as ap2s
from dipy.testing import (assert_greater, assert_less,
                          assert_greater_equal, assert_less_equal)
from numpy.testing import (assert_array_almost_equal,
                           assert_raises, assert_equal)
import pytest

from .test_patch2self import generate_gtab, rfiw_phantom

needs_sklearn = pytest.mark.skipif(not ap2s.has_sklearn,
                                   reason="Requires Scikit-Learn")


@needs_sklearn
def test_adaptive_patch2self_random_noise():
    S0 = 30 + 2 * np.random.standard_normal((20, 20, 20, 50))

    bvals = np.repeat(30, 50)

    # shift = True
    S0den_shift = ap2s.adaptive_patch2self(S0, bvals, model='ols', shift_intensity=True)

    assert_greater_equal(S0den_shift.min(), S0.min())
    assert_less_equal(np.round(S0den_shift.mean()), 30)

    # clip = True
    S0den_clip = ap2s.adaptive_patch2self(S0, bvals, model='ols',
                                          clip_negative_vals=True)

    assert_greater(S0den_clip.min(), S0.min())
    assert_equal(np.round(S0den_clip.mean()), 30)

    # both clip and shift = True, a mask, and site_weight_beam_arctan
    mask = np.zeros(S0.shape, dtype=np.bool)
    mask[1:-1, 2:-2, 3:-3] = True
    S0den_clip = ap2s.adaptive_patch2self(S0, bvals, patch_radius=0, model='ols',
                                          clip_negative_vals=True, mask=mask,
                                          shift_intensity=True,
                                          site_weight_func=ap2s.site_weight_beam_arctan)

    assert_greater(S0den_clip.min(), S0.min())
    assert_equal(np.round(S0den_clip.mean()), 30)

    # both clip and shift = False, a mask, and calcSVDU
    S0den_clip = ap2s.adaptive_patch2self(S0, bvals, model='ols',
                                          clip_negative_vals=False,
                                          shift_intensity=False, site_placer=ap2s.calcSVDU)

    assert_greater(S0den_clip.min(), S0.min())
    assert_equal(np.round(S0den_clip.mean()), 30)


@needs_sklearn
def test_adaptive_patch2self_boundary():
    # adaptive_patch2self preserves boundaries
    S0 = 100 + np.zeros((20, 20, 20, 20))
    noise = 2 * np.random.standard_normal((20, 20, 20, 20))
    S0 += noise
    S0[:10, :10, :10, :10] = 300 + noise[:10, :10, :10, :10]

    bvals = np.repeat(100, 20)

    ap2s.adaptive_patch2self(S0, bvals)
    assert_greater(S0[9, 9, 9, 9], 290)
    assert_less(S0[10, 10, 10, 10], 110)


@needs_sklearn
def test_phantom():
    gtab, bvals = generate_gtab()

    dwi, sigma = rfiw_phantom(gtab, snr=10)
    dwi_den1 = ap2s.adaptive_patch2self(dwi, model='ridge',
                                        bvals=bvals, alpha=1.0)

    assert_less(np.max(dwi_den1) / sigma, np.max(dwi) / sigma)
    dwi_den2 = ap2s.adaptive_patch2self(dwi, model='ridge',
                                        bvals=bvals, alpha=0.7)

    assert_less(np.max(dwi_den2) / sigma, np.max(dwi) / sigma)
    assert_array_almost_equal(dwi_den1, dwi_den2, decimal=0)

    assert_raises(ValueError, ap2s.adaptive_patch2self, dwi, model='empty',
                  bvals=bvals)

    # Try this with a sigma volume, instead of a scalar
    dwi_den = ap2s.adaptive_patch2self(dwi, bvals=bvals,
                                       model='ols')

    assert_less(np.max(dwi_den) / sigma, np.max(dwi) / sigma)
