import numpy as np
from dipy.denoise import adaptive_patch2self as ap2s
from dipy.testing import (assert_greater, assert_less, assert_greater_equal)
from numpy.testing import assert_raises
import pytest

from .test_patch2self import generate_gtab, rfiw_phantom

needs_sklearn = pytest.mark.skipif(not ap2s.has_sklearn,
                                   reason="Requires Scikit-Learn")


@needs_sklearn
def test_adaptive_patch2self_random_noise():
    nb0s = 5
    ndwis = 15
    bvals = [0] * nb0s + [1] * ndwis
    signal = 30.0 * np.ones((9, 9, 9, len(bvals)))
    for vi, b in enumerate(bvals):
        signal[..., vi] *= np.exp(-b)

    # Don't do Rician noise here because we want data to be unbiased for
    # comparison.
    noise = np.random.standard_normal(signal.shape)
    rmsnoise = np.mean(noise**2)**0.5
    data = signal + noise

    # ICA doesn't work without some nonGaussian signal, so stick to calcSVDU
    # here.

    # shift = True
    dataden_shift = ap2s.adaptive_patch2self(data, bvals, b0_threshold=0.5,
                                             model='ols', shift_intensity=True,
                                             site_placer=ap2s.calcSVDU)

    assert_greater_equal(dataden_shift.min(), data.min())
    assert_less(np.mean((dataden_shift - signal)**2)**0.5, rmsnoise)

    # clip = True
    dataden_clip = ap2s.adaptive_patch2self(data, bvals, b0_threshold=0.5,
                                            model='ols', n_comps=2,
                                            clip_negative_vals=True,
                                            site_placer=ap2s.calcSVDU)

    assert_greater_equal(dataden_clip.min(), data.min())
    assert_less(np.mean((dataden_clip - signal)**2)**0.5, rmsnoise)

    # both clip and shift = True (produces a warning), a mask, and
    # site_weight_beam_arctan
    mask = np.zeros(data.shape[:3], dtype=np.bool)
    mask[1:-1, 2:-2, 3:-3] = True
    dataden_clip = ap2s.adaptive_patch2self(data, bvals, b0_threshold=0.5,
                                            model='ols', n_comps=2,
                                            clip_negative_vals=True, mask=mask,
                                            shift_intensity=True,
                                            site_weight_func=ap2s.site_weight_beam_arctan,
                                            site_placer=ap2s.calcSVDU)

    assert_greater_equal(dataden_clip.min(), data.min())
    assert_less(np.mean((dataden_clip - signal)**2)**0.5, rmsnoise)

    # both clip and shift = False, + a mask
    dataden_clip = ap2s.adaptive_patch2self(data, bvals, b0_threshold=0.5,
                                            model='ols', mask=mask,
                                            clip_negative_vals=False,
                                            shift_intensity=False,
                                            site_placer=ap2s.calcSVDU)

    assert_greater_equal(dataden_clip.min(), data.min())
    assert_less(np.mean((dataden_clip - signal)**2)**0.5, rmsnoise)


@needs_sklearn
def test_adaptive_patch2self_boundary():
    # adaptive_patch2self preserves boundaries
    nb0s = 5
    ndwis = 15
    bvals = [0] * nb0s + [1000] * ndwis
    data = 100 + np.zeros((20, 20, 20, len(bvals)))
    noise = 2 * np.random.standard_normal(data.shape)
    data += noise
    data[:10, :10, :10, :10] = 300 + noise[:10, :10, :10, :10]

    den = ap2s.adaptive_patch2self(data, bvals, n_comps=1)
    assert_greater(den[9, 9, 9, 9], 290)
    assert_less(den[10, 10, 10, 10], 110)


@needs_sklearn
def test_phantom():
    gtab, bvals = generate_gtab()

    snr = 10.0   # must be > 2
    dwi, sigma = rfiw_phantom(gtab, snr=snr)
    avb0 = np.mean(dwi[..., bvals == 0], axis=-1)
    mask = np.zeros(avb0.shape, dtype=np.bool)
    thresh = 2.0 * np.mean(avb0) / snr
    mask[avb0 > thresh] = True
    maxdwi_ov_sigma = np.max(dwi) / sigma
    for mod in ('ols', 'ridge', 'lasso'):
        dwi_den = ap2s.adaptive_patch2self(dwi, model=mod, n_comps=1,
                                           mask=mask, bvals=bvals, alpha=1.0)
        assert_less(np.max(dwi_den) / sigma, maxdwi_ov_sigma)

    assert_raises(ValueError, ap2s.adaptive_patch2self, dwi, model='empty',
                  bvals=bvals)


@needs_sklearn
def test_doICA():
    gtab, bvals = generate_gtab()

    snr = 10.0   # must be > 2
    dwi, sigma = rfiw_phantom(gtab, snr=snr)

    x = ap2s.extract_data(dwi)
    ica = ap2s.doICA(x, 2, 20210522)
    assert ica.shape == (x.shape[0], 2)


@needs_sklearn
def test_get_coefs():
    gtab, bvals = generate_gtab()

    snr = 50.0   # must be > 2
    dwi, sigma = rfiw_phantom(gtab, snr=snr)

    x = ap2s.extract_data(dwi)

    # Test getting all the PCs
    ap = ap2s.AdaptivePatch2Self(x, n_comps=64, site_placer=ap2s.calcSVDU)

    coefs = ap.get_coefs(23, (1, 4, -1))

    # It would be nice to be more specific about the components, but I think
    # their sign is arbitrary, which means the ordering of the sites is not
    # completely predictable.
    assert coefs.shape == (3, 63)
