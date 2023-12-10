import warnings

import numpy as np
from dipy.denoise import patch2self as p2s
from dipy.testing import (assert_greater, assert_less,
                          assert_greater_equal, assert_less_equal)
from numpy.testing import (assert_array_almost_equal,
                           assert_raises, assert_equal)
import pytest
from dipy.sims.voxel import multi_tensor
from dipy.core.gradients import gradient_table, generate_bvecs
from dipy.testing.decorators import set_random_number_generator
from dipy.utils.optpkg import optional_package

sklearn, has_sklearn, _ = optional_package('sklearn')
needs_sklearn = pytest.mark.skipif(not has_sklearn, reason="Requires sklearn")


@needs_sklearn
@set_random_number_generator(1234)
def test_patch2self_random_noise(rng):
    S0 = 30 + 2 * rng.standard_normal((20, 20, 20, 50))

    bvals = np.repeat(30, 50)

    # shift = True
    S0den_shift = p2s.patch2self(S0, bvals, model='ols', shift_intensity=True)

    assert_greater_equal(S0den_shift.min(), S0.min())
    assert_less_equal(np.round(S0den_shift.mean()), 30)

    # clip = True
    msg = "Both `clip_negative_vals` and `shift_intensity` .*"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=msg, category=UserWarning)
        S0den_clip = p2s.patch2self(S0, bvals, model='ols',
                                    clip_negative_vals=True)

    assert_greater(S0den_clip.min(), S0.min())
    assert_equal(np.round(S0den_clip.mean()), 30)

    # both clip and shift = True, and int patch_radius
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=msg, category=UserWarning)
        S0den_clip = p2s.patch2self(S0, bvals, patch_radius=0, model='ols',
                                    clip_negative_vals=True,
                                    shift_intensity=True)

    assert_greater(S0den_clip.min(), S0.min())
    assert_equal(np.round(S0den_clip.mean()), 30)

    # both clip and shift = False
    S0den_clip = p2s.patch2self(S0, bvals, model='ols',
                                clip_negative_vals=False,
                                shift_intensity=False)

    assert_greater(S0den_clip.min(), S0.min())
    assert_equal(np.round(S0den_clip.mean()), 30)


@needs_sklearn
@set_random_number_generator(1234)
def test_patch2self_boundary(rng):
    # patch2self preserves boundaries
    S0 = 100 + np.zeros((20, 20, 20, 20))
    noise = 2 * rng.standard_normal((20, 20, 20, 20))
    S0 += noise
    S0[:10, :10, :10, :10] = 300 + noise[:10, :10, :10, :10]

    bvals = np.repeat(100, 20)

    p2s.patch2self(S0, bvals)
    assert_greater(S0[9, 9, 9, 9], 290)
    assert_less(S0[10, 10, 10, 10], 110)


def rfiw_phantom(gtab, snr=None, rng=None):
    """rectangle fiber immersed in water"""
    # define voxel index
    slice_ind = np.zeros((10, 10, 8))
    slice_ind[4:7, 4:7, :] = 1
    slice_ind[4:7, 7, :] = 2
    slice_ind[7, 7, :] = 3
    slice_ind[7, 4:7, :] = 4
    slice_ind[7, 3, :] = 5
    slice_ind[4:7, 3, :] = 6
    slice_ind[3, 3, :] = 7
    slice_ind[3, 4:7, :] = 8
    slice_ind[3, 7, :] = 9

    # Define tissue diffusion parameters
    # Restricted diffusion
    ADr = 0.99e-3
    RDr = 0.0
    # Hindered diffusion
    ADh = 2.26e-3
    RDh = 0.87e-3
    # S0 value for tissue
    S1 = 50
    # Fraction between Restricted and Hindered diffusion
    fia = 0.51

    # Define water diffusion
    Dwater = 3e-3
    S2 = 100  # S0 value for water

    # Define tissue volume fraction for each voxel type (in index order)
    f = np.array([0., 1., 0.6, 0.18, 0.30, 0.15, 0.50, 0.35, 0.70, 0.42])

    # Define S0 for each voxel (in index order)
    S0 = S1 * f + S2 * (1 - f)

    # multi tensor simulations assume that each water pull as constant S0
    # since I am assuming that tissue and water voxels have different S0,
    # tissue volume fractions have to be adjusted to the measured f values when
    # constant S0 are assumed constant. Doing this correction, simulations will
    # be analogous to simulates that S0 are different for each media. (For more
    # details on this contact the phantom designer)
    f1 = f * S1 / S0

    mevals = np.array([[ADr, RDr, RDr], [ADh, RDh, RDh],
                       [Dwater, Dwater, Dwater]])
    angles = [(0, 0, 1), (0, 0, 1), (0, 0, 1)]
    dwi = np.zeros(slice_ind.shape + (gtab.bvals.size, ))
    for i in range(10):
        fractions = [f1[i] * fia * 100, f1[i] *
                     (1 - fia) * 100, (1 - f1[i]) * 100]
        sig, direction = multi_tensor(gtab, mevals, S0=S0[i], angles=angles,
                                      fractions=fractions, snr=None)
        dwi[slice_ind == i, :] = sig
    if snr is None:
        return dwi
    else:
        sigma = S2 * 1.0 / snr
        n1 = rng.normal(0, sigma, size=dwi.shape)
        n2 = rng.normal(0, sigma, size=dwi.shape)
        return [np.sqrt((dwi / np.sqrt(2) + n1)**2 +
                        (dwi / np.sqrt(2) + n2)**2), sigma]


@needs_sklearn
@set_random_number_generator(4321)
def test_phantom(rng):

    # generate a gradient table for phantom data
    directions8 = generate_bvecs(8)
    directions30 = generate_bvecs(20)
    directions60 = generate_bvecs(30)
    # Create full dataset parameters
    # (6 b-values = 0, 8 directions for b-value 300, 30 directions for b-value
    # 1000 and 60 directions for b-value 2000)
    bvals = np.hstack((np.zeros(6),
                       300 * np.ones(8),
                       1000 * np.ones(20),
                       2000 * np.ones(30)))
    bvecs = np.vstack((np.zeros((6, 3)),
                       directions8, directions30, directions60))
    gtab = gradient_table(bvals, bvecs)

    dwi, sigma = rfiw_phantom(gtab, snr=10, rng=rng)
    dwi_den1 = p2s.patch2self(dwi, model='ridge',
                              bvals=bvals, alpha=1.0)

    assert_less(np.max(dwi_den1) / sigma, np.max(dwi) / sigma)
    dwi_den2 = p2s.patch2self(dwi, model='ridge',
                              bvals=bvals, alpha=0.7)

    assert_less(np.max(dwi_den2) / sigma, np.max(dwi) / sigma)
    assert_array_almost_equal(dwi_den1, dwi_den2, decimal=0)

    assert_raises(ValueError, p2s.patch2self, dwi, model='empty',
                  bvals=bvals)

    # Try this with a sigma volume, instead of a scalar
    dwi_den = p2s.patch2self(dwi, bvals=bvals,
                             model='ols')

    assert_less(np.max(dwi_den) / sigma, np.max(dwi) / sigma)
