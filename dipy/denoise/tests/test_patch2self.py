import numpy as np
from dipy.denoise import patch2self as p2s
from numpy.testing import (run_module_suite, assert_,
                           assert_equal,
                           assert_array_almost_equal,
                           assert_raises)
import pytest
from dipy.sims.voxel import multi_tensor
from dipy.core.gradients import gradient_table, generate_bvecs

needs_sklearn = pytest.mark.skipif(not p2s.has_sklearn,
                                   reason="Requires Scikit-Learn")


@needs_sklearn
def test_patch2self_random_noise():
    S0 = 100 + 2 * np.random.standard_normal((12, 13, 10, 10))

    S0nb = p2s.patch2self(S0)

    assert_(S0nb.min() > S0.min())
    assert_(S0nb.max() < S0.max())
    assert_equal(np.round(S0nb.mean()), 100)

    S0nb = p2s.patch2self(S0)

    assert_(S0nb.min() > S0.min())
    assert_(S0nb.max() < S0.max())
    assert_equal(np.round(S0nb.mean()), 100)


@needs_sklearn
def test_patch2self_boundary():
    # patch2self preserves boundaries
    S0 = 100 + np.zeros((20, 20, 20, 20))
    noise = 2 * np.random.standard_normal((20, 20, 20, 20))
    S0 += noise
    S0[:10, :10, :10, :10] = 300 + noise[:10, :10, :10, :10]
    p2s.patch2self(S0)
    assert_(S0[9, 9, 9, 9] > 290)
    assert_(S0[10, 10, 10, 10] < 110)


@needs_sklearn
def rfiw_phantom(gtab, snr=None):
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
    DWI = np.zeros(slice_ind.shape + (gtab.bvals.size, ))
    for i in range(10):
        fractions = [f1[i] * fia * 100, f1[i] *
                     (1 - fia) * 100, (1 - f1[i]) * 100]
        sig, direction = multi_tensor(gtab, mevals, S0=S0[i], angles=angles,
                                      fractions=fractions, snr=None)
        DWI[slice_ind == i, :] = sig
    if snr is None:
        return DWI
    else:
        sigma = S2 * 1.0 / snr
        n1 = np.random.normal(0, sigma, size=DWI.shape)
        n2 = np.random.normal(0, sigma, size=DWI.shape)
        return [np.sqrt((DWI / np.sqrt(2) + n1)**2 +
                        (DWI / np.sqrt(2) + n2)**2), sigma]


@needs_sklearn
def test_phantom():

    # generate a gradient table for phantom data
    directions8 = generate_bvecs(8)
    directions30 = generate_bvecs(30)
    directions60 = generate_bvecs(60)
    # Create full dataset parameters
    # (6 b-values = 0, 8 directions for b-value 300, 30 directions for b-value
    # 1000 and 60 directions for b-value 2000)
    bvals = np.hstack((np.zeros(6),
                       300 * np.ones(8),
                       1000 * np.ones(30),
                       2000 * np.ones(60)))
    bvecs = np.vstack((np.zeros((6, 3)),
                       directions8, directions30, directions60))
    gtab = gradient_table(bvals, bvecs)

    DWI, sigma = rfiw_phantom(gtab, snr=30)
    DWI_den1 = p2s.patch2self(DWI, patch_radius=[1, 1, 1], model='ridge')

    assert_(np.max(DWI_den1) / sigma < np.max(DWI) / sigma)
    DWI_den2 = p2s.patch2self(DWI, patch_radius=[1, 2, 1], model='ridge')

    assert_(np.max(DWI_den2) / sigma < np.max(DWI) / sigma)
    assert_array_almost_equal(DWI_den1, DWI_den2, decimal=1)

    assert_raises(AttributeError, p2s.patch2self, DWI, model='empty')

    # Try this with a sigma volume, instead of a scalar
    DWI_den = p2s.patch2self(DWI, patch_radius=[1, 1, 1])

    assert_(np.max(DWI_den) / sigma < np.max(DWI) / sigma)


if __name__ == '__main__':
    run_module_suite()
