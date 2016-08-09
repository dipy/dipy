import numpy as np
import scipy as sp
from numpy.testing import (run_module_suite,
                           assert_,
                           assert_equal,
                           assert_raises,
                           assert_array_almost_equal)
from dipy.denoise.localpca_slow import localpca_slow
from dipy.sims.voxel import multi_tensor
from dipy.core.gradients import gradient_table
from dipy.core.sphere import disperse_charges, HemiSphere
from dipy.sims.voxel import multi_tensor


def rfiw_phantom(gtab, snr=None):
    """retangle fiber immersed in water"""

    # define voxel index
    slice_ind = np.zeros((10, 10, 10))
    slice_ind[4:7, 4:7, :] = 1
    slice_ind[4:7, 7, :] = 2
    slice_ind[7, 7, :] = 3
    slice_ind[7, 4:7, :] = 4
    slice_ind[7, 3, :] = 5
    slice_ind[4:7, 3, :] = 6
    slice_ind[3, 3, :] = 7
    slice_ind[3, 4:7, :] = 8
    slice_ind[3, 7, :] = 9

    # Define tisse diffusion parameters
    # Restricted diffusion
    ADr = 0.99e-3
    RDr = 0.0
    # Hindered diffusion
    ADh = 2.26e-3
    RDh = 0.87
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
    # datails on this contact the phantom designer)
    f1 = f * S1 / S0

    mevals = np.array([[ADr, RDr, RDr], [ADh, RDh, RDh],
                       [Dwater, Dwater, Dwater]])
    angles = [(0, 0, 1), (0, 0, 1), (0, 0, 1)]
    DWI = np.zeros((10, 10, 10, gtab.bvals.size))
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


def gen_gtab():
    # generate the phantom data
    # Sample 8 diffusion-weighted directions for first shell
    n_pts = 8
    theta = np.pi * np.random.rand(n_pts)
    phi = 2 * np.pi * np.random.rand(n_pts)
    hsph_initial = HemiSphere(theta=theta, phi=phi)
    hsph_updated, potential = disperse_charges(hsph_initial, 5000)
    directions8 = hsph_updated.vertices  # directions for each shell

    # Sample 30 diffusion-weighted directions for second shell
    n_pts = 30
    theta = np.pi * np.random.rand(n_pts)
    phi = 2 * np.pi * np.random.rand(n_pts)
    hsph_initial = HemiSphere(theta=theta, phi=phi)
    hsph_updated, potential = disperse_charges(hsph_initial, 5000)
    directions30 = hsph_updated.vertices  # directions for each shell

    # Sample 60 diffusion-weighted directions for second shell
    n_pts = 60
    theta = np.pi * np.random.rand(n_pts)
    phi = 2 * np.pi * np.random.rand(n_pts)
    hsph_initial = HemiSphere(theta=theta, phi=phi)
    hsph_updated, potential = disperse_charges(hsph_initial, 5000)
    directions60 = hsph_updated.vertices  # directions for each shell

    # Create full dataset parameters
    # (6 b-values = 0, 8 directions for b-value 300, 30 directions for b-value
    # 1000 and 60 directions for b-value 2000)
    bvals = np.hstack((np.zeros(6), 300 * np.ones(8),
                       1000 * np.ones(30), 2000 * np.ones(60)))
    bvecs = np.vstack((np.zeros((6, 3)), directions8,
                       directions30, directions60))
    gtab = gradient_table(bvals, bvecs)

    return gtab


def test_lpca_static():
    S0 = 100 * np.ones((20, 20, 20, 20), dtype='f8')
    S0ns = localpca_slow(S0, sigma=np.ones((20, 20, 20), dtype=np.float64))
    assert_array_almost_equal(S0, S0ns)


def test_lpca_random_noise():
    S0 = 100 + 2 * np.random.standard_normal((22, 23, 30, 20))
    S0ns = localpca_slow(S0, sigma = np.std(S0))

    assert_(S0ns.min() > S0.min())
    assert_(S0ns.max() < S0.max())
    assert_equal(np.round(S0ns.mean()), 100)


def test_lpca_boundary_behaviour():
    # check is first slice is getting denoised or not ?
    S0 = 100 * np.ones((20, 20, 20, 20), dtype='f8')
    S0[:, :, 0, :] = S0[:, :, 0, :] + 2 * \
        np.random.standard_normal((20, 20, 20))
    S0_first = S0[:, :, 0, :]
    S0ns = localpca_slow(S0, sigma = np.std(S0))
    S0ns_first = S0ns[:, :, 0, :]
    rmses = np.sum(np.abs(S0ns_first - S0_first)) / (100.0 * 20.0 * 20.0 * 20.0)


    # shows that S0n_first is not very close to S0_first
    assert_(rmses > 0.0001)
    assert_equal(np.round(S0ns_first.mean()), 100)


def test_lpca_rmse():
    S0 = 100 + 2 * np.random.standard_normal((22, 23, 30, 20))
    S0ns = localpca_slow(S0, sigma = np.std(S0))
    rmses = np.sum(np.abs(S0ns - 100) / np.sum(100 * np.ones(S0.shape)))
    # error should be less than 5%
    assert_(rmses < 0.05)


def test_lpca_sharpness():
    S0 = np.ones((30, 30, 30, 20), dtype=np.float64) * 100
    S0[10:20, 10:20, 10:20, :] = 50
    S0[20:30, 20:30, 20:30, :] = 0
    S0 = S0 + 20 * np.random.standard_normal((30, 30, 30, 20))
    S0ns = localpca_slow(S0, sigma = 400.0)
    # check the edge gradient
    edgs = np.abs(np.mean(S0ns[8, 10:20, 10:20] - S0ns[12, 10:20, 10:20]) - 50)
    assert_(edgs < 2)

def test_lpca_dtype():

    S0 = 200 * np.ones((20, 20, 20, 3), dtype='f4')
    S0ns = localpca_slow(S0, sigma=1)
    assert_equal(S0.dtype, S0ns.dtype)

    S0 = 200 * np.ones((20, 20, 20, 20), dtype=np.uint16)
    S0ns = localpca_slow(S0, sigma=np.ones((20, 20, 20)))
    assert_equal(S0.dtype, S0ns.dtype)

def test_lpca_wrong():

    S0 = np.ones((20,20))
    assert_raises(ValueError, localpca_slow, S0, sigma=1)

def test_phantom():

    gtab = gen_gtab()
    DWI_clean = rfiw_phantom(gtab, snr=None)
    [DWI, sigma] = rfiw_phantom(gtab, snr=30)
    # To test without rician correction
    temp = (DWI_clean / sigma)**2
    DWI_clean_wrc = sigma * np.sqrt(np.pi / 2) * np.exp(-0.5 * temp) * ((1 + 0.5 * temp) * sp.special.iv(
        0, 0.25 * temp) + 0.5 * temp * sp.special.iv(1, 0.25 * temp))**2

    DWI_den = localpca_slow(DWI, sigma, patch_radius = 3)
    rmse_den = np.sum(np.abs(DWI_clean - DWI_den)) / np.sum(np.abs(DWI_clean))
    rmse_noisy = np.sum(np.abs(DWI_clean - DWI)) / np.sum(np.abs(DWI_clean))

    rmse_den_wrc = np.sum(np.abs(DWI_clean_wrc - DWI_den)
                          ) / np.sum(np.abs(DWI_clean_wrc))
    rmse_noisy_wrc = np.sum(np.abs(DWI_clean_wrc - DWI)) / \
        np.sum(np.abs(DWI_clean_wrc))
        
    assert_(np.max(DWI_clean) / sigma < np.max(DWI_den) / sigma)
    assert_(np.max(DWI_den) / sigma < np.max(DWI) / sigma)
    assert_(rmse_den < rmse_noisy)
    assert_(rmse_den_wrc < rmse_noisy_wrc)

if __name__ == '__main__':

    run_module_suite()
