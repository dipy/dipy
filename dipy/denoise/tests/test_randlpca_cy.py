import numpy as np
import scipy as sp
import scipy.special as sps
from numpy.testing import (run_module_suite,
                           assert_,
                           assert_equal,
                           assert_raises,
                           assert_array_almost_equal)
from dipy.denoise.randomlpca_denoise import randomlpca_denoise
from dipy.core.gradients import gradient_table, generate_bvecs
from dipy.sims.voxel import multi_tensor


# This is for python version testing
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


def gen_gtab():
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
    return gtab


def test_lpca_static():
    S0 = 100 * np.ones((20, 20, 20, 20), dtype='f8')
    S0ns, _, _ = randomlpca_denoise(S0)
    assert_array_almost_equal(S0, S0ns)


def test_lpca_random_noise():
    S0 = 100 + 2 * np.random.standard_normal((22, 23, 30, 20))
    S0ns, _, _ = randomlpca_denoise(S0)

    assert_(S0ns.min() > S0.min())
    assert_(S0ns.max() < S0.max())
    assert_equal(np.round(S0ns.mean()), 100)


def test_lpca_boundary_behaviour():
    # check is first slice is getting denoised or not ?
    S0 = 100 * np.ones((20, 20, 20, 20), dtype='f8')
    S0[:, :, 0, :] = S0[:, :, 0, :] + 2 * \
        np.random.standard_normal((20, 20, 20))
    S0_first = S0[:, :, 0, :]
    S0ns, _, _ = randomlpca_denoise(S0)
    S0ns_first = S0ns[:, :, 0, :]
    rmses = np.sum(np.abs(S0ns_first - S0_first)) / \
        (100.0 * 20.0 * 20.0 * 20.0)

    # shows that S0n_first is not very close to S0_first
    assert_(rmses > 0.0001)
    assert_equal(np.round(S0ns_first.mean()), 100)

    rmses = np.sum(np.abs(S0ns_first - S0_first)) / \
        (100.0 * 20.0 * 20.0 * 20.0)

    # shows that S0n_first is not very close to S0_first
    assert_(rmses > 0.0001)
    assert_equal(np.round(S0ns_first.mean()), 100)


def test_lpca_rmse():
    S0_w_noise = 100 + 2 * np.random.standard_normal((22, 23, 30, 20))
    rmse_w_noise = np.sqrt(np.mean((S0_w_noise - 100) ** 2))
    S0_denoised, _, _ = randomlpca_denoise(S0_w_noise)
    rmse_denoised = np.sqrt(np.mean((S0_denoised - 100) ** 2))
    # Denoising should always improve the RMSE:
    assert_(rmse_denoised < rmse_w_noise)


def test_lpca_sharpness():
    S0 = np.ones((30, 30, 30, 20), dtype=np.float64) * 100
    S0[10:20, 10:20, 10:20, :] = 50
    S0[20:30, 20:30, 20:30, :] = 0
    S0 = S0 + 20 * np.random.standard_normal((30, 30, 30, 20))
    S0ns, _, _ = randomlpca_denoise(S0)
    # check the edge gradient
    edgs = np.abs(np.mean(S0ns[8, 10:20, 10:20] - S0ns[12, 10:20, 10:20]) - 50)
    assert_(edgs < 2)


def test_lpca_dtype():
    # If out_dtype is not specified, we retain the original precision:
    S0 = 200 * np.ones((20, 20, 20, 3), dtype=np.float64)
    S0ns, _, _ = randomlpca_denoise(S0)
    assert_equal(S0.dtype, S0ns.dtype)

    S0 = 200 * np.ones((20, 20, 20, 20), dtype=np.uint16)
    S0ns, _, _ = randomlpca_denoise(S0)
    assert_equal(S0.dtype, S0ns.dtype)

    # If we set out_dtype, we get what we asked for:
    S0 = 200 * np.ones((20, 20, 20, 20), dtype=np.uint16)
    S0ns, _, _ = randomlpca_denoise(S0,
                    out_dtype=np.float32)
    assert_equal(np.float32, S0ns.dtype)

    # If we set a few entries to zero, this induces negative entries in the
    # Resulting denoised array:
    S0[5:8, 5:8, 5:8] = 0
    # But if we should always get all non-negative results:
    S0ns, _, _ = randomlpca_denoise(S0, out_dtype=np.uint16)
    assert_(np.all(S0ns >= 0))
    # And no wrap-around to crazy high values:
    assert_(np.all(S0ns <= 200))


def test_lpca_wrong():
    S0 = np.ones((20, 20))
    assert_raises(ValueError, randomlpca_denoise, S0)


def test_phantom():
    gtab = gen_gtab()
    DWI_clean = rfiw_phantom(gtab, snr=None)
    DWI, sigma = rfiw_phantom(gtab, snr=30)
    # To test without rician correction
    temp = (DWI_clean / sigma)**2
    DWI_clean_wrc = (sigma * np.sqrt(np.pi / 2) * np.exp(-0.5 * temp) *
                     ((1 + 0.5 * temp) * sps.iv(0, 0.25 * temp) + 0.5 * temp *
                     sps.iv(1, 0.25 * temp))**2)

    DWI_den, _, _ = randomlpca_denoise(DWI, patch_extent=3)
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

    # Try this with a sigma volume, instead of a scalar
    sigma_vol = sigma * np.ones(DWI.shape[:-1])
    mask = np.zeros_like(DWI, dtype=bool)[..., 0]
    mask[2:-2, 2:-2, 2:-2] = True
    DWI_den, _, _ = randomlpca_denoise(DWI, patch_extent=3)
    DWI_den[~mask] = 0
    DWI_clean_masked = DWI_clean.copy()
    DWI_clean_masked[~mask] = 0
    DWI_masked = DWI.copy()
    DWI_masked[~mask] = 0
    rmse_den = np.sum(np.abs(DWI_clean_masked - DWI_den)) / np.sum(np.abs(
            DWI_clean_masked))
    rmse_noisy = np.sum(np.abs(DWI_clean_masked - DWI_masked)) / np.sum(np.abs(
            DWI_clean_masked))

    DWI_clean_wrc_masked = DWI_clean_wrc.copy()
    DWI_clean_wrc_masked[~mask] = 0
    rmse_den_wrc = np.sum(np.abs(DWI_clean_wrc_masked - DWI_den)
                          ) / np.sum(np.abs(DWI_clean_wrc_masked))
    rmse_noisy_wrc = np.sum(np.abs(DWI_clean_wrc_masked - DWI_masked)) / \
        np.sum(np.abs(DWI_clean_wrc_masked))

    assert_(np.max(DWI_clean) / sigma < np.max(DWI_den) / sigma)
    assert_(np.max(DWI_den) / sigma < np.max(DWI) / sigma)
    assert_(rmse_den < rmse_noisy)
    assert_(rmse_den_wrc < rmse_noisy_wrc)

#
# def test_lpca_ill_conditioned():
#     gtab = gen_gtab()
#     DWI, sigma = rfiw_phantom(gtab, snr=30)
#     assert_raises(ValueError, randomlpca_denoise, DWI)

# Not input Sigma- so we are not testing this
# def test_lpca_sigma_wrong_shape():
#     gtab = gen_gtab()
#     DWI, sigma = rfiw_phantom(gtab, snr=30)
#     print(sigma)
#     # If sigma is 3D but shape is not like DWI.shape[:-1], an error is raised:
#     assert_raises(ValueError, randomlpca_denoise, sigma)


if __name__ == '__main__':
    run_module_suite()
