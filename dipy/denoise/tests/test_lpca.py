import warnings

import numpy as np
import scipy.special as sps
from numpy.testing import (assert_,
                           assert_equal,
                           assert_raises,
                           assert_array_almost_equal,
                           assert_warns)
from dipy.denoise.localpca import (
    dimensionality_problem_message, create_patch_radius_arr, compute_patch_size,
    compute_num_samples, compute_suggested_patch_radius, localpca, mppca,
    genpca, _pca_classifier)
from dipy.sims.voxel import multi_tensor
from dipy.core.gradients import gradient_table, generate_bvecs
from dipy.testing.decorators import set_random_number_generator


def setup_module():
    global gtab

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
        if rng is None:
            rng = np.random.default_rng()
        sigma = S2 * 1.0 / snr
        n1 = rng.normal(0, sigma, size=DWI.shape)
        n2 = rng.normal(0, sigma, size=DWI.shape)
        return [np.sqrt((DWI / np.sqrt(2) + n1)**2 +
                        (DWI / np.sqrt(2) + n2)**2), sigma]


def test_lpca_static():
    S0 = 100 * np.ones((20, 20, 20, 20), dtype='f8')
    S0ns = localpca(S0, sigma=np.ones((20, 20, 20), dtype=np.float64))
    assert_array_almost_equal(S0, S0ns)


@set_random_number_generator()
def test_lpca_random_noise(rng):
    S0 = 100 + 2 * rng.standard_normal((22, 23, 30, 20))
    S0ns = localpca(S0, sigma=np.std(S0))

    assert_(S0ns.min() > S0.min())
    assert_(S0ns.max() < S0.max())
    assert_equal(np.round(S0ns.mean()), 100)


@set_random_number_generator()
def test_lpca_boundary_behaviour(rng):
    # check is first slice is getting denoised or not ?
    S0 = 100 * np.ones((20, 20, 20, 20), dtype='f8')
    S0[:, :, 0, :] = S0[:, :, 0, :] + 2 * \
        rng.standard_normal((20, 20, 20))
    S0_first = S0[:, :, 0, :]
    S0ns = localpca(S0, sigma=np.std(S0))
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


@set_random_number_generator()
def test_lpca_rmse(rng):
    S0_w_noise = 100 + 2 * rng.standard_normal((22, 23, 30, 20))
    rmse_w_noise = np.sqrt(np.mean((S0_w_noise - 100) ** 2))
    S0_denoised = localpca(S0_w_noise, sigma=np.std(S0_w_noise))
    rmse_denoised = np.sqrt(np.mean((S0_denoised - 100) ** 2))
    # Denoising should always improve the RMSE:
    assert_(rmse_denoised < rmse_w_noise)


@set_random_number_generator()
def test_lpca_sharpness(rng):
    S0 = np.ones((30, 30, 30, 20), dtype=np.float64) * 100
    S0[10:20, 10:20, 10:20, :] = 50
    S0[20:30, 20:30, 20:30, :] = 0
    S0 = S0 + 20 * rng.standard_normal((30, 30, 30, 20))
    S0ns = localpca(S0, sigma=20.0)
    # check the edge gradient
    edgs = np.abs(np.mean(S0ns[8, 10:20, 10:20] - S0ns[12, 10:20, 10:20]) - 50)
    assert_(edgs < 2)


def test_lpca_dtype():
    # If out_dtype is not specified, we retain the original precision:
    S0 = 200 * np.ones((20, 20, 20, 3), dtype=np.float64)
    S0ns = localpca(S0, sigma=1)
    assert_equal(S0.dtype, S0ns.dtype)

    S0 = 200 * np.ones((20, 20, 20, 20), dtype=np.uint16)
    S0ns = localpca(S0, sigma=np.ones((20, 20, 20)))
    assert_equal(S0.dtype, S0ns.dtype)

    # If we set out_dtype, we get what we asked for:
    S0 = 200 * np.ones((20, 20, 20, 20), dtype=np.uint16)
    S0ns = localpca(S0, sigma=np.ones((20, 20, 20)),
                    out_dtype=np.float32)
    assert_equal(np.float32, S0ns.dtype)

    # If we set a few entries to zero, this induces negative entries in the
    # Resulting denoised array:
    S0[5:8, 5:8, 5:8] = 0
    # But if we should always get all non-negative results:
    S0ns = localpca(S0, sigma=np.ones((20, 20, 20)), out_dtype=np.uint16)
    assert_(np.all(S0ns >= 0))
    # And no wrap-around to crazy high values:
    assert_(np.all(S0ns <= 200))


def test_lpca_wrong():
    S0 = np.ones((20, 20))
    assert_raises(ValueError, localpca, S0, sigma=1)


@set_random_number_generator()
def test_phantom(rng):
    DWI_clean = rfiw_phantom(gtab, snr=None, rng=rng)
    DWI, sigma = rfiw_phantom(gtab, snr=30, rng=rng)
    # To test without Rician correction
    temp = (DWI_clean / sigma)**2
    DWI_clean_wrc = (sigma * np.sqrt(np.pi / 2) * np.exp(-0.5 * temp) *
                     ((1 + 0.5 * temp) * sps.iv(0, 0.25 * temp) + 0.5 * temp *
                     sps.iv(1, 0.25 * temp))**2)

    DWI_den = localpca(DWI, sigma, patch_radius=3)
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

    # Check if the results of different PCA methods (eig, svd) are similar
    DWI_den_svd = localpca(DWI, sigma, pca_method='svd', patch_radius=3)
    assert_array_almost_equal(DWI_den, DWI_den_svd)

    assert_raises(ValueError, localpca, DWI, sigma, pca_method='empty')

    # Try this with a sigma volume, instead of a scalar
    sigma_vol = sigma * np.ones(DWI.shape[:-1])
    mask = np.zeros_like(DWI, dtype=bool)[..., 0]
    mask[2:-2, 2:-2, 2:-2] = True
    DWI_den = localpca(DWI, sigma_vol, mask, patch_radius=3)
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


@set_random_number_generator()
def test_lpca_ill_conditioned(rng):
    DWI, sigma = rfiw_phantom(gtab, snr=30, rng=rng)
    for patch_radius in [1, [1, 1, 1]]:
        assert_warns(UserWarning, localpca, DWI, sigma,
                      patch_radius=patch_radius)


@set_random_number_generator()
def test_lpca_radius_wrong_shape(rng):
    DWI, sigma = rfiw_phantom(gtab, snr=30, rng=rng)
    for patch_radius in [[2, 2], [2, 2, 2, 2]]:
        assert_raises(ValueError, localpca, DWI, sigma,
                      patch_radius=patch_radius)


@set_random_number_generator()
def test_lpca_sigma_wrong_shape(rng):
    DWI, sigma = rfiw_phantom(gtab, snr=30, rng=rng)
    # If sigma is 3D but shape is not like DWI.shape[:-1], an error is raised:
    sigma = np.zeros((DWI.shape[0], DWI.shape[1] + 1, DWI.shape[2]))
    assert_raises(ValueError, localpca, DWI, sigma)


@set_random_number_generator()
def test_lpca_no_gtab_no_sigma(rng):
    DWI, sigma = rfiw_phantom(gtab, snr=30, rng=rng)
    assert_raises(ValueError, localpca, DWI, None, None)


@set_random_number_generator()
def test_pca_classifier(rng):
    # Produce small phantom with well aligned single voxels and ground truth
    # snr = 50, i.e signal std = 0.02 (Gaussian noise)
    std_gt = 0.02
    S0 = 1.0
    ndir = gtab.bvals.size
    signal_test = np.zeros((5, 5, 5, ndir))
    mevals = np.array([[0.99e-3, 0.0, 0.0], [2.26e-3, 0.87e-3, 0.87e-3]])
    sig, direction = multi_tensor(gtab, mevals, S0=S0,
                                  angles=[(0, 0, 1), (0, 0, 1)],
                                  fractions=(50, 50), snr=None)
    signal_test[..., :] = sig
    noise = std_gt*rng.standard_normal((5, 5, 5, ndir))
    dwi_test = signal_test + noise

    # Compute eigenvalues
    X = dwi_test.reshape(125, ndir)
    M = np.mean(X, axis=0)
    X = X - M
    [L, W] = np.linalg.eigh(np.dot(X.T, X)/125)

    # Find number of noise related eigenvalues
    var, c = _pca_classifier(L, 125)
    std = np.sqrt(var)

    # Expected number of signal components is 0 because the phantom only has
    # one voxel type and that information is captured by the mean of X.
    # Therefore, expected noise components should be equal to size of L.
    # To allow some margin of error let's assess if c is higher than
    # L.size - 3.
    assert_(c > L.size-3)

    # Let's check if noise std estimate as an error less than 5%
    std_error = abs(std - std_gt)/std_gt * 100
    assert_(std_error < 5)


@set_random_number_generator()
def test_mppca_in_phantom(rng):
    DWIgt = rfiw_phantom(gtab, snr=None, rng=rng)
    std_gt = 0.02
    noise = std_gt*rng.standard_normal(DWIgt.shape)
    DWInoise = DWIgt + noise

    # patch radius (2: #samples > #features, 1: #samples < #features)
    for PR in [2, 1]:
        if PR == 1:
            patch_radius_arr = create_patch_radius_arr(DWInoise, PR)
            patch_size = compute_patch_size(patch_radius_arr)
            num_samples = compute_num_samples(patch_size)
            spr = compute_suggested_patch_radius(DWInoise, patch_size)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=dimensionality_problem_message(
                        DWInoise, num_samples, spr),
                    category=UserWarning)
                DWIden = mppca(DWInoise, patch_radius=PR)
        else:
            DWIden = mppca(DWInoise, patch_radius=PR)

        # Test if denoised data is closer to ground truth than noisy data
        rmse_den = np.sum(np.abs(DWIgt - DWIden)) / np.sum(np.abs(DWIgt))
        rmse_noisy = np.sum(np.abs(DWIgt - DWInoise)) / np.sum(np.abs(DWIgt))
        assert_(rmse_den < rmse_noisy)


@set_random_number_generator()
def test_create_patch_radius_arr(rng):

    shape = (10, 10, 8, 104)
    arr = rng.standard_normal(shape)
    pr = 2
    expected_val = np.asarray([2, 2, 2])
    obtained_val = create_patch_radius_arr(arr, pr)
    assert np.array_equal(obtained_val, expected_val)


def test_compute_patch_size():

    patch_radius = 1
    expected_val = 3
    obtained_val = compute_patch_size(patch_radius)
    assert obtained_val == expected_val

    patch_radius = 2
    expected_val = 5
    obtained_val = compute_patch_size(patch_radius)
    assert obtained_val == expected_val


def test_compute_num_samples():

    patch_size = np.asarray([5, 5, 5])
    expected_val = 125
    obtained_val = compute_num_samples(patch_size)
    assert obtained_val == expected_val


@set_random_number_generator()
def test_compute_suggested_patch_radius(rng):

    shape = (10, 10, 8, 104)
    arr = rng.standard_normal(shape)
    patch_size = [3, 3, 3]
    expected_val = 2
    obtained_val = compute_suggested_patch_radius(arr, patch_size)
    assert obtained_val == expected_val

    patch_size = [5, 5, 5]
    obtained_val = compute_suggested_patch_radius(arr, patch_size)
    assert obtained_val == expected_val


@set_random_number_generator()
def test_mppca_returned_sigma(rng):
    DWIgt = rfiw_phantom(gtab, snr=None, rng=rng)
    std_gt = 0.02
    noise = std_gt*rng.standard_normal(DWIgt.shape)
    DWInoise = DWIgt + noise

    # patch radius (2: #samples > #features, 1: #samples < #features)
    for PR in [2, 1]:

        # Case that sigma is estimated using mpPCA
        if PR == 1:
            patch_radius_arr = create_patch_radius_arr(DWInoise, PR)
            patch_size = compute_patch_size(patch_radius_arr)
            num_samples = compute_num_samples(patch_size)
            spr = compute_suggested_patch_radius(DWInoise, patch_size)

        if PR == 1:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=dimensionality_problem_message(
                        DWInoise, num_samples, spr),
                    category=UserWarning)
                DWIden0, sigma = mppca(
                    DWInoise, patch_radius=PR, return_sigma=True)
        else:
            DWIden0, sigma = mppca(DWInoise, patch_radius=PR, return_sigma=True)
        msigma = np.mean(sigma)
        std_error = abs(msigma - std_gt)/std_gt * 100

        # if #noise_eigenvals >> #signal_eigenvals, variance should be well estimated
        # this is more likely achieved if #samples > #features
        if PR > 1:
            assert_(std_error < 5)
        else: # otherwise, variance estimate may be wrong
            pass

        # Case that sigma is inputted (sigma outputted should be the same as the
        # one inputted)
        if PR == 1:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=dimensionality_problem_message(DWInoise,
                                                           num_samples, spr),
                    category=UserWarning)
                DWIden1, rsigma = genpca(DWInoise, sigma=sigma, tau_factor=None,
                                         patch_radius=PR, return_sigma=True)
        else:
            DWIden1, rsigma = genpca(DWInoise, sigma=sigma, tau_factor=None,
                                     patch_radius=PR, return_sigma=True)

        assert_array_almost_equal(rsigma, sigma)

        # DWIden1 should be very similar to DWIden0
        rmse_den = np.sum(np.abs(DWIden1 - DWIden0)) / np.sum(np.abs(DWIden0))
        rmse_ref = np.sum(np.abs(DWIden1 - DWIgt)) / np.sum(np.abs(DWIgt))
        assert_(rmse_den < rmse_ref)
