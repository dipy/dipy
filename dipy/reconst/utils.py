from collections import namedtuple
import numpy as np
from scipy.stats import t
from scipy.linalg import solve_triangular


def dki_design_matrix(gtab):
    r"""Construct B design matrix for DKI.

    Parameters
    ----------
    gtab : GradientTable
        Measurement directions.

    Returns
    -------
    B : array (N, 22)
        Design matrix or B matrix for the DKI model
        B[j, :] = (Bxx, Bxy, Bzz, Bxz, Byz, Bzz,
                   Bxxxx, Byyyy, Bzzzz, Bxxxy, Bxxxz,
                   Bxyyy, Byyyz, Bxzzz, Byzzz, Bxxyy,
                   Bxxzz, Byyzz, Bxxyz, Bxyyz, Bxyzz,
                   BlogS0)

    """
    b = gtab.bvals
    bvec = gtab.bvecs

    B = np.zeros((len(b), 22))
    B[:, 0] = -b * bvec[:, 0] * bvec[:, 0]
    B[:, 1] = -2 * b * bvec[:, 0] * bvec[:, 1]
    B[:, 2] = -b * bvec[:, 1] * bvec[:, 1]
    B[:, 3] = -2 * b * bvec[:, 0] * bvec[:, 2]
    B[:, 4] = -2 * b * bvec[:, 1] * bvec[:, 2]
    B[:, 5] = -b * bvec[:, 2] * bvec[:, 2]
    B[:, 6] = b * b * bvec[:, 0]**4 / 6
    B[:, 7] = b * b * bvec[:, 1]**4 / 6
    B[:, 8] = b * b * bvec[:, 2]**4 / 6
    B[:, 9] = 4 * b * b * bvec[:, 0]**3 * bvec[:, 1] / 6
    B[:, 10] = 4 * b * b * bvec[:, 0]**3 * bvec[:, 2] / 6
    B[:, 11] = 4 * b * b * bvec[:, 1]**3 * bvec[:, 0] / 6
    B[:, 12] = 4 * b * b * bvec[:, 1]**3 * bvec[:, 2] / 6
    B[:, 13] = 4 * b * b * bvec[:, 2]**3 * bvec[:, 0] / 6
    B[:, 14] = 4 * b * b * bvec[:, 2]**3 * bvec[:, 1] / 6
    B[:, 15] = b * b * bvec[:, 0]**2 * bvec[:, 1]**2
    B[:, 16] = b * b * bvec[:, 0]**2 * bvec[:, 2]**2
    B[:, 17] = b * b * bvec[:, 1]**2 * bvec[:, 2]**2
    B[:, 18] = 2 * b * b * bvec[:, 0]**2 * bvec[:, 1] * bvec[:, 2]
    B[:, 19] = 2 * b * b * bvec[:, 1]**2 * bvec[:, 0] * bvec[:, 2]
    B[:, 20] = 2 * b * b * bvec[:, 2]**2 * bvec[:, 0] * bvec[:, 1]
    B[:, 21] = -np.ones(len(b))

    return B


def _roi_in_volume(data_shape, roi_center, roi_radii):
    """Ensures that a cuboid ROI is in a data volume.

    Parameters
    ----------
    data_shape : ndarray
        Shape of the data
    roi_center : ndarray, (3,)
        Center of ROI in data.
    roi_radii : ndarray, (3,)
        Radii of cuboid ROI

    Returns
    -------
    roi_radii : ndarray, (3,)
        Truncated radii of cuboid ROI. It remains unchanged if
        the ROI was already contained inside the data volume.
    """

    for i in range(len(roi_center)):
        inf_lim = int(roi_center[i] - roi_radii[i])
        sup_lim = int(roi_center[i] + roi_radii[i])
        if inf_lim < 0 or sup_lim >= data_shape[i]:
            roi_radii[i] = min(int(roi_center[i]),
                               int(data_shape[i] - roi_center[i]))
    return roi_radii


def _mask_from_roi(data_shape, roi_center, roi_radii):
    """Produces a mask from a cuboid ROI defined by center and radii.

    Parameters
    ----------
    data_shape : array-like, (3,)
        Shape of the data from which the ROI is taken.
    roi_center : array-like, (3,)
        Center of ROI in data.
    roi_radii : array-like, (3,)
        Radii of cuboid ROI.

    Returns
    -------
    mask : ndarray
        Mask of the cuboid ROI.
    """

    ci, cj, ck = roi_center
    wi, wj, wk = roi_radii
    interval_i = slice(int(ci - wi), int(ci + wi) + 1)
    interval_j = slice(int(cj - wj), int(cj + wj) + 1)
    interval_k = slice(int(ck - wk), int(ck + wk) + 1)

    if wi == 0:
        interval_i = ci
    elif wj == 0:
        interval_j = cj
    elif wk == 0:
        interval_k = ck

    mask = np.zeros(data_shape, dtype=np.int64)
    mask[interval_i, interval_j, interval_k] = 1

    return mask


probabilistic_ls_quantities = namedtuple("probabilistic_ls_quantities",
                                         "residual_variance,\
                                          degrees_of_freedom,\
                                          unscaled_posterior_covariance")


def probabilistic_least_squares(design_matrix, y, regularization_matrix=None):
    # Solve least-squares problem on the form
    # design_matrix * coef = y

    unscaled_posterior_covariance, pseudoInv, degrees_of_freedom = \
        get_data_independent_estimation_quantities(design_matrix, regularization_matrix)

    coef_posterior_mean = np.einsum('...ij, ...j->...i', pseudoInv, y)

    residuals = y - np.einsum('...ij, ...j->...i', design_matrix, coef_posterior_mean)
    residual_variance = (np.sum(residuals ** 2, axis=-1) / degrees_of_freedom)

    if y.ndim == 1:
        uncertainty_params = probabilistic_ls_quantities(residual_variance,
                                                         degrees_of_freedom,
                                                         unscaled_posterior_covariance)
    else:
        uncertainty_params = np.empty(y.shape[0], dtype=object)
        for i in range(y.shape[0]):
            if design_matrix.ndim == 2:
                # Ordinary least-squares: identical design matrix for all voxels
                uncertainty_params[i] = probabilistic_ls_quantities(residual_variance[i],
                                                                    degrees_of_freedom,
                                                                    unscaled_posterior_covariance)
            else:
                uncertainty_params[i] = probabilistic_ls_quantities(residual_variance[i],
                                                                    degrees_of_freedom[i],
                                                                    unscaled_posterior_covariance[i, ...])
    return coef_posterior_mean, uncertainty_params


def get_data_independent_estimation_quantities(design_matrix, regularization_matrix=None):

    S = compute_S(design_matrix, regularization_matrix)
    pseudoInv = np.linalg.solve(S, np.swapaxes(design_matrix, -1, -2))
    unscaled_posterior_covariance = np.einsum('...ik, ...jk->...ij', pseudoInv, pseudoInv)
    degrees_of_freedom = compute_degrees_of_freedom(design_matrix, pseudoInv)

    return unscaled_posterior_covariance, pseudoInv, degrees_of_freedom


def compute_S(design_matrix, regularization_matrix=None):

    if regularization_matrix is None:
        # In single voxel case: np.dot(design_matrix.T, design_matrix)
        S = np.einsum('...ki, ...kj->...ij', design_matrix, design_matrix)
    else:
        # In single voxel case: np.dot(design_matrix.T, design_matrix) + regularization_matrix
        S = (np.einsum('...ki, ...kj->...ij', design_matrix, design_matrix)
                                        + regularization_matrix)
    return S


def compute_degrees_of_freedom(design_matrix, pseudoInv):

    smoother_matrix = np.einsum('...ik, ...kj->...ij', design_matrix, pseudoInv)
    residual_matrix = np.eye(smoother_matrix.shape[-1]) - smoother_matrix
    degrees_of_freedom = np.sum(residual_matrix ** 2, axis=(-1, -2))

    return np.atleast_1d(degrees_of_freedom)


def t_confidence_interval(mean, scale, degrees_of_freedom, confidence=0.95):
    interval = t.interval(confidence, degrees_of_freedom, loc=mean, scale=scale)
    return interval


def t_quantile_function(mean, scale, degrees_of_freedom, quantile):
    out = t.ppf(quantile, degrees_of_freedom, loc=mean, scale=scale)
    return out


def percentiles_of_function(fun, mean, correlation_or_precision, degrees_of_freedom,
                            probabilities=None, n_samples=1000, use_precision=False):
    if probabilities is None:
        probabilities = np.arange(0.05, 0.95, 0.05)

    samples = sample_multivariate_t(mean,
                                    correlation_or_precision,
                                    degrees_of_freedom,
                                    n_samples=n_samples,
                                    use_precision=use_precision)

    empirical_percentiles = np.nanpercentile(fun(samples.T), probabilities * 100)
    return empirical_percentiles


def sample_multivariate_t(mean, correlation_or_precision, degrees_of_freedom,
                          n_samples=1, keepdims=False, use_precision=False):

    mean = np.atleast_2d(mean)
    n_voxels, n_coefs = mean.shape

    x = np.random.chisquare(degrees_of_freedom, (n_voxels, n_samples)) / degrees_of_freedom
    x = x[:, None, :]

    z = sample_multivariate_normal(np.zeros_like(mean), correlation_or_precision,
                                   n_samples, keepdims=True, use_precision=use_precision)

    samples = mean[..., None] + z / np.sqrt(x)

    if not keepdims:
        samples = np.squeeze(samples)

    return samples


def sample_multivariate_normal(mean, covariance_or_precision, n_samples,
                               keepdims=False, use_precision=False):
    mean = np.atleast_2d(mean)
    n_voxels, n_coefs = mean.shape
    covariance_or_precision = covariance_or_precision.reshape(n_voxels, n_coefs, n_coefs)
    samples = np.zeros((n_voxels, n_coefs, n_samples))

    # Loop over voxels and draw samples for each
    for i in range(n_voxels):
        if use_precision:
            standard_normal_samples = np.random.randn(n_coefs, n_samples)
            L = np.linalg.cholesky(covariance_or_precision[i, :, :])
            samples[i, :, :] = (mean[i, :, None] +
                                solve_triangular(L.T, standard_normal_samples))
        else:
            mvn_samples = np.random.multivariate_normal(mean[i, :], covariance_or_precision[i, :, :], (n_samples,)).T
            samples[i, :, :] = mvn_samples[None, ...]

    if not keepdims:
        samples = np.squeeze(samples)

    return samples