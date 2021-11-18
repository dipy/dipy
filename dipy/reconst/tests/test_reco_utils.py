"""Testing reconstruction utilities."""

import numpy as np

from dipy.reconst.recspeed import (adj_to_countarrs,
                                   argmax_from_countarrs)
from dipy.testing import assert_true, assert_false
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_equal, assert_almost_equal, assert_raises)
from dipy.reconst.utils import (probabilistic_least_squares,
                                compute_unscaled_posterior_precision,
                                sample_multivariate_normal,
                                sample_multivariate_t,
                                percentiles_of_function)

from scipy.stats import t as tstats


def test_probabilistic_least_squares():

    # Test case: linear regression,
    # y = c_1 + c_2 * x
    # where true values are c_1 = 1, c_2 = 2

    A = np.array([[1, 0], [1, 1], [1, 2]])
    y = np.array([1, 3, 5])
    coef_ground_truth = np.array([1, 2])

    # Noise-less case
    coef, uncertainty_quantities = probabilistic_least_squares(A, y)
    assert_array_almost_equal(coef, coef_ground_truth)
    assert_almost_equal(uncertainty_quantities.residual_variance, 0)

    # Noisy case
    y_noisy = y + np.array([1, 2, -2])*1e-4
    coef, uncertainty_quantities = probabilistic_least_squares(A, y_noisy)
    assert_array_almost_equal(coef, coef_ground_truth, decimal=3)
    assert(uncertainty_quantities.residual_variance > 0)

    regularization_matrix = np.diag([0, 1e8])
    # This should force the second coefficient to (virtually) zero
    coef_expected = np.array([3, 0])
    coef, uncertainty_quantities = probabilistic_least_squares(
        A, y, regularization_matrix=regularization_matrix)
    assert_array_almost_equal(coef, coef_expected)
    assert_almost_equal(uncertainty_quantities.residual_variance, 4)

    # Test case: y = c_1 * x + c_2 * x^2
    # Test posterior mean and residual variance correct when no model error
    np.random.seed(0)
    n_x = int(1e4)
    x = np.linspace(-3, 3, n_x).reshape(-1, 1)
    A = np.column_stack((x, x ** 2))
    variance_ground_truth = 0.1
    y_noisy = (np.dot(A, coef_ground_truth)
               + np.sqrt(variance_ground_truth)
               * np.random.randn(n_x))
    coef, uncertainty_quantities = probabilistic_least_squares(A, y_noisy)
    assert_array_almost_equal(coef, coef_ground_truth, decimal=3)
    assert_almost_equal(uncertainty_quantities.residual_variance,
                        variance_ground_truth,
                        decimal=2)


def test_unscaled_posterior_precision():
    A = np.array([[1, 0], [1, 1], [1, 2]])
    regularization_matrix = np.arange(4).reshape(2, 2)

    out = compute_unscaled_posterior_precision(A)
    out_expected = np.dot(A.T, A)
    assert_array_almost_equal(out, out_expected)

    out = compute_unscaled_posterior_precision(
        A, regularization_matrix=regularization_matrix)
    out_expected = np.dot(A.T, A) + regularization_matrix
    assert_array_almost_equal(out, out_expected)


def test_covariance_from_precision():
    # Single voxel case
    Q = np.diag([1, 2, 3])
    expected_inverse = np.diag([1, 1/2, 1/3])

    out = np.linalg.pinv(Q)
    assert_array_almost_equal(out, expected_inverse)

    # Multi voxel case
    Q = np.diag([1, 2, 3])
    Q = np.array([1, 2]).reshape(2, 1, 1) * Q[None, :, :]
    expected_inverse = np.stack((np.diag([1, 1/2, 1/3]),
                                 np.diag([1/2, 1/4, 1/6])), axis=0)
    out = np.linalg.pinv(Q)
    assert_array_almost_equal(out, expected_inverse)


def test_sample_multivariate_normal():
    np.random.seed(0)

    mean = np.array([1, 2])
    n_coefs = len(mean)
    precision = np.array([[10, 1], [1, 20]])
    covariance = np.linalg.inv(precision)

    for using_precision in [True, False]:
        # Test that sample mean matches posterior mean
        n_samples = int(1e6)
        if using_precision:
            samples = sample_multivariate_normal(mean,
                                                 precision,
                                                 n_samples,
                                                 use_precision=using_precision)
        else:
            samples = sample_multivariate_normal(mean,
                                                 covariance,
                                                 n_samples,
                                                 use_precision=using_precision)
        samples_mean = np.mean(samples, -1, keepdims=False)
        assert_array_almost_equal(samples_mean, mean, decimal=3)

        # Test that sample covariance matches posterior variance
        samples_centered = samples - samples_mean[:, None]
        sample_covariance = (1/(n_samples - 1) *
                             np.dot(samples_centered, samples_centered.T))
        assert (np.linalg.norm(np.dot(sample_covariance, precision)
                - np.eye(n_coefs)) < 0.005)


def test_sample_multivariate_t():
    np.random.seed(0)

    mean = np.array([1, 2])
    n_coefs = len(mean)
    precision = np.array([[10, 1], [1, 20]])
    correlation = np.linalg.inv(precision)
    df = 5  # Note: this is pretty far from a Gaussian

    for using_precision in [True, False]:
        # Test that sample mean matches theoretical mean
        n_samples = int(1e6)
        if using_precision:
            samples = sample_multivariate_t(mean, precision,
                                            df, n_samples=n_samples,
                                            use_precision=using_precision)
        else:
            samples = sample_multivariate_t(mean, correlation,
                                            df, n_samples=n_samples,
                                            use_precision=using_precision)
        samples_mean = np.mean(samples, -1, keepdims=False)
        assert_array_almost_equal(samples_mean, mean, decimal=3)

        # Test that sample covariance matches theoretical variance
        samples_centered = samples - samples_mean[:, None]
        sample_covariance = (1/(n_samples - 1) *
                             np.dot(samples_centered, samples_centered.T))
        assert (np.linalg.norm(sample_covariance - df/(df - 2) * correlation)
                < 0.001)


def test_percentiles():
    np.random.seed(0)

    A = np.array((2, 3)).reshape(1, 2)
    b = 1
    probabilities = np.array((0.1, 0.25, 0.5, 0.75, 0.9))

    def affine_function(coeff):
        coeff = np.atleast_2d(coeff)
        return np.dot(A, coeff.T) + b

    mean = np.array([1, 2])

    precision = np.array([[10, 1], [1, 20]])
    correlation = np.linalg.inv(precision)
    df = 5

    function_mean = np.dot(A, mean) + b
    function_scale = np.sqrt(np.dot(np.dot(A, correlation), A.T))
    function_df = df
    expected_quantiles = tstats.ppf(probabilities,
                                    function_df,
                                    loc=function_mean,
                                    scale=function_scale)

    n_samples = int(1e6)
    for using_precision in [True, False]:
        if using_precision:
            observed_quantiles = percentiles_of_function(
                affine_function, mean, precision, df,
                probabilities=probabilities, n_samples=n_samples,
                use_precision=using_precision)
        else:
            observed_quantiles = percentiles_of_function(
                affine_function, mean, correlation, df,
                probabilities=probabilities, n_samples=n_samples,
                use_precision=using_precision)

        assert_array_almost_equal(observed_quantiles,
                                  np.squeeze(expected_quantiles),
                                  2)


def test_adj_countarrs():
    adj = [[0, 1, 2],
           [2, 3],
           [4, 5, 6, 7]]
    counts, inds = adj_to_countarrs(adj)
    assert_array_equal(counts, [3, 2, 4])
    assert_equal(counts.dtype.type, np.uint32)
    assert_array_equal(inds, [0, 1, 2, 2, 3, 4, 5, 6, 7])
    assert_equal(inds.dtype.type, np.uint32)


def test_argmax_from_countarrs():
    # basic case
    vals = np.arange(10, dtype=float)
    vertinds = np.arange(10, dtype=np.uint32)
    adj_counts = np.ones((10,), dtype=np.uint32)
    adj_inds_raw = np.arange(10, dtype=np.uint32)[::-1]
    # when contiguous - OK
    adj_inds = adj_inds_raw.copy()
    argmax_from_countarrs(vals, vertinds, adj_counts, adj_inds)
    # yield assert_array_equal(inds, [5, 6, 7, 8, 9])
    # test for errors - first - not contiguous
    #
    # The tests below cause odd errors and segfaults with numpy SVN
    # vintage June 2010 (sometime after 1.4.0 release) - see
    # http://groups.google.com/group/cython-users/browse_thread/thread/624c696293b7fe44?pli=1
    """
    yield assert_raises(ValueError,
                        argmax_from_countarrs,
                        vals,
                        vertinds,
                        adj_counts,
                        adj_inds_raw)
    # too few vertices
    yield assert_raises(ValueError,
                        argmax_from_countarrs,
                        vals,
                        vertinds[:-1],
                        adj_counts,
                        adj_inds)
    # adj_inds too short
    yield assert_raises(IndexError,
                        argmax_from_countarrs,
                        vals,
                        vertinds,
                        adj_counts,
                        adj_inds[:-1])
    # vals too short
    yield assert_raises(IndexError,
                        argmax_from_countarrs,
                        vals[:-1],
                        vertinds,
                        adj_counts,
                        adj_inds)
                        """
