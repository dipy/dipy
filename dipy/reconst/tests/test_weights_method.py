"""Testing weights methods"""

import numpy as np
from numpy.testing import (
    assert_almost_equal,
    assert_equal,
    assert_raises,
)

import dipy.core.gradients as grad
from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
import dipy.reconst.dti as dti
from dipy.reconst.weights_method import (
    simple_cutoff,
    two_eyes_cutoff,
    weights_method_nlls_m_est,
    weights_method_wls_m_est,
)
from dipy.testing.decorators import set_random_number_generator

MIN_POSITIVE_SIGNAL = 0.0001


@set_random_number_generator()
def test_outlier_funcs(rng):
    """Test functions that define outliers."""

    # true signal
    b0 = 1000.0
    bval, bvecs = read_bvals_bvecs(*get_fnames(name="55dir_grad"))
    gtab = grad.gradient_table(bval, bvecs=bvecs)
    B = bval[1]
    D_orig = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 1.0, -np.log(b0) * B]) / B
    design_matrix = dti.design_matrix(gtab)
    log_pred_sig = np.dot(design_matrix, D_orig)
    pred_sig = np.exp(log_pred_sig)

    # noisy signal
    scale = 1
    error = rng.normal(scale=scale, size=pred_sig.shape)
    Y = pred_sig + error
    Y[Y < MIN_POSITIVE_SIGNAL] = MIN_POSITIVE_SIGNAL
    Y[0] = Y[0] * 100  # make 1 signal into a super outlier

    # other args
    residuals = Y - pred_sig  # may differ from residuals due to signal clip
    log_residuals = np.log(Y) - log_pred_sig
    # make some fake leverages (they should sum to npar=7)
    leverages = np.ones_like(Y) * D_orig.shape[0] / Y.shape[0]
    C = scale  # since we added noise, just set C to scale

    for outlier_func in [simple_cutoff, two_eyes_cutoff]:
        # use extremely generous cutoff of 6; find only the super outlier
        outlier = outlier_func(
            residuals, log_residuals, pred_sig, design_matrix, leverages, C, cutoff=6
        )
        assert_equal(outlier[0], True)
        assert_equal(outlier[1:], np.zeros_like(outlier[1:], dtype=bool))

        # make everything an outlier, by using a cut-off of zero
        outlier = outlier_func(
            residuals, log_residuals, pred_sig, design_matrix, leverages, C, cutoff=0.0
        )
        assert_equal(outlier.sum(), Y.shape[0])


@set_random_number_generator()
def test_weights_funcs(rng):
    """Test functions that define weights."""

    # true signal
    b0 = 1000.0
    bval, bvecs = read_bvals_bvecs(*get_fnames(name="55dir_grad"))
    gtab = grad.gradient_table(bval, bvecs=bvecs)
    B = bval[1]
    D_orig = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 1.0, -np.log(b0) * B]) / B
    design_matrix = dti.design_matrix(gtab)
    log_pred_sig = np.dot(design_matrix, D_orig)
    pred_sig = np.exp(log_pred_sig)

    # noisy signal
    scale = 1
    error = rng.normal(scale=scale, size=pred_sig.shape)
    Y = pred_sig + error
    Y[Y < MIN_POSITIVE_SIGNAL] = MIN_POSITIVE_SIGNAL
    Y[0] = Y[0] * 100  # make 1 signal into a super outlier

    # make some fake leverages (they should sum to npar=7)
    leverages = np.ones_like(Y) * D_orig.shape[0] / Y.shape[0]

    # set last robust to None ?
    # it's only currently used on the last iteration of WLS, because we need to
    # perform a clean OLS fit first
    # in general, weight functions for WLS will need to do this
    # makes sense that they generally have access to the defined 'last_robust'
    # from previous iterations, for the most generality

    # TODO: need to test about the number of iterations, to make sure errors are raised
    # TODO: robust return should be None expect on last for NLLS,
    #       and second to last and last on WLS

    NUM_ITER = 4
    for fit_type, weights_func in zip(
        ["wls", "nlls"], [weights_method_wls_m_est, weights_method_nlls_m_est]
    ):
        # invalid estimator choice
        assert_raises(
            ValueError,
            weights_func,
            Y,
            pred_sig,
            design_matrix,
            leverages,
            1,
            10,
            None,
            m_est="unknown_estimator",
        )

        # test error if <4 iters for wls, or <3 iters for nlls
        assert_raises(
            ValueError,
            weights_func,
            Y,
            pred_sig,
            design_matrix,
            leverages,
            1,
            3 if fit_type == "wls" else 2,
            None,
        )

        # test error if not enough data points
        assert_raises(
            ValueError,
            weights_func,
            Y[0:6],
            pred_sig[0:6],
            design_matrix[0:6],
            leverages[0:6],
            1,
            10,
            None,
        )

        for mest in ["gm", "cauchy"]:
            for outlier_func in [simple_cutoff, two_eyes_cutoff]:
                last_robust = None
                for idx in range(1, NUM_ITER + 1):
                    # neither last, nor second-to-last, iteration
                    weights, robust = weights_func(
                        Y,
                        pred_sig,
                        design_matrix,
                        leverages,
                        idx=idx,
                        total_idx=NUM_ITER,
                        last_robust=last_robust,
                        m_est=mest,
                        cutoff=3,
                        outlier_condition_func=outlier_func,
                    )
                    last_robust = robust

                    if idx >= NUM_ITER - 1 and fit_type == "wls":
                        # for second-to-last and last iter, robust not None
                        assert_equal(robust.shape, Y.shape)
                        # check that the super-outlier has 0 weight
                        assert_almost_equal(weights[0], 0)

                    if idx == NUM_ITER and fit_type == "nlls":
                        # for last iter, robust not None
                        assert_equal(robust.shape, Y.shape)
                        # check that the super-outlier has 0 weight
                        assert_almost_equal(weights[0], 0)

                    if idx < NUM_ITER - 1:  # iter okay for both wls & nlls
                        assert_equal(robust, None)
                        assert_equal(np.all(weights > 0), True)
