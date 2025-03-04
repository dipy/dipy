#!/usr/bin/python
"""Functions for defining weights for iterative fitting methods."""

import numpy as np

MIN_POSITIVE_SIGNAL = 0.0001


def simple_cutoff(
    residuals, log_residuals, pred_sig, design_matrix, leverages, C, cutoff
):
    """Define outliers based on the signal (rather than the log-signal).

    Parameters
    ----------
    residuals : ndarray
        Residuals of the signal (observed signal - fitted signal).
    log_residuals : ndarray
        Residuals of the log signal (log observed signal - fitted log signal).
    pred_sig : ndarray
        The predicted signal, given a previous fit.
    design_matrix : ndarray (g, ...)
        Design matrix holding the covariants used to solve for the
        regression coefficients.
    leverages : ndarray
        The leverages (diagonal of the 'hat matrix') of the fit.
    C : float
        Estimate of the standard deviation of the error.
    cutoff : float, optional
        Cut-off value for defining outliers based on fitting residuals.
        Here the condition is::

            |residuals| > cut_off x C x HAT_factor

        where HAT_factor = sqrt(1 - leverages) adjusts for leverage effects.

    """
    leverages[np.isclose(leverages, 1.0)] = 0.99  # avoids rare issues
    HAT_factor = np.sqrt(1 - leverages)
    cond = np.abs(residuals) > +cutoff * C * HAT_factor
    return cond


def two_eyes_cutoff(
    residuals, log_residuals, pred_sig, design_matrix, leverages, C, cutoff
):
    """Define outliers with two-eyes approach.

    see :footcite:p:`Collier2015` for more details.

    Parameters
    ----------
    residuals : ndarray
        Residuals of the signal (observed signal - fitted signal).
    log_residuals : ndarray
        Residuals of the log signal (log observed signal - fitted log signal).
    pred_sig : ndarray
        The predicted signal, given a previous fit.
    design_matrix : ndarray (g, ...)
        Design matrix holding the covariants used to solve for the
        regression coefficients.
    leverages : ndarray
        The leverages (diagonal of the 'hat matrix') of the fit.
    C : float
        Estimate of the standard deviation of the error.
    cutoff : float, optional
        Cut-off value for defining outliers based on fitting residuals,
        see :footcite:p:`Collier2015` for the two-eyes approached used here.

    References
    ----------
    .. footbibliography::

    """
    leverages[np.isclose(leverages, 1.0)] = 0.99  # avoids rare issues
    HAT_factor = np.sqrt(1 - leverages)
    cond = (residuals > +cutoff * C * HAT_factor) | (
        log_residuals < -cutoff * C * HAT_factor / pred_sig
    )
    return cond


def weights_method_wls_m_est(
    data,
    pred_sig,
    design_matrix,
    leverages,
    idx,
    total_idx,
    last_robust,
    *,
    m_est="gm",
    cutoff=3,
    outlier_condition_func=simple_cutoff,
):
    """Calculate M-estimator weights for WLS model.

    Parameters
    ----------
    data : ndarray
        The measured signal.
    pred_sig : ndarray
        The predicted signal, given a previous fit.
        Has the same shape as data.
    design_matrix : ndarray (g, ...)
        Design matrix holding the covariants used to solve for the
        regression coefficients.
    leverages : ndarray
        The leverages (diagonal of the 'hat matrix') of the fit.
    idx : int
        The current iteration number.
    total_idx : int
        The total number of iterations.
    last_robust : ndarray
        True for inlier indices and False for outlier indices. Must have the
        same shape as data.
    m_est : str, optional.
        M-estimator weighting scheme to use. Currently,
        'gm' (Geman-McClure) and 'cauchy' are provided.
    cutoff : float, optional
        Cut-off value for defining outliers based on fitting residuals.
        Will be passed to the outlier_condition_func.
        Typical example: ``|residuals| > cut_off x standard_deviation``
    outlier_condition_func : callable, optional
        A function with args and returns as follows::

            is_an_outlier = outlier_condition_func(residuals, log_residuals,
                pred_sig, design_matrix, leverages, C, cutoff)

    Notes
    -----
    Robust weights are calculated specifically for the WLS problem, i.e. the
    usual form of the WLS problem is accounted for when defining these new
    weights, see :footcite:p:`Collier2015`. On the second-to-last iteration,
    OLS is performed without outliers. On the last iteration, WLS is performed
    without outliers.

    References
    ----------
    .. footbibliography::

    """
    # check if M-estimator is valid (defined in this function)
    if m_est not in ["gm", "cauchy"]:
        raise ValueError("unknown M-estimator choice")

    # min 4 iters: (1) WLS (2) WLS with M-weights (3) clean OLS (3) clean WLS
    if total_idx < 4:
        raise ValueError("require 4+ iterations")

    p, N = design_matrix.shape[-1], data.shape[-1]
    if N <= p:
        raise ValueError("Fewer data points than parameters.")
    factor = 1.4826 * np.sqrt(N / (N - p))

    # handle potential zeros
    pred_sig[pred_sig < MIN_POSITIVE_SIGNAL] = MIN_POSITIVE_SIGNAL

    # calculate quantities needed for C and w
    log_pred_sig = np.log(pred_sig)
    residuals = data - pred_sig
    log_data = np.log(data)
    log_residuals = log_data - log_pred_sig
    z = pred_sig * log_residuals

    # IRLLS paper eq9 corrected (weights for log_residuals^2 are pred_sig^2)
    C = (
        factor
        * np.median(np.abs(z - np.median(z, axis=-1)[..., None]), axis=-1)[..., None]
    )
    C[C == 0] = np.nanmedian(C)  # C could be 0, if all signals = min_signal

    # NOTE: if more M-estimators are added, please update the docs!
    if m_est == "gm":
        w = (C / pred_sig) ** 2 / ((C / pred_sig) ** 2 + log_residuals**2) ** 2
    if m_est == "cauchy":
        w = C**2 / ((C / pred_sig) ** 2 + log_residuals**2)

    robust = None

    if idx == total_idx - 1:  # OLS without outliers
        cond = outlier_condition_func(
            residuals, log_residuals, pred_sig, design_matrix, leverages, C, cutoff
        )
        robust = np.logical_not(cond)

        w[~robust] = 0.0
        w[robust] = 1.0

    if idx == total_idx:  # WLS without outliers
        robust = last_robust
        w[~robust] = 0.0
        w[robust] = pred_sig[robust == 1] ** 2

    w[np.isinf(w)] = 0
    w[np.isnan(w)] = 0

    return w, robust


def weights_method_nlls_m_est(
    data,
    pred_sig,
    design_matrix,
    leverages,
    idx,
    total_idx,
    last_robust,
    *,
    m_est="gm",
    cutoff=3,
    outlier_condition_func=simple_cutoff,
):
    """Calculate M-estimator weights for NLLS model.

    Parameters
    ----------
    data : ndarray
        The measured signal.
    pred_sig : ndarray
        The predicted signal, given a previous fit.
        Has the same shape as data.
    design_matrix : ndarray (g, ...)
        Design matrix holding the covariants used to solve for the
        regression coefficients.
    leverages : ndarray
        The leverages (diagonal of the 'hat matrix') of the fit.
    idx : int
        The current iteration number.
    total_idx : int
        The total number of iterations.
    last_robust : ndarray
        True for inlier indices and False for outlier indices. Must have the
        same shape as data.
    m_est : str, optional.
        M-estimator weighting scheme to use. Currently,
        'gm' (Geman-McClure) and 'cauchy' are provided.
    cutoff : float, optional
        Cut-off value for defining outliers based on fitting residuals.
        Will be passed to the outlier_condition_func.
        Typical example: ``|residuals| > cut_off x standard_deviation``
    outlier_condition_func : callable, optional
        A function with args and returns as follows::

            is_an_outlier = outlier_condition_func(residuals, log_residuals,
                pred_sig, design_matrix, leverages, C, cutoff)

    Notes
    -----
    Robust weights are calculated specifically for the NLLS problem.
    On the last iteration, NLLS is performed without outliers.

    """
    # check if M-estimator is valid (defined in this function)
    if m_est not in ["gm", "cauchy"]:
        raise ValueError("unknown M-estimator choice")

    # min 3 iters: (1) NLLS (2) NLLS with M-weights (3) clean NLLS
    if total_idx < 3:
        raise ValueError("require 3+ iterations")

    p, N = design_matrix.shape[-1], data.shape[-1]
    if N <= p:
        raise ValueError("Fewer data points than parameters.")
    factor = 1.4826 * np.sqrt(N / (N - p))

    # handle potential zeros
    pred_sig[pred_sig < MIN_POSITIVE_SIGNAL] = MIN_POSITIVE_SIGNAL

    # calculate quantities needed for C and w
    log_pred_sig = np.log(pred_sig)
    residuals = data - pred_sig
    log_data = np.log(data)
    log_residuals = log_data - log_pred_sig

    C = (
        factor
        * np.median(np.abs(residuals - np.median(residuals)[..., None]), axis=-1)[
            ..., None
        ]
    )
    C[C == 0] = np.nanmedian(C)  # C could be 0, if all signals = min_signal

    # NOTE: if more M-estimators are added, please update the docs!
    if m_est == "gm":
        w = C**2 / (C**2 + residuals**2) ** 2
    if m_est == "cauchy":
        w = C**2 / (C**2 + residuals**2)

    robust = None

    if idx == total_idx:
        cond = outlier_condition_func(
            residuals, log_residuals, pred_sig, design_matrix, leverages, C, cutoff
        )
        robust = np.logical_not(cond)

        w[~robust] = 0.0
        w[robust] = 1.0

    w[np.isinf(w)] = 0
    w[np.isnan(w)] = 0

    return w, robust
