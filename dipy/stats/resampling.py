#!/usr/bin/python

import numpy as np
import scipy as sp


def bs_se(bs_pdf):
    """ Calculate the bootstrap standard error estimate of a statistic.
    """
    N = len(bs_pdf)
    return np.std(bs_pdf) * np.sqrt(N / (N - 1))


def bootstrap(x, *, statistic=bs_se, B=1000, alpha=0.95, rng=None):
    """

    Bootstrap resampling [1]_ to accurately estimate the standard error and
    confidence interval of a desired statistic of a probability distribution
    function (pdf).

    Parameters
    ----------
    x : ndarray (N, 1)
        Observable sample to resample. N should be reasonably large.
    statistic : method (optional)
        Method to calculate the desired statistic. (Default: calculate
        bootstrap standard error)
    B : integer (optional)
        Total number of bootstrap resamples in bootstrap pdf. (Default: 1000)
    alpha : float (optional)
        Percentile for confidence interval of the statistic. (Default: 0.05)
    rng : numpy.random.Generator
        Random number generator to use for sampling. If None, the generator
        is initialized using the default BitGenerator.

    Returns
    -------
    bs_pdf : ndarray (M, 1)
        Jackknife probability distribution function of the statistic.
    se : float
        Standard error of the statistic.
    ci : ndarray (2, 1)
        Confidence interval of the statistic.

    See Also
    --------
    numpy.std, numpy.random.random

    Notes
    -----
    Bootstrap resampling is non parametric. It is quite powerful in
    determining the standard error and the confidence interval of a sample
    distribution. The key characteristics of bootstrap is:

    1) uniform weighting among all samples (1/n)
    2) resampling with replacement

    In general, the sample size should be large to ensure accuracy of the
    estimates. The number of bootstrap resamples should be large as well as
    that will also influence the accuracy of the estimate.

    References
    ----------
    ..  [1] Efron, B., 1979. 1977 Rietz lecture--Bootstrap methods--Another
        look at the jackknife. Ann. Stat. 7, 1-26.

    """
    N = len(x)
    bs_pdf = np.empty((B,))

    rng = rng or np.random.default_rng()

    for ii in range(0, B):
        # resample with replacement
        rand_index = np.int16(np.round(rng.random(N) * (N - 1)))
        bs_pdf[ii] = statistic(x[rand_index])

    return bs_pdf, bs_se(bs_pdf), abc(x, statistic=statistic, alpha=alpha)


def abc(x, *, statistic=bs_se, alpha=0.05, eps=1e-5):
    """Calculate the bootstrap confidence interval by approximating the BCa.

    Parameters
    ----------
    x : np.ndarray
        Observed data (e.g. chosen gold standard estimate used for bootstrap)
    statistic : method
        Method to calculate the desired statistic given x and probability
        proportions (flat probability densities vector)
    alpha : float (0, 1)
        Desired confidence interval initial endpoint (Default: 0.05)
    eps : float (optional)
        Specifies step size in calculating numerical derivative T' and
        T''. Default: 1e-5

    See Also
    --------
    __tt, __tt_dot, __tt_dot_dot, __calc_z0

    Notes
    -----
    Unlike the BCa method of calculating the bootstrap confidence interval,
    the ABC method is computationally less demanding (about 3% computational
    power needed) and is fairly accurate (sometimes out performing BCa!). It
    does not require any bootstrap resampling and instead uses numerical
    derivatives via Taylor series to approximate the BCa calculation. However,
    the ABC method requires the statistic to be smooth and follow a
    multinomial distribution.

    References
    ----------
    ..  [2] DiCiccio, T.J., Efron, B., 1996. Bootstrap Confidence Intervals.
        Statistical Science. 11, 3, 189-228.

    """
    # define base variables -- n, p_0, sigma_hat, delta_hat
    n = len(x)
    p_0 = np.ones(x.shape) / n
    sigma_hat = np.zeros(x.shape)
    delta_hat = np.zeros(x.shape)
    for i in range(0, n):
        sigma_hat[i] = __tt_dot(i, x, p_0, statistic, eps)**2
        delta_hat[i] = __tt_dot(i, x, p_0, statistic, eps)
    sigma_hat = (sigma_hat / n**2)**0.5
    # estimate the bias (z_0) and the acceleration (a_hat)
    a_num = np.zeros(x.shape)
    a_dem = np.zeros(x.shape)
    for i in range(0, n):
        a_num[i] = __tt_dot(i, x, p_0, statistic, eps)**3
        a_dem[i] = __tt_dot(i, x, p_0, statistic, eps)**2
    a_hat = 1 / 6 * a_num / a_dem**1.5
    z_0 = __calc_z0(x, p_0, statistic, eps, a_hat, sigma_hat)
    # define helper variables -- w and l
    w = z_0 + __calc_z_alpha(1 - alpha)
    l = w / (1 - a_hat * w)**2
    return __tt(x, p_0 + l * delta_hat / sigma_hat, statistic)


def __calc_z_alpha(alpha):
    """ Calculate inverse of cdf of standard normal (quantile function).
    """
    return 2**0.5 * sp.special.erfinv(2 * alpha - 1)


def __calc_z0(x, p_0, statistic, eps, a_hat, sigma_hat):
    """ calculate the bias z_0 for abc method.

    See Also
    --------
    abc, __tt, __tt_dot, __tt_dot_dot

    """
    n = len(x)
    b_hat = np.ones(x.shape)
    tt_dot = np.ones(x.shape)
    for i in range(0, n):
        b_hat[i] = __tt_dot_dot(i, x, p_0, statistic, eps)
        tt_dot[i] = __tt_dot(i, x, p_0, statistic, eps)
    b_hat = b_hat / (2 * n**2)
    c_q_hat = (__tt(x, ((1 - eps) * p_0 + eps * tt_dot /
                        (n**2 * sigma_hat)), statistic) +
               __tt(x, ((1 - eps) * p_0 - eps * tt_dot /
                        (n**2 * sigma_hat)), statistic) -
               2 * __tt(x, p_0, statistic)) / eps**2
    return a_hat - (b_hat / sigma_hat - c_q_hat)


def __tt(x, p_0, statistic=bs_se):
    """Calculate desired statistic from observable data and a
    given proportional weighting.

    Parameters
    ----------
    x : np.ndarray
        Observable data (e.g. from gold standard).
    p_0 : np.ndarray
        Proportional weighting vector (Default: uniform weighting 1/n)

    Returns
    -------
    theta_hat : float
        Desired statistic of the observable data.

    See Also
    --------
    abc, __tt_dot, __tt_dot_dot
    """
    return statistic(x / p_0)


def __tt_dot(i, x, p_0, statistic, eps):
    """First numerical derivative of __tt.
    """
    e = np.zeros(x.shape)
    e[i] = 1
    return ((__tt(x, ((1 - eps) * p_0 + eps * e[i]), statistic) -
             __tt(x, p_0, statistic)) / eps)


def __tt_dot_dot(i, x, p_0, statistic, eps):
    """Second numerical derivative of __tt.
    """
    e = np.zeros(x.shape)
    e[i] = 1
    return (__tt_dot(i, x, p_0, statistic, eps) / eps +
            (__tt(x, ((1 - eps) * p_0 - eps * e[i]), statistic) -
             __tt(x, p_0, statistic)) / eps**2)


def jackknife(pdf, *, statistic=np.std, M=None, rng=None):
    """
    Jackknife resampling [3]_ to quickly estimate the bias and standard
    error of a desired statistic in a probability distribution function (pdf).

    Parameters
    ----------
    pdf : ndarray (N, 1)
        Probability distribution function to resample. N should be reasonably
        large.
    statistic : method (optional)
        Method to calculate the desired statistic. (Default: calculate
        standard deviation)
    M : integer (M < N)
        Total number of samples in jackknife pdf. (Default: M == N)

    Returns
    -------
    jk_pdf : ndarray (M, 1)
        Jackknife probability distribution function of the statistic.
    bias : float
        Bias of the jackknife pdf of the statistic.
    se : float
        Standard error of the statistic.
    rng : numpy.random.Generator
        Random number generator to use for sampling. If None, the generator
        is initialized using the default BitGenerator.

    See Also
    --------
    numpy.std, numpy.mean, numpy.random.random

    Notes
    -----
    Jackknife resampling like bootstrap resampling is non parametric. However,
    it requires a large distribution to be accurate and in some ways can be
    considered deterministic (if one removes the same set of samples,
    then one will get the same estimates of the bias and variance).

    In the context of this implementation, the sample size should be at least
    larger than the asymptotic convergence of the statistic (ACstat);
    preferably, larger than ACstat + np.greater(ACbias, ACvar)

    The clear benefit of using jackknife is its ability to estimate the bias
    of the statistic. The most powerful application of this is estimating the
    bias of a bootstrap-estimated standard error. In fact, one could
    "bootstrap the bootstrap" (nested bootstrap) of the estimated standard
    error, but the inaccuracy of the bootstrap to characterize the true mean
    would incur a poor estimate of the bias (recall: bias = mean[sample_est] -
    mean[true population])

    References
    ----------
    .. [3] Efron, B., 1979. 1977 Rietz lecture--Bootstrap methods--Another
           look at the jackknife. Ann. Stat. 7, 1-26.

    """
    N = len(pdf)
    pdf_mask = np.ones((N,), dtype='int16')  # keeps track of all n - 1 indexes
    mask_index = np.copy(pdf_mask)
    if M is None:
        M = N
    M = np.minimum(M, N - 1)
    jk_pdf = np.empty((M,))

    rng = rng or np.random.default_rng()

    for ii in range(0, M):
        rand_index = np.round(rng.random() * (N - 1))
        # choose a unique random sample to remove
        while pdf_mask[int(rand_index)] == 0:
            rand_index = np.round(rng.random() * (N - 1))
        # set mask to zero for chosen random index so not to choose again
        pdf_mask[int(rand_index)] = 0
        mask_index[int(rand_index)] = 0
        jk_pdf[ii] = statistic(pdf[mask_index > 0])  # compute n-1 statistic
        mask_index[int(rand_index)] = 1

    return (jk_pdf, (N - 1) * (np.mean(jk_pdf) - statistic(pdf)),
            np.sqrt(N - 1) * np.std(jk_pdf))


def residual_bootstrap(data):
    pass


def repetition_bootstrap(data):
    pass
