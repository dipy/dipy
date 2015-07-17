# cython: wraparound=False, cdivision=True, boundscheck=False

cimport cython
cimport numpy as np
import numpy as np

from libc.math cimport sqrt, exp, fabs, M_PI
from scipy.special import erfinv
from dipy.denoise.hyp1f1 import hyp1f1 as cython_mpmath1f1

from dipy.utils.optpkg import optional_package
mpmath, have_mpmath, _ = optional_package("mpmath")

try: 
    import cython_gsl
except ImportError:
    have_cython_gsl = False
else:
    have_cython_gsl = True
    
if not have_cython_gsl and not have_mpmath:
    raise ValueError('Cannot find cython_gsl nor mpmath package (required for hyp1f1). \
        Try pip install cythongsl (recommended : faster than mpmath, but you need to \
        install the GSL library also (sudo apt-get install libgsl0-dev libgsl0ldbl)) \
        or pip install mpmath (also pip install gmpy2 for faster performances.)')

IF have_cython_gsl:
    from cython_gsl cimport gsl_sf_hyperg_1F1 as gsl1f1
    print("The GNU GSL is subject to the GSL license, see \n\
        http://www.gnu.org/copyleft/gpl.html for more information")

if have_mpmath:
    import mpmath


cdef double hyp1f1(double a, int b, double x) nogil:
    """Wrapper for 1F1 hypergeometric series function
    http://en.wikipedia.org/wiki/Confluent_hypergeometric_function"""
    cdef double out

    # Has cython gsl version, use it since it's faster
    if have_cython_gsl:
        return gsl1f1(a, b, x)

    # Use mpmath cython version, and mpmath symbolic if no convergence
    with gil:
        out = cython_mpmath1f1(a, b, x)
        if np.isinf(out) or np.isnan(out):
            out = float(mpmath.hyp1f1(a, b, x))

        return out


cdef double _inv_cdf_gauss(double y, double eta, double sigma):
    """Helper function for _chi_to_gauss. Returns the gaussian distributed value
    associated to a given probability. See p. 4 of [1] eq. 13.

    y : float
        Probability of observing the desired value in the normal
        distribution N(eta, sigma**2)
    eta :
        Mean of the normal distribution N(eta, sigma**2)
    sigma : float
        Standard deviation of the normal distribution N(eta, sigma**2)

    return :
        Value associated to probability y given a normal distribution N(eta, sigma**2)
    """
    return eta + sigma * sqrt(2) * erfinv(2*y - 1)


def chi_to_gauss(m, eta, sigma, N, alpha=0.0001):

    """Maps the noisy signal intensity from a Rician/Non central chi distribution
    to its gaussian counterpart. See p. 4 of [1] eq. 12.

    m : float
        The noisy, Rician/Non central chi distributed value
    eta : float
        The underlying signal intensity estimated value
    sigma : float
        The gaussian noise estimated standard deviation
    N : int
        Number of coils of the acquision (N=1 for Rician noise)
    alpha : float
        Confidence interval for the cumulative distribution function.
        Clips the cdf to alpha/2 <= cdf <= 1-alpha/2

    return
        float : The noisy gaussian distributed signal intensity

    Reference:
    [1]. Koay CG, Ozarslan E and Basser PJ.
    A signal transformational framework for breaking the noise floor
    and its applications in MRI.
    Journal of Magnetic Resonance 2009; 197: 108-119.
    """
    return _chi_to_gauss(m, eta, sigma, N, alpha)


cdef double _chi_to_gauss(double m, double eta, double sigma, int N,
                          double alpha=0.0001) nogil:
    """Maps the noisy signal intensity from a Rician/Non central chi distribution
    to its gaussian counterpart. See p. 4 of [1] eq. 12.

    m : float
        The noisy, Rician/Non central chi distributed value
    eta : float
        The underlying signal intensity estimated value
    sigma : float
        The gaussian noise estimated standard deviation
    N : int
        Number of coils of the acquision (N=1 for Rician noise)
    alpha : float
        Confidence interval for the cumulative distribution function.
        Clips the cdf to alpha/2 <= cdf <= 1-alpha/2

    return
        float : The noisy gaussian distributed signal intensity

    Reference:
    [1]. Koay CG, Ozarslan E and Basser PJ.
    A signal transformational framework for breaking the noise floor
    and its applications in MRI.
    Journal of Magnetic Resonance 2009; 197: 108-119.
    """

    cdef double cdf

    with nogil:
        cdf = 1. - _marcumq_cython(eta/sigma, m/sigma, N)

    # clip cdf between alpha/2 and 1-alpha/2
    if cdf < alpha/2:
        cdf = alpha/2
    elif cdf > 1 - alpha/2:
        cdf = 1 - alpha/2

    with gil:
        return _inv_cdf_gauss(cdf, eta, sigma)


cdef double multifactorial(int N, int k=1) nogil:
    """Returns the multifactorial of order k of N.
    https://en.wikipedia.org/wiki/Factorial#Multifactorials

    N : int
        Number to compute the factorial of
    k : int
        Order of the factorial, default k=1

    return : double
        Return type is double, because multifactorial(21) > 2**64
        Same as scipy.special.factorialk
    """

    if N == 0:
        return 1.

    elif N < (k + 1):
        return N

    return N * multifactorial(N - k, k)


cdef double _marcumq_cython(double a, double b, int M, double eps=1e-8,
                            int max_iter=10000) nogil:
    """Computes the generalized Marcum Q function of order M.
    http://en.wikipedia.org/wiki/Marcum_Q-function

    a : float, eta/sigma
    b : float, m/sigma
    M : int, order of the function (Number of coils, N=1 for Rician noise)

    return : float
        Value of the function, always between 0 and 1 since it's a pdf.
    """
    cdef:
        double a2 = 0.5 * a**2
        double b2 = 0.5 * b**2
        double d = exp(-a2)
        double h = exp(-a2)
        double f = (b2**M) * exp(-b2) / multifactorial(M)
        double f_err = exp(-b2)
        double errbnd = 1. - f_err
        double  S = f * h
        double temp = 0.
        int k = 1
        int j = errbnd > 4*eps

    if fabs(a) < eps:

        for k in range(M):
            temp += b**(2*k) / (2**k * multifactorial(k))

        return exp(-b**2/2) * temp

    elif fabs(b) < eps:
        return 1.

    while j or k <= M:

        d *= a2 / k
        h += d
        f *= b2 / (k + M)
        S += f * h

        f_err *= b2 / k
        errbnd -= f_err

        j = errbnd > 4*eps
        k += 1

        if k > max_iter:
            break

    return 1. - S


def fixed_point_finder(m_hat, sigma, N, max_iter=100, eps=1e-4):
    """Fixed point formula for finding eta. Table 1 p. 11 of [1].
    This simply wraps the cython function _fixed_point_finder.

    m_hat : float
        Initial value for the estimation of eta.
    sigma : float
        Gaussian standard deviation of the noise.
    N : int
        Number of coils of the acquision (N=1 for Rician noise).
    max_iter : int, default=100
        Maximum number of iterations before breaking from the loop.
    eps : float, default = 1e-4
        Criterion for reaching convergence between two subsequent estimates of eta.

    return
    t1 : float
        Estimation of the underlying signal value

    Reference:
    [1]. Koay CG, Ozarslan E and Basser PJ.
    A signal transformational framework for breaking the noise floor
    and its applications in MRI.
    Journal of Magnetic Resonance 2009; 197: 108-119.
    """
    return _fixed_point_finder(m_hat, sigma, N, max_iter, eps)


cdef double _fixed_point_finder(double m_hat, double sigma, int N,
                                int max_iter=100, double eps=1e-4):
    """Fixed point formula for finding eta. Table 1 p. 11 of [1].

    m_hat : float
        Initial value for the estimation of eta
    sigma : float
        Gaussian standard deviation of the noise
    N : int
        Number of coils of the acquision (N=1 for Rician noise)
    max_iter : int, default=100
        Maximum number of iterations before breaking from the loop
    eps : float, default = 1e-4
        Criterion for reaching convergence between two subsequent estimates

    return
    t1 : float
        Estimation of the underlying signal value

    Notes
    --------
    This has the fix to return eta = 0 when below the noise floor instead of -eta.
    See Bai & al. 2014 A framework for accurate determination of the T2 distribution from
    multiple echo magnitude MRI images, Journal of Magnetic Resonance, Volume 244,
    July 2014, Pages 53-63, ISSN 1090-7807
    """

    cdef:
        double delta, m, t0, t1
        int cond = True
        int n_iter = 0

    with nogil:
        # Initial estimated value is below the noise floor, so just return 0
        if m_hat < sqrt(0.5 * M_PI) * sigma:
            return 0

        delta = _beta(N) * sigma - m_hat

        if fabs(delta) < 1e-15:
            return 0

        m = m_hat
        t0 = m
        t1 = _fixed_point_k(t0, m, sigma, N)

        while cond:

            t0 = t1
            t1 = _fixed_point_k(t0, m, sigma, N)
            n_iter += 1
            cond = fabs(t1 - t0) > eps

            if n_iter > max_iter:
                break

        return t1


cdef double _beta(int N) nogil:
    """Helper function for _xi, see p. 3 [1] just after eq. 8."""

    cdef:
        double factorialN_1 = multifactorial(N - 1)
        double factorial2N_1 = multifactorial(2*N - 1, 2)

    return sqrt(0.5 * M_PI) * (factorial2N_1 / (2**(N-1) * factorialN_1))


cdef double _fixed_point_g(double eta, double m, double sigma, int N) nogil:
    """Helper function for _fixed_point_k, see p. 3 [1] eq. 11."""
    return sqrt(m**2 + (_xi(eta, sigma, N) - 2*N) * sigma**2)


cdef double _fixed_point_k(double eta, double m, double sigma, int N) nogil:
    """Helper function for _fixed_point_finder, see p. 11 [1] eq. D2."""

    cdef:
        double fpg, num, denom
        double eta2sigma = -eta**2/(2*sigma**2)

    fpg = _fixed_point_g(eta, m, sigma, N)
    num = fpg * (fpg - eta)

    denom = eta * (1 - ((_beta(N)**2)/(2*N)) *
                   hyp1f1(-0.5, N, eta2sigma) *
                   hyp1f1(0.5, N+1, eta2sigma)) - fpg

    return eta - num / denom


cdef double _xi(double eta, double sigma, int N) nogil:
    """Standard deviation scaling factor formula, see p. 3 of [1], eq. 10.

    eta : float
        Signal intensity
    sigma : float
        Noise magnitude standard deviation
    N : int
        Number of coils of the acquisition (N=1 for Rician noise)

    return :
        The correction factor xi, where sigma_gaussian**2 = sigma**2 / xi
    """

    if fabs(sigma) < 1e-15:
        return 1.

    h1f1 = hyp1f1(-0.5, N, -eta**2/(2*sigma**2))
    return 2*N + eta**2/sigma**2 -(_beta(N) * h1f1)**2


# Test for cython functions
def _test_marcumq_cython(a, b, M, eps=1e-7, max_iter=10000):
    return _marcumq_cython(a, b, M, eps, max_iter)


def _test_beta(N):
    return _beta(N)


def _test_fixed_point_g(eta, m, sigma, N):
    return _fixed_point_g(eta, m, sigma, N)


def _test_fixed_point_k(eta, m, sigma, N):
    return _fixed_point_k(eta, m, sigma, N)


def _test_xi(eta, sigma, N):
    return _xi(eta, sigma, N)


def _test_multifactorial(N, k=1):
    return multifactorial(N, k)


def _test_inv_cdf_gauss(y, eta, sigma):
    return _inv_cdf_gauss(y, eta, sigma)
