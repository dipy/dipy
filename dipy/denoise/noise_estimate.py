from __future__ import division, print_function

import numpy as np

from scipy.special import gammainccinv


def _inv_nchi_cdf(N, K, alpha):
    """Inverse CDF for the noncentral chi distribution
    See [1]_ p.3 section 2.3"""
    return gammainccinv(N * K, 1 - alpha) / K


def piesno(data, N=1, alpha=0.01, l=100, itermax=100, eps=1e-10):
    """
    A routine for finding the underlying gaussian distribution standard
    deviation from magnitude signals.

    This is a re-implementation of [1]_ and the second step in the
    stabilisation framework of [2]_.

    Parameters
    -----------

    data : numpy array
        The magnitude signals to analyse. The last dimension must contain the
        same realisation of the volume, such as dMRI or fMRI data.

    N : int
        The number of phase array coils of the MRI scanner.

    alpha : float
        Probabilistic estimation threshold for the gamma function.

    l : int
        number of initial estimates for sigma to try.

    itermax : int
        Maximum number of iterations to execute if convergence
        is not reached.

    eps : float
        Tolerance for the convergence criterion. Convergence is
        reached if two subsequent estimates are smaller than eps.

    Returns
    --------
    sigma : float
        The estimated standard deviation of the gaussian noise.

    mask : ndarray
        A boolean mask indicating the voxels identified as pure noise.

    References
    ------------

    .. [1]. Koay CG, Ozarslan E and Pierpaoli C.
    "Probabilistic Identification and Estimation of Noise (PIESNO):
    A self-consistent approach and its applications in MRI."
    Journal of Magnetic Resonance 2009; 199: 94-103.

    .. [2] Koay CG, Ozarslan E and Basser PJ.
    "A signal transformational framework for breaking the noise floor
    and its applications in MRI."
    Journal of Magnetic Resonance 2009; 197: 108-119.
    """

    # Get optimal quantile for N if available, else use the median.
    opt_quantile = {1: 0.7968,
                    2: 0.7306,
                    4: 0.6722,
                    8: 0.6254,
                   16: 0.5900,
                   32: 0.5642,
                   64: 0.5456,
                  128: 0.5323}

    if N in opt_quantile:
        q = opt_quantile[N]
    else:
        q = 0.5

    # Initial estimation of sigma
    denom = np.sqrt(2 * _inv_nchi_cdf(N, 1, q))
    m = np.percentile(data, q*100) / denom
    phi = np.arange(1, l + 1) * m/l
    K = data.shape[-1]
    sum_m2 = np.sum(data**2, axis=-1)

    sigma = np.zeros_like(phi)
    mask = np.zeros(phi.shape + data.shape[:-1])

    lambda_minus = _inv_nchi_cdf(N, K, alpha/2)
    lambda_plus = _inv_nchi_cdf(N, K, 1 - alpha/2)

    pos = 0
    max_length_omega = 0

    for num, sig in enumerate(phi):

        sig_prev = 0
        omega_size = 1
        idx = np.zeros_like(sum_m2, dtype=np.bool)

        for n in range(itermax):

            if np.abs(sig - sig_prev) < eps:
                break

            s = sum_m2 / (2*K*sig**2)
            idx = np.logical_and(lambda_minus <= s, s <= lambda_plus)
            omega = data[idx, :]

            # If no point meets the criterion, exit
            if omega.size == 0:
                omega_size = 0
                break

            sig_prev = sig
            # Numpy percentile must range in 0 to 100, hence q*100
            sig = np.percentile(omega, q*100) / denom
            omega_size = omega.size/K

        # Remember the biggest omega array as giving the optimal
        # sigma amongst all initial estimates from phi
        if omega_size > max_length_omega:
            pos, max_length_omega, best_mask = num, omega_size, idx

        sigma[num] = sig
        mask[num] = idx

    return sigma[pos], mask[pos]
