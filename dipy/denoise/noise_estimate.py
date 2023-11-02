import numpy as np

from scipy.special import gammainccinv
from scipy.ndimage import convolve


def _inv_nchi_cdf(N, K, alpha):
    """Inverse CDF for the noncentral chi distribution
    See [1]_ p.3 section 2.3"""
    return gammainccinv(N * K, 1 - alpha) / K


# List of optimal quantile for PIESNO.
# Get optimal quantile for N if available, else use the median.
opt_quantile = {1: 0.79681213002002,
                2: 0.7306303027491917,
                4: 0.6721952960782169,
                8: 0.6254030432343569,
               16: 0.5900487123737876,
               32: 0.5641772300866416,
               64: 0.5455611840489607,
              128: 0.5322811923303339}


def piesno(data, N, alpha=0.01, l=100, itermax=100, eps=1e-5,
           return_mask=False):
    """
    Probabilistic Identification and Estimation of Noise (PIESNO).

    Parameters
    ----------
    data : ndarray
        The magnitude signals to analyse. The last dimension must contain the
        same realisation of the volume, such as dMRI or fMRI data.

    N : int
        The number of phase array coils of the MRI scanner.
        If your scanner does a SENSE reconstruction, ALWAYS use N=1, as the
        noise profile is always Rician.
        If your scanner does a GRAPPA reconstruction, set N as the number
        of phase array coils.

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

    return_mask : bool
        If True, return a mask identifying all the pure noise voxel
        that were found.

    Returns
    -------
    sigma : float
        The estimated standard deviation of the gaussian noise.

    mask : ndarray (optional)
        A boolean mask indicating the voxels identified as pure noise.

    Notes
    -----
    This function assumes two things : 1. The data has a noisy, non-masked
    background and 2. The data is a repetition of the same measurements
    along the last axis, i.e. dMRI or fMRI data, not structural data like
    T1/T2.

    This function processes the data slice by slice, as originally designed in
    the paper. Use it to get a slice by slice estimation of the noise, as in
    spinal cord imaging for example.

    References
    ----------

    .. [1] Koay CG, Ozarslan E and Pierpaoli C.
    "Probabilistic Identification and Estimation of Noise (PIESNO):
    A self-consistent approach and its applications in MRI."
    Journal of Magnetic Resonance 2009; 199: 94-103.

    .. [2] Koay CG, Ozarslan E and Basser PJ.
    "A signal transformational framework for breaking the noise floor
    and its applications in MRI."
    Journal of Magnetic Resonance 2009; 197: 108-119.
    """

    # This method works on a 2D array with repetitions as the third dimension,
    # so process the dataset slice by slice.
    if data.ndim < 3:
        e_s = "This function only works on datasets of at least 3 dimensions."
        raise ValueError(e_s)

    if N in opt_quantile:
        q = opt_quantile[N]
    else:
        q = 0.5

    # Initial estimation of sigma
    initial_estimation = (np.percentile(data, q * 100) /
                          np.sqrt(2 * _inv_nchi_cdf(N, 1, q)))

    if data.ndim == 4:

        sigma = np.zeros(data.shape[-2], dtype=np.float32)
        mask_noise = np.zeros(data.shape[:-1], dtype=bool)

        for idx in range(data.shape[-2]):
            sigma[idx], mask_noise[..., idx] = _piesno_3D(data[..., idx, :],
                                                          N,
                                                          alpha=alpha,
                                                          l=l,
                                                          itermax=itermax,
                                                          eps=eps,
                                                          return_mask=True,
                                                          initial_estimation=initial_estimation)

    else:
        sigma, mask_noise = _piesno_3D(data,
                                       N,
                                       alpha=alpha,
                                       l=l,
                                       itermax=itermax,
                                       eps=eps,
                                       return_mask=True,
                                       initial_estimation=initial_estimation)

    if return_mask:
        return sigma, mask_noise

    return sigma


def _piesno_3D(data, N, alpha=0.01, l=100, itermax=100, eps=1e-5,
               return_mask=False, initial_estimation=None):
    """
    Probabilistic Identification and Estimation of Noise (PIESNO).
    This is the slice by slice version for working on a 4D array.

    Parameters
    ----------
    data : ndarray
        The magnitude signals to analyse. The last dimension must contain the
        same realisation of the volume, such as dMRI or fMRI data.

    N : int
        The number of phase array coils of the MRI scanner.

    alpha : float (optional)
        Probabilistic estimation threshold for the gamma function.
        Default: 0.01.

    l : int (optional)
        number of initial estimates for sigma to try. Default: 100.

    itermax : int (optional)
        Maximum number of iterations to execute if convergence
        is not reached. Default: 100.

    eps : float (optional)
        Tolerance for the convergence criterion. Convergence is
        reached if two subsequent estimates are smaller than eps.
        Default: 1e-5.

    return_mask : bool (optional)
        If True, return a mask identifying all the pure noise voxel
        that were found. Default: False.

    initial_estimation : float (optional)
        Upper bound for the initial estimation of sigma. default : None,
        which computes the optimal quantile for N.

    Returns
    -------
    sigma : float
        The estimated standard deviation of the gaussian noise.

    mask : ndarray
        A boolean mask indicating the voxels identified as pure noise.

    Notes
    -----
    This function assumes two things : 1. The data has a noisy, non-masked
    background and 2. The data is a repetition of the same measurements
    along the last axis, i.e. dMRI or fMRI data, not structural data like
    T1/T2.

    References
    ----------

    .. [1] Koay CG, Ozarslan E and Pierpaoli C.
    "Probabilistic Identification and Estimation of Noise (PIESNO):
    A self-consistent approach and its applications in MRI."
    Journal of Magnetic Resonance 2009; 199: 94-103.

    .. [2] Koay CG, Ozarslan E and Basser PJ.
    "A signal transformational framework for breaking the noise floor
    and its applications in MRI."
    Journal of Magnetic Resonance 2009; 197: 108-119.
    """

    if np.all(data == 0):
        if return_mask:
            return 0, np.zeros(data.shape[:-1], dtype=bool)

        return 0

    if N in opt_quantile:
        q = opt_quantile[N]
    else:
        q = 0.5

    denom = np.sqrt(2 * _inv_nchi_cdf(N, 1, q))

    if initial_estimation is None:
        m = np.percentile(data, q * 100) / denom
    else:
        m = initial_estimation

    phi = np.arange(1, l + 1) * m / l
    K = data.shape[-1]
    sum_m2 = np.sum(data.astype(np.float32)**2, axis=2)

    sigma_prev = 0
    sigma = m
    prev_idx = 0
    mask = np.zeros(data.shape[:-1], dtype=bool)

    lambda_minus = _inv_nchi_cdf(N, K, alpha/2)
    lambda_plus = _inv_nchi_cdf(N, K, 1 - alpha/2)

    for sigma_init in phi:

        s = sum_m2 / (2 * K * sigma_init**2)
        found_idx = np.sum(np.logical_and(lambda_minus <= s, s <= lambda_plus),
                           dtype=np.int16)

        if found_idx > prev_idx:
            sigma = sigma_init
            prev_idx = found_idx

    for n in range(itermax):
        if np.abs(sigma - sigma_prev) < eps:
            break

        s = sum_m2 / (2 * K * sigma**2)
        mask[...] = np.logical_and(lambda_minus <= s, s <= lambda_plus)
        omega = data[mask, :]

        # If no point meets the criterion, exit
        if omega.size == 0:
            break

        sigma_prev = sigma

        # Numpy percentile must range in 0 to 100, hence q*100
        sigma = np.percentile(omega, q * 100) / denom

    if return_mask:
        return sigma, mask

    return sigma


def estimate_sigma(arr, disable_background_masking=False, N=0):
    """Standard deviation estimation from local patches

    Parameters
    ----------
    arr : 3D or 4D ndarray
        The array to be estimated

    disable_background_masking : bool, default False
        If True, uses all voxels for the estimation, otherwise, only non-zeros
        voxels are used. Useful if the background is masked by the scanner.

    N : int, default 0
        Number of coils of the receiver array. Use N = 1 in case of a SENSE
        reconstruction (Philips scanners) or the number of coils for a GRAPPA
        reconstruction (Siemens and GE). Use 0 to disable the correction factor,
        as for example if the noise is Gaussian distributed. See [1] for more
        information.

    Returns
    -------
    sigma : ndarray
        standard deviation of the noise, one estimation per volume.

    Notes
    -----
    This function is the same as manually taking the standard deviation of the
    background and gives one value for the whole 3D array.
    It also includes the coil-dependent correction factor of Koay 2006
    (see [1]_, equation 18) with theta = 0.
    Since this function was introduced in [2]_ for T1 imaging,
    it is expected to perform ok on diffusion MRI data, but might oversmooth
    some regions and leave others un-denoised for spatially varying noise
    profiles. Consider using :func:`piesno` to estimate sigma instead if visual
    inaccuracies are apparent in the denoised result.

    References
    ----------
    .. [1] Koay, C. G., & Basser, P. J. (2006). Analytically exact correction
    scheme for signal extraction from noisy magnitude MR signals.
    Journal of Magnetic Resonance), 179(2), 317-22.

    .. [2] Coupe, P., Yger, P., Prima, S., Hellier, P., Kervrann, C., Barillot,
    C., 2008. An optimized blockwise nonlocal means denoising filter for 3-D
    magnetic resonance images, IEEE Trans. Med. Imaging 27, 425-41.

    """
    k = np.zeros((3, 3, 3), dtype=np.int8)

    k[0, 1, 1] = 1
    k[2, 1, 1] = 1
    k[1, 0, 1] = 1
    k[1, 2, 1] = 1
    k[1, 1, 0] = 1
    k[1, 1, 2] = 1

    # Precomputed factor from Koay 2006, this corrects the bias of magnitude
    # image
    correction_factor = {0: 1,  # No correction
                         1: 0.42920367320510366,
                         4: 0.4834941393603609,
                         6: 0.4891759468548269,
                         8: 0.49195420135894175,
                        12: 0.4946862482541263,
                        16: 0.4960339908122364,
                        20: 0.4968365823718557,
                        24: 0.49736907650825657,
                        32: 0.49803177052530145,
                        64: 0.49901964176235936}

    if N in correction_factor:
        factor = correction_factor[N]
    else:
        raise ValueError("N = {0} is not supported! Please choose amongst \
{1}".format(N, sorted(list(correction_factor.keys()))))

    if arr.ndim == 3:
        sigma = np.zeros(1, dtype=np.float32)
        arr = arr[..., None]
    elif arr.ndim == 4:
        sigma = np.zeros(arr.shape[-1], dtype=np.float32)
    else:
        raise ValueError("Array shape is not supported!", arr.shape)

    if disable_background_masking:
        mask = arr[..., 0].astype(bool)
    else:
        mask = np.ones_like(arr[..., 0], dtype=bool)

    conv_out = np.zeros(arr[..., 0].shape, dtype=np.float64)

    for i in range(sigma.size):
        convolve(arr[..., i], k, output=conv_out)
        mean_block = np.sqrt(6/7) * (arr[..., i] - 1/6 * conv_out)
        sigma[i] = np.sqrt(np.mean(mean_block[mask]**2) / factor)

    return sigma
