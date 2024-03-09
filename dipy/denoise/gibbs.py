from functools import partial
from multiprocessing import Pool

import numpy as np
import scipy

from dipy.utils.multiproc import determine_num_processes

import scipy.fft
_fft = scipy.fft


def _image_tv(x, axis=0, n_points=3):
    """ Computes total variation (TV) of matrix x across a given axis and
    along two directions.

    Parameters
    ----------
    x : 2D ndarray
        matrix x
    axis : int (0 or 1)
        Axis which TV will be calculated. Default a is set to 0.
    n_points : int
        Number of points to be included in TV calculation.

    Returns
    -------
    ptv : 2D ndarray
        Total variation calculated from the right neighbours of each point.
    ntv : 2D ndarray
        Total variation calculated from the left neighbours of each point.

    """
    xs = x.copy() if axis else x.T.copy()

    # Add copies of the data so that data extreme points are also analysed
    xs = np.concatenate((xs[:, (-n_points-1):], xs, xs[:, 0:(n_points+1)]),
                        axis=1)

    ptv = np.absolute(xs[:, (n_points+1):(-n_points-1)] -
                      xs[:, (n_points+2):(-n_points)])
    ntv = np.absolute(xs[:, (n_points+1):(-n_points-1)] -
                      xs[:, n_points:(-n_points-2)])
    for n in range(1, n_points):
        ptv = ptv + np.absolute(xs[:, (n_points+1+n):(-n_points-1+n)] -
                                xs[:, (n_points+2+n):(-n_points+n)])
        ntv = ntv + np.absolute(xs[:, (n_points+1-n):(-n_points-1-n)] -
                                xs[:, (n_points-n):(-n_points-2-n)])

    if axis:
        return ptv, ntv
    else:
        return ptv.T, ntv.T


def _gibbs_removal_1d(x, axis=0, n_points=3):
    """Suppresses Gibbs ringing along a given axis using fourier sub-shifts.

    Parameters
    ----------
    x : 2D ndarray
        Matrix x.
    axis : int (0 or 1)
        Axis in which Gibbs oscillations will be suppressed.
        Default is set to 0.
    n_points : int, optional
        Number of neighbours to access local TV (see note).
        Default is set to 3.

    Returns
    -------
    xc : 2D ndarray
        Matrix with suppressed Gibbs oscillations along the given axis.

    Notes
    -----
    This function suppresses the effects of Gibbs oscillations based on the
    analysis of local total variation (TV). Although artefact correction is
    done based on two adjacent points for each voxel, total variation should be
    accessed in a larger range of neighbours. The number of neighbours to be
    considered in TV calculation can be adjusted using the parameter n_points.

    """
    dtype_float = np.promote_types(x.real.dtype, np.float32)

    ssamp = np.linspace(0.02, 0.9, num=45, dtype=dtype_float)

    xs = x.copy() if axis else x.T.copy()

    # TV for shift zero (baseline)
    tvr, tvl = _image_tv(xs, axis=1, n_points=n_points)
    tvp = np.minimum(tvr, tvl)
    tvn = tvp.copy()

    # Find optimal shift for gibbs removal
    isp = xs.copy()
    isn = xs.copy()
    sp = np.zeros(xs.shape, dtype=dtype_float)
    sn = np.zeros(xs.shape, dtype=dtype_float)
    N = xs.shape[1]
    c = _fft.fft(xs, axis=1)
    k = _fft.fftfreq(N, 1 / (2.0j * np.pi))
    k = k.astype(c.dtype, copy=False)
    for s in ssamp:
        ks = k * s
        # Access positive shift for given s
        img_p = abs(_fft.ifft(c * np.exp(ks), axis=1))

        tvsr, tvsl = _image_tv(img_p, axis=1, n_points=n_points)
        tvs_p = np.minimum(tvsr, tvsl)

        # Access negative shift for given s
        img_n = abs(_fft.ifft(c * np.exp(-ks), axis=1))
        tvsr, tvsl = _image_tv(img_n, axis=1, n_points=n_points)
        tvs_n = np.minimum(tvsr, tvsl)

        # Update positive shift params
        isp[tvp > tvs_p] = img_p[tvp > tvs_p]
        sp[tvp > tvs_p] = s
        tvp[tvp > tvs_p] = tvs_p[tvp > tvs_p]

        # Update negative shift params
        isn[tvn > tvs_n] = img_n[tvn > tvs_n]
        sn[tvn > tvs_n] = s
        tvn[tvn > tvs_n] = tvs_n[tvn > tvs_n]

    # check non-zero sub-voxel shifts
    idx = np.nonzero(sp + sn)

    # use positive and negative optimal sub-voxel shifts to interpolate to
    # original grid points
    xs[idx] = (isp[idx] - isn[idx])/(sp[idx] + sn[idx])*sn[idx] + isn[idx]

    return xs if axis else xs.T


def _weights(shape):
    """ Computes the weights necessary to combine two images processed by
    the 1D Gibbs removal procedure along two different axes [1]_.

    Parameters
    ----------
    shape : tuple
        shape of the image.

    Returns
    -------
    G0 : 2D ndarray
        Weights for the image corrected along axis 0.
    G1 : 2D ndarray
        Weights for the image corrected along axis 1.

    References
    ----------
    .. [1] Kellner E, Dhital B, Kiselev VG, Reisert M. Gibbs-ringing artifact
           removal based on local subvoxel-shifts. Magn Reson Med. 2016
           doi: 10.1002/mrm.26054.

    """
    G0 = np.zeros(shape)
    G1 = np.zeros(shape)
    k0 = np.linspace(-np.pi, np.pi, num=shape[0])
    k1 = np.linspace(-np.pi, np.pi, num=shape[1])

    # Middle points
    K1, K0 = np.meshgrid(k1[1:-1], k0[1:-1])
    cosk0 = 1.0 + np.cos(K0)
    cosk1 = 1.0 + np.cos(K1)
    G1[1:-1, 1:-1] = cosk0 / (cosk0 + cosk1)
    G0[1:-1, 1:-1] = cosk1 / (cosk0 + cosk1)

    # Boundaries
    G1[1:-1, 0] = G1[1:-1, -1] = 1
    G1[0, 0] = G1[-1, -1] = G1[0, -1] = G1[-1, 0] = 1/2
    G0[0, 1:-1] = G0[-1, 1:-1] = 1
    G0[0, 0] = G0[-1, -1] = G0[0, -1] = G0[-1, 0] = 1/2

    return G0, G1


def _gibbs_removal_2d(image, n_points=3, G0=None, G1=None):
    """ Suppress Gibbs ringing of a 2D image.

    Parameters
    ----------
    image : 2D ndarray
        Matrix containing the 2D image.
    n_points : int, optional
        Number of neighbours to access local TV (see note). Default is
        set to 3.
    G0 : 2D ndarray, optional.
        Weights for the image corrected along axis 0. If not given, the
        function estimates them using the function :func:`_weights`.
    G1 : 2D ndarray
        Weights for the image corrected along axis 1. If not given, the
        function estimates them using the function :func:`_weights`.

    Returns
    -------
    imagec : 2D ndarray
        Matrix with Gibbs oscillations reduced along axis a.

    Notes
    -----
    This function suppresses the effects of Gibbs oscillations based on the
    analysis of local total variation (TV). Although artefact correction is
    done based on two adjacent points for each voxel, total variation should be
    accessed in a larger range of neighbours. The number of neighbours to be
    considered in TV calculation can be adjusted using the parameter n_points.

    References
    ----------
    Please cite the following articles
    .. [1] Neto Henriques, R., 2018. Advanced Methods for Diffusion MRI Data
           Analysis and their Application to the Healthy Ageing Brain
           (Doctoral thesis). https://doi.org/10.17863/CAM.29356
    .. [2] Kellner E, Dhital B, Kiselev VG, Reisert M. Gibbs-ringing artifact
           removal based on local subvoxel-shifts. Magn Reson Med. 2016
           doi: 10.1002/mrm.26054.

    """
    if np.any(G0) is None or np.any(G1) is None:
        G0, G1 = _weights(image.shape)

    img_c1 = _gibbs_removal_1d(image, axis=1, n_points=n_points)
    img_c0 = _gibbs_removal_1d(image, axis=0, n_points=n_points)

    C1 = _fft.fft2(img_c1)
    C0 = _fft.fft2(img_c0)
    imagec = abs(_fft.ifft2(_fft.fftshift(C1)*G1 + _fft.fftshift(C0)*G0))

    return imagec


def gibbs_removal(vol, slice_axis=2, n_points=3, inplace=True,
                  num_processes=1):
    """Suppresses Gibbs ringing artefacts of images volumes.

    Parameters
    ----------
    vol : ndarray ([X, Y]), ([X, Y, Z]) or ([X, Y, Z, g])
        Matrix containing one volume (3D) or multiple (4D) volumes of images.
    slice_axis : int (0, 1, or 2)
        Data axis corresponding to the number of acquired slices.
        Default is set to the third axis.
    n_points : int, optional
        Number of neighbour points to access local TV (see note).
        Default is set to 3.
    inplace : bool, optional
        If True, the input data is replaced with results. Otherwise, returns
        a new array.
        Default is set to True.
    num_processes : int or None, optional
        Split the calculation to a pool of children processes. This only
        applies to 3D or 4D `data` arrays. Default is 1. If < 0 the maximal
        number of cores minus ``num_processes + 1`` is used (enter -1 to use
        as many cores as possible). 0 raises an error.

    Returns
    -------
    vol : ndarray ([X, Y]), ([X, Y, Z]) or ([X, Y, Z, g])
        Matrix containing one volume (3D) or multiple (4D) volumes of corrected
        images.

    Notes
    -----
    For 4D matrix last element should always correspond to the number of
    diffusion gradient directions.

    References
    ----------
    Please cite the following articles
    .. [1] Neto Henriques, R., 2018. Advanced Methods for Diffusion MRI Data
           Analysis and their Application to the Healthy Ageing Brain
           (Doctoral thesis). https://doi.org/10.17863/CAM.29356
    .. [2] Kellner E, Dhital B, Kiselev VG, Reisert M. Gibbs-ringing artifact
           removal based on local subvoxel-shifts. Magn Reson Med. 2016
           doi: 10.1002/mrm.26054.

    """
    nd = vol.ndim

    # check matrix dimension
    if nd > 4:
        raise ValueError("Data have to be a 4D, 3D or 2D matrix")
    elif nd < 2:
        raise ValueError("Data is not an image")

    if not isinstance(inplace, bool):
        raise TypeError("inplace must be a boolean.")

    num_processes = determine_num_processes(num_processes)

    # check the axis corresponding to different slices
    # 1) This axis cannot be larger than 2
    if slice_axis > 2:
        raise ValueError("Different slices have to be organized along" +
                         "one of the 3 first matrix dimensions")

    # 2) Reorder axis to allow iteration over the first axis
    elif nd == 3:
        vol = np.moveaxis(vol, slice_axis, 0)
    elif nd == 4:
        vol = np.moveaxis(vol, (slice_axis, 3), (0, 1))

    if nd == 4:
        inishap = vol.shape
        vol = vol.reshape((inishap[0] * inishap[1], inishap[2], inishap[3]))

    # Produce weighting functions for 2D Gibbs removal
    shap = vol.shape
    G0, G1 = _weights(shap[-2:])

    # Copy data if not inplace
    if not inplace:
        vol = vol.copy()

    # Run Gibbs removal of 2D images
    if nd == 2:
        vol[:, :] = _gibbs_removal_2d(vol, n_points=n_points, G0=G0, G1=G1)
    else:
        pool = Pool(num_processes)

        partial_func = partial(
            _gibbs_removal_2d, n_points=n_points, G0=G0, G1=G1
        )
        vol[:, :, :] = pool.map(partial_func, vol)
        pool.close()
        pool.join()

    # Reshape data to original format
    if nd == 3:
        vol = np.moveaxis(vol, 0, slice_axis)
    if nd == 4:
        vol = vol.reshape(inishap)
        vol = np.moveaxis(vol, (0, 1), (slice_axis, 3))

    return vol
