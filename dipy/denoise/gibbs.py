import numpy as np


def image_shift(x, s, a=0):
    """ Shift elements of matrix x by a position difference s along axis a.

    Parameters
    ----------
    x : 2D ndarray
        Original values of matrix x
    s : float
        Value of the shift
    a : int (0 or 1)
        Axis along which the shift will be applied.
        Default a is set to 0.

    Returns
    -------
    Shifted version of matrix x.

    Note
    ----
    The values of the new shifted matrix are calculated using linear
    interpolation of the values of the original matrix.
    """
    if a:
        xs = x.copy()
    else:
        xs = x.T.copy()

    if s >= 1 or s <= -1:
        raise ValueError('Shift should be a value between -1 and 1')
    elif s > 0:
        xs[:, :-1] = (xs[:, 1:] - xs[:, :-1]) * s + xs[:, :-1]
    else:
        xs[:, 1:] = (xs[:, 1:] - xs[:, :-1]) * (1+s) + xs[:, :-1]

    if a:
        return xs
    else:
        return xs.T


def image_tv(x, fn=0, nn=3, a=0):
    """ Computes total variation (TV) of matrix x along axis a in two
    directions.

    Parameters
    ----------
    x : 2D ndarray
        matrix x
    fn : int
        Distance of first neighbor to be included in TV calculation. If fn=0
        the own point is also included in the TV calculation.
    nn : int
        Number of points to be included in TV calculation.
    a : int (0 or 1)
        Axis along which TV will be calculated. Default a is set to 0.

    Returns
    -------
    PTV : 2D ndarray
        Total variation calculated from the right neighbors of each point
    NTV : 2D ndarray
        Total variation calculated from the left neighbors of each point
    Note
    ----
    This function was created to deal with gibbs artefacts of MR images.
    Assuming that MR images are reconstructed from estimates of their Fourier
    expansion coefficients, during TV calculation matrix x can taken as and
    periodic signal. In this way NTV values on the image left boundary is
    computed using the time series values on the right boundary and vice versa.
    """
    if a:
        xs = x.copy()
    else:
        xs = x.T.copy()

    xs = np.concatenate((xs[:, (-nn-fn):], xs, xs[:, 0:(nn+fn)]), axis=1)

    PTV = np.absolute(xs[:, (nn+fn):(-nn-fn)] - xs[:, (nn+fn+1):(-nn-fn+1)])
    NTV = np.absolute(xs[:, (nn+fn):(-nn-fn)] - xs[:, (nn+fn-1):(-nn-fn-1)])
    for n in np.linspace(fn+1, nn-2, num=nn-2):
        PTV = PTV + np.absolute(xs[:, (nn+fn+n):(-nn-fn+n)] -
                                xs[:, (nn+fn+n+1):(-nn-fn+n+1)])
        NTV = NTV + np.absolute(xs[:, (nn+fn-n):(-nn-fn-n)] -
                                xs[:, (nn+fn-n-1):(-nn-fn-n-1)])

    if a:
        return PTV, NTV
    else:
        return PTV.T, NTV.T


def gibbs_removal_1d(x, a=0, fn=0, nn=3):
    """ Decreases gibbs ringing along axis a.

    Parameters
    ----------
    x : 2D ndarray
        Matrix x.
    a : int (0 or 1)
        Axis along which gibbs oscilations will be reduced. Default a is set
        to 0 (i.e. gibbs are reduce along axis y).
    fn : int, optional
        Distance of first neighbour used to access local TV (see note).
        Default is set to 0 which means that the own point is also used to
        access local TV.
    nn : int, optional
        Number of neighbour points to access local TV (see note). Default is
        set to 3.

    Returns
    -------
    xc : 2D ndarray
        Matrix with gibbs oscilantions reduced along axis a.
    tv : 2D ndarray
        Global TV which show variation not removed (edges, anatomical
        variation, non-oscilatory component of gibbs artefact normally present
        in image background, etc.)
    Note
    ----
    This function decreases the effects of gibbs oscilations based on the
    analysis of local total variation (TV). Although artefact correction is
    done based on two adjanced points for each voxel, total variation should be
    accessed in a larger range of neigbors. If you want to adjust the number
    and index of the neigbors to be considered in TV calculation please change
    parameters nn and fn.
    """
    ssamp = np.linspace(0.02, 0.9, num=45)

    # TV for shift zero (baseline)
    TVR, TVL = image_tv(x, fn=fn, nn=nn, a=a)
    TVP = np.minimum(TVR, TVL)
    TVN = TVP.copy()

    # Find optimal shift for gibbs removal
    ISP = x.copy()
    ISN = x.copy()
    SP = np.zeros(x.shape)
    SN = np.zeros(x.shape)
    for s in ssamp:
        # Image shift using current pos shift
        Img_p = image_shift(x, s, a=a)
        TVSR, TVSL = image_tv(Img_p, fn=fn, nn=nn, a=a)
        TVS_p = np.minimum(TVSR, TVSL)
        # Image shift using current neg shift
        Img_n = image_shift(x, -s, a=a)
        TVSR, TVSL = image_tv(Img_n, fn=fn, nn=nn, a=a)
        TVS_n = np.minimum(TVSR, TVSL)
        # Update positive shift params
        ISP[TVP > TVS_p] = Img_p[TVP > TVS_p]
        SP[TVP > TVS_p] = s
        TVP[TVP > TVS_p] = TVS_p[TVP > TVS_p]
        # Update negative shift params
        ISN[TVN > TVS_n] = Img_n[TVN > TVS_n]
        SN[TVN > TVS_n] = s
        TVN[TVN > TVS_n] = TVS_n[TVN > TVS_n]

    # apply correction if SP and SN are not zeros
    xc = x.copy()
    idx = np.nonzero(SP + SN)
    xc[idx] = (ISP[idx] - ISN[idx])/(SP[idx] + SN[idx])*SN[idx] + ISN[idx]

    # Global minimum TV (can be useful as edge detector)
    tv = np.minimum(TVN, TVP)

    return xc, tv


def gibbs_removal_2d_weigthing_functions(shape):
    """ Computes the weights necessary to combine two images processed by
    the 1D gibbs removal procedure along two different axis [1]_.

    Parameters
    ----------
    shape : tuple
        shape of the image

    Returns
    -------
    G0 : 2D ndarray
        Weights for the image corrected along axis 0.
    G1 : 2D ndarray
        Weights for the image corrected along axis 1.

    References
    ----------
    .. [1] Kellner E, Dhital B, Kiselev VG, Reisert M. Gibbs-ringing artifact
           removal based on local subvoxel-shifts. Magn Reson Med. 2015
           doi: 10.1002/mrm.26054.
    """
    G0 = np.zeros(shape)
    G1 = np.zeros(shape)
    k = np.linspace(-np.pi, np.pi, num=shape[0])

    # Middle points
    K1, K0 = np.meshgrid(k[1:-1], k[1:-1])
    cosk0 = 1.0 + np.cos(K0)
    cosk1 = 1.0 + np.cos(K1)
    G1[1:-1, 1:-1] = cosk0 / (cosk0+cosk1)
    G0[1:-1, 1:-1] = cosk1 / (cosk0+cosk1)

    # Boundaries
    G1[1:-1, 0] = G1[1:-1, -1] = 1
    G1[0, 0] = G1[-1, -1] = G1[0, -1] = G1[-1, 0] = 1/2
    G0[0, 1:-1] = G0[-1, 1:-1] = 1
    G0[0, 0] = G0[-1, -1] = G0[0, -1] = G0[-1, 0] = 1/2

    return G0, G1


def gibbs_removal_2d(image, fn=0, nn=3, G0=None, G1=None):
    """ Decreases gibbs ringing of a 2D image.

    Parameters
    ----------
    image : 2D ndarray
        Matrix cotaining the 2D image.
    fn : int, optional
        Distance of first neighbour used to access local TV (see note).
        Default is set to 0 which means that the own point is also used to
        access local TV.
    nn : int, optional
        Number of neighbour points to access local TV (see note). Default is
        set to 3.
    G0 : 2D ndarray, optional.
        Weights for the image corrected along axis 1. If not given, the
        function estimates them using function:
            gibbs_removal_2d_weigthing_functions
    G1 : 2D ndarray
        Weights for the image corrected along axis 1. If not given, the
        function estimates them using function:
            gibbs_removal_2d_weigthing_functions

    Returns
    -------
    imagec : 2D ndarray
        Matrix with gibbs oscilantions reduced along axis a.
    tv : 2D ndarray
        Global TV which show variation not removed by the algorithm (edges,
        anatomical variation, non-oscilatory component of gibbs artefact
        normally present in image background, etc.)
    Note
    ----
    This function decreases the effects of gibbs oscilations based on the
    analysis of local total variation (TV) along the two axis of the image.
    Although artefact correction is done based on each point primary adjanced
    neighbors, total variation should be accessed in a larger range of
    neigbors. If you want to adjust the number and index of the neigbors to be
    considered in TV calculation please change parameters nn and fn.
    """
    if np.any(G0) == None or np.any(G1) == None:
        G0, G1 = gibbs_removal_2d_weigthing_functions(image.shape)

    img_c1, tv_c1 = gibbs_removal_1d(image, a=1, fn=fn, nn=nn)
    img_c0, tv_c0 = gibbs_removal_1d(image, a=0, fn=fn, nn=nn)

    C1 = np.fft.fft2(img_c1)
    C0 = np.fft.fft2(img_c0)
    imagec = abs(np.fft.ifft2(np.fft.fftshift(C1)*G1 + np.fft.fftshift(C0)*G0))

    # Just to access performance
    T1 = np.fft.fft2(tv_c1)
    T0 = np.fft.fft2(tv_c0)
    tv = abs(np.fft.ifft2(np.fft.fftshift(T1)*G1 + np.fft.fftshift(T0)*G0))

    return imagec, tv
