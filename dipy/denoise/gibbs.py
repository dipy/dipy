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
