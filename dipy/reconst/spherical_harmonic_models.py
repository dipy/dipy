from numpy import arange, arccos, arctan2, atleast_1d, broadcast_arrays, c_, \
                  diag, dot, empty, repeat, sqrt
from numpy.linalg import inv, pinv
from scipy.special import sph_harm, lpn
from .maskedview import MaskedView, _filled, _makearray
from .modelarray import ModelArray

def real_sph_harm(m, n, theta, phi):
    """
    Compute real spherical harmonics, where the real harmonic $Y^m_n$ is
    defined to be:
        Real($Y^m_n$) * sqrt(2) if m > 0
        $Y^m_n$                 if m == 0
        Imag($Y^m_n$) * sqrt(2) if m < 0

    This may take scalar or array arguments. The inputs will be broadcasted
    against each other.

    Parameters
    -----------
      - `m` : int |m| <= n
        The order of the harmonic.
      - `n` : int >= 0
        The degree of the harmonic.
      - `theta` : float [0, 2*pi]
        The azimuthal (longitudinal) coordinate.
      - `phi` : float [0, pi]
        The polar (colatitudinal) coordinate.

    Returns
    --------
      - `y_mn` : real float
        The real harmonic $Y^m_n$ sampled at `theta` and `phi`.

    :See also:
        scipy.special.sph_harm
    """
    m = atleast_1d(m)
    # find where m is =,< or > 0 and broadcasts to the size of the output
    m_eq0,junk,junk,junk = broadcast_arrays(m == 0, n, theta, phi)
    m_gt0,junk,junk,junk = broadcast_arrays(m > 0, n, theta, phi)
    m_lt0,junk,junk,junk = broadcast_arrays(m < 0, n, theta, phi)

    sh = sph_harm(m, n, theta, phi)
    real_sh = empty(sh.shape, 'double')
    real_sh[m_eq0] = sh[m_eq0].real
    real_sh[m_gt0] = sh[m_gt0].real * sqrt(2)
    real_sh[m_lt0] = sh[m_lt0].imag * sqrt(2)
    return real_sh

def sph_harm_ind_list(sh_order):
    """
    Returns the degree (n) and order (m) of all the symmetric spherical
    harmonics of degree less then or equal it sh_order. The results, m_list
    and n_list are kx1 arrays, where k depends on sh_order. They can be
    passed to real_sph_harm.

    Parameters
    ----------
    sh_order : int
        even int > 0, max degree to return

    Returns
    -------
    m_list : array
        orders of even spherical harmonics
    n_list : array
        degrees of even spherical hormonics

    See also
    --------
    real_sph_harm
    """
    if sh_order % 2 != 0:
        raise ValueError('sh_order must be an even integer >= 0')

    n_range = arange(0, sh_order+1, 2, dtype='int')
    n_list = repeat(n_range, n_range*2+1)

    ncoef = (sh_order + 2)*(sh_order + 1)/2
    offset = 0
    m_list = empty(ncoef, 'int')
    for ii in n_range:
        m_list[offset:offset+2*ii+1] = arange(-ii, ii+1)
        offset = offset + 2*ii + 1

    # makes the arrays ncoef by 1, allows for easy broadcasting later in code
    return (m_list, n_list)

def cartesian2polar(x=0, y=0, z=0):
    """Converts cartesian coordinates to polar coordinates

    converts a list of cartesian coordinates (x, y, z) to polar coordinates
    (R, theta, phi).

    """
    R = sqrt(x*x+y*y+z*z)
    theta = arctan2(y, x)
    phi = arccos(z)

    R, theta, phi = broadcast_arrays(R, theta, phi)

    return R, theta, phi

def reg_pinv(B, L):
    """Regularized psudo-inverse

    Computes a regularized least square inverse of B

    Parameters
    ----------
    B : array_like (n, m)
        Matrix to be inverted
    L : array_like (n,)

    Returns
    -------
    inv : ndarray (m, n)
        regularized least square inverse of B

    Notes
    -----
    In the literature this inverse is often written $(B^{T}B+L^{2})^{-1}B^{T}$.
    However here this inverse is implemented using the psudo-inverse because it
    is more numerically stable than the direct implementation of the matrix
    product. Also because of the transpose issue, B and the result are both
    transposed compared to the literature.

    """
    inv = pinv(c_[B, diag(L)])
    return inv[:B.shape[1]]

def blah(B, L):
    return dot(B.T, inv(dot(B, B.T) + diag(L*L)))

def qball_odf_fit(data, m, n, bvec, smooth):

    data, wrap = _makearray(data)
    R, theta, phi = cartesian2polar(*bvec)
    B = real_sph_harm(m[:, None], n[:, None], theta, phi)
    L = n * (n+1)
    lsqB = smooth_pinv(B, sqrt(smooth)*L)

    C = dot(data, lsqB)

    legendre0 = lpn(sh_order, 0)[0]
    F = legendre0[n]
    return wrap(F*C)

def qball_opdf_fit(data, m, n, bvec, smooth, min_signal=1e-5):
    """Fits an Orientation Probability Density Function to some diffusion data

    The OPDF is a
    """

    data, wrap = _makearray(data)
    R, theta, phi = cartesian2polar(*bvec)
    B = real_sph_harm(m[:, None], n[:, None], theta, phi)
    L = n * n+1
    lsqB = smooth_pinv(B, sqrt(smooth)*L)

    E = maximum(data, min_signal)
    D = log(E)
    C = dot(2*D*(3+2*D)*E, lsqB) - L*dot(E, lsqB)

    legendre0 = lpn(sh_order, 0)[0]
    F = legendre0[n]
    return wrap(F*C)

class SphHarmModels(ModelArray):

    def __init__(self, data, bvec, sh_order=6, mask=True, fit_method=OPDT,
                 *args, **kargs):

        mask = atleast_1d(mask)
        if not mask.any():
            raise ValueError('Mask cannot be all false')

        if not mask.all():
            data = data[mask]
            data = MaskedView(mask, data)
        m, n = sph_harm_ind_list(sh_order)
        sph_hamr_coef = fit_method(data, m, n, bvec, *args, **kargs)

        self.sh_order = sh_order
        self.m = m
        self.n = n
        self.model_params = sph_harm_coef

    def eval_at(self, points):
        R, theta, phi = cartesian2polar(*points)
        n = self.n[:,None]
        m = self.m[:,None]
        design = real_sph_harm(m, n, theta, phi)

        return dot(self.model_params, design)

