"""Tools for using spherical harmonic models to fit diffusion data.

References
----------
.. [1] Aganj, I., et al. 2009. ODF Reconstruction in Q-Ball Imaging With Solid
       Angle Consideration.
.. [2] Descoteaux, M., et al. 2007. Regularized, fast, and robust analytical
       Q-ball imaging.
.. [3] Tristan-Vega, A., et al. 2010. A new methodology for estimation of fiber
       populations in white matter of the brain with Funk-Radon transform.
.. [4] Tristan-Vega, A., et al. 2009. Estimation of fiber orientation
       probability density functions in high angular resolution diffusion
       imaging.


Note about the Transpose:
In the literature the matrix representation of these methods is often written
as Y = Bx where B is some design matrix and Y and x are column vectors. In our
case the input data, a dwi stored as a nifti file for example, is stored as row
vectors (ndarrays) of the form (x, y, z, n), where n is the number of diffusion
directions. We could transpose and reshape the data to be (n, x*y*z), so that
we could directly plug it into the above equation. However, I have chosen to
keep the data as is and implement the relevant equations rewritten in the
following form: Y.T = x.T B.T, or in python syntax data = np.dot(sh_coef, B.T)
where data is Y.T and sh_coef is x.T.
"""

from warnings import warn
import numpy as np
import scipy.special as sps

from numpy.random import randint

from dipy.utils.deprecator import deprecate_with_version
from dipy.reconst.odf import OdfModel, OdfFit
from dipy.core.geometry import cart2sphere
from dipy.core.onetime import auto_attr
from dipy.reconst.cache import Cache
from dipy.utils.deprecator import deprecated_params


descoteaux07_legacy_msg = \
    "The legacy descoteaux07 SH basis uses absolute values for negative " \
    "harmonic phase factors. It is outdated and will be deprecated in a "\
    "future DIPY release. Consider using the new descoteaux07 basis by setting"\
    "the `legacy` parameter to `False`."
tournier07_legacy_msg = \
    "The legacy tournier07 basis is not normalized. It is outdated and will " \
    "be deprecated in a future release of DIPY. Consider using the new " \
    "tournier07 basis by setting the `legacy` parameter to `False`."


def _copydoc(obj):
    def bandit(f):
        f.__doc__ = obj.__doc__
        return f
    return bandit


@deprecated_params('n', 'l_values', since='1.9', until='2.0')
def forward_sdeconv_mat(r_rh, l_values):
    """ Build forward spherical deconvolution matrix

    Parameters
    ----------
    r_rh : ndarray
        Rotational harmonics coefficients for the single fiber response
        function. Each element ``rh[i]`` is associated with spherical harmonics
        of order ``2*i``.
    l_values : ndarray
        The orders (l) of spherical harmonic function associated with each row
        of the deconvolution matrix. Only even orders are allowed

    Returns
    -------
    R : ndarray (N, N)
        Deconvolution matrix with shape (N, N)

    """

    if np.any(l_values % 2):
        raise ValueError("l_values has odd orders, expecting only even orders")
    return np.diag(r_rh[l_values // 2])


@deprecated_params(['m', 'n',], ['m_values', 'l_values'], since='1.9', until='2.0')
def sh_to_rh(r_sh, m_values, l_values):
    """ Spherical harmonics (SH) to rotational harmonics (RH)

    Calculate the rotational harmonic decomposition up to
    harmonic phase factor ``m``, order ``l`` for an axially and antipodally
    symmetric function. Note that all ``m != 0`` coefficients
    will be ignored as axial symmetry is assumed. Hence, there
    will be ``(sh_order/2 + 1)`` non-zero coefficients.

    Parameters
    ----------
    r_sh : ndarray (N,)
        ndarray of SH coefficients for the single fiber response function.
        These coefficients must correspond to the real spherical harmonic
        functions produced by `shm.real_sh_descoteaux_from_index`.
    m_values : ndarray (N,)
        The phase factors (m) of the spherical harmonic function associated with
        each coefficient.
    l_values : ndarray (N,)
        The orders (l) of the spherical harmonic function associated with each
        coefficient.

    Returns
    -------
    r_rh : ndarray (``(sh_order + 1)*(sh_order + 2)/2``,)
        Rotational harmonics coefficients representing the input `r_sh`

    See Also
    --------
    shm.real_sh_descoteaux_from_index, shm.real_sh_descoteaux

    References
    ----------
    .. [1] Tournier, J.D., et al. NeuroImage 2007. Robust determination of the
        fibre orientation distribution in diffusion MRI: Non-negativity
        constrained super-resolved spherical deconvolution

    """
    mask = m_values == 0
    # The delta function at theta = phi = 0 is known to have zero coefficients
    # where m != 0, therefore we need only compute the coefficients at m=0.
    dirac_sh = gen_dirac(0, l_values[mask], 0, 0)
    r_rh = r_sh[mask] / dirac_sh
    return r_rh


@deprecated_params(['m', 'n',], ['m_values', 'l_values'], since='1.9', until='2.0')
def gen_dirac(m_values, l_values, theta, phi, legacy=True):
    """ Generate Dirac delta function orientated in (theta, phi) on the sphere

    The spherical harmonics (SH) representation of this Dirac is returned as
    coefficients to spherical harmonic functions produced from ``descoteaux07``
    basis.

    Parameters
    ----------
    m_values : ndarray (N,)
        The phase factors of the spherical harmonic function associated with
        each coefficient.
    l_values : ndarray (N,)
        The order (l) of the spherical harmonic function associated with each
        coefficient.
    theta : float [0, pi]
        The polar (colatitudinal) coordinate.
    phi : float [0, 2*pi]
        The azimuthal (longitudinal) coordinate.
    legacy: bool, optional
        If true, uses DIPY's legacy descoteaux07 implementation (where |m|
        is used for m < 0). Else, implements the basis as defined in
        Descoteaux et al. 2007 (without the absolute value).

    See Also
    --------
    shm.real_sh_descoteaux_from_index, shm.real_sh_descoteaux

    Returns
    -------
    dirac : ndarray
        SH coefficients representing the Dirac function. The shape of this is
        `(m + 2) * (m + 1) / 2`.

    """
    return real_sh_descoteaux_from_index(m_values, l_values, theta, phi,
                                         legacy=legacy)


@deprecated_params(['m', 'n',], ['m_values', 'l_values'], since='1.9', until='2.0')
def spherical_harmonics(m_values, l_values, theta, phi, use_scipy=True):
    """Compute spherical harmonics.

    This may take scalar or array arguments. The inputs will be broadcast
    against each other.

    Parameters
    ----------
    m_values : array of int ``|m| <= l``
        The phase factors (m) of the harmonics.
    l_values : array of int ``l >= 0``
        The orders (l) of the harmonics.
    theta : float [0, 2*pi]
        The azimuthal (longitudinal) coordinate.
    phi : float [0, pi]
        The polar (colatitudinal) coordinate.
    use_scipy : bool, optional
        If True, use scipy implementation.

    Returns
    -------
    y_mn : complex float
        The harmonic $Y^m_l$ sampled at ``theta`` and ``phi``.

    Notes
    -----
    This is a faster implementation of scipy.special.sph_harm for
    scipy version < 0.15.0. For scipy 0.15 and onwards, we use the scipy
    implementation of the function.

    The usual definitions for ``theta` and `phi`` used in DIPY are interchanged
    in the method definition to agree with the definitions in
    scipy.special.sph_harm, where `theta` represents the azimuthal coordinate
    and `phi` represents the polar coordinate.

    Although scipy uses a naming convention where ``m`` is the order and ``n``
    is the degree of the SH, the opposite of DIPY's, their definition for
    both parameters is the same as ours, with ``l >= 0`` and ``|m| <= l``.
    """
    if use_scipy:
        return sps.sph_harm(m_values, l_values, theta, phi, dtype=complex)

    x = np.cos(phi)
    val = sps.lpmv(m_values, l_values, x).astype(complex)
    val *= np.sqrt((2 * l_values + 1) / 4.0 / np.pi)
    val *= np.exp(0.5 * (sps.gammaln(l_values - m_values + 1) - sps.gammaln(l_values + m_values + 1)))
    val = val * np.exp(1j * m_values * theta)
    return val


@deprecate_with_version('dipy.reconst.shm.real_sph_harm is deprecated, '
                        'Please use '
                        'dipy.reconst.shm.real_sh_descoteaux_from_index '
                        'instead', since='1.3', until='2.0')
@deprecated_params(['m', 'n',], ['m_values', 'l_values'], since='1.9', until='2.0')
def real_sph_harm(m_values, l_values, theta, phi):
    """ Compute real spherical harmonics.

    Where the real harmonic $Y^m_l$ is defined to be:

        Imag($Y^m_l$) * sqrt(2)     if m > 0
        $Y^0_l$                     if m = 0
        Real($Y^|m|_l$) * sqrt(2)   if m < 0

    This may take scalar or array arguments. The inputs will be broadcast
    against each other.

    Parameters
    ----------
    m_values : array of int ``|m| <= l``
        The phase factors (m) of the harmonics.
    l_values : array of int ``l >= 0``
        The orders (l) of the harmonics.
    theta : float [0, pi]
        The polar (colatitudinal) coordinate.
    phi : float [0, 2*pi]
        The azimuthal (longitudinal) coordinate.

    Returns
    -------
    y_mn : real float
        The real harmonic $Y^m_l$ sampled at `theta` and `phi`.

    See Also
    --------
    scipy.special.sph_harm
    """
    return real_sh_descoteaux_from_index(m_values, l_values, theta, phi, legacy=True)


@deprecated_params(['m', 'n',], ['m_values', 'l_values'], since='1.9', until='2.0')
def real_sh_tournier_from_index(m_values, l_values, theta, phi, legacy=True):
    """ Compute real spherical harmonics as initially defined in Tournier
    2007 [1]_ then updated in MRtrix3 [2]_, where the real harmonic $Y^m_l$
    is defined to be:

        Real($Y^m_l$) * sqrt(2)      if m > 0
        $Y^0_l$                      if m = 0
        Imag($Y^|m|_l$) * sqrt(2)    if m < 0

    This may take scalar or array arguments. The inputs will be broadcast
    against each other.

    Parameters
    ----------
    m_values : array of int ``|m| <= l``
        The phase factors (m) of the harmonics.
    l_values : array of int ``l >= 0``
        The orders (l) of the harmonics.
    theta : float [0, pi]
        The polar (colatitudinal) coordinate.
    phi : float [0, 2*pi]
        The azimuthal (longitudinal) coordinate.
    legacy: bool, optional
        If true, uses MRtrix 0.2 SH basis definition, where the ``sqrt(2)``
        factor is omitted. Else, uses the MRtrix 3 definition presented above.

    Returns
    -------
    real_sh : real float
        The real harmonics $Y^m_l$ sampled at ``theta`` and ``phi``.

    References
    ----------
    .. [1] Tournier J.D., Calamante F. and Connelly A. Robust determination
           of the fibre orientation distribution in diffusion MRI:
           Non-negativity constrained super-resolved spherical deconvolution.
           NeuroImage. 2007;35(4):1459-1472.
    .. [2] Tournier J-D, Smith R, Raffelt D, Tabbara R, Dhollander T,
           Pietsch M, et al. MRtrix3: A fast, flexible and open software
           framework for medical image processing and visualisation.
           NeuroImage. 2019 Nov 15;202:116-137.
    """
    # In the m < 0 case, Tournier basis considers |m|
    sh = spherical_harmonics(np.abs(m_values), l_values, phi, theta)
    real_sh = np.where(m_values < 0, sh.imag, sh.real)

    if not legacy:
        # The Tournier basis from MRtrix3 is normalized
        real_sh *= np.where(m_values == 0, 1., np.sqrt(2))
    else:
        warn(tournier07_legacy_msg, category=PendingDeprecationWarning)

    return real_sh


@deprecated_params(['m', 'n',], ['m_values', 'l_values'], since='1.9', until='2.0')
def real_sh_descoteaux_from_index(m_values, l_values, theta, phi, legacy=True):
    """ Compute real spherical harmonics as in Descoteaux et al. 2007 [1]_,
    where the real harmonic $Y^m_l$ is defined to be:

        Imag($Y^m_l$) * sqrt(2)      if m > 0
        $Y^0_l$                      if m = 0
        Real($Y^m_l$) * sqrt(2)      if m < 0

    This may take scalar or array arguments. The inputs will be broadcast
    against each other.

    Parameters
    ----------
    m_values : array of int ``|m| <= l``
        The phase factors (m) of the harmonics.
    l_values : array of int ``l >= 0``
        The orders (l) of the harmonics.
    theta : float [0, pi]
        The polar (colatitudinal) coordinate.
    phi : float [0, 2*pi]
        The azimuthal (longitudinal) coordinate.
    legacy: bool, optional
        If true, uses DIPY's legacy descoteaux07 implementation (where |m|
        is used for m < 0). Else, implements the basis as defined in
        Descoteaux et al. 2007 (without the absolute value).

    Returns
    -------
    real_sh : real float
        The real harmonic $Y^m_l$ sampled at ``theta`` and ``phi``.

    References
    ----------
     .. [1] Descoteaux, M., Angelino, E., Fitzgibbons, S. and Deriche, R.
           Regularized, Fast, and Robust Analytical Q-ball Imaging.
           Magn. Reson. Med. 2007;58:497-510.
    """
    if legacy:
        # In the case where m < 0, legacy descoteaux basis considers |m|
        warn(descoteaux07_legacy_msg, category=PendingDeprecationWarning)
        sh = spherical_harmonics(np.abs(m_values), l_values, phi, theta)
    else:
        # In the cited paper, the basis is defined without the absolute value
        sh = spherical_harmonics(m_values, l_values, phi, theta)

    real_sh = np.where(m_values > 0, sh.imag, sh.real)
    real_sh *= np.where(m_values == 0, 1., np.sqrt(2))

    return real_sh

@deprecated_params('sh_order', 'sh_order_max', since='1.9', until='2.0')
def real_sh_tournier(sh_order_max, theta, phi,
                     full_basis=False,
                     legacy=True):
    """ Compute real spherical harmonics as initially defined in Tournier
    2007 [1]_ then updated in MRtrix3 [2]_, where the real harmonic $Y^m_l$
    is defined to be:

        Real($Y^m_l$) * sqrt(2)      if m > 0
        $Y^0_l$                      if m = 0
        Imag($Y^|m|_l$) * sqrt(2)    if m < 0

    This may take scalar or array arguments. The inputs will be broadcast
    against each other.

    Parameters
    ----------
    sh_order_max : int
        The maximum order (l) of the spherical harmonic basis.
    theta : float [0, pi]
        The polar (colatitudinal) coordinate.
    phi : float [0, 2*pi]
        The azimuthal (longitudinal) coordinate.
    full_basis: bool, optional
        If true, returns a basis including odd order SH functions as well as
        even order SH functions. Else returns only even order SH functions.
    legacy: bool, optional
        If true, uses MRtrix 0.2 SH basis definition, where the ``sqrt(2)``
        factor is omitted. Else, uses MRtrix 3 definition presented above.

    Returns
    -------
    real_sh : real float
        The real harmonic $Y^m_l$ sampled at ``theta`` and ``phi``.
    m_values : array of int
        The phase factor (m) of the harmonics.
    l_values : array of int
        The order (l) of the harmonics.

    References
    ----------
    .. [1] Tournier J.D., Calamante F. and Connelly A. Robust determination
           of the fibre orientation distribution in diffusion MRI:
           Non-negativity constrained super-resolved spherical deconvolution.
           NeuroImage. 2007;35(4):1459-1472.
    .. [2] Tournier J-D, Smith R, Raffelt D, Tabbara R, Dhollander T,
           Pietsch M, et al. MRtrix3: A fast, flexible and open software
           framework for medical image processing and visualisation.
           NeuroImage. 2019 Nov 15;202:116-137.
    """
    m_values, l_values = sph_harm_ind_list(sh_order_max, full_basis)

    phi = np.reshape(phi, [-1, 1])
    theta = np.reshape(theta, [-1, 1])

    real_sh = real_sh_tournier_from_index(m_values, l_values, theta, phi, legacy)

    return real_sh, m_values, l_values

@deprecated_params('sh_order', 'sh_order_max', since='1.9', until='2.0')
def real_sh_descoteaux(sh_order_max, theta, phi,
                       full_basis=False,
                       legacy=True):
    """ Compute real spherical harmonics as in Descoteaux et al. 2007 [1]_,
    where the real harmonic $Y^m_l$ is defined to be:

        Imag($Y^m_l$) * sqrt(2)      if m > 0
        $Y^0_l$                      if m = 0
        Real($Y^m_l$) * sqrt(2)      if m < 0

    This may take scalar or array arguments. The inputs will be broadcast
    against each other.

    Parameters
    ----------
    sh_order_max : int
        The maximum order (l) of the spherical harmonic basis.
    theta : float [0, pi]
        The polar (colatitudinal) coordinate.
    phi : float [0, 2*pi]
        The azimuthal (longitudinal) coordinate.
    full_basis: bool, optional
        If true, returns a basis including odd order SH functions as well as
        even order SH functions. Otherwise returns only even order SH
        functions.
    legacy: bool, optional
        If true, uses DIPY's legacy descoteaux07 implementation (where |m|
        for m < 0). Else, implements the basis as defined in Descoteaux et al.
        2007.

    Returns
    -------
    real_sh : real float
        The real harmonic $Y^m_l$ sampled at ``theta`` and ``phi``.
    m_values : array of int
        The phase factor (m) of the harmonics.
    l_values : array of int
        The order (l) of the harmonics.

    References
    ----------
     .. [1] Descoteaux, M., Angelino, E., Fitzgibbons, S. and Deriche, R.
           Regularized, Fast, and Robust Analytical Q-ball Imaging.
           Magn. Reson. Med. 2007;58:497-510.
    """
    m_value, l_value = sph_harm_ind_list(sh_order_max, full_basis)

    phi = np.reshape(phi, [-1, 1])
    theta = np.reshape(theta, [-1, 1])

    real_sh = real_sh_descoteaux_from_index(m_value, l_value, theta, phi,
                                            legacy)

    return real_sh, m_value, l_value


@deprecate_with_version('dipy.reconst.shm.real_sym_sh_mrtrix is deprecated, '
                        'Please use dipy.reconst.shm.real_sh_tournier instead',
                        since='1.3', until='2.0')
@deprecated_params('sh_order', 'sh_order_max', since='1.9', until='2.0')
def real_sym_sh_mrtrix(sh_order_max, theta, phi):
    """
    Compute real symmetric spherical harmonics as in Tournier 2007 [2]_, where
    the real harmonic $Y^m_l$ is defined to be::

        Real($Y^m_l$)       if m > 0
        $Y^0_l$             if m = 0
        Imag($Y^|m|_l$)     if m < 0

    This may take scalar or array arguments. The inputs will be broadcast
    against each other.

    Parameters
    ----------
    sh_order_max : int
        The maximum order (l) of the spherical harmonic basis.
    theta : float [0, pi]
        The polar (colatitudinal) coordinate.
    phi : float [0, 2*pi]
        The azimuthal (longitudinal) coordinate.

    Returns
    -------
    y_mn : real float
        The real harmonic $Y^m_l$ sampled at ``theta`` and ``phi`` as
        implemented in mrtrix. Warning: the basis is Tournier et al.
        2007 [2]_; 2004 [1]_ is slightly different.
    m_values : array
        The phase factor (m) of the harmonics.
    l_values : array
        The order (l) of the harmonics.

    References
    ----------
    .. [1] Tournier J.D., Calamante F., Gadian D.G. and Connelly A.
           Direct estimation of the fibre orientation density function from
           diffusion-weighted MRI data using spherical deconvolution.
           NeuroImage. 2004;23:1176-1185.
    .. [2] Tournier J.D., Calamante F. and Connelly A. Robust determination
           of the fibre orientation distribution in diffusion MRI:
           Non-negativity constrained super-resolved spherical deconvolution.
           NeuroImage. 2007;35(4):1459-1472.

    """
    return real_sh_tournier(sh_order_max, theta, phi, legacy=True)


@deprecate_with_version('dipy.reconst.shm.real_sym_sh_basis is deprecated, '
                        'Please use dipy.reconst.shm.real_sh_descoteaux '
                        'instead', since='1.3', until='2.0')
@deprecated_params('sh_order', 'sh_order_max', since='1.9', until='2.0')
def real_sym_sh_basis(sh_order_max, theta, phi):
    """Samples a real symmetric spherical harmonic basis at point on the sphere

    Samples the basis functions up to order `sh_order_max` at points on the
    sphere given by `theta` and `phi`. The basis functions are defined here the
    same way as in Descoteaux et al. 2007 [1]_ where the real harmonic $Y^m_l$
    is defined to be:

        Imag($Y^m_l$) * sqrt(2)     if m > 0
        $Y^0_l$                     if m = 0
        Real($Y^|m|_l$) * sqrt(2)   if m < 0

    This may take scalar or array arguments. The inputs will be broadcast
    against each other.

    Parameters
    ----------
    sh_order_max : int
        The maximum order (l) of the spherical harmonic basis. Even int > 0,
        max spherical harmonic order
    theta : float [0, 2*pi]
        The azimuthal (longitudinal) coordinate.
    phi : float [0, pi]
        The polar (colatitudinal) coordinate.

    Returns
    -------
    y_mn : real float
        The real harmonic $Y^m_l$ sampled at ``theta`` and ``phi``
    m_values : array of int
        The phase factor (m) of the harmonics.
    l_values : array of int
        The order (l) of the harmonics.

    References
    ----------
    .. [1] Descoteaux, M., Angelino, E., Fitzgibbons, S. and Deriche, R.
           Regularized, Fast, and Robust Analytical Q-ball Imaging.
           Magn. Reson. Med. 2007;58:497-510.

    """
    return real_sh_descoteaux(sh_order_max, theta, phi, legacy=True)


sph_harm_lookup = {None: real_sh_descoteaux,
                   "tournier07": real_sh_tournier,
                   "descoteaux07": real_sh_descoteaux}

@deprecated_params('sh_order', 'sh_order_max', since='1.9', until='2.0')
def sph_harm_ind_list(sh_order_max, full_basis=False):
    """
    Returns the order (``l``) and phase_factor (``m``) of all the symmetric
    spherical harmonics of order less then or equal to ``sh_order_max``.
    The results, ``m_list`` and ``l_list`` are kx1 arrays, where k depends on
    ``sh_order_max``.
    They can be passed to :func:`real_sh_descoteaux_from_index` and
    :func:``real_sh_tournier_from_index``.

    Parameters
    ----------
    sh_order_max : int
        The maximum order (l) of the spherical harmonic basis.
        Even int > 0, max order to return
    full_basis: bool, optional
        True for SH basis with even and odd order terms

    Returns
    -------
    m_list : array of int
        phase factors (m) of even spherical harmonics
    l_list : array of int
        orders (l) of even spherical harmonics

    See Also
    --------
    shm.real_sh_descoteaux_from_index, shm.real_sh_tournier_from_index

    """
    if full_basis:
        l_range = np.arange(0, sh_order_max + 1, dtype=int)
        ncoef = int((sh_order_max + 1) * (sh_order_max + 1))
    else:
        if sh_order_max % 2 != 0:
            raise ValueError('sh_order_max must be an even integer >= 0')
        l_range = np.arange(0, sh_order_max + 1, 2, dtype=int)
        ncoef = int((sh_order_max + 2) * (sh_order_max + 1) // 2)

    l_list = np.repeat(l_range, l_range * 2 + 1)
    offset = 0
    m_list = np.empty(ncoef, 'int')
    for ii in l_range:
        m_list[offset:offset + 2 * ii + 1] = np.arange(-ii, ii + 1)
        offset = offset + 2 * ii + 1

    # makes the arrays ncoef by 1, allows for easy broadcasting later in code
    return m_list, l_list


def order_from_ncoef(ncoef, full_basis=False):
    """
    Given a number ``n`` of coefficients, calculate back the ``sh_order_max``

    Parameters
    ----------
    ncoef: int
        number of coefficients
    full_basis: bool, optional
        True when coefficients are for a full SH basis.

    Returns
    -------
    sh_order_max: int
        maximum order (l) of SH basis
    """
    if full_basis:
        # Solve the equation :
        # ncoef = (sh_order_max + 1) * (sh_order_max + 1)
        return int(np.sqrt(ncoef) - 1)

    # Solve the quadratic equation derived from :
    # ncoef = (sh_order_max + 2) * (sh_order_max + 1) / 2
    return -1 + int(np.sqrt(9 - 4 * (2-2*ncoef))//2)


def smooth_pinv(B, L):
    """Regularized pseudo-inverse

    Computes a regularized least square inverse of B

    Parameters
    ----------
    B : array_like (n, m)
        Matrix to be inverted
    L : array_like (m,)

    Returns
    -------
    inv : ndarray (m, n)
        regularized least square inverse of B

    Notes
    -----
    In the literature this inverse is often written $(B^{T}B+L^{2})^{-1}B^{T}$.
    However here this inverse is implemented using the pseudo-inverse because
    it is more numerically stable than the direct implementation of the matrix
    product.

    """
    L = np.diag(L)
    inv = np.linalg.pinv(np.concatenate((B, L)))
    return inv[:, :len(B)]


def lazy_index(index):
    """Produces a lazy index

    Returns a slice that can be used for indexing an array, if no slice can be
    made index is returned as is.
    """
    index = np.array(index)
    assert index.ndim == 1
    if index.dtype.kind == 'b':
        index = index.nonzero()[0]
    if len(index) == 1:
        return slice(index[0], index[0] + 1)
    step = np.unique(np.diff(index))
    if len(step) != 1 or step[0] == 0:
        return index
    else:
        return slice(index[0], index[-1] + 1, step[0])


def _gfa_sh(coef, sh0_index=0):
    """The gfa of the odf, computed from the spherical harmonic coefficients

    This is a private function because it only works for coefficients of
    normalized sh bases.

    Parameters
    ----------
    coef : array
        The coefficients, using a normalized sh basis, that represent each odf.
    sh0_index : int, optional
        The index of the coefficient associated with the 0th order sh harmonic.

    Returns
    -------
    gfa_values : array
        The gfa of each odf.

    """
    coef_sq = coef**2
    numer = coef_sq[..., sh0_index]
    denom = coef_sq.sum(-1)
    # The sum of the square of the coefficients being zero is the same as all
    # the coefficients being zero
    allzero = denom == 0
    # By adding 1 to numer and denom where both and are 0, we prevent 0/0
    numer = numer + allzero
    denom = denom + allzero
    return np.sqrt(1. - (numer / denom))


class SphHarmModel(OdfModel, Cache):
    """To be subclassed by all models that return a SphHarmFit when fit."""

    def sampling_matrix(self, sphere):
        """The matrix needed to sample ODFs from coefficients of the model.

        Parameters
        ----------
        sphere : Sphere
            Points used to sample ODF.

        Returns
        -------
        sampling_matrix : array
            The size of the matrix will be (N, M) where N is the number of
            vertices on sphere and M is the number of coefficients needed by
            the model.
        """
        sampling_matrix = self.cache_get("sampling_matrix", sphere)
        if sampling_matrix is None:
            sh_order = self.sh_order_max
            theta = sphere.theta
            phi = sphere.phi
            sampling_matrix, m_values, l_values = real_sh_descoteaux(sh_order,
                                                                     theta, phi)
            self.cache_set("sampling_matrix", sphere, sampling_matrix)
        return sampling_matrix


class QballBaseModel(SphHarmModel):
    """To be subclassed by Qball type models."""
    @deprecated_params('sh_order', 'sh_order_max', since='1.9', until='2.0')
    def __init__(self, gtab, sh_order_max, smooth=0.006, min_signal=1e-5,
                 assume_normed=False):
        """Creates a model that can be used to fit or sample diffusion data

        Parameters
        ----------
        gtab : GradientTable
            Diffusion gradients used to acquire data
        sh_order_max : even int >= 0
            the maximal spherical harmonic order (l) of the model
        smooth : float between 0 and 1, optional
            The regularization parameter of the model
        min_signal : float, > 0, optional
            During fitting, all signal values less than `min_signal` are
            clipped to `min_signal`. This is done primarily to avoid values
            less than or equal to zero when taking logs.
        assume_normed : bool, optional
            If True, clipping and normalization of the data with respect to the
            mean B0 signal are skipped during mode fitting. This is an advanced
            feature and should be used with care.

        See Also
        --------
        normalize_data

        """
        SphHarmModel.__init__(self, gtab)
        self._where_b0s = lazy_index(gtab.b0s_mask)
        self._where_dwi = lazy_index(~gtab.b0s_mask)
        self.assume_normed = assume_normed
        self.min_signal = min_signal
        x, y, z = gtab.gradients[self._where_dwi].T
        r, theta, phi = cart2sphere(x, y, z)
        B, m_values, l_values = real_sh_descoteaux(sh_order_max, theta[:, None],
                                                   phi[:, None])
        L = -l_values * (l_values + 1)
        legendre0 = sps.lpn(sh_order_max, 0)[0]
        F = legendre0[l_values]
        self.sh_order_max = sh_order_max
        self.B = B
        self.m_values = m_values
        self.l_values = l_values
        self._set_fit_matrix(B, L, F, smooth)

    def _set_fit_matrix(self, *args):
        """Should be set in a subclass and is called by __init__"""
        msg = "User must implement this method in a subclass"
        raise NotImplementedError(msg)

    def fit(self, data, mask=None):
        """Fits the model to diffusion data and returns the model fit"""
        # Normalize the data and fit coefficients
        if not self.assume_normed:
            data = normalize_data(data, self._where_b0s, self.min_signal)

        # Compute coefficients using abstract method
        coef = self._get_shm_coef(data)

        # Apply the mask to the coefficients
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            coef *= mask[..., None]
        return SphHarmFit(self, coef, mask)


class SphHarmFit(OdfFit):
    """Diffusion data fit to a spherical harmonic model"""

    def __init__(self, model, shm_coef, mask):
        self.model = model
        self._shm_coef = shm_coef
        self.mask = mask

    @property
    def shape(self):
        return self._shm_coef.shape[:-1]

    def __getitem__(self, index):
        """Allowing indexing into fit"""
        # Index shm_coefficients
        if isinstance(index, tuple):
            coef_index = index + (Ellipsis,)
        else:
            coef_index = index
        new_coef = self._shm_coef[coef_index]

        # Index mask
        if self.mask is not None:
            new_mask = self.mask[index]
            assert new_mask.shape == new_coef.shape[:-1]
        else:
            new_mask = None

        return SphHarmFit(self.model, new_coef, new_mask)

    def odf(self, sphere):
        """Samples the odf function on the points of a sphere

        Parameters
        ----------
        sphere : Sphere
            The points on which to sample the odf.

        Returns
        -------
        values : ndarray
            The value of the odf on each point of `sphere`.

        """
        B = self.model.sampling_matrix(sphere)
        return np.dot(self.shm_coeff, B.T)

    @auto_attr
    def gfa(self):
        return _gfa_sh(self.shm_coeff, 0)

    @property
    def shm_coeff(self):
        """The spherical harmonic coefficients of the odf

        Make this a property for now, if there is a use case for modifying
        the coefficients we can add a setter or expose the coefficients more
        directly
        """
        return self._shm_coef

    def predict(self, gtab=None, S0=1.0):
        """
        Predict the diffusion signal from the model coefficients.

        Parameters
        ----------
        gtab : a GradientTable class instance
            The directions and bvalues on which prediction is desired

        S0 : float array
           The mean non-diffusion-weighted signal in each voxel.
        """
        if not hasattr(self.model, 'predict'):
            msg = "This model does not have prediction implemented yet"
            raise NotImplementedError(msg)
        return self.model.predict(self._shm_coef, gtab, S0)


class CsaOdfModel(QballBaseModel):
    """Implementation of Constant Solid Angle reconstruction method.

    References
    ----------
    .. [1] Aganj, I., et al. 2009. ODF Reconstruction in Q-Ball Imaging With
           Solid Angle Consideration.
    """
    min = .001
    max = .999
    _n0_const = .5 / np.sqrt(np.pi)

    def _set_fit_matrix(self, B, L, F, smooth):
        """The fit matrix, is used by fit_coefficients to return the
        coefficients of the odf"""
        invB = smooth_pinv(B, np.sqrt(smooth) * L)
        L = L[:, None]
        F = F[:, None]
        self._fit_matrix = (F * L) / (8 * np.pi) * invB

    def _get_shm_coef(self, data, mask=None):
        """Returns the coefficients of the model"""
        data = data[..., self._where_dwi]
        data = data.clip(self.min, self.max)
        loglog_data = np.log(-np.log(data))
        sh_coef = np.dot(loglog_data, self._fit_matrix.T)
        sh_coef[..., 0] = self._n0_const
        return sh_coef


class OpdtModel(QballBaseModel):
    """Implementation of Orientation Probability Density Transform
    reconstruction method.

    References
    ----------
    .. [1] Tristan-Vega, A., et al. 2010. A new methodology for estimation of
           fiber populations in white matter of the brain with Funk-Radon
           transform.
    .. [2] Tristan-Vega, A., et al. 2009. Estimation of fiber orientation
           probability density functions in high angular resolution diffusion
           imaging.
    """
    def _set_fit_matrix(self, B, L, F, smooth):
        invB = smooth_pinv(B, np.sqrt(smooth) * L)
        L = L[:, None]
        F = F[:, None]
        delta_b = F * L * invB
        delta_q = 4 * F * invB
        self._fit_matrix = delta_b, delta_q

    def _get_shm_coef(self, data, mask=None):
        """Returns the coefficients of the model"""
        delta_b, delta_q = self._fit_matrix
        return _slowadc_formula(data[..., self._where_dwi], delta_b, delta_q)


def _slowadc_formula(data, delta_b, delta_q):
    """formula used in SlowAdcOpdfModel"""
    logd = -np.log(data)
    return (np.dot(logd * (1.5 - logd) * data, delta_q.T)
            - np.dot(data, delta_b.T))


class QballModel(QballBaseModel):
    """Implementation of regularized Qball reconstruction method.

    References
    ----------
    .. [1] Descoteaux, M., et al. 2007. Regularized, fast, and robust
           analytical Q-ball imaging.
    """

    def _set_fit_matrix(self, B, L, F, smooth):
        invB = smooth_pinv(B, np.sqrt(smooth) * L)
        F = F[:, None]
        self._fit_matrix = F * invB

    def _get_shm_coef(self, data, mask=None):
        """Returns the coefficients of the model"""
        return np.dot(data[..., self._where_dwi], self._fit_matrix.T)


def normalize_data(data, where_b0, min_signal=1e-5, out=None):
    """Normalizes the data with respect to the mean b0
    """
    if out is None:
        out = np.array(data, dtype='float32', copy=True)
    else:
        if out.dtype.kind != 'f':
            raise ValueError("out must be floating point")
        out[:] = data

    out.clip(min_signal, out=out)
    b0 = out[..., where_b0].mean(-1)
    out /= b0[..., None]
    return out


def hat(B):
    """Returns the hat matrix for the design matrix B
    """

    U, S, V = np.linalg.svd(B, False)
    H = np.dot(U, U.T)
    return H


def lcr_matrix(H):
    """Returns a matrix for computing leveraged, centered residuals from data

    if r = (d-Hd), the leveraged centered residuals are lcr = (r/l)-mean(r/l)
    returns the matrix R, such that lcr = Rd

    """
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError('H should be a square matrix')

    leverages = np.sqrt(1 - H.diagonal(), where=H.diagonal() < 1)
    leverages = leverages[:, None]
    R = (np.eye(len(H)) - H) / leverages
    return R - R.mean(0)


def bootstrap_data_array(data, H, R, permute=None):
    """Applies the Residual Bootstraps to the data given H and R

    data must be normalized, ie 0 < data <= 1

    This function, and the bootstrap_data_voxel function, calculate
    residual-bootstrap samples given a Hat matrix and a Residual matrix. These
    samples can be used for non-parametric statistics or for bootstrap
    probabilistic tractography:

    References
    ----------
    .. [1] J. I. Berman, et al., "Probabilistic streamline q-ball tractography
           using the residual bootstrap" 2008.
    .. [2] HA Haroon, et al., "Using the model-based residual bootstrap to
           quantify uncertainty in fiber orientations from Q-ball analysis"
           2009.
    .. [3] B. Jeurissen, et al., "Probabilistic Fiber Tracking Using the
           Residual Bootstrap with Constrained Spherical Deconvolution" 2011.
    """

    if permute is None:
        permute = randint(data.shape[-1], size=data.shape[-1])
    assert R.shape == H.shape
    assert len(permute) == R.shape[-1]
    R = R[permute]
    data = np.dot(data, (H + R).T)
    return data


def bootstrap_data_voxel(data, H, R, permute=None):
    """Like bootstrap_data_array but faster when for a single voxel

    data must be 1d and normalized
    """
    if permute is None:
        permute = randint(data.shape[-1], size=data.shape[-1])
    r = np.dot(data, R.T)
    boot_data = np.dot(data, H.T)
    boot_data += r[permute]
    return boot_data


class ResidualBootstrapWrapper:
    """Returns a residual bootstrap sample of the signal_object when indexed

    Wraps a signal_object, this signal object can be an interpolator. When
    indexed, the the wrapper indexes the signal_object to get the signal.
    There wrapper than samples the residual bootstrap distribution of signal and
    returns that sample.
    """
    def __init__(self, signal_object, B, where_dwi, min_signal=1e-5):
        """Builds a ResidualBootstrapWapper

        Given some linear model described by B, the design matrix, and a
        signal_object, returns an object which can sample the residual
        bootstrap distribution of the signal. We assume that the signals are
        normalized so we clip the bootstrap samples to be between `min_signal`
        and 1.

        Parameters
        ----------
        signal_object : some object that can be indexed
            This object should return diffusion weighted signals when indexed.
        B : ndarray, ndim=2
            The design matrix of the spherical harmonics model used to fit the
            data. This is the model that will be used to compute the residuals
            and sample the residual bootstrap distribution
        where_dwi :
            indexing object to find diffusion weighted signals from signal
        min_signal : float
            The lowest allowable signal.
        """
        self._signal_object = signal_object
        self._H = hat(B)
        self._R = lcr_matrix(self._H)
        self._min_signal = min_signal
        self._where_dwi = where_dwi
        self.data = signal_object.data
        self.voxel_size = signal_object.voxel_size

    def __getitem__(self, index):
        """Indexes self._signal_object and bootstraps the result"""
        signal = self._signal_object[index].copy()
        dwi_signal = signal[self._where_dwi]
        boot_signal = bootstrap_data_voxel(dwi_signal, self._H, self._R)
        boot_signal.clip(self._min_signal, 1., out=boot_signal)
        signal[self._where_dwi] = boot_signal
        return signal

@deprecated_params('sh_order', 'sh_order_max', since='1.9', until='2.0')
def sf_to_sh(sf, sphere, sh_order_max=4, basis_type=None, full_basis=False,
             legacy=True, smooth=0.0):
    """Spherical function to spherical harmonics (SH).

    Parameters
    ----------
    sf : ndarray
        Values of a function on the given ``sphere``.
    sphere : Sphere
        The points on which the sf is defined.
    sh_order_max : int, optional
        Maximum SH order (l) in the SH fit.  For ``sh_order_max``, there will be
        ``(sh_order_max + 1) * (sh_order_max + 2) / 2`` SH coefficients for a
        symmetric basis and ``(sh_order_max + 1) * (sh_order_max + 1)``
        coefficients for a full SH basis.
    basis_type : {None, 'tournier07', 'descoteaux07'}, optional
        ``None`` for the default DIPY basis,
        ``tournier07`` for the Tournier 2007 [2]_[3]_ basis,
        ``descoteaux07`` for the Descoteaux 2007 [1]_ basis,
        (``None`` defaults to ``descoteaux07``).
    full_basis: bool, optional
        True for using a SH basis containing even and odd order SH functions.
        False for using a SH basis consisting only of even order SH functions.
    legacy: bool, optional
        True to use a legacy basis definition for backward compatibility
        with previous ``tournier07`` and ``descoteaux07`` implementations.
    smooth : float, optional
        Lambda-regularization in the SH fit.

    Returns
    -------
    sh : ndarray
        SH coefficients representing the input function.

    References
    ----------
    .. [1] Descoteaux, M., Angelino, E., Fitzgibbons, S. and Deriche, R.
           Regularized, Fast, and Robust Analytical Q-ball Imaging.
           Magn. Reson. Med. 2007;58:497-510.
    .. [2] Tournier J.D., Calamante F. and Connelly A. Robust determination
           of the fibre orientation distribution in diffusion MRI:
           Non-negativity constrained super-resolved spherical deconvolution.
           NeuroImage. 2007;35(4):1459-1472.
    .. [3] Tournier J-D, Smith R, Raffelt D, Tabbara R, Dhollander T,
           Pietsch M, et al. MRtrix3: A fast, flexible and open software
           framework for medical image processing and visualisation.
           NeuroImage. 2019 Nov 15;202:116-137.
    """

    sph_harm_basis = sph_harm_lookup.get(basis_type)

    if sph_harm_basis is None:
        raise ValueError("Invalid basis name.")
    B, m_values, l_values = sph_harm_basis(sh_order_max, sphere.theta,
                                           sphere.phi,
                                           full_basis=full_basis,
                                           legacy=legacy)

    L = -l_values * (l_values + 1)
    invB = smooth_pinv(B, np.sqrt(smooth) * L)
    sh = np.dot(sf, invB.T)

    return sh

@deprecated_params('sh_order', 'sh_order_max', since='1.9', until='2.0')
def sh_to_sf(sh, sphere, sh_order_max=4, basis_type=None,
             full_basis=False, legacy=True):
    """Spherical harmonics (SH) to spherical function (SF).

    Parameters
    ----------
    sh : ndarray
        SH coefficients representing a spherical function.
    sphere : Sphere
        The points on which to sample the spherical function.
    sh_order_max : int, optional
        Maximum SH order (l) in the SH fit.  For ``sh_order_max``, there will be
        ``(sh_order_max + 1) * (sh_order_max + 2) / 2`` SH coefficients for a
        symmetric basis and ``(sh_order_max + 1) * (sh_order_max + 1)``
        coefficients for a full SH basis.
    basis_type : {None, 'tournier07', 'descoteaux07'}, optional
        ``None`` for the default DIPY basis,
        ``tournier07`` for the Tournier 2007 [2]_[3]_ basis,
        ``descoteaux07`` for the Descoteaux 2007 [1]_ basis,
        (``None`` defaults to ``descoteaux07``).
    full_basis: bool, optional
        True to use a SH basis containing even and odd order SH functions.
        Else, use a SH basis consisting only of even order SH functions.
    legacy: bool, optional
        True to use a legacy basis definition for backward compatibility
        with previous ``tournier07`` and ``descoteaux07`` implementations.

    Returns
    -------
    sf : ndarray
         Spherical function values on the ``sphere``.

    References
    ----------
    .. [1] Descoteaux, M., Angelino, E., Fitzgibbons, S. and Deriche, R.
           Regularized, Fast, and Robust Analytical Q-ball Imaging.
           Magn. Reson. Med. 2007;58:497-510.
    .. [2] Tournier J.D., Calamante F. and Connelly A. Robust determination
           of the fibre orientation distribution in diffusion MRI:
           Non-negativity constrained super-resolved spherical deconvolution.
           NeuroImage. 2007;35(4):1459-1472.
    .. [3] Tournier J-D, Smith R, Raffelt D, Tabbara R, Dhollander T,
           Pietsch M, et al. MRtrix3: A fast, flexible and open software
           framework for medical image processing and visualisation.
           NeuroImage. 2019 Nov 15;202:116-137.
    """

    sph_harm_basis = sph_harm_lookup.get(basis_type)

    if sph_harm_basis is None:
        raise ValueError("Invalid basis name.")
    B, m_values, l_values = sph_harm_basis(sh_order_max, sphere.theta,
                                           sphere.phi,
                                           full_basis=full_basis,
                                           legacy=legacy)

    sf = np.dot(sh, B.T)

    return sf

@deprecated_params('sh_order', 'sh_order_max', since='1.9', until='2.0')
def sh_to_sf_matrix(sphere, sh_order_max=4, basis_type=None, full_basis=False,
                    legacy=True, return_inv=True, smooth=0):
    """ Matrix that transforms Spherical harmonics (SH) to spherical
    function (SF).

    Parameters
    ----------
    sphere : Sphere
        The points on which to sample the spherical function.
    sh_order_max : int, optional
        Maximum SH order in the SH fit.  For ``sh_order_max``, there will be
        ``(sh_order_max + 1) * (sh_order_max + 2) / 2`` SH coefficients for a
        symmetric basis and ``(sh_order_max + 1) * (sh_order_max + 1)``
        coefficients for a full SH basis.
    basis_type : {None, 'tournier07', 'descoteaux07'}, optional
        ``None`` for the default DIPY basis,
        ``tournier07`` for the Tournier 2007 [2]_[3]_ basis,
        ``descoteaux07`` for the Descoteaux 2007 [1]_ basis,
        (``None`` defaults to ``descoteaux07``).
    full_basis: bool, optional
        If True, uses a SH basis containing even and odd order SH functions.
        Else, uses a SH basis consisting only of even order SH functions.
    legacy: bool, optional
        True to use a legacy basis definition for backward compatibility
        with previous ``tournier07`` and ``descoteaux07`` implementations.
    return_inv : bool, optional
        If True then the inverse of the matrix is also returned.
    smooth : float, optional
        Lambda-regularization in the SH fit.

    Returns
    -------
    B : ndarray
        Matrix that transforms spherical harmonics to spherical function
        ``sf = np.dot(sh, B)``.
    invB : ndarray
        Inverse of B.

    References
    ----------
    .. [1] Descoteaux, M., Angelino, E., Fitzgibbons, S. and Deriche, R.
           Regularized, Fast, and Robust Analytical Q-ball Imaging.
           Magn. Reson. Med. 2007;58:497-510.
    .. [2] Tournier J.D., Calamante F. and Connelly A. Robust determination
           of the fibre orientation distribution in diffusion MRI:
           Non-negativity constrained super-resolved spherical deconvolution.
           NeuroImage. 2007;35(4):1459-1472.
    .. [3] Tournier J-D, Smith R, Raffelt D, Tabbara R, Dhollander T,
           Pietsch M, et al. MRtrix3: A fast, flexible and open software
           framework for medical image processing and visualisation.
           NeuroImage. 2019 Nov 15;202:116-137.

    """

    sph_harm_basis = sph_harm_lookup.get(basis_type)

    if sph_harm_basis is None:
        raise ValueError("Invalid basis name.")
    B, m_values, l_values = sph_harm_basis(sh_order_max, sphere.theta,
                                           sphere.phi,
                                           full_basis=full_basis,
                                           legacy=legacy)

    if return_inv:
        L = -l_values * (l_values + 1)
        invB = smooth_pinv(B, np.sqrt(smooth) * L)
        return B.T, invB.T

    return B.T


def calculate_max_order(n_coeffs, full_basis=False):
    r"""Calculate the maximal harmonic order (l), given that you know the
    number of parameters that were estimated.

    Parameters
    ----------
    n_coeffs : int
        The number of SH coefficients
    full_basis: bool, optional
        True if the used SH basis contains even and odd order SH functions.
        False if the SH basis consists only of even order SH functions.

    Returns
    -------
    L : int
        The maximal SH order (l), given the number of coefficients

    Notes
    -----
    The calculation in this function for the symmetric SH basis
    proceeds according to the following logic:
    .. math::
        n = \frac{1}{2} (L+1) (L+2)
        \rarrow 2n = L^2 + 3L + 2
        \rarrow L^2 + 3L + 2 - 2n = 0
        \rarrow L^2 + 3L + 2(1-n) = 0
        \rarrow L_{1,2} = \frac{-3 \pm \sqrt{9 - 8 (1-n)}}{2}
        \rarrow L{1,2} = \frac{-3 \pm \sqrt{1 + 8n}}{2}

    Finally, the positive value is chosen between the two options.

    For a full SH basis, the calculation consists in solving the equation
    $n = (L + 1)^2$ for $L$, which gives $L = sqrt(n) - 1$.
    """

    # L2 is negative for all positive values of n_coeffs, so we don't
    # bother even computing it:
    # L2 = (-3 - np.sqrt(1 + 8 * n_coeffs)) / 2
    # L1 is always the larger value, so we go with that:
    if full_basis:
        L1 = np.sqrt(n_coeffs) - 1
        if L1.is_integer():
            return int(L1)
    else:
        L1 = (-3 + np.sqrt(1 + 8 * n_coeffs)) / 2.0
        # Check that it is a whole even number:
        if L1.is_integer() and not np.mod(L1, 2):
            return int(L1)

    # Otherwise, the input didn't make sense:
    raise ValueError(f"The input to ``calculate_max_order`` was"
                     f" {n_coeffs}, but that is not a valid number"
                     f" of coefficients for a spherical harmonics"
                     f" basis set.")


def anisotropic_power(sh_coeffs, norm_factor=0.00001, power=2,
                      non_negative=True):
    r"""Calculate anisotropic power map with a given SH coefficient matrix.

    Parameters
    ----------
    sh_coeffs : ndarray
        A ndarray where the last dimension is the
        SH coefficients estimates for that voxel.
    norm_factor: float, optional
        The value to normalize the ap values.
    power : int, optional
        The degree to which power maps are calculated.
    non_negative: bool, optional
        Whether to rectify the resulting map to be non-negative.

    Returns
    -------
    log_ap : ndarray
        The log of the resulting power image.

    Notes
    -----
    Calculate AP image based on a IxJxKxC SH coefficient matrix based on the
    equation:
    .. math::
        AP = \sum_{l=2,4,6,...}{\frac{1}{2l+1} \sum_{m=-l}^l{|a_{l,m}|^n}}

    Where the last dimension, C, is made of a flattened array of $l$x$m$
    coefficients, where $l$ are the SH orders, and $m = 2l+1$,
    So l=1 has 1 coefficient, l=2 has 5, ... l=8 has 17 and so on.
    A l=2 SH coefficient matrix will then be composed of a IxJxKx6 volume.
    The power, $n$ is usually set to $n=2$.

    The final AP image is then shifted by -log(norm_factor), to be strictly
    non-negative. Remaining values < 0 are discarded (set to 0), per default,
    and this option is controlled through the `non_negative` keyword argument.

    References
    ----------
    .. [1]  Dell'Acqua, F., Lacerda, L., Catani, M., Simmons, A., 2014.
            Anisotropic Power Maps: A diffusion contrast to reveal low
            anisotropy tissues from HARDI data,
            in: Proceedings of International Society for Magnetic Resonance in
            Medicine. Milan, Italy.

    """
    dim = sh_coeffs.shape[:-1]
    n_coeffs = sh_coeffs.shape[-1]
    max_order = calculate_max_order(n_coeffs)
    ap = np.zeros(dim)
    n_start = 1
    for L in range(2, max_order + 2, 2):
        n_stop = n_start + (2 * L + 1)
        ap_i = np.mean(np.abs(sh_coeffs[..., n_start:n_stop]) ** power, -1)
        ap += ap_i
        n_start = n_stop

    # Shift the map to be mostly non-negative,
    # only applying the log operation to positive elements
    # to avoid getting numpy warnings on log(0).
    # It is impossible to get ap values smaller than 0.
    # Also avoids getting voxels with -inf when non_negative=False.

    if ap.ndim < 1:
        # For the off chance we have a scalar on our hands
        ap = np.reshape(ap, (1, ))
    log_ap = np.zeros_like(ap)
    log_ap[ap > 0] = np.log(ap[ap > 0]) - np.log(norm_factor)

    # Deal with residual negative values:
    if non_negative:
        if isinstance(log_ap, np.ndarray):
            # zero all values < 0
            log_ap[log_ap < 0] = 0
        else:
            # assume this is a singleton float (input was 1D):
            if log_ap < 0:
                return 0
    return log_ap


def convert_sh_to_full_basis(sh_coeffs):
    """Given an array of SH coeffs from a symmetric basis, returns the
    coefficients for the full SH basis by filling odd order SH coefficients
    with zeros

    Parameters
    ----------
    sh_coeffs: ndarray
        A ndarray where the last dimension is the
        SH coefficients estimates for that voxel.

    Returns
    -------
    full_sh_coeffs: ndarray
        A ndarray where the last dimension is the
        SH coefficients estimates for that voxel in
        a full SH basis.
    """
    sh_order_max = calculate_max_order(sh_coeffs.shape[-1])
    _, n = sph_harm_ind_list(sh_order_max, full_basis=True)

    full_sh_coeffs =\
        np.zeros(np.append(sh_coeffs.shape[:-1], [n.size]).astype(int))
    mask = np.mod(n, 2) == 0

    full_sh_coeffs[..., mask] = sh_coeffs
    return full_sh_coeffs


def convert_sh_from_legacy(sh_coeffs, sh_basis, full_basis=False):
    """Convert SH coefficients in legacy SH basis to SH coefficients
    of the new SH basis for ``descoteaux07`` [1]_ or ``tournier07`` [2]_[3]_
    bases.

    Parameters
    ----------
    sh_coeffs: ndarray
        A ndarray where the last dimension is the
        SH coefficients estimates for that voxel.
    sh_basis: {'descoteaux07', 'tournier07'}
        ``tournier07`` for the Tournier 2007 [2]_[3]_ basis,
        ``descoteaux07`` for the Descoteaux 2007 [1]_ basis.
    full_basis: bool, optional
        True if the input SH basis includes both even and odd
        order SH functions, else False.

    Returns
    -------
    out_sh_coeffs: ndarray
        The array of coefficients expressed in the new SH basis.

    References
    ----------
    .. [1] Descoteaux, M., Angelino, E., Fitzgibbons, S. and Deriche, R.
           Regularized, Fast, and Robust Analytical Q-ball Imaging.
           Magn. Reson. Med. 2007;58:497-510.
    .. [2] Tournier J.D., Calamante F. and Connelly A. Robust determination
           of the fibre orientation distribution in diffusion MRI:
           Non-negativity constrained super-resolved spherical deconvolution.
           NeuroImage. 2007;35(4):1459-1472.
    .. [3] Tournier J-D, Smith R, Raffelt D, Tabbara R, Dhollander T,
           Pietsch M, et al. MRtrix3: A fast, flexible and open software
           framework for medical image processing and visualisation.
           NeuroImage. 2019 Nov 15;202:116-137.
    """
    sh_order_max = calculate_max_order(sh_coeffs.shape[-1],
                                       full_basis=full_basis)

    m_values, l_values = sph_harm_ind_list(sh_order_max, full_basis=full_basis)

    if sh_basis == 'descoteaux07':
        out_sh_coeffs = sh_coeffs * np.where(m_values < 0, (-1.)**m_values, 1.)
    elif sh_basis == 'tournier07':
        out_sh_coeffs = sh_coeffs * np.where(m_values == 0, 1., 1./np.sqrt(2))
    else:
        raise ValueError("Invalid basis name.")

    return out_sh_coeffs


def convert_sh_to_legacy(sh_coeffs, sh_basis, full_basis=False):
    """Convert SH coefficients in new SH basis to SH coefficients for
    the legacy SH basis for ``descoteaux07`` [1]_ or ``tournier07`` [2]_[3]_
    bases.

    Parameters
    ----------
    sh_coeffs: ndarray
        A ndarray where the last dimension is the
        SH coefficients estimates for that voxel.
    sh_basis: {'descoteaux07', 'tournier07'}
        ``tournier07`` for the Tournier 2007 [2]_[3]_ basis,
        ``descoteaux07`` for the Descoteaux 2007 [1]_ basis.
    full_basis: bool, optional
        True if the input SH basis includes both even and odd
        order SH functions.

    Returns
    -------
    out_sh_coeffs: ndarray
        The array of coefficients expressed in the legacy SH basis.

    References
    ----------
    .. [1] Descoteaux, M., Angelino, E., Fitzgibbons, S. and Deriche, R.
           Regularized, Fast, and Robust Analytical Q-ball Imaging.
           Magn. Reson. Med. 2007;58:497-510.
    .. [2] Tournier J.D., Calamante F. and Connelly A. Robust determination
           of the fibre orientation distribution in diffusion MRI:
           Non-negativity constrained super-resolved spherical deconvolution.
           NeuroImage. 2007;35(4):1459-1472.
    .. [3] Tournier J-D, Smith R, Raffelt D, Tabbara R, Dhollander T,
           Pietsch M, et al. MRtrix3: A fast, flexible and open software
           framework for medical image processing and visualisation.
           NeuroImage. 2019 Nov 15;202:116-137.
    """
    sh_order_max = calculate_max_order(sh_coeffs.shape[-1],
                                       full_basis=full_basis)

    m_values, l_values = sph_harm_ind_list(sh_order_max, full_basis=full_basis)

    if sh_basis == 'descoteaux07':
        out_sh_coeffs = sh_coeffs * np.where(m_values < 0, (-1.)**m_values, 1.)
    elif sh_basis == 'tournier07':
        out_sh_coeffs = sh_coeffs * np.where(m_values == 0, 1., np.sqrt(2))
    else:
        raise ValueError("Invalid basis name.")

    return out_sh_coeffs


def convert_sh_descoteaux_tournier(sh_coeffs):
    """Convert SH coefficients between legacy-descoteaux07 and tournier07.

    Convert SH coefficients between the legacy ``descoteaux07`` SH basis and
    the non-legacy ``tournier07`` SH basis. Because this conversion is equal to
    its own inverse, it can be used to convert in either direction:
    legacy-descoteaux to non-legacy-tournier or non-legacy-tournier to
    legacy-descoteaux.

    This can be used to convert SH representations between DIPY and MRtrix3.

    See [descoteaux07]_ and [tournier19]_ for the origin of these SH bases.
    See [mrtrixbasis]_ for a description of the basis used in MRtrix3.
    See [mrtrixdipybases]_ for more details on the conversion.

    Parameters
    ----------
    sh_coeffs: ndarray
        A ndarray where the last dimension is the
        SH coefficients estimates for that voxel.

    Returns
    -------
    out_sh_coeffs: ndarray
        The array of coefficients expressed in the "other" SH basis. If the
        input was in the legacy-descoteaux basis then the output will be in the
        non-legacy-tournier basis, and vice versa.

    References
    ----------
    .. [descoteaux07] Descoteaux, M., Angelino, E., Fitzgibbons, S. and
           Deriche, R. Regularized, Fast, and Robust Analytical Q-ball Imaging.
           Magn. Reson. Med. 2007;58:497-510.
    .. [tournier19] Tournier J-D, Smith R, Raffelt D, Tabbara R, Dhollander T,
           Pietsch M, et al. MRtrix3: A fast, flexible and open software
           framework for medical image processing and visualisation.
           NeuroImage. 2019 Nov 15;202:116-137.
    .. [mrtrixbasis] https://mrtrix.readthedocs.io/en/latest/concepts/spherical_harmonics.html
    .. [mrtrixdipybases] https://github.com/dipy/dipy/discussions/2959#discussioncomment-7481675
    """  # noqa: E501

    sh_order_max = calculate_max_order(sh_coeffs.shape[-1])
    m_values, l_values = sph_harm_ind_list(sh_order_max)
    basis_indices = list(zip(l_values, m_values))  # dipy basis ordering
    basis_indices_permuted = list(zip(l_values, -m_values))  # mrtrix basis ordering
    permutation = [
        basis_indices.index(basis_indices_permuted[i])
        for i in range(len(basis_indices))
    ]
    return sh_coeffs[..., permutation]
