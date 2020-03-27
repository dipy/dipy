""" Tools for using spherical harmonic models to fit diffusion data

References
----------
Aganj, I., et al. 2009. ODF Reconstruction in Q-Ball Imaging With Solid
    Angle Consideration.
Descoteaux, M., et al. 2007. Regularized, fast, and robust analytical
    Q-ball imaging.
Tristan-Vega, A., et al. 2010. A new methodology for estimation of fiber
    populations in white matter of the brain with Funk-Radon transform.
Tristan-Vega, A., et al. 2009. Estimation of fiber orientation probability
    density functions in high angular resolution diffusion imaging.


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

import numpy as np
from numpy import concatenate, diag, diff, empty, eye, sqrt, unique, dot
from numpy.linalg import pinv, svd
from numpy.random import randint

from dipy.reconst.odf import OdfModel, OdfFit
from dipy.core.geometry import cart2sphere
from dipy.core.onetime import auto_attr
from dipy.reconst.cache import Cache

from distutils.version import LooseVersion
import scipy
from scipy.special import lpn, lpmv, gammaln

if LooseVersion(scipy.version.short_version) >= LooseVersion('0.15.0'):
    SCIPY_15_PLUS = True
    import scipy.special as sps
else:
    SCIPY_15_PLUS = False


def _copydoc(obj):
    def bandit(f):
        f.__doc__ = obj.__doc__
        return f
    return bandit


def forward_sdeconv_mat(r_rh, n):
    """ Build forward spherical deconvolution matrix

    Parameters
    ----------
    r_rh : ndarray
        Rotational harmonics coefficients for the single fiber response
        function. Each element `rh[i]` is associated with spherical harmonics
        of degree `2*i`.
    n : ndarray
        The degree of spherical harmonic function associated with each row of
        the deconvolution matrix. Only even degrees are allowed

    Returns
    -------
    R : ndarray (N, N)
        Deconvolution matrix with shape (N, N)

    """

    if np.any(n % 2):
        raise ValueError("n has odd degrees, expecting only even degrees")
    return np.diag(r_rh[n // 2])


def sh_to_rh(r_sh, m, n):
    """ Spherical harmonics (SH) to rotational harmonics (RH)

    Calculate the rotational harmonic decomposition up to
    harmonic order `m`, degree `n` for an axially and antipodally
    symmetric function. Note that all ``m != 0`` coefficients
    will be ignored as axial symmetry is assumed. Hence, there
    will be ``(sh_order/2 + 1)`` non-zero coefficients.

    Parameters
    ----------
    r_sh : ndarray (N,)
        ndarray of SH coefficients for the single fiber response function.
        These coefficients must correspond to the real spherical harmonic
        functions produced by `shm.real_sph_harm`.
    m : ndarray (N,)
        The order of the spherical harmonic function associated with each
        coefficient.
    n : ndarray (N,)
        The degree of the spherical harmonic function associated with each
        coefficient.

    Returns
    -------
    r_rh : ndarray (``(sh_order + 1)*(sh_order + 2)/2``,)
        Rotational harmonics coefficients representing the input `r_sh`

    See Also
    --------
    shm.real_sph_harm, shm.real_sym_sh_basis

    References
    ----------
    .. [1] Tournier, J.D., et al. NeuroImage 2007. Robust determination of the
        fibre orientation distribution in diffusion MRI: Non-negativity
        constrained super-resolved spherical deconvolution

    """
    mask = m == 0
    # The delta function at theta = phi = 0 is known to have zero coefficients
    # where m != 0, therefore we need only compute the coefficients at m=0.
    dirac_sh = gen_dirac(0, n[mask], 0, 0)
    r_rh = r_sh[mask] / dirac_sh
    return r_rh


def gen_dirac(m, n, theta, phi):
    """ Generate Dirac delta function orientated in (theta, phi) on the sphere

    The spherical harmonics (SH) representation of this Dirac is returned as
    coefficients to spherical harmonic functions produced by
    `shm.real_sph_harm`.

    Parameters
    ----------
    m : ndarray (N,)
        The order of the spherical harmonic function associated with each
        coefficient.
    n : ndarray (N,)
        The degree of the spherical harmonic function associated with each
        coefficient.
    theta : float [0, 2*pi]
        The azimuthal (longitudinal) coordinate.
    phi : float [0, pi]
        The polar (colatitudinal) coordinate.

    See Also
    --------
    shm.real_sph_harm, shm.real_sym_sh_basis

    Returns
    -------
    dirac : ndarray
        SH coefficients representing the Dirac function. The shape of this is
        `(m + 2) * (m + 1) / 2`.

    """
    return real_sph_harm(m, n, theta, phi)


def spherical_harmonics(m, n, theta, phi):
    x = np.cos(phi)
    val = lpmv(m, n, x).astype(complex)
    val *= np.sqrt((2 * n + 1) / 4.0 / np.pi)
    val *= np.exp(0.5 * (gammaln(n - m + 1) - gammaln(n + m + 1)))
    val = val * np.exp(1j * m * theta)
    return val

if SCIPY_15_PLUS:
    def spherical_harmonics(m, n, theta, phi):
        return sps.sph_harm(m, n, theta, phi, dtype=complex)

spherical_harmonics.__doc__ = r""" Compute spherical harmonics

    This may take scalar or array arguments. The inputs will be broadcasted
    against each other.

    Parameters
    ----------
    m : int ``|m| <= n``
        The order of the harmonic.
    n : int ``>= 0``
        The degree of the harmonic.
    theta : float [0, 2*pi]
        The azimuthal (longitudinal) coordinate.
    phi : float [0, pi]
        The polar (colatitudinal) coordinate.

    Returns
    -------
    y_mn : complex float
        The harmonic $Y^m_n$ sampled at `theta` and `phi`.

    Notes
    -----
    This is a faster implementation of scipy.special.sph_harm for
    scipy version < 0.15.0. For scipy 0.15 and onwards, we use the scipy
    implementation of the function
    """


def real_sph_harm(m, n, theta, phi):
    r""" Compute real spherical harmonics.

    Where the real harmonic $Y^m_n$ is defined to be:

        Imag($Y^m_n$) * sqrt(2)     if m > 0
        $Y^0_n$                     if m = 0
        Real($Y^|m|_n$) * sqrt(2)   if m < 0

    This may take scalar or array arguments. The inputs will be broadcasted
    against each other.

    Parameters
    ----------
    m : int ``|m| <= n``
        The order of the harmonic.
    n : int ``>= 0``
        The degree of the harmonic.
    theta : float [0, 2*pi]
        The azimuthal (longitudinal) coordinate.
    phi : float [0, pi]
        The polar (colatitudinal) coordinate.

    Returns
    --------
    y_mn : real float
        The real harmonic $Y^m_n$ sampled at `theta` and `phi`.

    See Also
    --------
    scipy.special.sph_harm
    """
    # dipy uses a convention for theta and phi that is reversed with respect to
    # function signature of scipy.special.sph_harm
    sh = spherical_harmonics(np.abs(m), n, phi, theta)

    real_sh = np.where(m > 0, sh.imag, sh.real)
    real_sh *= np.where(m == 0, 1., np.sqrt(2))
    return real_sh


def real_sym_sh_mrtrix(sh_order, theta, phi):
    """
    Compute real spherical harmonics as in Tournier 2007 [2]_, where the real
    harmonic $Y^m_n$ is defined to be::

        Real($Y^m_n$)       if m > 0
        $Y^0_n$             if m = 0
        Imag($Y^|m|_n$)     if m < 0

    This may take scalar or array arguments. The inputs will be broadcasted
    against each other.

    Parameters
    -----------
    sh_order : int
        The maximum degree or the spherical harmonic basis.
    theta : float [0, pi]
        The polar (colatitudinal) coordinate.
    phi : float [0, 2*pi]
        The azimuthal (longitudinal) coordinate.

    Returns
    --------
    y_mn : real float
        The real harmonic $Y^m_n$ sampled at `theta` and `phi` as
        implemented in mrtrix. Warning: the basis is Tournier et al.
        2007 [2]_; 2004 [1]_ is slightly different.
    m : array
        The order of the harmonics.
    n : array
        The degree of the harmonics.

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
    m, n = sph_harm_ind_list(sh_order)
    phi = np.reshape(phi, [-1, 1])
    theta = np.reshape(theta, [-1, 1])

    m = -m
    real_sh = real_sph_harm(m, n, theta, phi)
    real_sh /= np.where(m == 0, 1., np.sqrt(2))
    return real_sh, m, n


def real_sym_sh_basis(sh_order, theta, phi):
    """Samples a real symmetric spherical harmonic basis at point on the sphere

    Samples the basis functions up to order `sh_order` at points on the sphere
    given by `theta` and `phi`. The basis functions are defined here the same
    way as in Descoteaux et al. 2007 [1]_ where the real harmonic $Y^m_n$ is
    defined to be:

        Imag($Y^m_n$) * sqrt(2)     if m > 0
        $Y^0_n$                     if m = 0
        Real($Y^|m|_n$) * sqrt(2)   if m < 0

    This may take scalar or array arguments. The inputs will be broadcasted
    against each other.

    Parameters
    -----------
    sh_order : int
        even int > 0, max spherical harmonic degree
    theta : float [0, 2*pi]
        The azimuthal (longitudinal) coordinate.
    phi : float [0, pi]
        The polar (colatitudinal) coordinate.

    Returns
    --------
    y_mn : real float
        The real harmonic $Y^m_n$ sampled at `theta` and `phi`
    m : array
        The order of the harmonics.
    n : array
        The degree of the harmonics.

    References
    ----------
    .. [1] Descoteaux, M., Angelino, E., Fitzgibbons, S. and Deriche, R.
           Regularized, Fast, and Robust Analytical Q-ball Imaging.
           Magn. Reson. Med. 2007;58:497-510.

    """
    m, n = sph_harm_ind_list(sh_order)
    phi = np.reshape(phi, [-1, 1])
    theta = np.reshape(theta, [-1, 1])

    real_sh = real_sph_harm(m, n, theta, phi)
    return real_sh, m, n


sph_harm_lookup = {None: real_sym_sh_basis,
                   "tournier07": real_sym_sh_mrtrix,
                   "descoteaux07": real_sym_sh_basis}


def sph_harm_ind_list(sh_order):
    """
    Returns the degree (n) and order (m) of all the symmetric spherical
    harmonics of degree less then or equal to `sh_order`. The results, `m_list`
    and `n_list` are kx1 arrays, where k depends on sh_order. They can be
    passed to :func:`real_sph_harm`.

    Parameters
    ----------
    sh_order : int
        even int > 0, max degree to return

    Returns
    -------
    m_list : array
        orders of even spherical harmonics
    n_list : array
        degrees of even spherical harmonics

    See also
    --------
    real_sph_harm
    """
    if sh_order % 2 != 0:
        raise ValueError('sh_order must be an even integer >= 0')

    n_range = np.arange(0, sh_order + 1, 2, dtype=int)
    n_list = np.repeat(n_range, n_range * 2 + 1)

    ncoef = int((sh_order + 2) * (sh_order + 1) // 2)
    offset = 0
    m_list = empty(ncoef, 'int')
    for ii in n_range:
        m_list[offset:offset + 2 * ii + 1] = np.arange(-ii, ii + 1)
        offset = offset + 2 * ii + 1

    # makes the arrays ncoef by 1, allows for easy broadcasting later in code
    return (m_list, n_list)


def order_from_ncoef(ncoef):
    """
    Given a number n of coefficients, calculate back the sh_order
    """
    # Solve the quadratic equation derived from :
    # ncoef = (sh_order + 2) * (sh_order + 1) / 2
    return int(-3 + np.sqrt(9 - 4 * (2-2*ncoef)))/2


def smooth_pinv(B, L):
    """Regularized pseudo-inverse

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
    However here this inverse is implemented using the pseudo-inverse because
    it is more numerically stable than the direct implementation of the matrix
    product.

    """
    L = diag(L)
    inv = pinv(concatenate((B, L)))
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
    step = unique(diff(index))
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
    sh0_index : int
        The index of the coefficient associated with the 0th order sh harmonic.

    Returns
    -------
    gfa_values : array
        The gfa of each odf.

    """
    coef_sq = coef**2
    numer = coef_sq[..., sh0_index]
    denom = (coef_sq).sum(-1)
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
            sh_order = self.sh_order
            theta = sphere.theta
            phi = sphere.phi
            sampling_matrix, m, n = real_sym_sh_basis(sh_order, theta, phi)
            self.cache_set("sampling_matrix", sphere, sampling_matrix)
        return sampling_matrix


class QballBaseModel(SphHarmModel):
    """To be subclassed by Qball type models."""
    def __init__(self, gtab, sh_order, smooth=0.006, min_signal=1e-5,
                 assume_normed=False):
        """Creates a model that can be used to fit or sample diffusion data

        Parameters
        ------------
        gtab : GradientTable
            Diffusion gradients used to acquire data
        sh_order : even int >= 0
            the spherical harmonic order of the model
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
        B, m, n = real_sym_sh_basis(sh_order, theta[:, None], phi[:, None])
        L = -n * (n + 1)
        legendre0 = lpn(sh_order, 0)[0]
        F = legendre0[n]
        self.sh_order = sh_order
        self.B = B
        self.m = m
        self.n = n
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
        return dot(self.shm_coeff, B.T)

    @auto_attr
    def gfa(self):
        return _gfa_sh(self.shm_coeff, 0)

    @property
    def shm_coeff(self):
        """The spherical harmonic coefficients of the odf

        Make this a property for now, if there is a usecase for modifying
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
           Default: 1.0 in all voxels
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
        invB = smooth_pinv(B, sqrt(smooth) * L)
        L = L[:, None]
        F = F[:, None]
        self._fit_matrix = (F * L) / (8 * np.pi) * invB

    def _get_shm_coef(self, data, mask=None):
        """Returns the coefficients of the model"""
        data = data[..., self._where_dwi]
        data = data.clip(self.min, self.max)
        loglog_data = np.log(-np.log(data))
        sh_coef = dot(loglog_data, self._fit_matrix.T)
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
        invB = smooth_pinv(B, sqrt(smooth) * L)
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
    return dot(logd * (1.5 - logd) * data, delta_q.T) - dot(data, delta_b.T)


class QballModel(QballBaseModel):
    """Implementation of regularized Qball reconstruction method.

    References
    ----------
    .. [1] Descoteaux, M., et al. 2007. Regularized, fast, and robust
           analytical Q-ball imaging.
    """

    def _set_fit_matrix(self, B, L, F, smooth):
        invB = smooth_pinv(B, sqrt(smooth) * L)
        F = F[:, None]
        self._fit_matrix = F * invB

    def _get_shm_coef(self, data, mask=None):
        """Returns the coefficients of the model"""
        return dot(data[..., self._where_dwi], self._fit_matrix.T)


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

    U, S, V = svd(B, False)
    H = dot(U, U.T)
    return H


def lcr_matrix(H):
    """Returns a matrix for computing leveraged, centered residuals from data

    if r = (d-Hd), the leveraged centered residuals are lcr = (r/l)-mean(r/l)
    ruturns the matrix R, such lcr = Rd

    """
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError('H should be a square matrix')

    leverages = sqrt(1 - H.diagonal())
    leverages = leverages[:, None]
    R = (eye(len(H)) - H) / leverages
    return R - R.mean(0)


def bootstrap_data_array(data, H, R, permute=None):
    """Applies the Residual Bootstraps to the data given H and R

    data must be normalized, ie 0 < data <= 1

    This function, and the bootstrap_data_voxel function, calculate
    residual-bootsrap samples given a Hat matrix and a Residual matrix. These
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
    data = dot(data, (H + R).T)
    return data


def bootstrap_data_voxel(data, H, R, permute=None):
    """Like bootstrap_data_array but faster when for a single voxel

    data must be 1d and normalized
    """
    if permute is None:
        permute = randint(data.shape[-1], size=data.shape[-1])
    r = dot(data, R.T)
    boot_data = dot(data, H.T)
    boot_data += r[permute]
    return boot_data


class ResidualBootstrapWrapper(object):
    """Returns a residual bootstrap sample of the signal_object when indexed

    Wraps a signal_object, this signal object can be an interpolator. When
    indexed, the the wrapper indexes the signal_object to get the signal.
    There wrapper than samples the residual boostrap distribution of signal and
    returns that sample.
    """
    def __init__(self, signal_object, B, where_dwi, min_signal=1e-5):
        """Builds a ResidualBootstrapWapper

        Given some linear model described by B, the design matrix, and a
        signal_object, returns an object which can sample the residual
        bootstrap distribution of the signal. We assume that the signals are
        normalized so we clip the bootsrap samples to be between `min_signal`
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

def just_sh_basis(sphere, sh_order=4, basis_type=None):
    """Spherical function to spherical harmonics (SH).

    Parameters
    ----------
    sphere : Sphere
        The points on which the sf is defined.
    sh_order : int, optional
        Maximum SH order in the SH fit.  For `sh_order`, there will be
        ``(sh_order + 1) * (sh_order_2) / 2`` SH coefficients (default 4).
    basis_type : {None, 'tournier07', 'descoteaux07'}
        ``None`` for the default DIPY basis,
        ``tournier07`` for the Tournier 2007 [2]_ basis, and
        ``descoteaux07`` for the Descoteaux 2007 [1]_ basis
        (``None`` defaults to ``descoteaux07``).

    Returns
    -------
    sh_basis : ndarray
        SH Basis based on the sphere sampling.

    References
    ----------
    .. [1] Descoteaux, M., Angelino, E., Fitzgibbons, S. and Deriche, R.
           Regularized, Fast, and Robust Analytical Q-ball Imaging.
           Magn. Reson. Med. 2007;58:497-510.
    .. [2] Tournier J.D., Calamante F. and Connelly A. Robust determination
           of the fibre orientation distribution in diffusion MRI:
           Non-negativity constrained super-resolved spherical deconvolution.
           NeuroImage. 2007;35(4):1459-1472.

    """

    sph_harm_basis = sph_harm_lookup.get(basis_type)

    if sph_harm_basis is None:
        raise ValueError("Invalid basis name.")
    B, m, n = sph_harm_basis(sh_order, sphere.theta, sphere.phi)

    return B

def sf_to_sh(sf, sphere, sh_order=4, basis_type=None, smooth=0.0):
    """Spherical function to spherical harmonics (SH).

    Parameters
    ----------
    sf : ndarray
        Values of a function on the given `sphere`.
    sphere : Sphere
        The points on which the sf is defined.
    sh_order : int, optional
        Maximum SH order in the SH fit.  For `sh_order`, there will be
        ``(sh_order + 1) * (sh_order_2) / 2`` SH coefficients (default 4).
    basis_type : {None, 'tournier07', 'descoteaux07'}
        ``None`` for the default DIPY basis,
        ``tournier07`` for the Tournier 2007 [2]_ basis, and
        ``descoteaux07`` for the Descoteaux 2007 [1]_ basis
        (``None`` defaults to ``descoteaux07``).
    smooth : float, optional
        Lambda-regularization in the SH fit (default 0.0).

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

    """

    sph_harm_basis = sph_harm_lookup.get(basis_type)

    if sph_harm_basis is None:
        raise ValueError("Invalid basis name.")
    B, m, n = sph_harm_basis(sh_order, sphere.theta, sphere.phi)

    L = -n * (n + 1)
    invB = smooth_pinv(B, sqrt(smooth) * L)
    sh = np.dot(sf, invB.T)

    return sh


def sh_to_sf(sh, sphere, sh_order, basis_type=None):
    """Spherical harmonics (SH) to spherical function (SF).

    Parameters
    ----------
    sh : ndarray
        SH coefficients representing a spherical function.
    sphere : Sphere
        The points on which to sample the spherical function.
    sh_order : int, optional
        Maximum SH order in the SH fit.  For `sh_order`, there will be
        ``(sh_order + 1) * (sh_order_2) / 2`` SH coefficients (default 4).
    basis_type : {None, 'tournier07', 'descoteaux07'}
        ``None`` for the default DIPY basis,
        ``tournier07`` for the Tournier 2007 [2]_ basis, and
        ``descoteaux07`` for the Descoteaux 2007 [1]_ basis
        (``None`` defaults to ``descoteaux07``).

    Returns
    -------
    sf : ndarray
         Spherical function values on the `sphere`.

    References
    ----------
    .. [1] Descoteaux, M., Angelino, E., Fitzgibbons, S. and Deriche, R.
           Regularized, Fast, and Robust Analytical Q-ball Imaging.
           Magn. Reson. Med. 2007;58:497-510.
    .. [2] Tournier J.D., Calamante F. and Connelly A. Robust determination
           of the fibre orientation distribution in diffusion MRI:
           Non-negativity constrained super-resolved spherical deconvolution.
           NeuroImage. 2007;35(4):1459-1472.

    """

    sph_harm_basis = sph_harm_lookup.get(basis_type)

    if sph_harm_basis is None:
        raise ValueError("Invalid basis name.")
    B, m, n = sph_harm_basis(sh_order, sphere.theta, sphere.phi)

    sf = np.dot(sh, B.T)

    return sf


def sh_to_sf_matrix(sphere, sh_order, basis_type=None, return_inv=True,
                    smooth=0):
    """ Matrix that transforms Spherical harmonics (SH) to spherical
    function (SF).

    Parameters
    ----------
    sphere : Sphere
        The points on which to sample the spherical function.
    sh_order : int, optional
        Maximum SH order in the SH fit.  For `sh_order`, there will be
        ``(sh_order + 1) * (sh_order_2) / 2`` SH coefficients (default 4).
    basis_type : {None, 'tournier07', 'descoteaux07'}
        ``None`` for the default DIPY basis,
        ``tournier07`` for the Tournier 2007 [2]_ basis, and
        ``descoteaux07`` for the Descoteaux 2007 [1]_ basis
        (``None`` defaults to ``descoteaux07``).
    return_inv : bool
        If True then the inverse of the matrix is also returned
    smooth : float, optional
        Lambda-regularization in the SH fit (default 0.0).

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

    """

    sph_harm_basis = sph_harm_lookup.get(basis_type)

    if sph_harm_basis is None:
        raise ValueError("Invalid basis name.")
    B, m, n = sph_harm_basis(sh_order, sphere.theta, sphere.phi)

    if return_inv:
        L = -n * (n + 1)
        invB = smooth_pinv(B, np.sqrt(smooth) * L)
        return B.T, invB.T

    return B.T


def calculate_max_order(n_coeffs):
        """Calculate the maximal harmonic order, given that you know the
        number of parameters that were estimated.

        Parameters
        ----------
        n_coeffs : int
            The number of SH coefficients

        Returns
        -------
        L : int
            The maximal SH order, given the number of coefficients

        Notes
        -----
        The calculation in this function proceeds according to the following
        logic:
        .. math::
           n = \frac{1}{2} (L+1) (L+2)
           \rarrow 2n = L^2 + 3L + 2
           \rarrow L^2 + 3L + 2 - 2n = 0
           \rarrow L^2 + 3L + 2(1-n) = 0
           \rarrow L_{1,2} = \frac{-3 \pm \sqrt{9 - 8 (1-n)}}{2}
           \rarrow L{1,2} = \frac{-3 \pm \sqrt{1 + 8n}}{2}

        Finally, the positive value is chosen between the two options.
        """

        # L2 is negative for all positive values of n_coeffs, so we don't
        # bother even computing it:
        # L2 = (-3 - np.sqrt(1 + 8 * n_coeffs)) / 2
        # L1 is always the larger value, so we go with that:
        L1 = (-3 + np.sqrt(1 + 8 * n_coeffs)) / 2.0
        # Check that it is a whole even number:
        if L1.is_integer() and not np.mod(L1, 2):
            return int(L1)
        else:
            # Otherwise, the input didn't make sense:
            raise ValueError("The input to ``calculate_max_order`` was ",
                             "%s, but that is not a valid number" % n_coeffs,
                             "of coefficients for a spherical harmonics ",
                             "basis set.")


def anisotropic_power(sh_coeffs, norm_factor=0.00001, power=2,
                      non_negative=True):
    """Calculates anisotropic power map with a given SH coefficient matrix

    Parameters
    ----------
    sh_coeffs : ndarray
        A ndarray where the last dimension is the
        SH coefficients estimates for that voxel.
    norm_factor: float, optional
        The value to normalize the ap values. Default is 10^-5.
    power : int, optional
        The degree to which power maps are calculated. Default: 2.
    non_negative: bool, optional
        Whether to rectify the resulting map to be non-negative.
        Default: True.

    Returns
    -------
    log_ap : ndarray
        The log of the resulting power image.

    Notes
    ----------
    Calculate AP image based on a IxJxKxC SH coefficient matrix based on the
    equation:
    .. math::
        AP = \sum_{l=2,4,6,...}{\frac{1}{2l+1} \sum_{m=-l}^l{|a_{l,m}|^n}}

    Where the last dimension, C, is made of a flattened array of $l$x$m$
    coefficients, where $l$ are the SH orders, and $m = 2l+1$,
    So l=1 has 1 coeffecient, l=2 has 5, ... l=8 has 17 and so on.
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
