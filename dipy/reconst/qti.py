"""Classes and functions for fitting the covariance tensor model of q-space
trajectory imaging."""

from warnings import warn

import numpy as np

from dipy.reconst.base import ReconstModel
from dipy.core.ndindex import ndindex
from dipy.reconst.dti import auto_attr


def from_3x3_to_6x1(T):
    """Convert symmetric 3 x 3 matrices into 6 x 1 vectors.

    Parameters
    ----------
    T : numpy.ndarray
        An array of size (..., 3, 3).

    Returns
    -------
    V : numpy.ndarray
        Converted vectors of size (..., 6, 1).
    """
    if T.shape[-2::] != (3, 3):
        raise ValueError('The shape of the input array must be (..., 3, 3).')
    if not np.all(np.isclose(T, np.swapaxes(T, -1, -2))):
        warn('All input tensors are not symmetric.')
    C = np.sqrt(2)
    V = np.stack((T[..., 0, 0],
                  T[..., 1, 1],
                  T[..., 2, 2],
                  C * T[..., 1, 2],
                  C * T[..., 0, 2],
                  C * T[..., 0, 1]), axis=-1)[..., np.newaxis]
    return V


def from_6x1_to_3x3(V):
    """Convert 6 x 1 vectors into symmetric 3 x 3 matrices.

    Parameters
    ----------
    V : numpy.ndarray
        An array of size (..., 6, 1).

    Returns
    -------
    T : numpy.ndarray
        Converted matrices of size (..., 3, 3).
    """
    if V.shape[-2::] != (6, 1):
        raise ValueError('The shape of the input array must be (..., 6, 1).')
    C = 1 / np.sqrt(2)
    T = np.array(([V[..., 0, 0], C * V[..., 5, 0], C * V[..., 4, 0]],
                  [C * V[..., 5, 0], V[..., 1, 0], C * V[..., 3, 0]],
                  [C * V[..., 4, 0], C * V[..., 3, 0], V[..., 2, 0]]))
    T = np.moveaxis(T, (0, 1), (-2, -1))
    return T


def from_6x6_to_21x1(T):
    """Convert symmetric 6 x 6 matrices into 21 x 1 vectors.

    Parameters
    ----------
    T : numpy.ndarray
        An array of size (..., 6, 6).

    Returns
    -------
    V : numpy.ndarray
        Converted vectors of size (..., 21, 1).
    """
    if T.shape[-2::] != (6, 6):
        raise ValueError('The shape of the input array must be (..., 6, 6).')
    if not np.all(np.isclose(T, np.swapaxes(T, -1, -2))):
        warn('All input tensors are not symmetric.')
    C = np.sqrt(2)
    V = np.stack(([T[..., 0, 0], T[..., 1, 1], T[..., 2, 2],
                   C * T[..., 1, 2], C * T[..., 0, 2], C * T[..., 0, 1],
                   C * T[..., 0, 3], C * T[..., 0, 4], C * T[..., 0, 5],
                   C * T[..., 1, 3], C * T[..., 1, 4], C * T[..., 1, 5],
                   C * T[..., 2, 3], C * T[..., 2, 4], C * T[..., 2, 5],
                   T[..., 3, 3], T[..., 4, 4], T[..., 5, 5],
                   C * T[..., 3, 4], C * T[..., 4, 5], C * T[..., 5, 3]]),
                 axis=-1)[..., np.newaxis]
    return V


def from_21x1_to_6x6(V):
    """Convert 21 x 1 vectors into symmetric 3 x 3 matrices.

    Parameters
    ----------
    V : numpy.ndarray
        An array of size (..., 21, 1).

    Returns
    -------
    T : numpy.ndarray
        Converted matrices of size (..., 6, 6).
    """
    if V.shape[-2::] != (21, 1):
        raise ValueError('The shape of the input array must be (..., 21, 1).')
    C = 1 / np.sqrt(2)
    T = np.array(
        ([V[..., 0, 0], C * V[..., 5, 0], C * V[..., 4, 0],
          C * V[..., 6, 0], C * V[..., 7, 0], C * V[..., 8, 0]],
         [C * V[..., 5, 0], V[..., 1, 0], C * V[..., 3, 0],
          C * V[..., 9, 0], C * V[..., 10, 0], C * V[..., 11, 0]],
         [C * V[..., 4, 0], C * V[..., 3, 0], V[..., 2, 0],
          C * V[..., 12, 0], C * V[..., 13, 0], C * V[..., 14, 0]],
         [C * V[..., 6, 0], C * V[..., 9, 0], C * V[..., 12, 0],
          V[..., 15, 0], C * V[..., 18, 0], C * V[..., 20, 0]],
         [C * V[..., 7, 0], C * V[..., 10, 0], C * V[..., 13, 0],
          C * V[..., 18, 0], V[..., 16, 0], C * V[..., 19, 0]],
         [C * V[..., 8, 0], C * V[..., 11, 0], C * V[..., 14, 0],
          C * V[..., 20, 0], C * V[..., 19, 0], V[..., 17, 0]]))
    T = np.moveaxis(T, (0, 1), (-2, -1))
    return T


# These tensors are used in the calculation of QTI parameters
e_iso = np.eye(3) / 3
E_iso = np.eye(6) / 3
E_bulk = np.matmul(from_3x3_to_6x1(e_iso), from_3x3_to_6x1(e_iso).T)
E_shear = E_iso - E_bulk
E_tsym = E_bulk + .4 * E_shear


def design_matrix(btens):
    """Calculate the QTI design matrix from the b-tensors.

    Parameters
    ----------
    btens : numpy.ndarray
        An array of b-tensors of shape (number of acquisitions, 3, 3).
    
    Returns
    -------
    X : numpy.ndarray
        QTI design matrix.
    """
    X = np.zeros((btens.shape[0], 28))
    for i, bten in enumerate(btens):
        b = from_3x3_to_6x1(bten)
        b_sq = from_6x6_to_21x1(np.matmul(b, b.T))
        X[i] = np.concatenate(
            ([1], (-b.T)[0, :], (0.5 * b_sq.T)[0, :]))
    return X


def _ols_fit(data, mask, X):
    """Estimate the QTI model parameters using ordinary least squares.

    Parameters
    ----------
    data : numpy.ndarray
        Array of shape (..., number of acquisitions).
    mask : numpy.ndarray
        Array with the same shape as the data array of a single acquisition.
    X : numpy.ndarray
        Design matrix of shape (number of acquisitions, 28).

    Returns
    -------
    params : numpy.ndarray
        Array of shape (..., 28) containing the estimated model parameters.
        Element 0 is the estimated signal without diffusion-weighting, elements
        1-6 are the estimated diffusion tensor parameters, and elements 7-28 are
        the estimated covariance tensor parameters.
    """
    params = np.zeros(mask.shape + (28,)) * np.nan
    X_inv = np.linalg.pinv(np.matmul(X.T, X))  # Independent of data
    index = ndindex(mask.shape)
    for v in index:  # This loop is slow
        if not mask[v]:
            continue
        S = np.log(data[v])[:, np.newaxis]
        params[v] = np.matmul(X_inv, np.matmul(X.T, S))[:, 0]
    return params


def _wls_fit(data, mask, X):
    """Estimate the QTI model parameters using weighted least squares where the
    signal magnitudes are used as the weights.

    Parameters
    ----------
    data : numpy.ndarray
        Array of shape (..., number of acquisitions).
    mask : numpy.ndarray
        Array with the same shape as the data array of a single acquisition.
    X : numpy.ndarray
        Design matrix of shape (number of acquisitions, 28).

    Returns
    -------
    params : numpy.ndarray
        Array of shape (..., 28) containing the estimated model parameters.
        Element 0 is the estimated signal without diffusion-weighting, elements
        1-6 are the estimated diffusion tensor parameters, and elements 7-27 are
        the estimated covariance tensor parameters.
    """
    params = np.zeros(mask.shape + (28,)) * np.nan
    index = ndindex(mask.shape)
    for v in index:  # This loop is slow loop
        if not mask[v]:
            continue
        S = np.log(data[v])[:, np.newaxis]
        B = X.T * data[v][np.newaxis, :]
        A = np.matmul(B, X)
        params[v] = np.matmul(np.matmul(np.linalg.pinv(A), B), S)[:, 0]
    return params


class QtiModel(ReconstModel):

    def __init__(self, gtab, fit_method='OLS'):
        """Covariance tensor model of q-space trajectory imaging [1]_.

        Parameters
        ----------
        gtab : dipy.core.gradients.GradientTable
            Gradient table with b-tensors.
        fit_method : str, optional
            Must be one of the followng:
            'OLS' for ordinary least squares
            'WLS' for weighted least squares.

        References
        ----------
        .. [1] Westin, Carl-Fredrik, et al. "Q-space trajectory imaging for
           multidimensional diffusion MRI of the human brain." Neuroimage 135
           (2016): 345-362. doi.org/10.1016/j.neuroimage.2016.02.039.
        """
        ReconstModel.__init__(self, gtab)

        if self.gtab.btens is None:
            raise ValueError(
                'QTI requires b-tensors to be defined in the gradient table.'
            )
        self.X = design_matrix(self.gtab.btens)
        rank = np.linalg.matrix_rank(np.matmul(self.X.T, self.X))
        if rank < 28:
            warn(
                'The combination of the b-tensor shapes, sizes, and ' +
                'orientations does not enable all elements of the covariance ' +
                'tensor to be estimated (rank(X.T, X) = %s < 28).' % rank
            )

        if fit_method != 'OLS' and fit_method != 'WLS':
            raise ValueError(
                'Invalid value (%s) for \'fit_method\'.' % fit_method
                + ' Options: \'OLS\', \'WLS\'.'
            )
        self.fit_method = fit_method

    def fit(self, data, mask=None):
        """Fit method of the QTI model class.

        Parameters
        ----------
        data : numpy.ndarray
            Array of shape (..., number of acquisitions).
        mask : numpy.ndarray, optional
            Array with the same shape as the data array of a single acquisition.

        Returns
        -------
        qtifit : dipy.reconst.qti.QtiFit instance
            The fitted model.
        """
        if mask is None:
            mask = np.ones(data.shape[:-1], dtype=bool)
        else:
            if mask.shape != data.shape[:-1]:
                raise ValueError('Mask is not the same shape as data.')
            mask = np.array(mask, dtype=bool, copy=False)

        if self.fit_method == 'OLS':
            params = _ols_fit(data, mask, self.X)
        elif self.fit_method == 'WLS':
            params = _wls_fit(data, mask, self.X)

        return QtiFit(params)


class QtiFit(object):

    def __init__(self, params):
        """Class for the fitted QTI model.

        Parameters
        ----------
        params : numpy.ndarray
            Array of shape (..., 28) containing the estimated model parameters.
            Element 0 is the estimated signal without diffusion-weighting,
            elements 1-6 are the estimated diffusion tensor parameters, and
            elements 7-27 are the estimated covariance tensor parameters.
        """
        self.params = params
        return

    @auto_attr
    def S0_hat(self):
        """Predicted signal at b = 0.

        Returns
        -------
        S0 : numpy.ndarray
        """
        S0 = np.exp(self.params[:, :, :, 0])
        return S0

    @auto_attr
    def md(self):
        """Mean diffusivity.

        Returns
        -------
        md : numpy.ndarray
        """
        md = np.matmul(
            self.params[..., np.newaxis, 1:7],
            from_3x3_to_6x1(e_iso)
        )[..., 0, 0]
        return md

    @auto_attr
    def v_md(self):
        """Variance of microscopic mean diffusivities.

        Returns
        -------
        v_md : numpy.ndarray
        """
        v_md = np.matmul(
            self.params[..., np.newaxis, 7::],
            from_6x6_to_21x1(E_bulk)
        )[..., 0, 0]
        return v_md

    @auto_attr
    def v_shear(self):
        """Shear variance.

        Returns
        -------
        v_shear : numpy.ndarray
        """
        v_shear = np.matmul(
            self.params[..., np.newaxis, 7::],
            from_6x6_to_21x1(E_shear)
        )[..., 0, 0]
        return v_shear

    @auto_attr
    def v_iso(self):
        """Total isotropic variance.

        Returns
        -------
        v_iso : numpy.ndarray
        """
        v_iso = np.matmul(
            self.params[..., np.newaxis, 7::],
            from_6x6_to_21x1(E_iso)
        )[..., 0, 0]
        return v_iso

    @auto_attr
    def d_sq(self):
        """Diffusion tensor's outer product with itself.

        Returns
        -------
        d_sq : numpy.ndarray
        """
        d_sq = np.matmul(
            self.params[..., 1:7, np.newaxis],
            self.params[..., np.newaxis, 1:7]
        )
        return d_sq

    @auto_attr
    def mean_d_sq(self):
        """Average of microscopic diffusion tensors' outer products with
        themselves.

        Returns
        -------
        mean_d_sq : numpy.ndarray
        """
        mean_d_sq = from_21x1_to_6x6(
            self.params[..., 7::, np.newaxis]) + self.d_sq
        return mean_d_sq

    @auto_attr
    def c_md(self):
        """Normalized variance of mean diffusivities.

        Returns
        -------
        c_md : numpy.ndarray
        """
        c_md = self.v_md / np.matmul(
            np.swapaxes(from_6x6_to_21x1(self.mean_d_sq), -1, -2),
            from_6x6_to_21x1(E_bulk))[..., 0, 0]
        return c_md

    @auto_attr
    def c_mu(self):
        """Normalized variance of microscopic diffusion tensor eigenvalues.

        Returns
        -------
        c_mu : numpy.ndarray
        """
        c_mu = (1.5 * np.matmul(
            np.swapaxes(from_6x6_to_21x1(self.mean_d_sq), -1, -2),
            from_6x6_to_21x1(E_shear)) / np.matmul(
            np.swapaxes(from_6x6_to_21x1(self.mean_d_sq), -1, -2),
            from_6x6_to_21x1(E_iso)))[..., 0, 0]
        return c_mu

    @auto_attr
    def ufa(self):
        """Microscopic fractional anisotropy.

        Returns
        -------
        ufa : numpy.ndarray
        """
        ufa = np.sqrt(self.c_mu)
        return ufa

    @auto_attr
    def c_m(self):
        """Normalized variance of diffusion tensor eigenvalues.

        Returns
        -------
        c_m : numpy.ndarray
        """
        c_m = (1.5 * np.matmul(
            np.swapaxes(from_6x6_to_21x1(self.d_sq), -1, -2),
            from_6x6_to_21x1(E_shear)) / np.matmul(
            np.swapaxes(from_6x6_to_21x1(self.d_sq), -1, -2),
            from_6x6_to_21x1(E_iso)))[..., 0, 0]
        return c_m

    @auto_attr
    def fa(self):
        """Fractional anisotropy.

        Returns
        -------
        fa : numpy.ndarray
        """
        fa = np.sqrt(self.c_m)
        return fa

    @auto_attr
    def c_c(self):
        """Microscopic orientation coherence.

        Returns
        -------
        c_c : numpy.ndarray
        """
        c_c = self.c_m / self.c_mu
        return c_c

    @auto_attr
    def fa(self):
        """Fractional anisotropy.

        Returns
        -------
        fa : numpy.ndarray
        """
        fa = np.sqrt(self.c_m)
        return fa

    @auto_attr
    def mk(self):
        """Total kurtosis.

        Returns
        -------
        mk : numpy.ndarray
        """
        mk = (3 * np.matmul(
            self.params[..., np.newaxis, 7::],
            from_6x6_to_21x1(E_tsym)) / np.matmul(
            np.swapaxes(from_6x6_to_21x1(self.d_sq), -1, -2),
            from_6x6_to_21x1(E_bulk)))[..., 0, 0]
        return mk

    @auto_attr
    def k_bulk(self):
        """Bulk kurtosis.

        Returns
        -------
        k_bulk : numpy.ndarray
        """
        k_bulk = (3 * np.matmul(
            self.params[..., np.newaxis, 7::],
            from_6x6_to_21x1(E_bulk)) / np.matmul(
            np.swapaxes(from_6x6_to_21x1(self.d_sq), -1, -2),
            from_6x6_to_21x1(E_bulk)))[..., 0, 0]
        return k_bulk

    @auto_attr
    def k_shear(self):
        """Shear kurtosis.

        Returns
        -------
        k_shear : numpy.ndarray
        """
        k_shear = (6 / 5 * np.matmul(
            self.params[..., np.newaxis, 7::],
            from_6x6_to_21x1(E_shear)) / np.matmul(
            np.swapaxes(from_6x6_to_21x1(self.d_sq), -1, -2),
            from_6x6_to_21x1(E_bulk)))[..., 0, 0]
        return k_shear

    @auto_attr
    def k_mu(self):
        """Microscopic kurtosis.

        Returns
        -------
        k_mu : numpy.ndarray
        """
        k_mu = (6 / 5 * np.matmul(
            np.swapaxes(from_6x6_to_21x1(self.mean_d_sq), -1, -2),
            from_6x6_to_21x1(E_shear)) / np.matmul(
            np.swapaxes(from_6x6_to_21x1(self.d_sq), -1, -2),
            from_6x6_to_21x1(E_bulk)))[..., 0, 0]
        return k_mu
