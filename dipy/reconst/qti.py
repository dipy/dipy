"""Classes and functions for fitting the covariance tensor model of q-space
trajectory imaging."""

import warnings
import numpy as np
from dipy.reconst.base import ReconstModel
from dipy.core.ndindex import ndindex


def from_3x3_to_6x1(T):
    """Convert symmetric 3 x 3 matrices into 6 x 1 vectors.

    Parameters
    ----------
    T : np.ndarray
        An array of size (3, 3, ...). 

    Returns
    -------    
    V : np.ndarray
        Converted vectors of size (6, 1, ...).
    """
    if T.shape[0:2] != (3, 3):
        raise ValueError('The shape of the input array must be (3, 3, ...).')
    if not np.all(np.isclose(T, np.swapaxes(T, 0, 1))):
        warnings.warn('All input tensors are not symmetric.', Warning)
    C = np.sqrt(2)
    V = np.array([[T[0, 0],
                   T[1, 1],
                   T[2, 2],
                   C * T[1, 2],
                   C * T[0, 2],
                   C * T[0, 1]]])
    V = np.swapaxes(V, 0, 1)
    return V


def from_6x1_to_3x3(V):
    """Convert 6 x 1 vectors into symmetric 3 x 3 matrices.

    Parameters
    ----------
    V : np.ndarray
        An array of size (6, 1, ...).

    Returns
    ------- 
    T : np.ndarray
        Converted matrices of size (3, 3, ...).
    """
    if V.shape[0:2] != (6, 1):
        raise ValueError('The shape of the input array must be (6, 1, ...).')
    C = 1 / np.sqrt(2)
    T = np.array([[V[0, 0], C * V[5, 0], C * V[4, 0]],
                  [C * V[5, 0], V[1, 0], C * V[3, 0]],
                  [C * V[4, 0], C * V[3, 0], V[2, 0]]])
    return T


def from_6x6_to_21x1(T):
    """Convert symmetric 6 x 6 matrices into 21 x 1 vectors.

    Parameters
    ----------
    T : np.ndarray
        An array of size (6, 6, ...). 

    Returns
    -------    
    V : np.ndarray
        Converted vectors of size (21, 1, ...).
    """
    if T.shape[0:2] != (6, 6):
        raise ValueError('The shape of the input array must be (6, 6, ...).')
    if not np.all(np.isclose(T, np.swapaxes(T, 0, 1))):
        warnings.warn('All input tensors are not symmetric.', Warning)
    C = np.sqrt(2)
    V = np.array([[T[0, 0], T[1, 1], T[2, 2],
                   C * T[1, 2], C * T[0, 2], C * T[0, 1],
                   C * T[0, 3], C * T[0, 4], C * T[0, 5],
                   C * T[1, 3], C * T[1, 4], C * T[1, 5],
                   C * T[2, 3], C * T[2, 4], C * T[2, 5],
                   T[3, 3], T[4, 4], T[5, 5],
                   C * T[3, 4], C * T[4, 5], C * T[5, 3]]])
    V = np.swapaxes(V, 0, 1)
    return V


def from_21x1_to_6x6(V):
    """Convert 21 x 1 vectors into symmetric 6 x 6 matrices.

    Parameters
    ----------
    V : np.ndarray
        An array of size (21, 1, ...).

    Returns
    ------- 
    T : np.ndarray
        Converted matrices of size (6, 6, ...).
    """
    if V.shape != (21, 1):
        raise ValueError('The shape of the input array must be (21, 1, ...).')
    V = V[:, 0]  # Code is easier to read without the extra dimension
    C = 1 / np.sqrt(2)
    T = np.array(
        [[V[0], C * V[5], C * V[4], C * V[6], C * V[7], C * V[8]],
         [C * V[5], V[1], C * V[3], C * V[9], C * V[10], C * V[11]],
         [C * V[4], C * V[3], V[2], C * V[12], C * V[13], C * V[14]],
         [C * V[6], C * V[9], C * V[12], V[15], C * V[18], C * V[20]],
         [C * V[7], C * V[10], C * V[13], C * V[18], V[16], C * V[19]],
         [C * V[8], C * V[11], C * V[14], C * V[20], C * V[19], V[17]]])
    return T


# These tensors enable the calculation of the QTI parameter maps
e_iso = np.eye(3) / 3
E_iso = np.eye(6) / 3
E_bulk = np.matmul(from_3x3_to_6x1(e_iso), from_3x3_to_6x1(e_iso).T)
E_shear = E_iso - E_bulk


class QtiModel(ReconstModel):

    def __init__(self, gtab, fit_method='WLS'):
        """Covariance tensor model of q-space trajectory imaging [1].

        Parameters
        ----------
        gtab : dipy.core.gradients.GradientTable instance
            Gradient table.
        fit_method : str, must be one of the following
            'OLS' for ordinary least squares

        References
        ----------
        .. [1] Westin, Carl-Fredrik, et al. "Q-space trajectory imaging for
           multidimensional diffusion MRI of the human brain." Neuroimage 135
            (2016): 345-362. doi.org/10.1016/j.neuroimage.2016.02.039.

        Notes
        -----
        QTI requires at least two b-tensor shapes and a minimum of 28
        acquisitions.
        """
        ReconstModel.__init__(self, gtab)

        if self.gtab == None:
            raise ValueError(
                'QTI requires b-tensors to be defined in the gradient table.')

        self.fit_method = fit_method
        if self.fit_method == 'OLS':  # Design matrix is independent of data
            self.X = np.zeros((self.gtab.btens.shape[0], 28))
            for i, bten in enumerate(self.gtab.btens):
                b = from_3x3_to_6x1(bten)
                b_sq = from_6x6_to_21x1(np.matmul(b, b.T))
                self.X[i] = np.concatenate(
                    ([1], (-b.T)[0, :], (0.5 * b_sq.T)[0, :]))
                self.X_inv = np.linalg.pinv(np.matmul(self.X.T, self.X))

    def fit(self, data, mask=None):
        """Fit method of the QTI model class.

        Parameters
        ----------
        data : array
            The measured signal from one voxel.

        mask : array
            A boolean array used to mark the coordinates in the data that
            should be analyzed that has the shape data.shape[:-1]

        Returns
        -------
        qtifit : dipy.reconst.qti.QtiFit instance
        """
        if mask is None:
            mask = np.ones(data.shape[:-1], dtype=bool)
        else:
            if mask.shape != data.shape[:-1]:
                raise ValueError("Mask is not the same shape as data.")
            mask = np.array(mask, dtype=bool, copy=False)
        data_in_mask = np.reshape(data[mask], (-1, data.shape[-1]))

        params = np.zeros(mask.shape + (28,))
        index = ndindex(mask.shape)
        for v in index:  # Loop over data
            if not mask[v]:
                continue
            S = np.log(data[v])[:, np.newaxis]
            params[v] = np.matmul(self.X_inv, np.matmul(self.X.T, S))[0]

        return QtiFit(params)


class QtiFit(object):

    def __init__(self, params):
        """Initialize QtiFit instance."""
        self.params = params
        return

    @property
    def S0_hat(self):
        """Predicted signal at b = 0."""
        return self.params[:, :, :, 0]

    @property
    def md(self):
        """Mean diffusivity."""
        return

    @property
    def ufa(self):
        """Microscopic fractional anisotropy."""
        return
