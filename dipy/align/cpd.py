"""

Note
----

This file is copied (possibly with major modifications) from the
sources of the pycpd project - https://github.com/siavashk/pycpd.
It remains licensed as the rest of PyCPD (MIT license as of October 2010).

# ## ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyCPD package for the
#   copyright and license terms.
#
# ## ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""

import numpy as np
import numbers
from warnings import warn


def gaussian_kernel(X, beta, Y=None):
    if Y is None:
        Y = X
    diff = X[:, None, :] - Y[None, :, :]
    diff = np.square(diff)
    diff = np.sum(diff, 2)
    return np.exp(-diff / (2 * beta**2))


def low_rank_eigen(G, num_eig):
    """Calculate num_eig eigenvectors and eigenvalues of gaussian matrix G.

    Enables lower dimensional solving.

    """
    S, Q = np.linalg.eigh(G)
    eig_indices = list(np.argsort(np.abs(S))[::-1][:num_eig])
    Q = Q[:, eig_indices]  # eigenvectors
    S = S[eig_indices]  # eigenvalues.
    return Q, S


def initialize_sigma2(X, Y):
    """Initialize the variance (sigma2).

    Parameters
    ----------
    X: numpy array
        NxD array of points for target.

    Y: numpy array
        MxD array of points for source.

    Returns
    -------
    sigma2: float
        Initial variance.
    """
    (N, D) = X.shape
    (M, _) = Y.shape
    diff = X[None, :, :] - Y[:, None, :]
    err = diff ** 2
    return np.sum(err) / (D * M * N)


def lowrankQS(G, beta, num_eig, eig_fgt=False):
    """Calculate eigenvectors and eigenvalues of gaussian matrix G.

    !!!
    This function is a placeholder for implementing the fast
    gauss transform. It is not yet implemented.
    !!!

    Parameters
    ----------
    G: numpy array
        Gaussian kernel matrix.

    beta: float
        Width of the Gaussian kernel.

    num_eig: int
        Number of eigenvectors to use in lowrank calculation of G

    eig_fgt: bool
        If True, use fast gauss transform method to speed up.

    """
    # if we do not use FGT we construct affinity matrix G and find the
    # first eigenvectors/values directly

    if eig_fgt is False:
        S, Q = np.linalg.eigh(G)
        eig_indices = list(np.argsort(np.abs(S))[::-1][:num_eig])
        Q = Q[:, eig_indices]  # eigenvectors
        S = S[eig_indices]  # eigenvalues.

        return Q, S

    elif eig_fgt is True:
        raise Exception('Fast Gauss Transform Not Implemented!')


class DeformableRegistration:
    """
    Deformable point cloud registration.

    Attributes
    ----------
    X: numpy array
        NxD array of target points.

    Y: numpy array
        MxD array of source points.

    TY: numpy array
        MxD array of transformed source points.

    sigma2: float (positive)
        Initial variance of the Gaussian mixture model.

    N: int
        Number of target points.

    M: int
        Number of source points.

    D: int
        Dimensionality of source and target points

    iteration: int
        The current iteration throughout registration.

    max_iterations: int
        Registration will terminate once the algorithm has taken this
        many iterations.

    tolerance: float (positive)
        Registration will terminate once the difference between
        consecutive objective function values falls within this tolerance.

    w: float (between 0 and 1)
        Contribution of the uniform distribution to account for outliers.
        Valid values span 0 (inclusive) and 1 (exclusive).

    q: float
        The objective function value that represents the misalignment between
        source and target point clouds.

    diff: float (positive)
        The absolute difference between the current and previous objective
        function values.

    P: numpy array
        MxN array of probabilities.
        P[m, n] represents the probability that the m-th source point
        corresponds to the n-th target point.

    Pt1: numpy array
        Nx1 column array. Multiplication result between the transpose of P
        and a column vector of all 1s.

    P1: numpy array
        Mx1 column array.
        Multiplication result between P and a column vector of all 1s.

    Np: float (positive)
        The sum of all elements in P.

    alpha: float (positive)
        Represents the trade-off between the goodness of maximum likelihoo
        fit and regularization.

    beta: float(positive)
        Width of the Gaussian kernel.

    low_rank: bool
        Whether to use low rank approximation.

    num_eig: int
        Number of eigenvectors to use in lowrank calculation.
    """

    def __init__(self, X, Y, sigma2=None, alpha=None, beta=None,
                 low_rank=False, num_eig=100, max_iterations=None,
                 tolerance=None,  w=None, *args, **kwargs):
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise ValueError(
                "The target point cloud (X) must be at a 2D numpy array.")

        if not isinstance(Y, np.ndarray) or Y.ndim != 2:
            raise ValueError(
                "The source point cloud (Y) must be a 2D numpy array.")

        if X.shape[1] != Y.shape[1]:
            msg = "Both point clouds need to have the same number "
            msg += "of dimensions."
            raise ValueError(msg)

        if sigma2 is not None and (not isinstance(sigma2, numbers.Number)
                                   or sigma2 <= 0):
            msg = f"Expected a positive value for sigma2 instead got: {sigma2}"
            raise ValueError(msg)

        if max_iterations is not None and (not isinstance(max_iterations,
                                                          numbers.Number)
                                           or max_iterations < 0):
            msg = "Expected a positive integer for max_iterations "
            msg += f"instead got: {max_iterations}"
            raise ValueError(msg)
        elif isinstance(max_iterations, numbers.Number) and \
                not isinstance(max_iterations, int):
            msg = "Received a non-integer value for max_iterations: "
            msg += f"{max_iterations}. Casting to integer."
            warn(msg)
            max_iterations = int(max_iterations)

        if tolerance is not None and (not isinstance(tolerance, numbers.Number)
                                      or tolerance < 0):
            msg = "Expected a positive float for tolerance "
            msg += f"instead got: {tolerance}"
            raise ValueError(msg)

        if w is not None and (not isinstance(w, numbers.Number)
                              or w < 0 or w >= 1):
            msg = "Expected a value between 0 (inclusive) and 1 (exclusive) "
            msg += f"for w instead got: {w}"
            raise ValueError(msg)

        self.X = X
        self.Y = Y
        self.TY = Y
        self.sigma2 = initialize_sigma2(X, Y) if sigma2 is None else sigma2
        (self.N, self.D) = self.X.shape
        (self.M, _) = self.Y.shape
        self.tolerance = 0.001 if tolerance is None else tolerance
        self.w = 0.0 if w is None else w
        self.max_iterations = 100 if max_iterations is None else max_iterations
        self.iteration = 0
        self.diff = np.inf
        self.q = np.inf
        self.P = np.zeros((self.M, self.N))
        self.Pt1 = np.zeros((self.N, ))
        self.P1 = np.zeros((self.M, ))
        self.PX = np.zeros((self.M, self.D))
        self.Np = 0

        if alpha is not None and (not isinstance(alpha, numbers.Number)
                                  or alpha <= 0):
            msg = "Expected a positive value for regularization parameter "
            msg += f"alpha. Instead got: {alpha}"
            raise ValueError(msg)

        if beta is not None and (not isinstance(beta, numbers.Number)
                                 or beta <= 0):
            msg = "Expected a positive value for the width of the coherent "
            msg += f"Gaussian kernel. Instead got: {beta}"

        self.alpha = 2 if alpha is None else alpha
        self.beta = 2 if beta is None else beta
        self.W = np.zeros((self.M, self.D))
        self.G = gaussian_kernel(self.Y, self.beta)
        self.low_rank = low_rank
        self.num_eig = num_eig
        if self.low_rank is True:
            self.Q, self.S = low_rank_eigen(self.G, self.num_eig)
            self.inv_S = np.diag(1./self.S)
            self.S = np.diag(self.S)
            self.E = 0.

    def register(self, callback=lambda **kwargs: None):
        """
        Perform the EM registration.

        Parameters
        ----------
        callback: function
            A function that will be called after each iteration.
            Can be used to visualize the registration process.

        Returns
        -------
        self.TY: numpy array
            MxD array of transformed source points.

        registration_parameters:
            Returned params dependent on registration method used.
        """
        self.transform_point_cloud()
        while self.iteration < self.max_iterations and \
                self.diff > self.tolerance:
            self.iterate()
            if callable(callback):
                kwargs = {'iteration': self.iteration,
                          'error': self.q, 'X': self.X, 'Y': self.TY}
                callback(**kwargs)

        return self.TY, self.get_registration_parameters()

    def update_transform(self):
        """
        Calculate a new estimate of the deformable transformation.
        See Eq. 22 of https://arxiv.org/pdf/0905.2635.pdf.

        """
        if self.low_rank is False:
            A = np.dot(np.diag(self.P1), self.G) + \
                self.alpha * self.sigma2 * np.eye(self.M)
            B = self.PX - np.dot(np.diag(self.P1), self.Y)
            self.W = np.linalg.solve(A, B)

        elif self.low_rank is True:
            # Matlab code equivalent can be found here:
            # https://github.com/markeroon/matlab-computer-vision-routines/tree/master/third_party/CoherentPointDrift
            dP = np.diag(self.P1)
            dPQ = np.matmul(dP, self.Q)
            F = self.PX - np.matmul(dP, self.Y)

            self.W = 1 / (self.alpha * self.sigma2) * (F - np.matmul(dPQ, (
                np.linalg.solve((self.alpha * self.sigma2 * self.inv_S + np.matmul(self.Q.T, dPQ)),
                                (np.matmul(self.Q.T, F))))))
            QtW = np.matmul(self.Q.T, self.W)
            self.E = self.E + self.alpha / 2 * np.trace(np.matmul(QtW.T, np.matmul(self.S, QtW)))

    def transform_point_cloud(self, Y=None):
        """Update a point cloud using the new estimate of the deformable
        transformation.

        Parameters
        ----------
        Y: numpy array, optional
            Array of points to transform - use to predict on new set of points.
            Best for predicting on new points not used to run initial
            registration. If None, self.Y used.

        Returns
        -------
        If Y is None, returns None.
        Otherwise, returns the transformed Y.


        """
        if Y is not None:
            G = gaussian_kernel(X=Y, beta=self.beta, Y=self.Y)
            return Y + np.dot(G, self.W)
        else:
            if self.low_rank is False:
                self.TY = self.Y + np.dot(self.G, self.W)

            elif self.low_rank is True:
                self.TY = self.Y + np.matmul(
                    self.Q,
                    np.matmul(self.S, np.matmul(self.Q.T, self.W)))
                return

    def update_variance(self):
        """Update the variance of the mixture model.

        This is using the new estimate of the deformable transformation.
        See the update rule for sigma2 in
        Eq. 23 of of https://arxiv.org/pdf/0905.2635.pdf.

        """
        qprev = self.sigma2

        # The original CPD paper does not explicitly calculate the objective
        # functional. This functional will include terms from both the negative
        # log-likelihood and the Gaussian kernel used for regularization.
        self.q = np.inf

        xPx = np.dot(np.transpose(self.Pt1), np.sum(
            np.multiply(self.X, self.X), axis=1))
        yPy = np.dot(np.transpose(self.P1),  np.sum(
            np.multiply(self.TY, self.TY), axis=1))
        trPXY = np.sum(np.multiply(self.TY, self.PX))

        self.sigma2 = (xPx - 2 * trPXY + yPy) / (self.Np * self.D)

        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10

        # Here we use the difference between the current and previous
        # estimate of the variance as a proxy to test for convergence.
        self.diff = np.abs(self.sigma2 - qprev)

    def get_registration_parameters(self):
        """Return the current estimate of the deformable transformation
        parameters.

        Returns
        -------
        self.G: numpy array
            Gaussian kernel matrix.

        self.W: numpy array
            Deformable transformation matrix.
        """
        return self.G, self.W

    def iterate(self):
        """Perform one iteration of the EM algorithm."""
        self.expectation()
        self.maximization()
        self.iteration += 1

    def expectation(self):
        """Compute the expectation step of the EM algorithm."""
        # (M, N)
        P = np.sum((self.X[None, :, :] - self.TY[:, None, :])**2, axis=2)
        P = np.exp(-P/(2*self.sigma2))
        c = (2*np.pi*self.sigma2)**(self.D/2)*self.w/(1. - self.w)*self.M/self.N

        den = np.sum(P, axis=0, keepdims=True)  # (1, N)
        den = np.clip(den, np.finfo(self.X.dtype).eps, None) + c

        self.P = np.divide(P, den)
        self.Pt1 = np.sum(self.P, axis=0)
        self.P1 = np.sum(self.P, axis=1)
        self.Np = np.sum(self.P1)
        self.PX = np.matmul(self.P, self.X)

    def maximization(self):
        """Compute the maximization step of the EM algorithm."""
        self.update_transform()
        self.transform_point_cloud()
        self.update_variance()
