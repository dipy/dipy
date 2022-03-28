"""A unified interface for performing and debugging optimization problems."""

import abc
import numpy as np
import scipy.sparse as sps
import scipy.optimize as opt
from scipy.optimize import minimize


class Optimizer(object):

    def __init__(self, fun,  x0, args=(), method='L-BFGS-B', jac=None,
                 hess=None, hessp=None, bounds=None, constraints=(),
                 tol=None, callback=None, options=None, evolution=False):
        """ A class for handling minimization of scalar function of one or more
        variables.

        Parameters
        ----------
        fun : callable
            Objective function.

        x0 : ndarray
            Initial guess.

        args : tuple, optional
            Extra arguments passed to the objective function and its
            derivatives (Jacobian, Hessian).

        method : str, optional
            Type of solver.  Should be one of

                - 'Nelder-Mead'
                - 'Powell'
                - 'CG'
                - 'BFGS'
                - 'Newton-CG'
                - 'Anneal'
                - 'L-BFGS-B'
                - 'TNC'
                - 'COBYLA'
                - 'SLSQP'
                - 'dogleg'
                - 'trust-ncg'

        jac : bool or callable, optional
            Jacobian of objective function. Only for CG, BFGS, Newton-CG,
            dogleg, trust-ncg.
            If `jac` is a Boolean and is True, `fun` is assumed to return the
            value of Jacobian along with the objective function. If False, the
            Jacobian will be estimated numerically.
            `jac` can also be a callable returning the Jacobian of the
            objective. In this case, it must accept the same arguments
            as `fun`.

        hess, hessp : callable, optional
            Hessian of objective function or Hessian of objective function
            times an arbitrary vector p.  Only for Newton-CG,
            dogleg, trust-ncg.
            Only one of `hessp` or `hess` needs to be given.  If `hess` is
            provided, then `hessp` will be ignored.  If neither `hess` nor
            `hessp` is provided, then the hessian product will be approximated
            using finite differences on `jac`. `hessp` must compute the Hessian
            times an arbitrary vector.

        bounds : sequence, optional
            Bounds for variables (only for L-BFGS-B, TNC and SLSQP).
            ``(min, max)`` pairs for each element in ``x``, defining
            the bounds on that parameter. Use None for one of ``min`` or
            ``max`` when there is no bound in that direction.

        constraints : dict or sequence of dict, optional
            Constraints definition (only for COBYLA and SLSQP).
            Each constraint is defined in a dictionary with fields:
                type : str
                    Constraint type: 'eq' for equality, 'ineq' for inequality.
                fun : callable
                    The function defining the constraint.
                jac : callable, optional
                    The Jacobian of `fun` (only for SLSQP).
                args : sequence, optional
                    Extra arguments to be passed to the function and Jacobian.
            Equality constraint means that the constraint function result is to
            be zero whereas inequality means that it is to be non-negative.
            Note that COBYLA only supports inequality constraints.

        tol : float, optional
            Tolerance for termination. For detailed control, use
            solver-specific options.

        callback : callable, optional
            Called after each iteration, as ``callback(xk)``, where ``xk`` is
            the current parameter vector. Only available using Scipy >= 0.12.

        options : dict, optional
            A dictionary of solver options. All methods accept the following
            generic options:
                maxiter : int
                    Maximum number of iterations to perform.
                disp : bool
                    Set to True to print convergence messages.
            For method-specific options, see
            `show_options('minimize', method)`.

        evolution : bool, optional
            save history of x for each iteration. Only available using Scipy
            >= 0.12.

        See Also
        --------
        scipy.optimize.minimize

        """
        self.size_of_x = len(x0)
        self._evol_kx = None

        if evolution is True:

            self._evol_kx = []

            def history_of_x(kx):
                self._evol_kx.append(kx)
            res = minimize(fun, x0, args, method, jac, hess, hessp, bounds,
                           constraints, tol, callback=history_of_x,
                           options=options)

        else:

            res = minimize(fun, x0, args, method, jac, hess, hessp, bounds,
                           constraints, tol, callback, options)

        self.res = res

    @property
    def xopt(self):

        return self.res['x']

    @property
    def fopt(self):

        return self.res['fun']

    @property
    def nit(self):

        return self.res['nit']

    @property
    def nfev(self):

        return self.res['nfev']

    @property
    def message(self):

        return self.res['message']

    def print_summary(self):

        print(self.res)

    @property
    def evolution(self):
        if self._evol_kx is not None:
            return np.asarray(self._evol_kx)
        else:
            return None


def spdot(A, B):
    """The same as np.dot(A, B), except it works even if A or B or both
    are sparse matrices.

    Parameters
    ----------
    A, B : arrays of shape (m, n), (n, k)

    Returns
    -------
    The matrix product AB. If both A and B are sparse, the result will be a
    sparse matrix. Otherwise, a dense result is returned

    See discussion here:
    http://mail.scipy.org/pipermail/scipy-user/2010-November/027700.html

    """
    if sps.issparse(A) and sps.issparse(B):
        return A * B
    elif sps.issparse(A) and not sps.issparse(B):
        return (A * B).view(type=B.__class__)
    elif not sps.issparse(A) and sps.issparse(B):
        return (B.T * A.T).T.view(type=A.__class__)
    else:
        return np.dot(A, B)


def sparse_nnls(y, X,
                momentum=1,
                step_size=0.01,
                non_neg=True,
                check_error_iter=10,
                max_error_checks=10,
                converge_on_sse=0.99):
    """
    Solve y=Xh for h, using gradient descent, with X a sparse matrix.

    Parameters
    ----------
    y : 1-d array of shape (N)
        The data. Needs to be dense.

    X : ndarray. May be either sparse or dense. Shape (N, M)
       The regressors

    momentum : float, optional (default: 1).
        The persistence of the gradient.

    step_size : float, optional (default: 0.01).
        The increment of parameter update in each iteration

    non_neg : Boolean, optional (default: True)
        Whether to enforce non-negativity of the solution.

    check_error_iter : int (default:10)
        How many rounds to run between error evaluation for
        convergence-checking.

    max_error_checks : int (default: 10)
        Don't check errors more than this number of times if no improvement in
        r-squared is seen.

    converge_on_sse : float (default: 0.99)
      a percentage improvement in SSE that is required each time to say
      that things are still going well.

    Returns
    -------
    h_best : The best estimate of the parameters.

    """
    num_regressors = X.shape[1]
    # Initialize the parameters at the origin:
    h = np.zeros(num_regressors)
    # If nothing good happens, we'll return that:
    h_best = h
    iteration = 1
    ss_residuals_min = np.inf  # This will keep track of the best solution
    sse_best = np.inf   # This will keep track of the best performance so far
    count_bad = 0  # Number of times estimation error has gone up.
    error_checks = 0  # How many error checks have we done so far

    while 1:
        if iteration > 1:
            # The gradient is (Kay 2008 supplemental page 27):
            gradient = spdot(X.T, spdot(X, h) - y)
            gradient += momentum * gradient
            # Normalize to unit-length
            unit_length_gradient = (gradient /
                                    np.sqrt(np.dot(gradient, gradient)))
            # Update the parameters in the direction of the gradient:
            h -= step_size * unit_length_gradient
            if non_neg:
                # Set negative values to 0:
                h[h < 0] = 0

        # Every once in a while check whether it's converged:
        if np.mod(iteration, check_error_iter):
            # This calculates the sum of squared residuals at this point:
            sse = np.sum((y - spdot(X, h)) ** 2)
            # Did we do better this time around?
            if sse < ss_residuals_min:
                # Update your expectations about the minimum error:
                ss_residuals_min = sse
                h_best = h  # This holds the best params we have so far
                # Are we generally (over iterations) converging on
                # sufficient improvement in r-squared?
                if sse < converge_on_sse * sse_best:
                    sse_best = sse
                    count_bad = 0
                else:
                    count_bad += 1
            else:
                count_bad += 1

            if count_bad >= max_error_checks:
                return h_best
            error_checks += 1
        iteration += 1


class SKLearnLinearSolver(object, metaclass=abc.ABCMeta):
    """
    Provide a sklearn-like uniform interface to algorithms that solve problems
    of the form: $y = Ax$ for $x$

    Sub-classes of SKLearnLinearSolver should provide a 'fit' method that have
    the following signature: `SKLearnLinearSolver.fit(X, y)`, which would set
    an attribute `SKLearnLinearSolver.coef_`, with the shape (X.shape[1],),
    such that an estimate of y can be calculated as:
    `y_hat = np.dot(X, SKLearnLinearSolver.coef_.T)`
    """
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    @abc.abstractmethod
    def fit(self, X, y):
        """Implement for all derived classes """

    def predict(self, X):
        """
        Predict using the result of the model

        Parameters
        ----------
        X : array-like (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape = (n_samples,)
            Predicted values.
        """
        X = np.asarray(X)
        return np.dot(X, self.coef_.T)


class NonNegativeLeastSquares(SKLearnLinearSolver):
    """
    A sklearn-like interface to scipy.optimize.nnls

    """
    def fit(self, X, y):
        """
        Fit the NonNegativeLeastSquares linear model to data

        Parameters
        ----------

        """
        coef, rnorm = opt.nnls(X, y)
        self.coef_ = coef
        return self
