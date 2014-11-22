""" A unified interface for performing and debugging optimization problems.

Only L-BFGS-B and Powell is supported in this class for versions of
Scipy < 0.12. All optimizers are available for scipy >= 0.12.
"""

from distutils.version import LooseVersion
import numpy as np
import scipy

SCIPY_LESS_0_12 = LooseVersion(scipy.__version__) < '0.12'

if not SCIPY_LESS_0_12:
    from scipy.optimize import minimize
else:
    from scipy.optimize import fmin_l_bfgs_b, fmin_powell


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
            the current parameter vector. Only available using Scipy 0.12+.

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
            0.12+.

        See also
        ---------
        scipy.optimize.minimize
        """

        self.size_of_x = len(x0)
        self._evol_kx = None

        _eps = np.finfo(float).eps

        if SCIPY_LESS_0_12:

            if evolution is True:
                print('Saving the history is only available with Scipy 0.12+.')

            if method == 'L-BFGS-B':
                default_options = {'maxcor': 10, 'ftol': 1e-7, 'gtol': 1e-5,
                                   'eps': 1e-8, 'maxiter': 1000}

                if jac is None:
                    approx_grad = True
                else:
                    approx_grad = False

                if options is None:
                    options = default_options

                if options is not None:
                    for key in options:
                        default_options[key] = options[key]
                    options = default_options

                try:
                    out = fmin_l_bfgs_b(fun, x0, args,
                                        approx_grad=approx_grad,
                                        bounds=bounds,
                                        m=options['maxcor'],
                                        factr=options['ftol']/_eps,
                                        pgtol=options['gtol'],
                                        epsilon=options['eps'],
                                        maxiter=options['maxiter'])
                except TypeError:

                    msg = 'In Scipy ' + scipy.__version__ + ' `maxiter` '
                    msg += 'parameter is not available for L-BFGS-B. \n Using '
                    msg += '`maxfun` instead with value twice of maxiter.'

                    print(msg)
                    out = fmin_l_bfgs_b(fun, x0, args,
                                        approx_grad=approx_grad,
                                        bounds=bounds,
                                        m=options['maxcor'],
                                        factr=options['ftol']/_eps,
                                        pgtol=options['gtol'],
                                        epsilon=options['eps'],
                                        maxfun=options['maxiter'] * 2)

                res = {'x': out[0], 'fun': out[1], 'nfev': out[2]['funcalls']}
                try:
                    res['nit'] = out[2]['nit']
                except KeyError:
                    res['nit'] = None

            elif method == 'Powell':

                default_options = {'xtol': 0.0001, 'ftol': 0.0001,
                                   'maxiter': None}

                if options is None:
                    options = default_options

                if options is not None:
                    for key in options:
                        default_options[key] = options[key]
                    options = default_options

                out = fmin_powell(fun, x0, args,
                                  xtol=options['xtol'],
                                  ftol=options['ftol'],
                                  maxiter=options['maxiter'],
                                  full_output=True,
                                  disp=False,
                                  retall=True)

                xopt, fopt, direc, iterations, funcs, warnflag, allvecs = out
                res = {'x': xopt, 'fun': fopt,
                       'nfev': funcs, 'nit': iterations}

            else:

                msg = 'Only L-BFGS-B and Powell is supported in this class '
                msg += 'for versions of Scipy < 0.11.'
                raise ValueError(msg)

        if not SCIPY_LESS_0_12:

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
