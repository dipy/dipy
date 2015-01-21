import numpy as np
import scipy.sparse as sps

import numpy.testing as npt
from dipy.core.optimize import Optimizer, SCIPY_LESS_0_12, sparse_nnls, spdot
import dipy.core.optimize as opt


def func(x):
    return x[0]**2 + x[1]**2 + x[2]**2


def func2(x):
    return x[0]**2 + 0.5 * x[1]**2 + 0.2 * x[2]**2 + 0.2 * x[3]**2


@npt.dec.skipif(SCIPY_LESS_0_12)
def test_optimize_new_scipy():
    opt = Optimizer(fun=func, x0=np.array([1., 1., 1.]), method='Powell')

    npt.assert_array_almost_equal(opt.xopt, np.array([0, 0, 0]))
    npt.assert_almost_equal(opt.fopt, 0)

    opt = Optimizer(fun=func, x0=np.array([1., 1., 1.]), method='L-BFGS-B',
                    options={'maxcor': 10, 'ftol': 1e-7,
                             'gtol': 1e-5, 'eps': 1e-8})

    npt.assert_array_almost_equal(opt.xopt, np.array([0, 0, 0]))
    npt.assert_almost_equal(opt.fopt, 0)
    npt.assert_equal(opt.evolution, None)

    npt.assert_equal(opt.evolution, None)

    opt = Optimizer(fun=func, x0=np.array([1., 1., 1.]), method='L-BFGS-B',
                    options={'maxcor': 10, 'ftol': 1e-7,
                             'gtol': 1e-5, 'eps': 1e-8},
                    evolution=False)

    npt.assert_array_almost_equal(opt.xopt, np.array([0, 0, 0]))
    npt.assert_almost_equal(opt.fopt, 0)

    opt.print_summary()

    opt = Optimizer(fun=func2, x0=np.array([1., 1., 1., 5.]),
                    method='L-BFGS-B',
                    options={'maxcor': 10, 'ftol': 1e-7,
                             'gtol': 1e-5, 'eps': 1e-8},
                    evolution=True)

    npt.assert_equal(opt.evolution.shape, (opt.nit, 4))

    opt = Optimizer(fun=func2, x0=np.array([1., 1., 1., 5.]),
                    method='Powell',
                    options={'xtol': 1e-6, 'ftol': 1e-6, 'maxiter': 1e6},
                    evolution=True)

    npt.assert_array_almost_equal(opt.xopt, np.array([0, 0, 0, 0.]))


@npt.dec.skipif(not SCIPY_LESS_0_12)
def test_optimize_old_scipy():

    opt = Optimizer(fun=func, x0=np.array([1., 1., 1.]),
                    method='L-BFGS-B',
                    options={'maxcor': 10, 'ftol': 1e-7,
                             'gtol': 1e-5, 'eps': 1e-8})

    npt.assert_array_almost_equal(opt.xopt, np.array([0, 0, 0]))
    npt.assert_almost_equal(opt.fopt, 0)

    opt = Optimizer(fun=func2, x0=np.array([1., 1., 1., 5.]),
                    method='Powell',
                    options={'xtol': 1e-6, 'ftol': 1e-6, 'maxiter': 1e6},
                    evolution=True)

    npt.assert_array_almost_equal(opt.xopt, np.array([0, 0, 0, 0.]))

    opt = Optimizer(fun=func, x0=np.array([1., 1., 1.]),
                    method='L-BFGS-B',
                    options={'maxcor': 10, 'eps': 1e-8})

    npt.assert_array_almost_equal(opt.xopt, np.array([0, 0, 0]))
    npt.assert_almost_equal(opt.fopt, 0)

    opt = Optimizer(fun=func, x0=np.array([1., 1., 1.]),
                    method='L-BFGS-B',
                    options=None)

    npt.assert_array_almost_equal(opt.xopt, np.array([0, 0, 0]))
    npt.assert_almost_equal(opt.fopt, 0)

    opt = Optimizer(fun=func2, x0=np.array([1., 1., 1., 5.]),
                    method='L-BFGS-B',
                    options={'gtol': 1e-7, 'ftol': 1e-7, 'maxiter': 10000})

    npt.assert_array_almost_equal(opt.xopt, np.array([0, 0, 0, 0.]), 4)
    npt.assert_almost_equal(opt.fopt, 0)

    opt = Optimizer(fun=func2, x0=np.array([1., 1., 1., 5.]),
                    method='Powell',
                    options={'maxiter': 1e6},
                    evolution=True)

    npt.assert_array_almost_equal(opt.xopt, np.array([0, 0, 0, 0.]))

    opt = Optimizer(fun=func2, x0=np.array([1., 1., 1., 5.]),
                    method='Powell',
                    options={'maxiter': 1e6},
                    evolution=True)

    npt.assert_array_almost_equal(opt.xopt, np.array([0, 0, 0, 0.]))


def test_sklearn_linear_solver():
    class SillySolver(opt.SKLearnLinearSolver):
        def fit(self, X, y):
            self.coef_ = np.ones(X.shape[-1])

    MySillySolver = SillySolver()
    n_samples = 100
    n_features = 20
    y = np.random.rand(n_samples)
    X = np.ones((n_samples, n_features))
    MySillySolver.fit(X, y)
    npt.assert_equal(MySillySolver.coef_, np.ones(n_features))
    npt.assert_equal(MySillySolver.predict(X), np.ones(n_samples) * 20)


def test_nonnegativeleastsquares():
    n = 100
    X = np.eye(n)
    beta = np.random.rand(n)
    y = np.dot(X, beta)
    my_nnls = opt.NonNegativeLeastSquares()
    my_nnls.fit(X, y)
    npt.assert_equal(my_nnls.coef_, beta)
    npt.assert_equal(my_nnls.predict(X), y)


def test_spdot():
    n = 100
    m = 20
    k = 10
    A = np.random.randn(n, m)
    B = np.random.randn(m, k)
    A_sparse = sps.csr_matrix(A)
    B_sparse = sps.csr_matrix(B)
    dense_dot = np.dot(A, B)
    # Try all the different variations:
    npt.assert_array_almost_equal(dense_dot,
                                  spdot(A_sparse, B_sparse).todense())
    npt.assert_array_almost_equal(dense_dot, spdot(A, B_sparse))
    npt.assert_array_almost_equal(dense_dot, spdot(A_sparse, B))


def test_sparse_nnls():
    # Set up the regression:
    beta = np.random.rand(10)
    X = np.random.randn(1000, 10)
    y = np.dot(X, beta)
    beta_hat = sparse_nnls(y, X)
    beta_hat_sparse = sparse_nnls(y, sps.csr_matrix(X))
    # We should be able to get back the right answer for this simple case
    npt.assert_array_almost_equal(beta, beta_hat, decimal=1)
    npt.assert_array_almost_equal(beta, beta_hat_sparse, decimal=1)


if __name__ == '__main__':
    npt.run_module_suite()
