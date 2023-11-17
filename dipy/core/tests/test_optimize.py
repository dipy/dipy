import numpy as np
import scipy.sparse as sps

import numpy.testing as npt
from dipy.core.optimize import Optimizer, sparse_nnls, spdot
import dipy.core.optimize as opt
from dipy.testing.decorators import set_random_number_generator


def func(x):
    return x[0]**2 + x[1]**2 + x[2]**2


def func2(x):
    return x[0]**2 + 0.5 * x[1]**2 + 0.2 * x[2]**2 + 0.2 * x[3]**2


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


@set_random_number_generator()
def test_sklearn_linear_solver(rng):
    class SillySolver(opt.SKLearnLinearSolver):
        def fit(self, X, y):
            self.coef_ = np.ones(X.shape[-1])

    MySillySolver = SillySolver()
    n_samples = 100
    n_features = 20
    y = rng.random(n_samples)
    X = np.ones((n_samples, n_features))
    MySillySolver.fit(X, y)
    npt.assert_equal(MySillySolver.coef_, np.ones(n_features))
    npt.assert_equal(MySillySolver.predict(X), np.ones(n_samples) * 20)


@set_random_number_generator()
def test_nonnegativeleastsquares(rng):
    n = 100
    X = np.eye(n)
    beta = rng.random(n)
    y = np.dot(X, beta)
    my_nnls = opt.NonNegativeLeastSquares()
    my_nnls.fit(X, y)
    npt.assert_equal(my_nnls.coef_, beta)
    npt.assert_equal(my_nnls.predict(X), y)


@set_random_number_generator()
def test_spdot(rng):
    n = 100
    m = 20
    k = 10
    A = rng.standard_normal((n, m))
    B = rng.standard_normal((m, k))
    A_sparse = sps.csr_matrix(A)
    B_sparse = sps.csr_matrix(B)
    dense_dot = np.dot(A, B)
    # Try all the different variations:
    npt.assert_array_almost_equal(dense_dot,
                                  spdot(A_sparse, B_sparse).todense())
    npt.assert_array_almost_equal(dense_dot, spdot(A, B_sparse))
    npt.assert_array_almost_equal(dense_dot, spdot(A_sparse, B))


@set_random_number_generator()
def test_sparse_nnls(rng):
    # Set up the regression:
    beta = rng.random(10)
    X = rng.standard_normal((1000, 10))
    y = np.dot(X, beta)
    beta_hat = sparse_nnls(y, X)
    beta_hat_sparse = sparse_nnls(y, sps.csr_matrix(X))
    # We should be able to get back the right answer for this simple case
    npt.assert_array_almost_equal(beta, beta_hat, decimal=1)
    npt.assert_array_almost_equal(beta, beta_hat_sparse, decimal=1)
