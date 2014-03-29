import numpy as np
from numpy.testing import (assert_almost_equal,
                           assert_array_almost_equal)


from dipy.core.optimize import Optimizer, scipy_less_0_11



def test_optimize():

    def func(x):

        return x[0]**2 + x[1]**2 + x[2]**2

    def func2(x):

        return x[0]**2 + 0.5 * x[1]**2 + 0.2 * x[2]**2 + 0.2 * x[3]**2


    if not scipy_less_0_11:

        opt = Optimizer(fun=func, x0=np.array([1., 1., 1.]), method='Powell')

        assert_array_almost_equal(opt.xopt, np.array([0, 0, 0]))

        assert_almost_equal(opt.fopt, 0)


        opt = Optimizer(fun=func, x0=np.array([1., 1., 1.]), method='L-BFGS-B',
                        options={'maxcor':10, 'ftol':1e-7, 'gtol':1e-5, 'eps':1e-8})

        assert_array_almost_equal(opt.xopt, np.array([0, 0, 0]))

        assert_almost_equal(opt.fopt, 0)

        opt = Optimizer(fun=func, x0=np.array([1., 1., 1.]), method='L-BFGS-B',
                        options={'maxcor':10, 'ftol':1e-7, 'gtol':1e-5, 'eps':1e-8},
                        history=True)

        assert_array_almost_equal(opt.xopt, np.array([0, 0, 0]))

        assert_almost_equal(opt.fopt, 0)

        opt.info

        opt = Optimizer(fun=func2, x0=np.array([1., 1., 1., 5.]),
                        method='L-BFGS-B',
                        options={'maxcor':10, 'ftol':1e-7, 'gtol':1e-5, 'eps':1e-8},
                        history=True)
        print(opt.nit)

        print(opt.hist)


    if scipy_less_0_11:

        opt = Optimizer(fun=func, x0=np.array([1., 1., 1.]), method='L-BFGS-B',
                        options={'maxcor':10, 'ftol':1e-7, 'gtol':1e-5, 'eps':1e-8})

        assert_array_almost_equal(opt.xopt, np.array([0, 0, 0]))

        assert_almost_equal(opt.fopt, 0)





test_optimize()
