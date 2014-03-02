import numpy as np


class SymmetricDiffeomorphicRegistration(object):

    def __init__(   self,
                    metric,
                    opt_iters = [25, 100, 100],
                    opt_tol = 1e-4,
                    inv_iters = 20,
                    inv_tol = 1e-3
                    energy_window = 12):
        """ Symmetric Diffeomorphic Registration

        Parameters
        ----------
        metric : Similarity object

        opt_iters : list
            maximum number of iterations at each level of the Gaussian Pyramid 
            (multi-resolution), opt_iters[0] corresponds the finest resolution

        opt_tol: float
            tolerance for the optimization algorithm, the algorithm stops when
            the derivative of the energy profile w.r.t. time falls below opt_tol

        inv_iters : int
            maximum number of iterations of the displacement field inversion 
            algorithm 

        inv_tol : float
            tolerance for the displacement field inversion algorithm

        energy_window: int
            minimum number of iterations to be considered when estimating the
            derivative of energy over time

        """

        self.metric = metric
        self.opt_iters = opt_iters
        self.opt_tol = opt_tol
        self.inv_iters = inv_iters
        self.inv_tol = inv_tol

    def optimize(self, static, moving):

        return SymmetricDiffeomorficMap(self)

class SymmetricDiffeomorficMap(object):

    def __init__(self, model):

        pass

    def transform(moving, interpolation='tri'):

        pass

    def warp_direct():

        pass

    def warp_inverse():

        pass



