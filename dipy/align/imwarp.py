import numpy as np


class SymmetricDiffeomorphicRegistration(object):

    def __init__(self, metric, iters):
        """ Symmetric Diffeomorphic Registration

        Parameters
        ----------
        metric : Similarity object

        iters : int

        inv_iters : int

        tol_inv : float

        tol_opt : float

        """

        self.iters = iters

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



