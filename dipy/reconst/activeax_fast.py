# -*- coding: utf-8 -*-
from dipy.reconst.base import ReconstModel
import numpy as np
import cvxpy as cvx
from scipy.optimize import least_squares
from scipy.optimize import differential_evolution
from dipy.reconst.recspeed import S1, S2, S2_new, activeax_cost_one
from dipy.data import get_data

gamma = 2.675987 * 10 ** 8
D_intra = 0.6 * 10 ** 3
fname, fscanner = get_data('ActiveAx_synth_2d')
params = np.loadtxt(fscanner)
G = params[:, 3] / 10 ** 6  # gradient strength
D_iso = 2 * 10 ** 3
am = np.array([1.84118307861360])


class ActiveAxModel(ReconstModel):

    def __init__(self, gtab, fit_method='MIX'):
        r""" MIX framework (MIX) [1]_.

        The MIX computes the ActiveAx parameters. ActiveAx is a multi
        compartment model, (sum of exponentials).
        This algorithm uses three different optimizer. It starts with a
        differential evolutionary algorithm and fits the parameters in the
        power of exponentials. Then the fitted parameters in the first step are
        utilized to make a linear convex problem. Using a convex optimization,
        the volume fractions are determined. Then the last step is non linear
        least square fitting on all the parameters. The results of the first
        and second step are utilized as the initial values for the last step
        of the algorithm. (see [1]_ for a comparison and a through discussion).

        Parameters
        ----------
        gtab : GradientTable

        fit_method : str or callable

        Returns
        -------
        ActiveAx parameters

        References
        ----------
        .. [1] Farooq, Hamza, et al. "Microstructure Imaging of Crossing (MIX)
               White Matter Fibers from diffusion MRI." Scientific reports 6
               (2016).

        """

        self.maxiter = 1  # The maximum number of generations, genetic
#        algorithm 11 default
        self.xtol = 1e-3  # Tolerance for termination, nonlinear least square
#        1e-5 default
        self.gtab = gtab
        self.big_delta = gtab.big_delta
        self.small_delta = gtab.small_delta
        self.gamma = gamma
        self.G = G
        self.G2 = self.G ** 2
        D_iso = 2 * 10 ** 3
        self.yhat_ball = D_iso * self.gtab.bvals
        self.L = self.gtab.bvals * D_intra
        self.phi_inv = np.zeros((4, 4))
        self.yhat_zeppelin = np.zeros(self.small_delta.shape[0])
        self.yhat_cylinder = np.zeros(self.small_delta.shape[0])
        self.yhat_dot = np.zeros(self.gtab.bvals.shape)
        self.exp_phi1 = np.zeros((self.small_delta.shape[0], 4))
        self.exp_phi1[:, 2] = np.exp(-self.yhat_ball)
        self.exp_phi1[:, 3] = np.ones(self.gtab.bvals.shape)
        self.x2 = np.zeros(3)

    def fit(self, data):
        """ Fit method of the ActiveAx model class

        Parameters
        ----------
        data : array
            The measured signal from one voxel.

        """
        bounds = [(0.01, np.pi), (0.01, np.pi), (0.1, 11), (0.1, 0.8)]
        res_one = differential_evolution(self.stoc_search_cost, bounds,
                                         maxiter=self.maxiter, args=(data,))
        x = res_one.x
        phi = self.Phi(x)
        fe = self.cvx_fit(data, phi)
        x_fe = self.x_and_fe_to_x_fe(x, fe)
        bounds = ([0.01, 0.01,  0.01, 0.01, 0.01, 0.1, 0.01], [0.9,  0.9,  0.9,
                  np.pi, np.pi, 11, 0.9])
        res = least_squares(self.nlls_cost, x_fe, bounds=(bounds),
                            xtol=self.xtol, args=(data,))
        result = res.x
        return result

    def stoc_search_cost(self, x, signal):
        """
        Aax_exvivo_nlin

        Cost function for genetic algorithm

        Parameters
        ----------
        x : array
            x.shape = 4x1
            x(0) theta (radian)
            x(1) phi (radian)
            x(2) R (micrometers)
            x(3) v=f1/(f1+f2) (0.1 - 0.8)

        bvals
        bvecs
        G: gradient strength
        small_delta
        big_delta
        gamma: gyromagnetic ratio (2.675987 * 10 ** 8 )
        D_intra= intrinsic free diffusivity (0.6 * 10 ** 3 mircometer^2/sec)
        D_iso= isotropic diffusivity, (2 * 10 ** 3 mircometer^2/sec)

        Returns
        -------
        (signal -  S)^T(signal -  S)

        Notes
        --------
        cost function for genetic algorithm:

        .. math::

            (signal -  S)^T(signal -  S)
        """
        phi = self.Phi(x)
        return activeax_cost_one(phi, signal)

    def cvx_fit(self, signal, phi):
        """
        Linear parameters fit using cvx

        Parameters
        ----------
        phi : array
            phi.shape = number of data points x 4
            signal : array
            signal.shape = number of data points x 1

        Returns
        -------
        f1, f2, f3, f4 (volume fractions)
        f1 = fe[0]
        f2 = fe[1]
        f3 = fe[2]
        f4 = fe[3]

        Notes
        --------
        cost function for genetic algorithm:

        .. math::

            minimize(norm((signal)- (phi*fe)))
        """

        # Create four scalar optimization variables.
        fe = cvx.Variable(4)
        # Create four constraints.
        constraints = [cvx.sum_entries(fe) == 1,
                       fe[0] >= 0.011,
                       fe[1] >= 0.011,
                       fe[2] >= 0.011,
                       fe[3] >= 0.011]

        # Form objective.
        obj = cvx.Minimize(cvx.sum_entries(cvx.square(phi * fe - signal)))

        # Form and solve problem.
        prob = cvx.Problem(obj, constraints)
        prob.solve()  # Returns the optimal value.
        return np.array(fe.value)

    def nlls_cost(self, x_fe, signal):
        """
        cost function for the least square problem

        Parameters
        ----------
        x_fe : array
            x_fe(0) x_fe(1) x_fe(2)  are f1 f2 f3
            x_fe(3) theta
            x_fe(4) phi
            x_fe(5) R
            x_fe(6) as f4

        signal_param : array
            signal_param.shape = number of data points x 7

            signal_param = np.hstack([signal[:, None], bvals[:, None], bvecs,
                                  G[:, None], small_delta[:, None],
                                  big_delta[:, None]])

        Returns
        -------
        sum{(signal -  phi*fe)^2}

        Notes
        --------
        cost function for the least square problem

        .. math::

            sum{(signal -  phi*fe)^2}
        """

        x, fe = self.x_fe_to_x_and_fe(x_fe)
        phi = self.Phi2(x_fe)
        return np.sum((np.dot(phi, fe) - signal) ** 2)

    def x_to_xs(self, x):
        x1 = x[0:3]
        self.x2[0:2] = x[0:2]
        self.x2[2] = x[3]
        return x1, self.x2

    def S3(self):
        # ball
        yhat_ball = D_iso * self.gtab.bvals
        return yhat_ball

    def S4(self):
        # dot
        yhat_dot = np.zeros(self.gtab.bvals.shape)
        return yhat_dot

    def x_fe_to_x_and_fe(self, x_fe):
        fe = np.zeros((1, 4))
        fe = fe[0]
        fe[0:3] = x_fe[0:3]
        fe[3] = x_fe[6]
        x = x_fe[3:6]
        return x, fe

    def x_and_fe_to_x_fe(self, x, fe):
        x_fe = np.zeros(7)
        fe = fe[:, 0]
        x_fe[:3] = fe[:3]
        x_fe[3:6] = x[:3]
        x_fe[6] = fe[3]
        return x_fe

    def Phi(self, x):
        x1, self.x2 = self.x_to_xs(x)
        S2(self.x2, self.gtab.bvals, self.gtab.bvecs, self.yhat_zeppelin)
        S1(x1, am, self.gtab.bvecs, self.gtab.bvals, self.small_delta,
           self.big_delta, self.G2, self.L, self.yhat_cylinder)
        self.exp_phi1[:, 0] = np.exp(-self.yhat_cylinder)
        self.exp_phi1[:, 1] = np.exp(-self.yhat_zeppelin)
        return self.exp_phi1

    def Phi2(self, x_fe):
        x, fe = self.x_fe_to_x_and_fe(x_fe)
        x1 = x[0:3]
        S2_new(x_fe, self.gtab.bvals,  self.gtab.bvecs, self.yhat_zeppelin)
        S1(x1, am, self.gtab.bvecs, self.gtab.bvals, self.gtab.small_delta,
           self.gtab.big_delta, self.G2, self.L, self.yhat_cylinder)
        self.exp_phi1[:, 0] = np.exp(-self.yhat_cylinder)
        self.exp_phi1[:, 1] = np.exp(-self.yhat_zeppelin)
        return self.exp_phi1

    def estimate_signal(self, x_fe):
        x, fe = self.x_fe_to_x_and_fe(x_fe)
        x1, x2 = self.x_to_xs(x)
        S = fe[0] * self.S1_slow(x1) + fe[1] * self.S2_slow(x2)
        + fe[2] * self.S3() + fe[3] * self.S4()
        return S
