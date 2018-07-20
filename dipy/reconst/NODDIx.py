from dipy.reconst.base import ReconstModel
import numpy as np
import cvxpy as cvx
import dipy.reconst.noddi_speed as noddixspeed
from scipy.optimize import least_squares
from scipy.optimize import differential_evolution
from scipy import special

r"""

gamma: gyromagnetic ratio
D_intra= intrinsic free diffusivity
D_iso= isotropic diffusivity
"""
gamma = 2.675987 * 10 ** 8
D_intra = 1.7 * 10 ** 3  # (mircometer^2/sec for in vivo human)
D_iso = 3 * 10 ** 3


class NODDIxModel(ReconstModel):
    r""" MIX framework (MIX) [1]_.
    The MIX computes the NODDIx parameters. NODDIx is a multi
    compartment model, (sum of exponentials).
    This algorithm uses three different optimizers. It starts with a
    differential evolution algorithm and fits the parameters in the
    power of exponentials. Then the fitted parameters in the first step are
    utilized to make a linear convex problem. Using a convex optimization,
    the volume fractions are determined. The last step of this algorithm
    is non linear least square fitting on all the parameters.
    The results of the first and second step are utilized as the initial
    values for the last step of the algorithm.
    (see [1]_ for a comparison and a thorough discussion).
    Parameters
    ----------
    gtab : GradientTable
    fit_method : str or callable
    Returns  the 11 parameters of the model
    -------
    References
    ----------
    .. [1] Farooq, Hamza, et al. "Microstructure Imaging of Crossing (MIX)
           White Matter Fibers from diffusion MRI." Scientific reports 6
           (2016).
    """

    def __init__(self, gtab, params, fit_method='MIX'):
        # The maximum number of generations, genetic algorithm 1000 default, 1
        self.maxiter = 1000
        # Tolerance for termination, nonlinear least square 1e-8 default, 1e-3
        self.xtol = 1e-8
        self.gtab = gtab
        self.big_delta = gtab.big_delta
        self.small_delta = gtab.small_delta
        self.gamma = gamma
        self.G = params[:, 3] / 10 ** 6  # gradient strength (Tesla/micrometer)
        self.G2 = self.G ** 2
        self.yhat_ball = D_iso * self.gtab.bvals
        self.L = self.gtab.bvals * D_intra
        self.phi_inv = np.zeros((4, 4))
        self.yhat_zeppelin = np.zeros(self.small_delta.shape[0])
        self.yhat_cylinder = np.zeros(self.small_delta.shape[0])
        self.yhat_dot = np.zeros(self.gtab.bvals.shape)
        self.exp_phi1 = np.zeros((self.small_delta.shape[0], 5))
        self.exp_phi1[:, 4] = np.exp(-self.yhat_ball)

    def fit(self, data):
        """ Fit method of the NODDIx model class
        Parameters
        ----------
        The 11 parameters that the model outputs after fitting are:
        Volume Fraction 1 - Intracellular 1
        Volume Fraction 2 - Intracellular 2
        Volume Fraction 3 - Extracellular 1
        Volume Fraction 4 - Extracellular 2
        Volume Fraction 5 - CSF: Isotropic
        Orientation Dispersion 1
        Theta 1
        Phi 1
        Orientation Dispersion 2
        Theta 2
        Phi 2
        ----------
        data : array
        The measured signal from one voxel.
        """
        bounds = [(0.011, 0.98), (0.011, np.pi), (0.011, np.pi), (0.11, 1),
                  (0.011, 0.98), (0.011, np.pi), (0.011, np.pi), (0.11, 1)]
        # can we limit this..
        res_one = differential_evolution(self.stoc_search_cost, bounds,
                                         maxiter=self.maxiter, args=(data,),
                                         tol=0.001, seed=200,
                                         mutation=(0.0, 1.05),
                                         disp=True, polish=True, popsize=5,
                                         init='latinhypercube')
        x = res_one.x
        phi = self.Phi(x)
        f = self.cvx_fit(data, phi)
        x_f = self.x_and_f_to_x_f(x, f)

        bounds = ([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                   0.01], [0.9, 0.9, 0.9, 0.9, 0.9, 0.99, np.pi, np.pi, 0.99,
                           np.pi, np.pi])
        res = least_squares(self.nlls_cost, x_f, bounds=(bounds),
                            xtol=self.xtol, args=(data,))
        result = res.x
        return result

    def stoc_search_cost(self, x, signal):
        """
        Cost function for the differential evolution
        Calls another function described by:
            differential_evol_cost
        Returns
        -------
        (signal -  S)^T(signal -  S)
        Notes
        --------
        cost function for differential evolution algorithm:
        .. math::
            (signal -  S)^T(signal -  S)
        """
        phi = self.Phi(x)
        return self.differential_evol_cost(phi, signal)

    def differential_evol_cost(self, phi, signal):

        """
        To make the cost function for differential evolution algorithm
        """
        #  moore-penrose inverse
        phi_mp = np.dot(np.linalg.inv(np.dot(phi.T, phi)), phi.T)
        #  sigma
        f = np.dot(phi_mp, signal)
        yhat = np.dot(phi, f)
        return np.dot((signal - yhat).T, signal - yhat)

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
        f0, f1, f2, f3, f4 (volume fractions)
        f0 = f[0]
        f1 = f[1]
        f2 = f[2]
        f3 = f[3]
        f4 = f[4]
        Notes
        --------
        cost function for genetic algorithm:
        .. math::
            minimize(norm((signal)- (phi*f)))
        """

        # Create 5 scalar optimization variables.
        f = cvx.Variable(5)
        # Create four constraints.
        constraints = [cvx.sum_entries(f) == 1,
                       f[0] >= 0.011,
                       f[1] >= 0.011,
                       f[2] >= 0.011,
                       f[3] >= 0.011,
                       f[4] >= 0.011,
                       f[0] <= 0.89,
                       f[1] <= 0.89,
                       f[2] <= 0.89,
                       f[3] <= 0.89,
                       f[4] <= 0.89]

        # Form objective.
        obj = cvx.Minimize(cvx.sum_entries(cvx.square(phi * f - signal)))

        # Form and solve problem.
        prob = cvx.Problem(obj, constraints)
        # Returns the optimal value
        prob.solve()
        return np.array(f.value)

    def nlls_cost(self, x_f, signal):
        """
        cost function for the least square problem
        Parameters
        ----------
        x_f : array
            x_f(0) x_f(1) x_f(2) x_f(3) x_f(4) are f1 f2 f3 f4 f5(volfractions)
            x_f(5) Orintation Dispersion 1
            x_f(6) Theta1
            x_f(7) Phi1
            x_f(8) Orintation Dispersion 2
            x_f(9) Theta2
            x_f(10) Phi2
        Returns
        -------
        sum{(signal -  phi*f)^2}
        Notes
        --------
        cost function for the least square problem
        .. math::
            sum{(signal -  phi*f)^2}
        """
        x, f = self.x_f_to_x_and_f(x_f)
        phi = self.Phi2(x_f)
        return np.sum((np.dot(phi, f) - signal) ** 2)

    def S_ic1(self, x):
        """
        This function models the intracellular component.
        The intra-cellular compartment refrs to the space bounded by the
        membrane of neurites. We model this space as a set of sticks, i.e.,
        cylinders of zero radius, to capture the highly restricted nature of
        diffusion perpendicular to neurites and unhindered diffusion along
        them. (see [2]_ for a comparison and a thorough discussion)

        ----------
        References
        ----------
        .. [2] Zhang, H. et. al. NeuroImage NODDI : Practical in vivo neurite
               orientation dispersion and density imaging of the human brain.
               NeuroImage, 61(4), 1000–1016.

        """
        OD1 = x[0]
        sinT1 = np.sin(x[1])
        cosT1 = np.cos(x[1])
        sinP1 = np.sin(x[2])
        cosP1 = np.cos(x[2])
        n1 = [cosP1*sinT1, sinP1*sinT1, cosT1]
        kappa1 = 1/np.tan(OD1*np.pi/2)
        x1 = [D_intra, 0, kappa1]
        signal_ic1 = self.SynthMeasWatsonSHCylNeuman_PGSE(x1, n1)
        return signal_ic1

    def S_ec1(self, x):
        """
        This function models the extracellular component.
        The extra-cellular compartment refers to the space around the
        neurites, which is occupied by various types of glial cells and,
        additionally in gray matter, cell bodies (somas). In this space, the
        diffusion of water molecules is hindered by the presence of neurites
        but not restricted, hence is modeled with simple (Gaussian)
        anisotropic diffusion.
        (see [2]_ for a comparison and a thorough discussion)
        ----------
        References
        ----------
        .. [2] Zhang, H. et. al. NeuroImage NODDI : Practical in vivo neurite
               orientation dispersion and density imaging of the human brain.
               NeuroImage, 61(4), 1000–1016.
        """
        OD1 = x[0]
        sinT1 = np.sin(x[1])
        cosT1 = np.cos(x[1])
        sinP1 = np.sin(x[2])
        cosP1 = np.cos(x[2])
        v_ic1 = x[3]
        n1 = [cosP1*sinT1, sinP1*sinT1, cosT1]
        kappa1 = 1/np.tan(OD1*np.pi/2)
        d_perp = D_intra * (1 - v_ic1)
        signal_ec1 = self.SynthMeasWatsonHinderedDiffusion_PGSE([D_intra,
                                                                 d_perp,
                                                                 kappa1], n1)
        return signal_ec1
        """
        We extend the NODDI model as presented in [2] for two fiber
        orientations. Therefore we have 2 intracellular and extracellular
        components to account for this.
        (see Supplimentary note 6: [1]_ for a comparison and a thorough
        discussion)
        ----------
        References
        ----------
        .. [1] Farooq, Hamza, et al. "Microstructure Imaging of Crossing (MIX)
               White Matter Fibers from diffusion MRI." Scientific reports 6
               (2016).
        """

    def S_ic2(self, x):
        OD2 = x[4]
        sinT2 = np.sin(x[5])
        cosT2 = np.cos(x[5])
        sinP2 = np.sin(x[6])
        cosP2 = np.cos(x[6])
        n2 = [cosP2 * sinT2, sinP2 * sinT2, cosT2]
        kappa2 = 1/np.tan(OD2*np.pi/2)
        x2 = [D_intra, 0, kappa2]
        signal_ic2 = self.SynthMeasWatsonSHCylNeuman_PGSE(x2, n2)
        return signal_ic2

    def S_ec2(self, x):
        OD2 = x[4]
        sinT2 = np.sin(x[5])
        cosT2 = np.cos(x[5])
        sinP2 = np.sin(x[6])
        cosP2 = np.cos(x[6])
        n2 = [cosP2 * sinT2, sinP2 * sinT2, cosT2]
        v_ic2 = x[7]
        d_perp2 = D_intra * (1 - v_ic2)
        kappa2 = 1/np.tan(OD2*np.pi/2)
        signal_ec2 = self.SynthMeasWatsonHinderedDiffusion_PGSE([D_intra,
                                                                 d_perp2,
                                                                 kappa2], n2)
        return signal_ec2

    def S_ec1_new(self, x, f):
        OD1 = x[0]
        sinT1 = np.sin(x[1])
        cosT1 = np.cos(x[1])
        sinP1 = np.sin(x[2])
        cosP1 = np.cos(x[2])
        v_ic1 = f[0]
        n1 = [cosP1*sinT1, sinP1*sinT1, cosT1]
        kappa1 = 1/np.tan(OD1*np.pi/2)
        d_perp = D_intra * (1 - v_ic1)
        signal_ec1 = self.SynthMeasWatsonHinderedDiffusion_PGSE([D_intra,
                                                                 d_perp,
                                                                 kappa1], n1)
        return signal_ec1

    def S_ec2_new(self, x, f):
        OD2 = x[3]
        sinT2 = np.sin(x[4])
        cosT2 = np.cos(x[4])
        sinP2 = np.sin(x[5])
        cosP2 = np.cos(x[5])
        n2 = [cosP2 * sinT2, sinP2 * sinT2, cosT2]
        v_ic2 = f[2]
        d_perp2 = D_intra * (1 - v_ic2)
        kappa2 = 1/np.tan(OD2*np.pi/2)
        signal_ec2 = self.SynthMeasWatsonHinderedDiffusion_PGSE([D_intra,
                                                                 d_perp2,
                                                                 kappa2], n2)
        return signal_ec2

    def S_ic2_new(self, x):
        OD2 = x[3]
        sinT2 = np.sin(x[4])
        cosT2 = np.cos(x[4])
        sinP2 = np.sin(x[5])
        cosP2 = np.cos(x[5])
        n2 = [cosP2 * sinT2, sinP2 * sinT2, cosT2]
        kappa2 = 1/np.tan(OD2*np.pi/2)
        x2 = [D_intra, 0, kappa2]
        signal_ic2 = self.SynthMeasWatsonSHCylNeuman_PGSE(x2, n2)
        return signal_ic2

    def SynthMeasWatsonSHCylNeuman_PGSE(self, x, fiberdir):
        d = x[0]
        kappa = x[2]

        l_q = self.gtab.bvecs.shape[0]

        # parallel component
        LePar = self.CylNeumanLePar_PGSE(d)

        # Perpendicular component
        LePerp = np.zeros((self.G.shape[0]))
        ePerp = np.exp(LePerp)

        # Compute the Legendre weighted signal
        Lpmp = LePerp - LePar
        lgi = noddixspeed.legendre_gauss_integral(Lpmp, 6)

        # Compute the SH coefficients of the Watson's distribution
        coeff = noddixspeed.watson_sh_coeff(kappa)
        coeffMatrix = np.tile(coeff, [l_q, 1])

        cosTheta = np.dot(self.gtab.bvecs, fiberdir)
        badCosTheta = np.where(abs(cosTheta) > 1)
        cosTheta[badCosTheta] = \
            cosTheta[badCosTheta] / abs(cosTheta[badCosTheta])

        # Compute the SH values at cosTheta
        sh = np.zeros(coeff.shape[0])
        shMatrix = np.tile(sh, [l_q, 1])

        # Computes a for loop for the Legendre matrix and evaulates the
        # Legendre Integral at a Point : Cython Code
        noddixspeed.synthMeasSHFor(cosTheta, shMatrix)
        E = np.sum(lgi * coeffMatrix * shMatrix, 1)
        E[E <= 0] = min(E[E > 0]) * 0.1
        E = 0.5 * E * ePerp
        return E

    def CylNeumanLePar_PGSE(self, d):
        # Radial wavenumbers
        modQ = gamma * self.small_delta * self.G
        modQ_Sq = modQ ** 2
        # Diffusion time for PGSE, in a matrix for the computation below.
        difftime = (self.big_delta - self.small_delta / 3)
        # Parallel component
        LE = -modQ_Sq * difftime * d
        return LE

    def SynthMeasWatsonHinderedDiffusion_PGSE(self, x, fibredir):
        dPar = x[0]
        dPerp = x[1]
        kappa = x[2]
        dw = self.WatsonHinderedDiffusionCoeff(dPar, dPerp, kappa)
        xh = [dw[0], dw[1]]
        E = self.SynthMeasHinderedDiffusion_PGSE(xh, fibredir)
        return E

    def WatsonHinderedDiffusionCoeff(self, dPar, dPerp, kappa):
        dw = np.zeros((2, 1))
        dParMdPerp = dPar - dPerp

        if kappa < 1e-5:
            dParP2dPerp = dPar + 2 * dPerp
            k2 = kappa * kappa
            dw[0] = dParP2dPerp / 3 + 4 * dParMdPerp * kappa / 45 \
                + 8 * dParMdPerp * k2 / 945
            dw[1] = dParP2dPerp / 3 - 2 * dParMdPerp * kappa / 45 \
                - 4 * dParMdPerp * k2 / 945
        else:
            sk = np.sqrt(kappa)
            dawsonf = 0.5 * np.exp(-kappa) * np.sqrt(np.pi) * special.erfi(sk)
            factor = sk / dawsonf
            dw[0] = (-dParMdPerp + 2 * dPerp * kappa +
                     dParMdPerp * factor) / (2*kappa)
            dw[1] = (dParMdPerp + 2 * (dPar + dPerp) * kappa -
                     dParMdPerp * factor) / (4 * kappa)
        return dw

    def SynthMeasHinderedDiffusion_PGSE(self, x, fibredir):
        dPar = x[0]
        dPerp = x[1]
        # Angles between gradient directions and fibre direction.
        cosTheta = np.dot(self.gtab.bvecs, fibredir)
        cosThetaSq = cosTheta ** 2
        # Find hindered signals
        E = np.exp(- self.gtab.bvals * ((dPar - dPerp) * cosThetaSq + dPerp))
        return E

    def x_f_to_x_and_f(self, x_f):
        f = np.zeros((1, 5))
        f = x_f[0:5]
        x = x_f[5:12]
        return x, f

    def x_and_f_to_x_f(self, x, f):
        x_f = np.zeros(11)
        f = np.squeeze(f)
        f11ga = x[3]
        f12ga = x[7]
        x_f[0] = (f[0] + f11ga) / 2
        x_f[1] = f[1]
        x_f[2] = (f[2] + f12ga) / 2
        x_f[3:5] = f[3:5]
        x_f[5:8] = x[0:3]
        x_f[8:11] = x[4:7]
        return x_f

    def Phi(self, x):
        self.exp_phi1[:, 0] = self.S_ic1(x)
        self.exp_phi1[:, 1] = self.S_ec1(x)
        self.exp_phi1[:, 2] = self.S_ic2(x)
        self.exp_phi1[:, 3] = self.S_ec2(x)
        return self.exp_phi1

    def Phi2(self, x_f):
        x, f = self.x_f_to_x_and_f(x_f)
        self.exp_phi1[:, 0] = self.S_ic1(x)
        self.exp_phi1[:, 1] = self.S_ec1_new(x, f)
        self.exp_phi1[:, 2] = self.S_ic2_new(x)
        self.exp_phi1[:, 3] = self.S_ec2_new(x, f)
        return self.exp_phi1

    def estimate_signal(self, x_f):
        x, f = self.x_f_to_x_and_f(x_f)
        x1, x2 = self.x_to_xs(x)
        S = f[0] * self.S1_slow(x1) + f[1] * self.S2_slow(x2)
        + f[2] * self.S3() + f[3] * self.S4()
        return S
