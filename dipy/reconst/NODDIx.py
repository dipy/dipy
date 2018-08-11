from dipy.reconst.base import ReconstModel as model
import numpy as np
import cvxpy as cvx
from dipy.reconst.multi_voxel import multi_voxel_fit
import dipy.reconst.noddi_speed as noddixspeed
from scipy.optimize import least_squares
from scipy.optimize import differential_evolution
from dipy.utils.optpkg import optional_package
from scipy import special

cvxpy, have_cvxpy, _ = optional_package("cvxpy")

gamma = 2.675987 * 10 ** 8  # gyromagnetic ratio for Hydrogen
D_intra = 1.7 * 10 ** 3  # intrinsic free diffusivity
D_iso = 3 * 10 ** 3  # isotropic diffusivity


class NODDIxModel(model):
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
    ReconstModel of DIPY
    Returns the signal with the following 11 parameters of the model

    Parameters
    ----------
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

    References
    ----------
    .. [1] Farooq, Hamza, et al. "Microstructure Imaging of Crossing (MIX)
           White Matter Fibers from diffusion MRI." Scientific reports 6
           (2016).

    Notes
    -----
    The implementation of NODDIx may require CVXPY (http://www.cvxpy.org/).
    """

    def __init__(self, gtab, params, fit_method='MIX'):
        # The maximum number of generations, genetic algorithm 1000 default, 1
        self.maxiter = 100
        # Tolerance for termination, nonlinear least square 1e-8 default, 1e-3
        self.xtol = 1e-8
        self.gtab = gtab
        self.big_delta = gtab.big_delta
        self.small_delta = gtab.small_delta
        self.gamma = gamma
        self.G = params[:, 3] / 10 ** 6  # gradient strength (Tesla/micrometer)
        self.yhat_ball = D_iso * self.gtab.bvals
        self.L = self.gtab.bvals * D_intra
        self.phi_inv = np.zeros((4, 4))
        self.yhat_zeppelin = np.zeros(self.gtab.bvals.shape[0])
        self.yhat_cylinder = np.zeros(self.gtab.bvals.shape[0])
        self.yhat_dot = np.zeros(self.gtab.bvals.shape)
        self.exp_phi1 = np.zeros((self.gtab.bvals.shape[0], 5))
        self.exp_phi1[:, 4] = np.exp(-self.yhat_ball)

    @multi_voxel_fit
    def fit(self, data):
        r""" Fit method of the NODDIx model class

        data : array
        The measured signal from one voxel.

        """
        bounds = [(0.011, 0.98), (0.011, np.pi), (0.011, np.pi), (0.11, 1),
                  (0.011, 0.98), (0.011, np.pi), (0.011, np.pi), (0.11, 1)]

        diff_res = differential_evolution(self.stoc_search_cost, bounds,
                                          maxiter=self.maxiter, args=(data,),
                                          tol=0.001, seed=200,
                                          mutation=(0, 1.05),
                                          strategy='best1bin',
                                          disp=False, polish=True, popsize=14)

        # Step 1: store the results of the differential evolution in x
        x = diff_res.x
        phi = self.Phi(x)
        # Step 2: perform convex optimization
        f = self.cvx_fit(data, phi)
        # Combine all 10 parameters of the model into a single array
        x_f = self.x_and_f_to_x_f(x, f)

        bounds = ([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                   0.01], [0.9, 0.9, 0.9, 0.9, 0.9, 0.99, np.pi, np.pi, 0.99,
                           np.pi, np.pi])
        res = least_squares(self.nlls_cost, x_f, xtol=self.xtol, args=(data,))
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
        f0 = f[0]: Volume Fraction of Intra-Cellular Region 1
        f1 = f[1]: Volume Fraction of Extra-Cellular Region 1
        f2 = f[2]: Volume Fraction of Intra-Cellular Region 2
        f3 = f[3]: Volume Fraction of Extra-Cellular Region 2
        f4 = f[4]: Volume Fraction for region containing CSF
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

    def Phi(self, x):
        """
        Constructs the Signal from the intracellular and extracellular compart-
        ments for the Differential Evolution and Variable Separation.
        """
        self.exp_phi1[:, 0] = self.S_ic1(x)
        self.exp_phi1[:, 1] = self.S_ec1(x)
        self.exp_phi1[:, 2] = self.S_ic2(x)
        self.exp_phi1[:, 3] = self.S_ec2(x)
        return self.exp_phi1

    def Phi2(self, x_f):
        """
        Constructs the Signal from the intracellular and extracellular compart-
        ments: Convex Fitting + NLLS - LM method.
        """
        x, f = self.x_f_to_x_and_f(x_f)
        self.exp_phi1[:, 0] = self.S_ic1(x)
        self.exp_phi1[:, 1] = self.S_ec1_new(x, f)
        self.exp_phi1[:, 2] = self.S_ic2_new(x)
        self.exp_phi1[:, 3] = self.S_ec2_new(x, f)
        return self.exp_phi1

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

    def S_ic2(self, x):
        """
        We extend the NODDI model as presented in [2]_ for two fiber
        orientations. Therefore we have 2 intracellular and extracellular
        components to account for this.

        S_ic2 corresponds to the second intracellular component in the NODDIx
        model

        (see Supplimentary note from 6: [1]_ for a comparison and a thorough
        discussion)
        ----------
        References
        ----------
        .. [1] Farooq, Hamza, et al. "Microstructure Imaging of Crossing (MIX)
               White Matter Fibers from diffusion MRI." Scientific reports 6
               (2016).
        .. [2] Zhang, H. et. al. NeuroImage NODDI : Practical in vivo neurite
               orientation dispersion and density imaging of the human brain.
               NeuroImage, 61(4), 1000–1016.
        """
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
        """
        We extend the NODDI model as presented in [2] for two fiber
        orientations. Therefore we have 2 extracellular and extracellular
        components to account for this.

        S_ic2 corresponds to the second extracellular component in the NODDIx
        model

        (see Supplimentary note 6: [1]_ for a comparison and a thorough
        discussion)
        ----------
        References
        ----------
        .. [1] Farooq, Hamza, et al. "Microstructure Imaging of Crossing (MIX)
               White Matter Fibers from diffusion MRI." Scientific reports 6
               (2016).
        .. [2] Zhang, H. et. al. NeuroImage NODDI : Practical in vivo neurite
               orientation dispersion and density imaging of the human brain.
               NeuroImage, 61(4), 1000–1016.
        """
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
        """
        This function is used in the second step of the MIX framework to
        construct the Phi when the data is fitted using the Differential
        Evolution. It is used to calculate the cost for non-linear least
        squares.

        Computes the extracellular component for the first fiber.

        Refer to the nlls_cost() function.
        """
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
        """
        This function is used in the second step of the MIX framework to
        construct the Phi when the data is fitted using the Differential
        Evolution. It is used to calculate the cost for non-linear least
        squares.

        Computes the extracellular component for the second fiber.

        Refer to the nlls_cost() function.
        """
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
        """
        This function is used in the second step of the MIX framework to
        construct the Phi when the data is fitted using the Differential
        Evolution. It is used to calculate the cost for non-linear least
        squares.

        Computes the intracellular component for the second fiber.

        Refer to the nlls_cost() function.
        """
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
        """
        Substrate: Impermeable cylinders with one radius in an empty background
        Orientation distribution: Watson's distribution with SH approximation
        Pulse sequence: Pulsed gradient spin echo
        Signal approximation: Gaussian phase distribution.

        This returns the measurements E according to the model and the Jacobian
        J of the measurements with respect to the parameters.  The Jacobian
        does not include derivates with respect to the fibre direction.

        x is the list of model parameters in SI units:
            x(1) is the diffusivity of the material inside the cylinders.
            x(2) is the radius of the cylinders.
            x(3) is the concentration parameter of the Watson's distribution
        fibredir is a unit vector along the symmetry axis of the Watson's
        distribution.  It must be in Cartesian coordinates [x y z]' with size
        [3 1]. [1]_

        References
        ----------
        .. [1] Zhang, H. et. al. NeuroImage NODDI : Practical in vivo neurite
               orientation dispersion and density imaging of the human brain.
               NeuroImage, 61(4), 1000–1016.
        """
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

        # The Legendre Gauss Integran is computed from Cython
        # Please Refere: noddi_speed.pyx
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
        E[np.isnan(E)] = 0.1
        E[E <= 0] = min(E[E > 0]) * 0.1
        E = 0.5 * E * ePerp
        return E

    def CylNeumanLePar_PGSE(self, d):
        r"""
        Substrate: Parallel, impermeable cylinders with one radius in an empty
        background.
        Pulse sequence: Pulsed gradient spin echo
        Signal approximation: Gaussian phase distribution.

        This function returns the log signal attenuation in parallel direction
        (LePar) according to the Neuman model and the Jacobian J of LePar with
        respect to the parameters.  The Jacobian does not include derivates
        with respect to the fibre direction.

        d is the diffusivity of the material inside the cylinders.

        G, delta and smalldel are the gradient strength, pulse separation and
        pulse length of each measurement in the protocol. [1]_

        References
        ----------
        .. [1] Zhang, H. et. al. NeuroImage NODDI : Practical in vivo neurite
               orientation dispersion and density imaging of the human brain.
               NeuroImage, 61(4), 1000–1016.
        """
        # Radial wavenumbers
        modQ = gamma * self.small_delta * self.G
        modQ_Sq = modQ ** 2
        # Diffusion time for PGSE, in a matrix for the computation below.
        difftime = (self.big_delta - self.small_delta / 3)
        # Parallel component
        LE = -modQ_Sq * difftime * d
        return LE

    def SynthMeasWatsonHinderedDiffusion_PGSE(self, x, fibredir):
        """
        Substrate: Anisotropic hindered diffusion compartment
        Orientation distribution: Watson's distribution
        Pulse sequence: Pulsed gradient spin echo
        Signal approximation: N/A
        returns the measurements E according to the model and the Jacobian J of
        the measurements with respect to the parameters.  The Jacobian does not
        include derivates with respect to the fibre direction.

        x is the list of model parameters in SI units:
        x(0) is the free diffusivity of the material inside and outside the
        cylinders.
        x(1): is the hindered diffusivity outside the cylinders in
              perpendicular directions.
        x(2) is the concentration parameter of the Watson's distribution

        fibredir is a unit vector along the symmetry axis of the Watson's
        distribution. [1]_

        References
        ----------
        .. [1] Zhang, H. et. al. NeuroImage NODDI : Practical in vivo neurite
               orientation dispersion and density imaging of the human brain.
               NeuroImage, 61(4), 1000–1016.
        """
        dPar = x[0]
        dPerp = x[1]
        kappa = x[2]
        dw = self.WatsonHinderedDiffusionCoeff(dPar, dPerp, kappa)
        xh = [dw[0], dw[1]]
        E = self.SynthMeasHinderedDiffusion_PGSE(xh, fibredir)
        return E

    def WatsonHinderedDiffusionCoeff(self, dPar, dPerp, kappa):
        """
        Substrate: Anisotropic hindered diffusion compartment
        Orientation distribution: Watson's distribution
        WatsonHinderedDiffusionCoeff(dPar, dPerp, kappa)
        returns the equivalent parallel and perpendicular diffusion
        coefficients for hindered compartment with impermeable cylinder's
        oriented with a Watson's distribution with a cocentration parameter of
        kappa.

        dPar is the free diffusivity of the material inside and outside the
        cylinders.
        dPerp is the hindered diffusivity outside the cylinders in
        perpendicular directions.
        kappa is the concentration parameter of the Watson's distribution. [1]_

        References
        ----------
        .. [1] Zhang, H. et. al. NeuroImage NODDI : Practical in vivo neurite
               orientation dispersion and density imaging of the human brain.
               NeuroImage, 61(4), 1000–1016.
        """
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
        """
        Substrate: Anisotropic hindered diffusion compartment
        Pulse sequence: Pulsed gradient spin echo
        Signal approximation: N/A

        This function returns the measurements E according to the model and the
        Jacobian J of the measurements with respect to the parameters. The
        Jacobian does not include derivates with respect to the fibre
        direction.

        x is the list of model parameters in SI units:
        x(0): is the free diffusivity of the material inside and outside the
             cylinders.
        x(1): is the hindered diffusivity outside the cylinders in
              perpendicular directions.

        fibredir is a unit vector along the cylinder axis. [1]_

        References
        ----------
        .. [1] Zhang, H. et. al. NeuroImage NODDI : Practical in vivo neurite
               orientation dispersion and density imaging of the human brain.
               NeuroImage, 61(4), 1000–1016.
        """
        dPar = x[0]
        dPerp = x[1]
        # Angles between gradient directions and fibre direction.
        cosTheta = np.dot(self.gtab.bvecs, fibredir)
        cosThetaSq = cosTheta ** 2
        # Find hindered signals
        E = np.exp(- self.gtab.bvals * ((dPar - dPerp) * cosThetaSq + dPerp))
        return E

    def x_f_to_x_and_f(self, x_f):
        """
        The MIX framework makes use of Variable Projections (VarPro) to
        separately fit the Volume Fractions and the other parameters that
        involve exponential functions.

        This function performs this task of taking the 11 input parameters of
        the signal and creates 2 separate lists:
            f: Volume  Fractions
            x: Other Signal Params [1]_

        References
        ----------
        .. [1] Farooq, Hamza, et al. "Microstructure Imaging of Crossing (MIX)
               White Matter Fibers from diffusion MRI." Scientific reports 6
               (2016).
        """
        f = np.zeros((1, 5))
        f = x_f[0:5]
        x = x_f[5:12]
        return x, f

    def x_and_f_to_x_f(self, x, f):
        """
        The MIX framework makes use of Variable Projections (VarPro) to
        separately fit the Volume Fractions and the other parameters that
        involve exponential functions.

        This function performs this task of taking the 11 input parameters of
        the signal and creates 2 separate lists:
            f: Volume  Fractions
            x: Other Signal Params [1]_

        References
        ----------
        .. [1] Farooq, Hamza, et al. "Microstructure Imaging of Crossing (MIX)
               White Matter Fibers from diffusion MRI." Scientific reports 6
               (2016).
        """
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
