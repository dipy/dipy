""" Classes and functions for fitting ivim model """
import warnings

import numpy as np
from scipy.optimize import least_squares, differential_evolution

from dipy.reconst.base import ReconstModel
from dipy.reconst.multi_voxel import multi_voxel_fit
from dipy.utils.optpkg import optional_package

cvxpy, have_cvxpy, _ = optional_package("cvxpy", min_version="1.4.1")

# global variable for bounding least_squares in both models
BOUNDS = ([0., 0., 0., 0.], [np.inf, .2, 1., 1.])


def ivim_prediction(params, gtab):
    """The Intravoxel incoherent motion (IVIM) model function.

    Parameters
    ----------
    params : array
        An array of IVIM parameters - [S0, f, D_star, D].

    gtab : GradientTable class instance
        Gradient directions and bvalues.

    S0 : float, optional
        This has been added just for consistency with the existing
        API. Unlike other models, IVIM predicts S0 and this is over written
        by the S0 value in params.

    Returns
    -------
    S : array
        An array containing the IVIM signal estimated using given parameters.

    """
    b = gtab.bvals
    S0, f, D_star, D = params

    S = S0 * (f * np.exp(-b * D_star) + (1 - f) * np.exp(-b * D))

    return S


def _ivim_error(params, gtab, signal):
    """Error function to be used in fitting the IVIM model.

    Parameters
    ----------
    params : array
        An array of IVIM parameters - [S0, f, D_star, D]

    gtab : GradientTable class instance
        Gradient directions and bvalues.

    signal : array
        Array containing the actual signal values.

    Returns
    -------
    residual : array
        An array containing the difference between actual and estimated signal.

    """
    residual = signal - ivim_prediction(params, gtab)
    return residual


def f_D_star_prediction(params, gtab, S0, D):
    """Function used to predict IVIM signal when S0 and D are known
    by considering f and D_star as the unknown parameters.

    Parameters
    ----------
    params : array
        The value of f and D_star.

    gtab : GradientTable class instance
        Gradient directions and bvalues.

    S0 : float
        The parameters S0 obtained from a linear fit.

    D : float
        The parameters D obtained from a linear fit.

    Returns
    -------
    S : array
        An array containing the IVIM signal estimated using given parameters.

    """
    f, D_star = params
    b = gtab.bvals
    S = S0 * (f * np.exp(-b * D_star) + (1 - f) * np.exp(-b * D))
    return S


def f_D_star_error(params, gtab, signal, S0, D):
    """Error function used to fit f and D_star keeping S0 and D fixed

    Parameters
    ----------
    params : array
        The value of f and D_star.

    gtab : GradientTable class instance
        Gradient directions and bvalues.

    signal : array
        Array containing the actual signal values.

    S0 : float
        The parameters S0 obtained from a linear fit.

    D : float
        The parameters D obtained from a linear fit.

    Returns
    -------
    residual : array
        An array containing the difference of actual and estimated signal.

    """
    f, D_star = params
    return signal - f_D_star_prediction([f, D_star], gtab, S0, D)


def ivim_model_selector(gtab, fit_method='trr', **kwargs):
    """
    Selector function to switch between the 2-stage Trust-Region Reflective
    based NLLS fitting method (also containing the linear fit): `trr` and the
    Variable Projections based fitting method: `varpro`.

    Parameters
    ----------
    fit_method : string, optional
        The value fit_method can either be 'trr' or 'varpro'.
        default : trr

    """
    bounds_warning = 'Bounds for this fit have been set from experiments '
    bounds_warning += 'and literature survey. To change the bounds, please '
    bounds_warning += 'input your bounds in model definition...'

    if fit_method.lower() == 'trr':
        ivimmodel_trr = IvimModelTRR(gtab, **kwargs)
        if 'bounds' not in kwargs:
            warnings.warn(bounds_warning, UserWarning)
        return ivimmodel_trr

    elif fit_method.lower() == 'varpro':
        ivimmodel_vp = IvimModelVP(gtab, **kwargs)
        if 'bounds' not in kwargs:
            warnings.warn(bounds_warning, UserWarning)
        return ivimmodel_vp

    else:
        opt_msg = 'The fit_method option chosen was not correct. '
        opt_msg += 'Using fit_method: TRR instead...'
        warnings.warn(opt_msg, UserWarning)
        return IvimModelTRR(gtab, **kwargs)


IvimModel = ivim_model_selector


class IvimModelTRR(ReconstModel):
    """Ivim model
    """
    def __init__(self, gtab, split_b_D=400.0, split_b_S0=200., bounds=None,
                 two_stage=True, tol=1e-15,
                 x_scale=(1000., 0.1, 0.001, 0.0001),
                 gtol=1e-15, ftol=1e-15, eps=1e-15, maxiter=1000):

        r"""
        Initialize an IVIM model.

        The IVIM model assumes that biological tissue includes a volume
        fraction 'f' of water flowing with a pseudo-diffusion coefficient
        D* and a fraction (1-f) of static (diffusion only), intra and
        extracellular water, with a diffusion coefficient D. In this model
        the echo attenuation of a signal in a single voxel can be written as

            .. math::

            S(b) = S_0[f*e^{(-b*D\*)} + (1-f)e^{(-b*D)}]

            Where:
            .. math::

            S_0, f, D\* and D are the IVIM parameters.

        Parameters
        ----------
        gtab : GradientTable class instance
            Gradient directions and bvalues

        split_b_D : float, optional
            The b-value to split the data on for two-stage fit. This will be
            used while estimating the value of D. The assumption is that at
            higher b values the effects of perfusion is less and hence the
            signal can be approximated as a mono-exponential decay.
            default : 400.

        split_b_S0 : float, optional
            The b-value to split the data on for two-stage fit for estimation
            of S0 and initial guess for D_star. The assumption here is that
            at low bvalues the effects of perfusion are more.
            default : 200.

        bounds : tuple of arrays with 4 elements, optional
            Bounds to constrain the fitted model parameters. This is only
            supported for Scipy version > 0.17. When using a older Scipy
            version, this function will raise an error if bounds are different
            from None. This parameter is also used to fill nan values for out
            of bounds parameters in the `IvimFit` class using the method
            fill_na. default : ([0., 0., 0., 0.], [np.inf, .3, 1., 1.])

        two_stage : bool
            Argument to specify whether to perform a non-linear fitting of all
            parameters after the linear fitting by splitting the data based on
            bvalues. This gives more accurate parameters but takes more time.
            The linear fit can be used to get a quick estimation of the
            parameters. default : False

        tol : float, optional
            Tolerance for convergence of minimization.
            default : 1e-15

        x_scale : array-like, optional
            Scaling for the parameters. This is passed to `least_squares` which
            is only available for Scipy version > 0.17.
            default: [1000, 0.01, 0.001, 0.0001]

        gtol : float, optional
            Tolerance for termination by the norm of the gradient.
            default : 1e-15

        ftol : float, optional
            Tolerance for termination by the change of the cost function.
            default : 1e-15

        eps : float, optional
            Step size used for numerical approximation of the jacobian.
            default : 1e-15

        maxiter : int, optional
            Maximum number of iterations to perform.
            default : 1000

        References
        ----------
        .. [1] Le Bihan, Denis, et al. "Separation of diffusion and perfusion
               in intravoxel incoherent motion MR imaging." Radiology 168.2
               (1988): 497-505.
        .. [2] Federau, Christian, et al. "Quantitative measurement of brain
               perfusion with intravoxel incoherent motion MR imaging."
               Radiology 265.3 (2012): 874-881.
        """
        if not np.any(gtab.b0s_mask):
            e_s = "No measured signal at bvalue == 0."
            e_s += "The IVIM model requires signal measured at 0 bvalue"
            raise ValueError(e_s)

        if gtab.b0_threshold > 0:
            b0_s = "The IVIM model requires a measurement at b==0. As of "
            b0_s += "version 0.15, the default b0_threshold for the "
            b0_s += "GradientTable object is set to 50, so if you used the "
            b0_s += "default settings to initialize the gtab input to the "
            b0_s += "IVIM model, you may have provided a gtab with "
            b0_s += "b0_threshold larger than 0. Please initialize the gtab "
            b0_s += "input with b0_threshold=0"
            raise ValueError(b0_s)

        ReconstModel.__init__(self, gtab)
        self.split_b_D = split_b_D
        self.split_b_S0 = split_b_S0
        self.bounds = bounds
        self.two_stage = two_stage
        self.tol = tol
        self.options = {'gtol': gtol, 'ftol': ftol,
                        'eps': eps, 'maxiter': maxiter}
        self.x_scale = x_scale

        self.bounds = bounds or BOUNDS

    @multi_voxel_fit
    def fit(self, data):
        """ Fit method of the IvimModelTRR class.

        The fitting takes place in the following steps: Linear fitting for D
        (bvals > `split_b_D` (default: 400)) and store S0_prime. Another linear
        fit for S0 (bvals < split_b_S0 (default: 200)). Estimate f using
        1 - S0_prime/S0. Use non-linear least squares to fit D_star and f.

        We do a final non-linear fitting of all four parameters and select the
        set of parameters which make sense physically. The criteria for
        selecting a particular set of parameters is checking the
        pseudo-perfusion fraction. If the fraction is more than `f_threshold`
        (default: 25%), we will reject the solution obtained from non-linear
        least squares fitting and consider only the linear fit.


        Parameters
        ----------
        data : array
            The measured signal from one voxel. A multi voxel decorator
            will be applied to this fit method to scale it and apply it
            to multiple voxels.


        Returns
        -------
        IvimFit object
        """
        # Get S0_prime and D - parameters assuming a single exponential decay
        # for signals for bvals greater than `split_b_D`
        S0_prime, D = self.estimate_linear_fit(
            data, self.split_b_D, less_than=False)

        # Get S0 and D_star_prime - parameters assuming a single exponential
        # decay for for signals for bvals greater than `split_b_S0`.

        S0, D_star_prime = self.estimate_linear_fit(data, self.split_b_S0,
                                                    less_than=True)
        # Estimate f
        f_guess = 1 - S0_prime / S0

        # Fit f and D_star using leastsq.
        params_f_D_star = [f_guess, D_star_prime]
        f, D_star = self.estimate_f_D_star(params_f_D_star, data, S0, D)
        params_linear = np.array([S0, f, D_star, D])
        # Fit parameters again if two_stage flag is set.
        if self.two_stage:
            params_two_stage = self._leastsq(data, params_linear)
            bounds_violated = ~(np.all(params_two_stage >= self.bounds[0]) and
                                (np.all(params_two_stage <= self.bounds[1])))
            if bounds_violated:
                warningMsg = "Bounds are violated for leastsq fitting. "
                warningMsg += "Returning parameters from linear fit"
                warnings.warn(warningMsg, UserWarning)
                return IvimFit(self, params_linear)
            else:
                return IvimFit(self, params_two_stage)
        else:
            return IvimFit(self, params_linear)

    def estimate_linear_fit(self, data, split_b, less_than=True):
        """Estimate a linear fit by taking log of data.

        Parameters
        ----------
        data : array
            An array containing the data to be fit

        split_b : float
            The b value to split the data

        less_than : bool
            If True, splitting occurs for bvalues less than split_b

        Returns
        -------
        S0 : float
            The estimated S0 value. (intercept)

        D : float
            The estimated value of D.
        """
        if less_than:
            bvals_split = self.gtab.bvals[self.gtab.bvals <= split_b]
            D, neg_log_S0 = np.polyfit(bvals_split,
                                       -np.log(data[self.gtab.bvals <=
                                                    split_b]), 1)
        else:
            bvals_split = self.gtab.bvals[self.gtab.bvals >= split_b]
            D, neg_log_S0 = np.polyfit(bvals_split,
                                       -np.log(data[self.gtab.bvals >=
                                                    split_b]), 1)

        S0 = np.exp(-neg_log_S0)
        return S0, D

    def estimate_f_D_star(self, params_f_D_star, data, S0, D):
        """Estimate f and D_star using the values of all the other parameters
        obtained from a linear fit.

        Parameters
        ----------
        params_f_D_star: array
            An array containing the value of f and D_star.

        data : array
            Array containing the actual signal values.

        S0 : float
            The parameters S0 obtained from a linear fit.

        D : float
            The parameters D obtained from a linear fit.

        Returns
        -------
        f : float
           Perfusion fraction estimated from the fit.
        D_star :
            The value of D_star estimated from the fit.
        """
        gtol = self.options["gtol"]
        ftol = self.options["ftol"]
        xtol = self.tol
        maxfev = self.options["maxiter"]

        try:
            res = least_squares(f_D_star_error,
                                params_f_D_star,
                                bounds=((0., 0.), (self.bounds[1][1],
                                                   self.bounds[1][2])),
                                args=(self.gtab, data, S0, D),
                                ftol=ftol,
                                xtol=xtol,
                                gtol=gtol,
                                max_nfev=maxfev)
            f, D_star = res.x
            return f, D_star
        except ValueError:
            warningMsg = "x0 obtained from linear fitting is not feasible"
            warningMsg += " as initial guess for leastsq while estimating "
            warningMsg += "f and D_star. Using parameters from the "
            warningMsg += "linear fit."
            warnings.warn(warningMsg, UserWarning)
            f, D_star = params_f_D_star
            return f, D_star

    def predict(self, ivim_params, gtab, S0=1.):
        """
        Predict a signal for this IvimModel class instance given parameters.

        Parameters
        ----------
        ivim_params : array
            The ivim parameters as an array [S0, f, D_star and D]

        gtab : GradientTable class instance
            Gradient directions and bvalues.

        S0 : float, optional
            This has been added just for consistency with the existing
            API. Unlike other models, IVIM predicts S0 and this is over written
            by the S0 value in params.

        Returns
        -------
        ivim_signal : array
            The predicted IVIM signal using given parameters.
        """
        return ivim_prediction(ivim_params, gtab)

    def _leastsq(self, data, x0):
        """Use leastsq to find ivim_params

        Parameters
        ----------
        data : array, (len(bvals))
            An array containing the signal from a voxel.
            If the data was a 3D image of 10x10x10 grid with 21 bvalues,
            the multi_voxel decorator will run the single voxel fitting
            on all the 1000 voxels to get the parameters in
            IvimFit.model_paramters. The shape of the parameter array
            will be (data[:-1], 4).

        x0 : array
            Initial guesses for the parameters S0, f, D_star and D
            calculated using a linear fitting.

        Returns
        -------
        x0 : array
            Estimates of the parameters S0, f, D_star and D.
        """
        gtol = self.options["gtol"]
        ftol = self.options["ftol"]
        xtol = self.tol
        maxfev = self.options["maxiter"]
        bounds = self.bounds

        try:
            res = least_squares(_ivim_error,
                                x0,
                                bounds=bounds,
                                ftol=ftol,
                                xtol=xtol,
                                gtol=gtol,
                                max_nfev=maxfev,
                                args=(self.gtab, data),
                                x_scale=self.x_scale)
            ivim_params = res.x
            if np.all(np.isnan(ivim_params)):
                return np.array([-1, -1, -1, -1])
            return ivim_params
        except ValueError:
            warningMsg = "x0 is unfeasible for leastsq fitting."
            warningMsg += " Returning x0 values from the linear fit."
            warnings.warn(warningMsg, UserWarning)
            return x0


class IvimModelVP(ReconstModel):

    def __init__(self, gtab, bounds=None, maxiter=10, xtol=1e-8):
        r""" Initialize an IvimModelVP class.

        The IVIM model assumes that biological tissue includes a volume
        fraction 'f' of water flowing with a pseudo-diffusion coefficient
        D* and a fraction (1-f: treated as a separate fraction in the variable
        projection method) of static (diffusion only), intra and
        extracellular water, with a diffusion coefficient D. In this model
        the echo attenuation of a signal in a single voxel can be written as

            .. math::

            S(b) = S_0*[f*e^{(-b*D\*)} + (1-f)e^{(-b*D)}]

            Where:
            .. math::

            S_0, f, D\* and D are the IVIM parameters.


        maxiter: int, optional
            Maximum number of iterations for the Differential Evolution in
            SciPy.
            default : 10

        xtol : float, optional
            Tolerance for convergence of minimization.
            default : 1e-8

        References
        ----------
        .. [1] Le Bihan, Denis, et al. "Separation of diffusion and perfusion
               in intravoxel incoherent motion MR imaging." Radiology 168.2
               (1988): 497-505.
        .. [2] Federau, Christian, et al. "Quantitative measurement of brain
               perfusion with intravoxel incoherent motion MR imaging."
               Radiology 265.3 (2012): 874-881.
        .. [3] Fadnavis, Shreyas et.al. "MicroLearn: Framework for machine
               learning, reconstruction, optimization and microstructure
               modeling, Proceedings of: International Society of Magnetic
               Resonance in Medicine (ISMRM), Montreal, Canada, 2019.
        """

        self.maxiter = maxiter
        self.xtol = xtol
        self.bvals = gtab.bvals
        self.yhat_perfusion = np.zeros(self.bvals.shape[0])
        self.yhat_diffusion = np.zeros(self.bvals.shape[0])
        self.exp_phi1 = np.zeros((self.bvals.shape[0], 2))
        self.bounds = bounds or (BOUNDS[0][1:], BOUNDS[1][1:])

    @multi_voxel_fit
    def fit(self, data, bounds_de=None):
        r""" Fit method of the IvimModelVP model class

        MicroLearn framework (VarPro)[1]_.

        The VarPro computes the IVIM parameters using the MIX approach.
        This algorithm uses three different optimizers. It starts with a
        differential evolution algorithm and fits the parameters in the
        power of exponentials. Then the fitted parameters in the first step are
        utilized to make a linear convex problem. Using a convex optimization,
        the volume fractions are determined. Then the last step is non linear
        least square fitting on all the parameters. The results of the first
        and second step are utilized as the initial values for the last step
        of the algorithm. (see [1]_ and [2]_ for a comparison and a through
        discussion).

        References
        ----------
        .. [1] Fadnavis, Shreyas et.al. "MicroLearn: Framework for machine
               learning, reconstruction, optimization and microstructure
               modeling, Proceedings of: International Society of Magnetic
               Resonance in Medicine (ISMRM), Montreal, Canada, 2019.
        .. [2] Farooq, Hamza, et al. "Microstructure Imaging of Crossing (MIX)
               White Matter Fibers from diffusion MRI." Scientific reports 6
               (2016).

        """
        data_max = data.max()
        data = data / data_max
        b = self.bvals

        # Setting up the bounds for differential_evolution
        bounds_de = np.array([(0.005, 0.01), (10**-4, 0.001)])

        # Optimizer #1: Differential Evolution
        res_one = differential_evolution(self.stoc_search_cost, bounds_de,
                                         maxiter=self.maxiter, args=(data,),
                                         disp=False, polish=True, popsize=28)
        x = res_one.x
        phi = self.phi(x)

        # Optimizer #2: Convex Optimizer
        f = self.cvx_fit(data, phi)
        x_f = self.x_and_f_to_x_f(x, f)

        # Setting up the bounds for least_squares
        bounds = self.bounds

        # Optimizer #3: Nonlinear-Least Squares
        res = least_squares(self.nlls_cost, x_f, bounds=bounds,
                            xtol=self.xtol, args=(data,))
        result = res.x
        f_est = result[0]
        D_star_est = result[1]
        D_est = result[2]

        S0 = data / (f_est * np.exp(-b * D_star_est) + (1 - f_est) *
                     np.exp(-b * D_est))
        S0_est = S0 * data_max

        # final result containing the four fit parameters: S0, f, D* and D
        result = np.insert(result, 0, np.mean(S0_est), axis=0)
        return IvimFit(self, result)

    def stoc_search_cost(self, x, signal):
        """
        Cost function for differential evolution algorithm. Performs a
        stochastic search for the non-linear parameters 'x'. The objective
        function is calculated in the :func: `ivim_mix_cost_one`. The function
        constructs the parameters using :func: `phi`.

        Parameters
        ----------
        x : array
            input from the Differential Evolution optimizer.

        signal : array
            The signal values measured for this model.

        Returns
        -------
        :func: `ivim_mix_cost_one`

        """
        phi = self.phi(x)
        return self.ivim_mix_cost_one(phi, signal)

    def ivim_mix_cost_one(self, phi, signal):
        """
        Constructs the objective for the :func: `stoc_search_cost`.

        First calculates the Moore-Penrose inverse of the input `phi` and takes
        a dot product with the measured signal. The result obtained is again
        multiplied with `phi` to complete the projection of the variable into
        a transformed space. (see [1]_ and [2]_ for thorough discussion on
        Variable Projections and relevant cost functions).

        Parameters
        ----------
        phi : array
            Returns an array calculated from :func: `Phi`.

        signal : array
            The signal values measured for this model.

        Returns
        -------
        (signal -  S)^T(signal -  S)

        Notes
        -----
        to make cost function for Differential Evolution algorithm:
        .. math::

            (signal -  S)^T(signal -  S)

        References
        ----------
        .. [1] Fadnavis, Shreyas et.al. "MicroLearn: Framework for machine
               learning, reconstruction, optimization and microstructure
               modeling, Proceedings of: International Society of Magnetic
               Resonance in Medicine (ISMRM), Montreal, Canada, 2019.
        .. [2] Farooq, Hamza, et al. "Microstructure Imaging of Crossing (MIX)
               White Matter Fibers from diffusion MRI." Scientific reports 6
               (2016).

        """
        # Moore-Penrose
        phi_mp = np.dot(np.linalg.inv(np.dot(phi.T, phi)), phi.T)
        f = np.dot(phi_mp, signal)
        yhat = np.dot(phi, f)  # - sigma
        return np.dot((signal - yhat).T, signal - yhat)

    def cvx_fit(self, signal, phi):
        """
        Performs the constrained search for the linear parameters `f` after
        the estimation of `x` is done. Estimation of the linear parameters `f`
        is a constrained linear least-squares optimization problem solved by
        using a convex optimizer from cvxpy. The IVIM equation contains two
        parameters that depend on the same volume fraction. Both are estimated
        as separately in the convex optimizer.

        Parameters
        ----------
        phi : array
            Returns an array calculated from :func: `phi`.

        signal : array
            The signal values measured for this model.

        Returns
        -------
        f1, f2 (volume fractions)

        Notes
        -----
        cost function for differential evolution algorithm:

        .. math::

            minimize(norm((signal)- (phi*f)))
        """

        # Create four scalar optimization variables.
        f = cvxpy.Variable(2)
        # Constraints have been set similar to the MIX paper's
        # Supplementary Note 2: Synthetic Data Experiments, experiment 2
        constraints = [cvxpy.sum(f) == 1,
                       f[0] >= 0.011,
                       f[1] >= 0.011,
                       f[0] <= self.bounds[1][0],
                       f[1] <= 0.89]

        # Form objective.
        obj = cvxpy.Minimize(cvxpy.sum(cvxpy.square(phi @ f - signal)))

        # Form and solve problem.
        prob = cvxpy.Problem(obj, constraints)
        prob.solve()  # Returns the optimal value.
        return np.array(f.value)

    def nlls_cost(self, x_f, signal):
        """
        Cost function for the least square problem. The cost function is used
        in the Least Squares function of SciPy in :func: `fit`. It guarantees
        that stopping point of the algorithm is at least a stationary point
        with reduction in the the number of iterations required by the
        differential evolution optimizer.

        Parameters
        ----------
        x_f : array
            Contains the parameters 'x' and 'f' combines in the same array.

        signal : array
            The signal values measured for this model.

        Returns
        -------
        sum{(signal -  phi*f)^2}

        Notes
        -----
        cost function for the least square problem.

        .. math::

            sum{(signal -  phi*f)^2}
        """

        x, f = self.x_f_to_x_and_f(x_f)
        f1 = np.array([f, 1 - f])
        phi = self.phi(x)
        return np.sum((np.dot(phi, f1) - signal) ** 2)

    def x_f_to_x_and_f(self, x_f):
        """
        Splits the array of parameters in x_f to 'x' and 'f' for performing
        a search on the both of them independently using the Trust Region
        Method.

        Parameters
        ----------
        x_f : array
            Combined array of parameters 'x' and 'f' parameters.

        Returns
        -------
        x, f : array
            Split parameters into two separate arrays

        """
        x = np.zeros(2)
        f = x_f[0]
        x = x_f[1:3]
        return x, f

    def x_and_f_to_x_f(self, x, f):
        """
        Combines the array of parameters 'x' and 'f' into x_f for performing
        NLLS on the final stage of optimization.

        Parameters
        ----------
         x, f : array
            Split parameters into two separate arrays

        Returns
        -------
        x_f : array
            Combined array of parameters 'x' and 'f' parameters.

        """
        x_f = np.zeros(3)
        x_f[0] = f[0]
        x_f[1:3] = x
        return x_f

    def phi(self, x):
        """
        Creates a structure for the combining the diffusion and pseudo-
        diffusion by multiplying with the bvals and then exponentiating each of
        the two components for fitting as per the IVIM- two compartment model.

        Parameters
        ----------
         x : array
            input from the Differential Evolution optimizer.

        Returns
        -------
        exp_phi1 : array
            Combined array of parameters perfusion/pseudo-diffusion
            and diffusion parameters.

        """
        self.yhat_perfusion = self.bvals * x[0]
        self.yhat_diffusion = self.bvals * x[1]
        self.exp_phi1[:, 0] = np.exp(-self.yhat_perfusion)
        self.exp_phi1[:, 1] = np.exp(-self.yhat_diffusion)
        return self.exp_phi1


class IvimFit:

    def __init__(self, model, model_params):
        """ Initialize a IvimFit class instance.

        Parameters
        ----------
        model : Model class

        model_params : array
            The parameters of the model. In this case it is an
            array of ivim parameters. If the fitting is done
            for multi_voxel data, the multi_voxel decorator will
            run the fitting on all the voxels and model_params
            will be an array of the dimensions (data[:-1], 4),
            i.e., there will be 4 parameters for each of the voxels.

        """
        self.model = model
        self.model_params = model_params

    def __getitem__(self, index):
        model_params = self.model_params
        N = model_params.ndim
        if type(index) is not tuple:
            index = (index,)
        elif len(index) >= model_params.ndim:
            raise IndexError("IndexError: invalid index")
        index = index + (slice(None),) * (N - len(index))
        return type(self)(self.model, model_params[index])

    @property
    def S0_predicted(self):
        return self.model_params[..., 0]

    @property
    def perfusion_fraction(self):
        return self.model_params[..., 1]

    @property
    def D_star(self):
        return self.model_params[..., 2]

    @property
    def D(self):
        return self.model_params[..., 3]

    @property
    def shape(self):
        return self.model_params.shape[:-1]

    def predict(self, gtab, S0=1.):
        """Given a model fit, predict the signal.

        Parameters
        ----------
        gtab : GradientTable class instance
               Gradient directions and bvalues

        S0 : float
            S0 value here is not necessary and will not be used to predict the
            signal. It has been added to conform to the structure of the
            predict method in multi_voxel which requires a keyword argument S0.

        Returns
        -------
        signal : array
            The signal values predicted for this model using its parameters.

        """
        return ivim_prediction(self.model_params, gtab)
