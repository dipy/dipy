""" Classes and functions for fitting ivim model """
from __future__ import division, print_function, absolute_import

from distutils.version import LooseVersion

import numpy as np
import scipy
import warnings
from dipy.reconst.base import ReconstModel
from dipy.reconst.multi_voxel import multi_voxel_fit

SCIPY_LESS_0_17 = (LooseVersion(scipy.version.short_version) <
                   LooseVersion('0.17'))

if SCIPY_LESS_0_17:
    from scipy.optimize import leastsq
else:
    from scipy.optimize import least_squares


def ivim_prediction(params, gtab, S0=1.):
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
    S0, f, D_star, D = params
    b = gtab.bvals
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


class IvimModel(ReconstModel):
    """Ivim model
    """

    def __init__(self, gtab, split_b_D=400.0, split_b_S0=200., bounds=None,
                 two_stage=True, tol=1e-15,
                 x_scale=[1000., 0.1, 0.001, 0.0001],
                 options={'gtol': 1e-15, 'ftol': 1e-15,
                          'eps': 1e-15, 'maxiter': 1000}):
        """
        Initialize an IVIM model.

        The IVIM model assumes that biological tissue includes a volume
        fraction 'f' of water flowing with a pseudo-perfusion coefficient
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

        x_scale : array, optional
            Scaling for the parameters. This is passed to `least_squares` which
            is only available for Scipy version > 0.17.
            default: [1000, 0.01, 0.001, 0.0001]

        options : dict, optional
            Dictionary containing gtol, ftol, eps and maxiter. This is passed
            to leastsq.
            default : options={'gtol': 1e-15, 'ftol': 1e-15, 'eps': 1e-15,
                      'maxiter': 1000}

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

        ReconstModel.__init__(self, gtab)
        self.split_b_D = split_b_D
        self.split_b_S0 = split_b_S0
        self.bounds = bounds
        self.two_stage = two_stage
        self.tol = tol
        self.options = options
        self.x_scale = x_scale

        if SCIPY_LESS_0_17 and self.bounds is not None:
            e_s = "Scipy versions less than 0.17 do not support "
            e_s += "bounds. Please update to Scipy 0.17 to use bounds"
            raise ValueError(e_s)
        elif self.bounds is None:
            self.bounds = ((0., 0., 0., 0.), (np.inf, .3, 1., 1.))
        else:
            self.bounds = bounds

    @multi_voxel_fit
    def fit(self, data, mask=None):
        """ Fit method of the Ivim model class.

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

        mask : array
            A boolean array used to mark the coordinates in the data that
            should be analyzed that has the shape data.shape[:-1]

        Returns
        -------
        IvimFit object
        """
        # Get S0_prime and D - paramters assuming a single exponential decay
        # for signals for bvals greater than `split_b_D`
        S0_prime, D = self.estimate_linear_fit(
            data, self.split_b_D, less_than=False)

        # Get S0 and D_star_prime - paramters assuming a single exponential
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
        epsfcn = self.options["eps"]
        maxfev = self.options["maxiter"]

        if SCIPY_LESS_0_17:
            try:
                res = leastsq(f_D_star_error,
                              params_f_D_star,
                              args=(self.gtab, data, S0, D),
                              gtol=gtol,
                              xtol=xtol,
                              ftol=ftol,
                              epsfcn=epsfcn,
                              maxfev=maxfev)
                f, D_star = res[0]
                return f, D_star
            except ValueError:
                warningMsg = "x0 obtained from linear fitting is not feasibile"
                warningMsg += " as initial guess for leastsq. Parameters are"
                warningMsg += " returned only from the linear fit."
                warnings.warn(warningMsg, UserWarning)
                f, D_star = params_f_D
                return f, D_star
        else:
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
                warningMsg = "x0 obtained from linear fitting is not feasibile"
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
        epsfcn = self.options["eps"]
        maxfev = self.options["maxiter"]
        bounds = self.bounds

        if SCIPY_LESS_0_17:
            try:
                res = leastsq(_ivim_error,
                              x0,
                              args=(self.gtab, data),
                              gtol=gtol,
                              xtol=xtol,
                              ftol=ftol,
                              epsfcn=epsfcn,
                              maxfev=maxfev)
                ivim_params = res[0]
                if np.all(np.isnan(ivim_params)):
                    return np.array([-1, -1, -1, -1])
                return ivim_params
            except ValueError:
                warningMsg = "x0 is unfeasible for leastsq fitting."
                warningMsg += " Returning x0 values from the linear fit."
                warnings.warn(warningMsg, UserWarning)
                return x0
        else:
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


class IvimFit(object):

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
