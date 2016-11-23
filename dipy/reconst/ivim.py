""" Classes and functions for fitting ivim model """
from __future__ import division, print_function, absolute_import

from distutils.version import LooseVersion

import numpy as np
import scipy
import warnings
from dipy.reconst.base import ReconstModel
from dipy.reconst.multi_voxel import multi_voxel_fit
import itertools as it

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

def _AIC(rss, k, n):
    """
    Calculates the AIC of a regression

    Parameters
    ----------
    rss : float
        Residual sum of squares of the data and the fit
    k : float
        Number of parameters in the fit
    n : float
        Number of points in the data

    Returns
    -------
    AIC : float
        The AIC value of the fit
    """
    return 2 * k + n * np.log(rss / n)


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
                 two_stage=True, fast_linear_fit=False, tol=1e-15,
                 noneg_exp=2.0,
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

        fast_linear_fit : bool
            A boolean to select the fast, linear-only fitting for the first stage fit.
            This replaces a slower non-linear fitting of S0 and D*.

        tol : float, optional
            Tolerance for convergence of minimization.
            default : 1e-15

        noneg_exp : float
            Only for fast linear fitting.
            A value of 2 is used to prevent negative values in the residual fitting.
            This works because (x*e^a)^b = x^b*e^(a*b), but of course is only approximate because of noise.
            A value of 1 is a standard log linear fitting with negative values removed.
            So if you have noisy data it may be better to retain points and use 2, but if your data aren't noisy
            then use 1 and a few points may be dropped but it's better than not.

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
        self.fast_linear_fit = fast_linear_fit
        self.tol = tol
        self.noneg_exp = noneg_exp
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
        if self.fast_linear_fit:
            params_linear = self.fit_linear(data, mask)
        else:
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

    def fit_linear(self, data, mask=None):
        """ Linear fit method of the Ivim model class.

        The fitting takes place in the following steps:
        1. Linear fit for D (bvals > `split_b_D` (default: 400)) and store S0_prime.
            Assuming we have a single exponential decay for bvals greater than split_b_D
        2. Calculate the residuals for bvals <= split_b_S0
        3. Linear fit for D_star on the residuals in step 2 for bvals < `split_b_S0` (default: 400) and store S0_resid.
            Assuming the remaining signal is from a single exponential decay for bvals less than split_b_S0
        4. Calculate S0 = S0_prime + S0_resid
        5. Calculate f = S0_resid / S0


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
        params : array
            An array of the model parameters
        """

        # setup
        high_bvals = self.gtab.bvals > self.split_b_D
        low_bvals = self.gtab.bvals <= self.split_b_S0

        # step 1
        S0_prime, D = self.estimate_linear_fit(data, b_selection=high_bvals)
        params_high_b = np.array([S0_prime, 0, 0, D])

        # step 2
        resid = _ivim_error(params_high_b, self.gtab, data) ** self.noneg_exp  # params=[S0, f, D*, D]
        low_and_positive = np.logical_and(resid > 0, low_bvals)

        #I'm leaving shifted and pure and complex fitting in here for now in case I want to compare later
        #resid_shifted = _ivim_error(params_high_b, self.gtab, data) # params=[S0, f, D*, D]
        #resid_shift = np.min(resid_shifted) - 1
        #resid_shifted -= resid_shift

        #resid_pure = _ivim_error(params_high_b, self.gtab, data)   # params=[S0, f, D*, D]
        #low_and_positive_pure = np.logical_and(resid_pure > 0, low_bvals)

        # step 3
        S0_resid, D_star = self.estimate_linear_fit(resid, b_selection=low_and_positive)
        S0_resid **= 1.0 / self.noneg_exp
        D_star /= self.noneg_exp

        #S0_resid_shifed, D_star_shifted = self.estimate_linear_fit(resid_shifted, b_selection=low_bvals)

        #S0_resid_pure, D_star_pure = self.estimate_linear_fit(resid_pure, b_selection=low_and_positive_pure)

        #S0_resid_pure_cplx, D_star_pure_cplx = self.estimate_linear_fit(resid_pure+0j, b_selection=low_bvals)

        # step 4
        S0 = S0_resid + S0_prime

        # step 5
        f = S0_resid / S0

        # return the values
        params = np.array([S0, f, D_star, D])
        return params

    def estimate_linear_fit(self, data, split_b=None, less_than=True, b_selection=None):
        """Estimate a linear fit by taking log of data.

        Parameters
        ----------
        data : array
            An array containing the data to be fit

        split_b : float
            The b value to split the data
            split_b or b_selection is required, if both exist b_selection takes precedence

        less_than : bool
            If True, splitting occurs for bvalues less than split_b

        b_selection : True/False array
            The array which selects the desired data
            split_b or b_selection is required, if both exist b_selection takes precedence

        Returns
        -------
        S0 : float
            The estimated S0 value. (intercept)

        D : float
            The estimated value of D.
        """
        if b_selection is None:
            if split_b is not None:
                if less_than:
                    b_selection = self.gtab.bvals <= split_b
                else:
                    b_selection = self.gtab.bvals >= split_b
            else:
                warningMsg = "In estimate_linear_fit, either split_b or b_selection is required!"
                warnings.warn(warningMsg, UserWarning)

        D, neg_log_S0 = np.polyfit(self.gtab.bvals[b_selection],
                                       -np.log(data[b_selection]), 1)

        S0 = np.exp(-neg_log_S0)
        return S0, D

    def aic_fit_compare(self, data, param_iter):
        """
        Parameters
        ----------
        data : array
            Array containing the actual signal values.

        param_iter : iterable (tuple/list) of arrays
            An iterable containing the parameters of all the curves to fit

        Returns
        -------
        aic_arr : array
            Array of AIC values for the given models, smaller is a better model
        """
        aic_arr = np.array([])
        n = data.size
        for param in param_iter:
            k = np.sum(param != 0)  # number of nonzero elements is the number of parameters
            rss = np.sum(np.square(_ivim_error(param, self.gtab, data)))
            aic = _AIC(rss, k, n)
            aic_arr = np.append(aic_arr, aic)  # maybe should pre-allocate for speed
        return aic_arr

    def aic_relative_likelihood(self, aic_iter):
        """
        Parameters
        ----------
        aic_iter : iterable (tuple/list)
            An iterable containing the AIC values to compare

        Returns
        -------
            A numpy array will all possible relatively likelihood permutations
        """
        aic_iter_combos = list(it.combinations(aic_iter, 2))
        rlike = np.zeros(len(aic_iter_combos))
        idx = 0
        for combo in aic_iter_combos:
            d = combo[1] - combo[0] # traditonal: combo[0] - combo[1]  # note: combo[0] must be less than combo[1] for the traditional definition
            if d == 0:  # have to do this to avoid np.sign(0)=0, which would give 0 rather than 1
                rlike[idx] = 1.
            else:
                rlike[idx] = np.sign(d)*np.exp(-np.abs(d)/2)
            idx += 1
        return rlike




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
    def S0(self):
        return self.model_params[..., 0]

    @property
    def perfusion_fraction(self):
        return self.model_params[..., 1]

    @property
    def f(self):
        return self.model_params[..., 1]

    @property
    def D_star(self):
        return self.model_params[..., 2]

    @property
    def D(self):
        return self.model_params[..., 3]

    @property
    def S0_f_Dstar_D(self):
        return self.model_params

    @property
    def shape(self):
        return self.model_params.shape[:-1]

    @property
    def S0_resid(self):
        return self.model_params[..., 0] * self.model_params[..., 1]

    @property
    def S0_prime(self):
        return self.model_params[..., 0] - self.S0_resid

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

    def check_fit_bounds(self, bounds=None):
        """Check the fit parameter bounds

        Parameters
        ----------
        bounds : tuple of arrays with 4 elements, optional
            Note that this defaults to what was supplied IvimModel.
            Bounds to constrain the fitted model parameters.
            default : ([0., 0., 0., 0.], [np.inf, .3, 1., 1.])

        Returns
        -------
        in_bounds : array
            The parameter array of [S0, f, D*, D] with True indicating
            it is in bounds, False if it exceeds the bounds in
            either positive or negative direction.
        """
        if bounds is not None:
            internal_bounds = bounds
        else:
            internal_bounds = self.model.bounds
        # note that this works because it's multi voxel decorated here!
        in_bounds = np.logical_and(np.greater_equal(self.model_params,internal_bounds[0]),np.less_equal(self.model_params,internal_bounds[1]))
        return in_bounds

    def enforce_fit_bounds(self, bounds=None):
        """Enforce (clamp/clip) the fit parameters to the bounds

        Parameters
        ----------
        bounds : tuple of arrays with 4 elements, optional
            Note that this defaults to what was supplied IvimModel.
            Bounds to constrain the fitted model parameters.
            default : ([0., 0., 0., 0.], [np.inf, .3, 1., 1.])
        """
        if bounds is not None:
            internal_bounds = bounds
        else:
            internal_bounds = self.model.bounds
        # note that this works because it's multi voxel decorated here!
        self.model_params = np.clip(self.model_params, internal_bounds[0], internal_bounds[1])
