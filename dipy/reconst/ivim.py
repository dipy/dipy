#!/usr/bin/python
""" Classes and functions for fitting ivim model """
from __future__ import division, print_function, absolute_import
import numpy as np

from dipy.core.gradients import gradient_table
from dipy.reconst.base import ReconstModel
from dipy.reconst.dti import _min_positive_signal, apparent_diffusion_coef
from dipy.reconst.dti import TensorModel, mean_diffusivity
from dipy.reconst.vec_val_sum import vec_val_vect
from dipy.core.sphere import Sphere

from distutils.version import LooseVersion
import scipy

SCIPY_LESS_0_17 = LooseVersion(scipy.version.short_version) < '0.17'

if not SCIPY_LESS_0_17:
    least_squares = scipy.optimize.least_squares
else:
    leastsq = scipy.optimize.leastsq


def ivim_function(params, bvals):
    """The Intravoxel incoherent motion (IVIM) model function.

    The IVIM model assumes that biological tissue includes a volume fraction
    f of water flowing in perfused capillaries, with a perfusion
    coefficient D* and a fraction (1-f) of static (diffusion only), intra and
    extracellular water, with a diffusion coefficient D. In this model the
    echo attenuation of a signal in a single voxel can be written as

        .. math::

        S(b) = S_0[f*e^{(-b*D\*)} + (1-f)e^{(-b*D)}]

        Where:
        .. math::

        S_0, f, D\* and D are the IVIM parameters.

    Parameters
    ----------
    params : array
             An array of IVIM parameters - S0, f, D_star, D

    bvals  : array
             bvalues

    References
    ----------
    .. [1] Le Bihan, Denis, et al. "Separation of diffusion
               and perfusion in intravoxel incoherent motion MR
               imaging." Radiology 168.2 (1988): 497-505.
    .. [2] Federau, Christian, et al. "Quantitative measurement
               of brain perfusion with intravoxel incoherent motion
               MR imaging." Radiology 265.3 (2012): 874-881.
    """
    S0, f, D_star, D = params
    S = S0 * (f * np.exp(-bvals * D_star) + (1 - f) * np.exp(-bvals * D))
    return S


def _ivim_error(params, bvals, signal):
    """Error function to be used in fitting the IVIM model
    """
    return (signal - ivim_function(params, bvals))


class IvimModel(ReconstModel):
    """Ivim model
    """

    def __init__(self, gtab, split_b=200.0, min_signal=None,
                 x0=[1.0, 0.10, 0.001, 0.0009], fit_method="one_stage",
                 jac=True, bounds=None,
                 tol=1e-7, options={'gtol': 1e-12, 'ftol': 1e-12,
                                    'eps': 1e-12, 'maxiter': 1000}):
        """
        Initialize an IVIM model.

        Parameters
        ----------
        gtab : GradientTable class instance
            Gradient directions and bvalues

        split_b : float
            The b-value to split the data on for two-stage fit

        min_signal : float
            The minimum signal value in the data.
        """
        ReconstModel.__init__(self, gtab)
        self.split_b = split_b
        self.min_signal = min_signal
        self.x0 = x0
        self.fit_method = fit_method
        self.jac = jac
        self.bounds = bounds
        self.tol = tol
        self.options = options

        if self.min_signal is not None and self.min_signal <= 0:
            e_s = "The `min_signal` key-word argument needs to be strictly"
            e_s += " positive."
            raise ValueError(e_s)

        if self.jac == True:
            self.jac = _ivim_jacobian_func
        else:
            self.jac = None

        if SCIPY_LESS_0_17 and self.bounds is not None:
            e_s = "Scipy versions less than 0.17 do not support "
            e_s += "bounds. Please update to Scipy 0.17 to use bounds"
            raise ValueError(e_s)
        else:
            self.bounds = ([0., 0., 0., 0.], [np.inf, 1., 1., 1.])

        if self.fit_method is not "one_stage" and self.fit_method is not "two_stage":
            e_s = "Fit method must be either "
            e_s += "'one_stage' or 'two_stage'"
            raise ValueError(e_s)

    def fit(self, data, mask=None):
        """ Fit method of the Ivim model class

        Parameters
        ----------
        data : array
            The measured signal from voxels.
        mask : array
            A boolean array used to mark the coordinates in the data that
            should be analyzed that has the shape data.shape[:-1]
        x0 : array, optional
            Initial guesses for the parameters S0, f, D_star and D
            Default : [1.0, 0.10, 0.001, 0.0009]
            These initial parameters are taken from [1]_
            Dimension can either be 1 or (N, 4) where N is the number of
            voxels in the data.
        fit_method: str
            Use one-stage fitting or two-stage fitting. The two stage fitting
            first fits a tensor model and uses the parameters from it to get
            the initial guesses for the IVIM model.
        jac : Boolean, optional
            Default : True
            Use the Jacobian. If true, use the Jacobian function defined in
            `_ivim_jacobian_func` to calculate the Jacobian for minimization.
        bounds : tuple, optional
            Bounds for the various parameters,
        tol : float, optional
            Default : 1e-25
            Tolerance for convergence of minimization
        routine : str, optional
            Default : minimize
            Specify whether to use leastsq or minimize for fitting.

        Returns
        -------
        IvimFit object

        References
        ----------
        .. [1] Federau, Christian, et al. "Quantitative measurement
                   of brain perfusion with intravoxel incoherent motion
                   MR imaging." Radiology 265.3 (2012): 874-881.
        """
        if mask is None:
            # Flatten it to 2D either way:
            data_in_mask = np.reshape(data, (-1, data.shape[-1]))
        else:
            # Check for valid shape of the mask
            if mask.shape != data.shape[:-1]:
                raise ValueError("Mask is not the same shape as data.")
            mask = np.array(mask, dtype=bool, copy=False)
            data_in_mask = np.reshape(data[mask], (-1, data.shape[-1]))
        if self.min_signal is None:
            min_signal = _min_positive_signal(data)
        else:
            min_signal = self.min_signal

        if len(self.x0) == 4:
            x0 = np.ones(
                (data.shape[:-1] + (4,))) * self.x0
        else:
            if self.x0.shape != (data.shape[0], 4):
                raise ValueError(
                    "Initial guess params should be of shape (num_voxels,4)")

        data_in_mask = np.maximum(data_in_mask, min_signal)
        # Get the x0 parameters for the masked data
        x0_mask = np.reshape(x0[mask], (-1, x0.shape[-1]))

        if self.fit_method == "one_stage":
            params_in_mask, fit_stats_in_mask = one_stage(data_in_mask,
                                                          self.gtab, x0_mask,
                                                          self.jac, self.bounds, self.tol,
                                                          self.options)
        elif self.fit_method == "two_stage":
            params_in_mask, fit_stats_in_mask = two_stage(data_in_mask,
                                                          self.gtab, x0_mask,
                                                          self.split_b, self.jac,
                                                          self.bounds, self.tol,
                                                          self.options)

        if mask is None:
            out_shape = data.shape[:-1] + (-1, )
            ivim_params = params_in_mask.reshape(out_shape)
            fit_stats = fit_stats_in_mask.reshape(out_shape)
        else:
            ivim_params = np.zeros(data.shape[:-1] + (4,))
            ivim_params[mask, :] = params_in_mask
            fit_stats = np.zeros(data.shape[:-1] + (4,))
            fit_stats[mask, :] = fit_stats_in_mask

        return IvimFit(self, ivim_params, fit_stats)

    def predict(self, ivim_params):
        """
        Predict a signal for this IvimModel class instance given parameters.
        Parameters
        ----------
        ivim_params : ndarray
            The last dimension should have 4 parameters: S0, f, D_star and D
        """
        return ivim_function(ivim_params, self.gtab.bvals)


class IvimFit(object):

    def __init__(self, model, model_params, fit_stats):
        """ Initialize a IvimFit class instance.
            Parameters
            ----------
            The model parameters are S0, f, D_star, D
        """
        self.model = model
        self.model_params = model_params
        self.fit_stats = fit_stats

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
    def shape(self):
        return self.model_params.shape[:-1]

    def predict(self, gtab, step=None, S0=1.0):
        r"""
        Given a model fit, predict the signal.

        Parameters
        ----------
        gtab : a GradientTable class instance
            This encodes the directions for which a prediction is made
        """
        return ivim_function(self.model_params, gtab.bvals)


def one_stage(data, gtab, x0, jac, bounds, tol, options):
    """
    Fit the ivim params using minimize

    Parameters
    ----------
    data : array ([X, Y, Z, ...], g)
        Data or response variables holding the data. Note that the last
        dimension should contain the data. It makes no copies of data.

    jac : bool, optional
        Use the Jacobian? Default: False

    Returns
    -------
    ivim_params: the S0, f, D_star, D value for each voxel.

    """
    flat_data = data.reshape((-1, data.shape[-1]))
    S_normalization = flat_data[:, 0]
    flat_data = (flat_data.T / S_normalization).T
    flat_x0 = x0.reshape(-1, x0.shape[-1])
    # Flatten for the iteration over voxels:
    bvals = gtab.bvals
    ivim_params = np.empty((flat_data.shape[0], 4))
    fit_stats = np.empty((flat_data.shape[0], 4))
    flat_x0[..., 0] = flat_data[..., 0]

    fit_stats = _leastsq(flat_data, bvals, flat_x0,
                         ivim_params, options, tol, bounds, jac)
    ivim_params.shape = data.shape[:-1] + (4,)
    ivim_params[:, 0] = ivim_params[:, 0] * S_normalization
    return ivim_params, fit_stats


def _leastsq(flat_data, bvals, flat_x0, ivim_params, options, tol, bounds, jac):
    """Use leastsq for finding ivim_params"""
    num_voxels = flat_data.shape[0]
    result = np.empty(flat_data.shape[0], dtype=object)
    gtol = options["gtol"]
    ftol = options["ftol"]
    xtol = tol
    epsfcn = options["eps"]
    maxfev = options["maxiter"]
    if not SCIPY_LESS_0_17:
        for vox in range(num_voxels):
            res = least_squares(_ivim_error,
                                flat_x0[vox],
                                jac=jac,
                                bounds=bounds,
                                ftol=ftol,
                                xtol=xtol,
                                gtol=gtol,
                                max_nfev=maxfev,
                                args=(bvals, flat_data[vox]))
            ivim_params[vox, :4] = res.x
            result[vox] = res
    else:
        for vox in range(num_voxels):
            res = leastsq(_ivim_error,
                          flat_x0[vox],
                          args=(bvals, flat_data[vox]),
                          Dfun=jac,
                          gtol=gtol,
                          xtol=xtol,
                          ftol=ftol,
                          epsfcn=epsfcn,
                          maxfev=maxfev)
            ivim_params[vox, :4] = res[0]
            result[vox] = res
    return result


def two_stage(data, gtab, x0,
              split_b, jac, bounds, tol,
              options):
    """
    Fit the ivim params using a two stage fit.

    In the two stage fitting routine, initially, we fit the signal
    at bvals less than the specified split_b using the TensorModel
    and get an intial guess for f and D. Then, using these parameters
    we fit the entire data for all bvalues. The default split_b is
    200.

    Parameters
    ----------
    data : array ([X, Y, Z, ...], g)
        Data or response variables holding the data. Note that the last
        dimension should contain the data. It makes no copies of data.

    jac : bool, optional
        Use the Jacobian? Default: False

    Returns
    -------
    ivim_params: the S0, f, D_star, D value for each voxel.

    """
    flat_data = data.reshape((-1, data.shape[-1]))
    S_normalization = flat_data[:, 0]
    flat_data = (flat_data.T / S_normalization).T

    flat_x0 = x0.reshape(-1, x0.shape[-1])
    # Flatten for the iteration over voxels:
    bvals = gtab.bvals
    ivim_params = np.empty((flat_data.shape[0], 4))
    fit_stats = np.empty((flat_data.shape[0], 4))
    flat_x0[..., 0] = flat_data[..., 0]

    S0_hat, D_guess = _get_S0_D_guess(flat_data, gtab, split_b)
    f_guess = 1 - S0_hat / flat_data[..., 0]

    flat_x0[..., 1] = f_guess
    flat_x0[..., 3] = D_guess

    fit_stats = _leastsq(flat_data, bvals, flat_x0,
                         ivim_params, options, tol, bounds, jac)
    ivim_params.shape = data.shape[:-1] + (4,)
    ivim_params[:, 0] = ivim_params[:, 0] * S_normalization

    return ivim_params, fit_stats


def _get_S0_D_guess(data, gtab, split_b):
    """
    Obtain initial guess for S0 and D for two stage fitting.

    Using TensorModel from reconst.dti, we fit an exponential
    for those signals which are less than a particular bvalue
    (split_b). The apparent diffusion coefficient gives us an
    initial estimate for D.

    Parameters
    ----------
    data : array ([X, Y, Z, ...], g)
        Data or response variables holding the data. Note that the last
        dimension should contain the data. It makes no copies of data.

    gtab : GradientTable class instance
            Gradient directions and bvalues

    split_b : float
        The b-value to split the data on for two-stage fit

    Returns
    -------
    S0_hat : array ([S0_1, S0_2, ...])
        An array which gives the guess for all the S0 values.

    D_guess : array ([D_1, D_2, ...])
        An array which gives the guess for all the D values.
    """
    bvals_ge_split = gtab.bvals[gtab.bvals > split_b]
    bvecs_ge_split = gtab.bvecs[gtab.bvals > split_b]
    gtab_ge_split = gradient_table(bvals_ge_split, bvecs_ge_split.T)
    tensor_model = TensorModel(gtab_ge_split)
    tenfit = tensor_model.fit(data[..., gtab.bvals > split_b])
    D_guess = mean_diffusivity(tenfit.evals)
    dti_params = tenfit.model_params
    evecs = dti_params[..., 3:12].reshape((dti_params.shape[:-1] + (3, 3)))
    evals = dti_params[..., :3]
    qform = vec_val_vect(evecs, evals)

    sphere = Sphere(xyz=gtab.bvecs[~gtab.b0s_mask])
    ADC = apparent_diffusion_coef(qform, sphere)

    S0_hat = np.mean(data[..., ~gtab.b0s_mask] /
                     np.exp(-gtab.bvals[~gtab.b0s_mask] * ADC),
                     -1)
    return S0_hat, D_guess


def _ivim_jacobian_func(params, bvals, signal):
    """The Jacobian is the first derivative of the error function.

    Parameters
    ----------
    params : array (N, 4)
             An array of IVIM parameters - S0, f, D_star, D

    bvals  : array
             bvalues

    Returns
    -------
    jacobian : array (N, 4)
        The Jacobian for the error function.
    Notes
    -----
    The worked out Jacobian can be derived as follows.
    Let `i` be the index which denotes the number of points to fit and `j` be
    the number of parameters. In this case `i` is the number of bvalues.
    Let :math:`x_i` be the independent parameter (bvalue) and :math:`\beta`
    denote the vector of parameters with :math:`\beta_j = (S0, f, D*, D)`.
    Thus we have `j = 0,1,2,3`.

    The IVIM signal is:
    .. math:: S(x_i, \beta) = \beta_0[\beta_1e^{(-x_i\beta_2)} + (1-\beta_1)e^{(-x_i\beta_3)}]

    Here:
    .. math:: \beta_0 = S0, \beta_1 = f, \beta_2 = D*, \beta_3 = D and x_i = b

    The residual that we need to find the Jacobian for is: 
    .. math:: r_i = [y_i - S(x_i, \beta)]

    The Jacobian is defined as:
    .. math:: - \frac{\partial r_i}{\partial \beta_j}

    Thus we will have : 
    .. math:: J_{i,j} = -\frac{\partial S}{\partial \beta_j}

    The various terms in the Jacobian are thus:
    .. math:: \frac{\partial S}{\partial \beta_0} =  [\beta_1e^{(-x_i\beta_2)} + (1-\beta_1)e^{(-x_i\beta_3)}]
              \frac{\partial S}{\partial \beta_1} = \beta_0[e^{-x_i\beta_2} - e^{-x_i\beta_3}]
              \frac{\partial S}{\partial \beta_2} = \beta_0[-x_i\beta_1e^{-x_i\beta_2}]
              \frac{\partial S}{\partial \beta_3} = \beta_0[-x_i(1- \beta_1)e^{-x_i\beta_3}]

    This should give us the Jacobian
    """
    S0, f, D_star, D = params

    bvals_D_star = np.exp(-bvals * D_star)
    bvals_D = np.exp(-bvals * D)

    derv_S0 = f * bvals_D_star + (1 - D) * bvals_D
    derv_f = S0 * (bvals_D_star - bvals_D)
    derv_D_star = S0 * (-bvals * f * bvals_D_star)
    derv_D = S0 * (-bvals * (1 - f) * bvals_D)

    return (np.array([-derv_S0,
                      -derv_f,
                      -derv_D_star,
                      -derv_D])).T
