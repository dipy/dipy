""" Classes and functions for fitting ivim model """
from __future__ import division, print_function, absolute_import
import numpy as np

from dipy.core.gradients import gradient_table
from dipy.reconst.base import ReconstModel
from dipy.reconst.dti import _min_positive_signal, apparent_diffusion_coef
from dipy.reconst.dti import TensorModel, mean_diffusivity
from dipy.reconst.multi_voxel import multi_voxel_fit
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
    'f' of water flowing in perfused capillaries, with a perfusion
    coefficient D* and a fraction (1-f) of static (diffusion only), intra and
    extracellular water, with a diffusion coefficient D. In this model the
    echo attenuation of a signal in a single voxel can be written as

        .. math::

        S(b) = S_0[f*e^{(-b*D\*)} + (1-f)e^{(-b*D)}]

        Where:
        .. math::

        S_0, f, D* and D are the IVIM parameters.

    Parameters
    ----------
    params : array
        An array of IVIM parameters - S0, f, D_star, D

    bvals : array
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

    Parameters
    ----------
    params : array
        An array of IVIM parameters. [S0, f, D_star, D]

    bvals : array
        bvalues

    signal : array
        Array containing the actual signal values.

    """
    return (signal - ivim_function(params, bvals))


class IvimModel(ReconstModel):
    """Ivim model
    """

    def __init__(self, gtab, split_b=200.0,
                 bounds=None, tol=1e-7,
                 options={'gtol': 1e-7, 'ftol': 1e-7,
                          'eps': 1e-7, 'maxiter': 1000}):
        """
        Initialize an IVIM model.

        The IVIM model assumes that biological tissue includes a volume
        fraction 'f' of water flowing in perfused capillaries, with a
        perfusion coefficient D* and a fraction (1-f) of static (diffusion
        only), intra and extracellular water, with a diffusion coefficient
        D. In this model the echo attenuation of a signal in a single voxel
        can be written as

            .. math::

            S(b) = S_0[f*e^{(-b*D\*)} + (1-f)e^{(-b*D)}]

            Where:
            .. math::

            S_0, f, D\* and D are the IVIM parameters.

        Parameters
        ----------
        gtab : GradientTable class instance
            Gradient directions and bvalues

        split_b : float
            The b-value to split the data on for two-stage fit

        bounds : array, (4,4)
            Bounds for the parameters. This is applicable only for Scipy
            version > 0.17 as we use least_squares for the fitting which
            supports bounds. For versions less than Scipy 0.17, this is
            by default set to None. Setting a bound on a Scipy version less
            than 0.17 will raise an error.

            This can be supplied as a tuple for Scipy versions 0.17. It is
            recommended to upgrade to Scipy 0.17 for bounded fitting. The default
            bounds are set to ([0., 0., 0., 0.], [np.inf, 1., 1., 1.])

        tol : float, optional
            Default : 1e-7
            Tolerance for convergence of minimization.

        options : dict, optional
            Dictionary containing gtol, ftol, eps and maxiter. This is passed
            to leastsq. By default these values are set to
            options={'gtol': 1e-7, 'ftol': 1e-7, 'eps': 1e-7, 'maxiter': 1000}

        References
        ----------
        .. [1] Le Bihan, Denis, et al. "Separation of diffusion
                   and perfusion in intravoxel incoherent motion MR
                   imaging." Radiology 168.2 (1988): 497-505.
        .. [2] Federau, Christian, et al. "Quantitative measurement
                   of brain perfusion with intravoxel incoherent motion
                   MR imaging." Radiology 265.3 (2012): 874-881.
        """
        ReconstModel.__init__(self, gtab)
        self.split_b = split_b
        self.bounds = bounds
        self.tol = tol
        self.options = options

        if SCIPY_LESS_0_17 and self.bounds is not None:
            e_s = "Scipy versions less than 0.17 do not support "
            e_s += "bounds. Please update to Scipy 0.17 to use bounds"
            raise ValueError(e_s)
        else:
            self.bounds = (np.array([0., 0., 0., 0.]),
                           np.array([np.inf, 1., 0.1, 0.1]))

    @multi_voxel_fit
    def fit(self, data, mask=None):
        """ Fit method of the Ivim model class

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

        References
        ----------
        .. [1] Federau, Christian, et al. "Quantitative measurement
                   of brain perfusion with intravoxel incoherent motion
                   MR imaging." Radiology 265.3 (2012): 874-881.
        """
        # Call the function _estimate_S0_D to get initial x0 guess.
        x0 = self.estimate_x0(data)
        # Use leastsq to get ivim_params
        params_in_mask = self._leastsq(data, x0)
        return IvimFit(self, params_in_mask)

    def predict(self, ivim_params):
        """
        Predict a signal for this IvimModel class instance given parameters.

        Parameters
        ----------
        ivim_params : array
            The ivim parameters as an array [S0, f, D_star and D]

        Returns
        -------
        ivim_signal : array
            The predicted IVIM signal using given parameters.
        """
        return ivim_function(ivim_params, self.gtab.bvals)

    def estimate_x0(self, data):
        """
        Fit the ivim params using a two stage fit.

        In the two stage fitting routine, initially, we fit the signal
        at bvals less than the specified split_b using the TensorModel
        and get an intial guess for f and D. Then, using these parameters
        we fit the entire data for all bvalues. The default split_b is
        200.

        Parameters
        ----------
        data : array
            The measured signal from one voxel. A multi voxel decorator
            will be applied to this fit method to scale it and apply it
            to multiple voxels.

        Returns
        -------
        x0_guess : array
            An array with initial values of S0, f, D_star, D for each voxel.
        """
        S0_hat, D_guess = self._estimate_S0_D(data)
        f_guess = 1 - S0_hat / data[0]
        # We set the S0 guess as the signal value at b=0
        # The D* guess is roughly 10 times the D value
        x0 = np.array([data[0], f_guess, 10 * D_guess, D_guess])

        if self.bounds is None:
            bounds_check = [np.inf, 1., 0.1, 0.1]
        else:
            bounds_check = self.bounds

        x0 = np.where(x0 > bounds_check[0], x0, bounds_check[0])
        x0 = np.where(x0 < bounds_check[1], x0, bounds_check[1])

        return x0

    def _estimate_S0_D(self, data):
        """
        Obtain initial guess for S0 and D for two stage fitting.

        Using TensorModel from reconst.dti, we fit an exponential
        for those signals which are greater than a particular bvalue
        (split_b). The apparent diffusion coefficient gives us an
        initial estimate for D and S0.

        Parameters
        ----------
        data : array
            The measured signal from one voxel.

        Returns
        -------
        S0_hat : float
            Initial S0 guess for this voxel.

        D_guess : float
            Initial D_guess for this voxel.
        """
        gtab = self.gtab
        split_b = self.split_b
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
        return np.array([S0_hat, D_guess])

    def _leastsq(self, data, x0):
        """
        Use leastsq for finding ivim_params

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
            calculated using the function `_estimate_S0_D`
        """
        gtol = self.options["gtol"]
        ftol = self.options["ftol"]
        xtol = self.tol
        epsfcn = self.options["eps"]
        maxfev = self.options["maxiter"]
        bounds = self.bounds
        bvals = self.gtab.bvals
        if not SCIPY_LESS_0_17:
            res = least_squares(_ivim_error,
                                x0,
                                bounds=bounds,
                                ftol=ftol,
                                xtol=xtol,
                                gtol=gtol,
                                max_nfev=maxfev,
                                args=(bvals, data))
            ivim_params = res.x
            return ivim_params
        else:
            res = leastsq(_ivim_error,
                          x0,
                          args=(bvals, data),
                          gtol=gtol,
                          xtol=xtol,
                          ftol=ftol,
                          epsfcn=epsfcn,
                          maxfev=maxfev)
            ivim_params = res[0]
            return ivim_params


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
    def shape(self):
        return self.model_params.shape[:-1]

    def predict(self, gtab, S0=1.):
        r"""
        Given a model fit, predict the signal.

        Parameters
        ----------
        gtab : GradientTable class instance
               Gradient directions and bvalues

        S0 : float
            S0 value here is not necessary and will
            not be used to predict the signal. It has
            been added to conform to the structure
            of the predict method in multi_voxel which
            requires a keyword argument S0.

        Returns
        -------
        signal : array
            The signal values predicted for this model using
            its parameters.
        """
        return ivim_function(self.model_params, gtab.bvals)
