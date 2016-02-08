""" Classes and functions for fitting tensors without free water
contamination """
from __future__ import division, print_function, absolute_import

import warnings

import numpy as np

import scipy.optimize as opt

from .base import ReconstModel

from dipy.reconst.dti import (TensorFit, design_matrix, _min_positive_signal,
                              decompose_tensor, from_lower_triangular,
                              lower_triangular, apparent_diffusion_coef)
from dipy.reconst.dki import _positive_evals

from dipy.core.sphere import Sphere
from .vec_val_sum import vec_val_vect
from dipy.core.ndindex import ndindex


def fwdti_prediction(params, gtab, Diso=3.0e-3):
    """
    Predict a signal given the parameters of the free water diffusion tensor
    model.
    Parameters
    ----------
    params : ndarray
        Model parameters. The last dimension should have the 12 tensor
        parameters (3 eigenvalues, followed by the 3 corresponding
        eigenvectors) the volume fraction of the free water compartment, and
        the non diffusion-weighted signal S0
    gtab : a GradientTable class instance
        The gradient table for this prediction
    Diso : float, optional
        Value of the free water isotropic diffusion. Default is set to 3e-3
        $mm^{2}.s^{-1}$. Please ajust this value if you are assuming different
        units of diffusion.
    Notes
    -----
    The predicted signal is given by:
    $S(\theta, b) = S_0 * [(1-f) * e^{-b ADC} + f * e^{-b D_{iso}]$, where
    $ADC = \theta Q \theta^T$, $\theta$ is a unit vector pointing at any
    direction on the sphere for which a signal is to be predicted, $b$ is the b
    value provided in the GradientTable input for that direction, $Q$ is the
    quadratic form of the tensor determined by the input parameters, $f$ is the
    free water diffusion compartment, $D_{iso}$ is the free water diffusivity
    which is equal to $3 * 10^{-3} mm^{2}s^{-1} [1]_.
    References
    ----------
    .. [1] Hoy, A.R., Koay, C.G., Kecskemeti, S.R., Alexander, A.L., 2014.
           Optimization of a free water elimination two-compartmental model
           for diffusion tensor imaging. NeuroImage 103, 323-333.
           doi: 10.1016/j.neuroimage.2014.09.053
    """
    evals = params[..., :3]
    evecs = params[..., 3:-2].reshape(params.shape[:-1] + (3, 3))
    f = params[..., 12]
    S0 = params[..., 13]
    qform = vec_val_vect(evecs, evals)
    sphere = Sphere(xyz=gtab.bvecs[~gtab.b0s_mask])
    adc = apparent_diffusion_coef(qform, sphere)
    mask = _positive_evals(evals[..., 0], evals[..., 1], evals[..., 2])

    # First do the calculation for the diffusion weighted measurements:
    pred_sig = np.zeros(f.shape + (gtab.bvals.shape[0],))
    index = ndindex(f.shape)
    for v in index:
        if mask[v]:
            pre_pred_sig = S0[v] * \
            ((1 - f[v]) * np.exp(-gtab.bvals[~gtab.b0s_mask] * adc[v]) +
            f[v] * np.exp(-gtab.bvals[~gtab.b0s_mask] * Diso))

            # Then we need to sort out what goes where:
            pred_s = np.zeros(pre_pred_sig.shape[:-1] + (gtab.bvals.shape[0],))

            # These are the diffusion-weighted values
            pred_s[..., ~gtab.b0s_mask] = pre_pred_sig

            # For completeness, we predict the mean S0 for the non-diffusion
            # weighted measurements, which is our best guess:
            pred_s[..., gtab.b0s_mask] = S0[v]
            pred_sig[v] = pred_s

    return pred_sig


class FreeWaterTensorModel(ReconstModel):
    """ Class for the Free Water Elimitation Diffusion Tensor Model
    """
    def __init__(self, gtab, fit_method="WLS", *args, **kwargs):
        """ Free Water Diffusion Tensor Model [1]_.
        Parameters
        ----------
        gtab : GradientTable class instance
        fit_method : str or callable
            str can be one of the following:
            'WLS' for weighted linear least square fit according to [1]_
                fwdti.wls_fit_tensor
            'NLS' for non-linear least square fit according to [1]_
                fwdti.wls_fit_tensor
            callable has to have the signature:
              fit_method(design_matrix, data, *args, **kwargs)
        args, kwargs : arguments and key-word arguments passed to the
           fit_method. See fwdti.wls_fit_tensor, fwdti.nls_fit_tensor for
           details
        min_signal : float
            The minimum signal value. Needs to be a strictly positive
            number. Default: minimal signal in the data provided to `fit`.
        References
        ----------
        .. [1] Hoy, A.R., Koay, C.G., Kecskemeti, S.R., Alexander, A.L., 2014.
               Optimization of a free water elimination two-compartmental model
               for diffusion tensor imaging. NeuroImage 103, 323-333.
               doi: 10.1016/j.neuroimage.2014.09.053
        """
        ReconstModel.__init__(self, gtab)

        if not callable(fit_method):
            try:
                fit_method = common_fit_methods[fit_method]
            except KeyError:
                e_s = '"' + str(fit_method) + '" is not a known fit '
                e_s += 'method, the fit method should either be a '
                e_s += 'function or one of the common fit methods'
                raise ValueError(e_s)
        self.fit_method = fit_method
        self.design_matrix = design_matrix(self.gtab)
        self.args = args
        self.kwargs = kwargs
        self.min_signal = self.kwargs.pop('min_signal', None)
        if self.min_signal is not None and self.min_signal <= 0:
            e_s = "The `min_signal` key-word argument needs to be strictly"
            e_s += " positive."
            raise ValueError(e_s)

    def fit(self, data, mask=None):
        """ Fit method of the free water elimination DTI model class
        Parameters
        ----------
        data : array
            The measured signal from one voxel.
        mask : array
            A boolean array used to mark the coordinates in the data that
            should be analyzed that has the shape data.shape[:-1]
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

        data_in_mask = np.maximum(data_in_mask, min_signal)
        params_in_mask = self.fit_method(self.design_matrix, data_in_mask,
                                         *self.args, **self.kwargs)

        if mask is None:
            out_shape = data.shape[:-1] + (-1, )
            fwdti_params = params_in_mask.reshape(out_shape)
        else:
            fwdti_params = np.zeros(data.shape[:-1] + (13,))
            fwdti_params[mask, :] = params_in_mask

        return FreeWaterTensorFit(self, fwdti_params)

    def predict(self, fwdti_params):
        """
        Predict a signal for this TensorModel class instance given parameters.
        Parameters
        ----------
        fwdti_params : ndarray
            The last dimension should have 13 parameters: the 12 tensor
            parameters (3 eigenvalues, followed by the 3 corresponding
            eigenvectors) and the volume fraction of the free water compartment
        S0 : float or ndarray
            The non diffusion-weighted signal in every voxel, or across all
            voxels. Default: 1
        """
        return fwdti_prediction(fwdti_params, self.gtab)


class FreeWaterTensorFit(TensorFit):
    """ Class for fitting the Free Water Tensor Model """
    def __init__(self, model, model_params):
        """ Initialize a FreeWaterTensorFit class instance.
        Since the free water tensor model is an extension of DTI, class
        instance is defined as subclass of the TensorFit from dti.py
        Parameters
        ----------
        model : FreeWaterTensorModel Class instance
            Class instance containing the free water tensor model for the fit
        model_params : ndarray (x, y, z, 13) or (n, 13)
            All parameters estimated from the free water tensor model.
            Parameters are ordered as follows:
                1) Three diffusion tensor's eigenvalues
                2) Three lines of the eigenvector matrix each containing the
                   first, second and third coordinates of the eigenvector
                3) The volume fraction of the free water compartment
        """
        TensorFit.__init__(self, model, model_params)

    @property
    def f(self):
        """
        Returns the free water diffusion volume fraction f
        """
        return self.model_params[..., 12]

    def predict(self, gtab):
        r""" Given a free water tensor model fit, predict the signal on the
        vertices of a gradient table
        Parameters
        ----------
        fwdti_params : ndarray (x, y, z, 13) or (n, 13)
            All parameters estimated from the free water tensor model.
            Parameters are ordered as follows:
                1) Three diffusion tensor's eigenvalues
                2) Three lines of the eigenvector matrix each containing the
                   first, second and third coordinates of the eigenvector
                3) The volume fraction of the free water compartment
        gtab : a GradientTable class instance
            The gradient table for this prediction
        S0 : float or ndarray (optional)
            The non diffusion-weighted signal in every voxel, or across all
            voxels. Default: 1
        Notes
        -----
        The predicted signal is given by:
        .. math::
            edit
        """
        return fwdti_prediction(self.model_params, gtab)


def wls_fit_tensor(design_matrix, data, Diso=3e-3, piterations=3,
                   riterations=2):
    r""" Computes weighted least squares (WLS) fit to calculate self-diffusion
    tensor using a linear regression model [1]_.
    Parameters
    ----------
    design_matrix : array (g, 7)
        Design matrix holding the covariants used to solve for the regression
        coefficients.
    data : array ([X, Y, Z, ...], g)
        Data or response variables holding the data. Note that the last
        dimension should contain the data. It makes no copies of data.
    Diso : float, optional
        Value of the free water isotropic diffusion. Default is set to 3e-3
        $mm^{2}.s^{-1}$. Please ajust this value if you are assuming different
        units of diffusion.
    piterations : inter, optional
        Number of iterations used to refine the precision of f. Default is set
        to 3 corresponding to a precision of 0.01.
    riterations : inter, optional
        Number of iteration repetitions with adapted S0. To insure that S0 is
        taken as a model free parameter Each precision iteration is repeated
        riterations times with adapted S0.
    Returns
    -------
    All parameters estimated from the free water tensor model.
    Parameters are ordered as follows:
        1) Three diffusion tensor's eigenvalues
        2) Three lines of the eigenvector matrix each containing the
           first, second and third coordinates of the eigenvector
        3) The volume fraction of the free water compartment
        4) The estimate of the non diffusion-weighted signal S0
    References
    ----------
    .. [1] Chung, SW., Lu, Y., Henry, R.G., 2006. Comparison of bootstrap
       approaches for estimation of uncertainties of DTI parameters.
       NeuroImage 33, 531-541.
    """
    tol = 1e-6

    # preparing data and initializing parametres
    data = np.asarray(data)
    data_flat = data.reshape((-1, data.shape[-1]))
    fw_params = np.empty((len(data_flat), 14))

    # inverting design matrix and defining minimun diffusion aloud
    min_diffusivity = tol / -design_matrix.min()
    inv_design = np.linalg.pinv(design_matrix)

    # lopping WLS solution on all data voxels
    for vox in range(len(data_flat)):
        fw_params[vox] = _wls_iter(design_matrix, inv_design, data_flat[vox],
                                    min_diffusivity, Diso=Diso,
                                    piterations=piterations,
                                    riterations=riterations)

    # Reshape data according to the input data shape
    fw_params = fw_params.reshape((data.shape[:-1]) + (14,))

    return fw_params


def _wls_iter(design_matrix, inv_design, sig, min_diffusivity, Diso=3e-3,
              piterations=3, riterations=2):
    """ Helper function used by wls_fit_tensor - Applies WLS fit of the
    water free elimination model to single voxel signals.
    Parameters
    ----------
    design_matrix : array (g, 7)
        Design matrix holding the covariants used to solve for the regression
        coefficients.
    inv_design : array (g, 7)
        Inverse of the design matrix.
    sig : array (g, )
        Diffusion-weighted signal for a single voxel data.
    min_diffusivity : float
        Because negative eigenvalues are not physical and small eigenvalues,
        much smaller than the diffusion weighting, cause quite a lot of noise
        in metrics such as fa, diffusivity values smaller than
        `min_diffusivity` are replaced with `min_diffusivity`.
    Diso : float, optional
        Value of the free water isotropic diffusion. Default is set to 3e-3
        $mm^{2}.s^{-1}$. Please ajust this value if you are assuming different
        units of diffusion.
    piterations : inter, optional
        Number of iterations used to refine the precision of f. Default is set
        to 3 corresponding to a precision of 0.01.
    riterations : inter, optional
        Number of iteration repetitions with adapted S0. To insure that S0 is
        taken as a model free parameter Each precision iteration is repeated
        riterations times with adapted S0.
    Returns
    -------
    All parameters estimated from the free water tensor model.
    Parameters are ordered as follows:
        1) Three diffusion tensor's eigenvalues
        2) Three lines of the eigenvector matrix each containing the
           first, second and third coordinates of the eigenvector
        3) The volume fraction of the free water compartment
        4) The estimate of the non diffusion-weighted signal S0
    """
    W = design_matrix

    # DTI ordinary linear least square solution
    log_s = np.log(sig)
    ols_result = np.dot(inv_design, log_s)

    # Define weights as diag(yn**2)
    S2 = np.diag(np.exp(2 * np.dot(W, ols_result)))

    # DTI weighted linear least square solution
    WTS2 = np.dot(W.T, S2)
    inv_WT_S2_W = np.linalg.pinv(np.dot(WTS2, W))
    invWTS2W_WTS2 = np.dot(inv_WT_S2_W, WTS2)
    params = np.dot(invWTS2W_WTS2, log_s)

    # General free-water signal contribution
    fwsig = np.exp(np.dot(design_matrix, 
                          np.array([Diso, 0, Diso, 0, 0, Diso, 0])))

    df = 1  # initialize precision
    flow = 0  # lower f evaluated
    fhig = 1  # higher f evaluated
    ns = 9  # initial number of samples per iteration
    nvol = len(sig)
    for p in range(piterations):
        df = df * 0.1
        fs = np.linspace(flow+df, fhig-df, num=ns)  # sampling f
        # repeat fw contribution for all the samples
        SFW = np.array([fwsig,]*ns)
        FS, SI = np.meshgrid(fs, sig)
        for r in range(riterations):
            # Free-water adjusted signal
            S0 = np.exp(-params[6])
            SA = SI - FS*S0*SFW.T
            # SA < 0 means that the signal components from the free water
            # component is larger than the total fiber. This cases are present
            # for inapropriate large volume fractions (given the current S0
            # value estimated). To avoid the log of negative values:
            SA[SA <= 0] = 0.0001  # same min signal assumed in dti.py
            y = np.log(SA / (1-FS))

            # Estimate tissue's tensor from inv(A.T*S2*A)*A.T*S2*y
            S2 = np.diag(np.square(np.dot(W, params)))
            WTS2 = np.dot(W.T, S2)
            inv_WT_S2_W = np.linalg.pinv(np.dot(WTS2, W))
            invWTS2W_WTS2 = np.dot(inv_WT_S2_W, WTS2)
            all_new_params = np.dot(invWTS2W_WTS2, y)

            # compute F2
            S0r = np.exp(-np.array([all_new_params[6],]*nvol))
            SIpred = (1-FS)*np.exp(np.dot(W, all_new_params)) + FS*S0r*SFW.T
            F2 = np.sum(np.square(SI - SIpred), axis=0)

            # Select params for lower F2
            Mind = np.argmin(F2)
            params = all_new_params[:, Mind]
        # Updated f
        f = fs[Mind]
        # refining precision
        flow = f - df
        fhig = f + df
        ns = 19

    S0 = np.exp(-params[6])

    evals, evecs = decompose_tensor(from_lower_triangular(params),
                                    min_diffusivity=min_diffusivity)
    fw_params = np.concatenate((evals, evecs[0], evecs[1], evecs[2], 
                                np.array([f]), np.array([S0])), axis=0)
    return fw_params


def _nlls_err_func(tensor_elements, design_matrix, data, Diso=3e-3,
                   weighting=None, sigma=None):
    """
    Error function for the non-linear least-squares fit of the tensor water
    elimination model.

    Parameters
    ----------
    tensor_elements : array (8, )
        The six independent elements of the diffusion tensor followed by
        -log(S0) and the volume fraction f of the water elimination compartment

    design_matrix : array
        The design matrix

    data : array
        The voxel signal in all gradient directions

    weighting : str (optional).
         Whether to use the Geman McClure weighting criterion (see [1]_
         for details)

    sigma : float or float array (optional)
        If 'sigma' weighting is used, we will weight the error function
        according to the background noise estimated either in aggregate over
        all directions (when a float is provided), or to an estimate of the
        noise in each diffusion-weighting direction (if an array is
        provided). If 'gmm', the Geman-Mclure M-estimator is used for
        weighting.
    """
    f = tensor_elements[7]
    # This is the predicted signal given the params:
    y = (1-f) * np.exp(np.dot(design_matrix, tensor_elements[:7])) + \
        f * np.exp(np.dot(design_matrix,
                          np.array([Diso, 0, Diso, 0, 0, Diso, 
                                    tensor_elements[6]])))

    # Compute the residuals
    residuals = data - y

    # If we don't want to weight the residuals, we are basically done:
    if weighting is None:
        # And we return the SSE:
        return residuals
    se = residuals ** 2
    # If the user provided a sigma (e.g 1.5267 * std(background_noise), as
    # suggested by Chang et al.) we will use it:
    if weighting == 'sigma':
        if sigma is None:
            e_s = "Must provide sigma value as input to use this weighting"
            e_s += " method"
            raise ValueError(e_s)
        w = 1/(sigma**2)

    elif weighting == 'gmm':
        # We use the Geman McClure M-estimator to compute the weights on the
        # residuals:
        C = 1.4826 * np.median(np.abs(residuals - np.median(residuals)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = 1/(se + C**2)
            # The weights are normalized to the mean weight (see p. 1089):
            w = w/np.mean(w)

    # Return the weighted residuals:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.sqrt(w * se)


def nlls_fit_tensor(design_matrix, data, fw_params=None, Diso=3e-3,
                    weighting=None, sigma=None, jac=False):
    """
    Fit the water elimination tensor model using the non-linear least-squares.

    Parameters
    ----------
    design_matrix : array (g, 7)
        Design matrix holding the covariants used to solve for the regression
        coefficients.

    data : array ([X, Y, Z, ...], g)
        Data or response variables holding the data. Note that the last
        dimension should contain the data. It makes no copies of data.

    fw_params: ([X, Y, Z, ...], 13), optional
           A first model parameters guess (3 eigenvalues follow the coordinates
           of 3 eigenvalues and the volume fraction of the free water
           compartment). If the initial fw_paramters are not given, function
           will use the WLS free water elimination algorithm to estimate the
           parameters first guess.
           Default: None

    Diso : float, optional
        Value of the free water isotropic diffusion. Default is set to 3e-3
        $mm^{2}.s^{-1}$. Please ajust this value if you are assuming different
        units of diffusion.

    weighting: str
           the weighting scheme to use in considering the
           squared-error. Default behavior is to use uniform weighting. Other
           options: 'sigma' 'gmm'

    sigma: float
        If the 'sigma' weighting scheme is used, a value of sigma needs to be
        provided here. According to [Chang2005]_, a good value to use is
        1.5267 * std(background_noise), where background_noise is estimated
        from some part of the image known to contain no signal (only noise).

    jac : bool
        Use the Jacobian? Default: False

    Returns
    -------
    nlls_params: for each voxel the eigen-values and eigen-vectors of the
        tissue tensor and the volume fraction of the free water compartment.
    """
    # Flatten for the iteration over voxels:
    flat_data = data.reshape((-1, data.shape[-1]))

    # Use the WLS method parameters as the starting point if fw_params is None:
    if fw_params==None:
        fw_params = wls_fit_tensor(design_matrix, flat_data,  Diso=Diso)

    for vox in range(flat_data.shape[0]):
        if np.all(flat_data[vox] == 0):
            raise ValueError("The data in this voxel contains only zeros")

        params = fw_params[vox]

        # converting evals and evecs to diffusion tensor elements
        evals = params[:3]
        evecs = params[3:12].reshape((3, 3))
        dt = lower_triangular(np.dot(np.dot(evecs, np.diag(evals)), evecs.T))
        f = params[12]
        s0 = params[13]
        start_params = np.concatenate((dt, [-np.log(s0), f]), axis=0)
        # Do the optimization in this voxel:
        # if jac:
            #this_tensor, status = opt.leastsq(_nlls_err_func, start_params,
            #                                  args=(design_matrix,
            #                                        flat_data[vox],
            #                                        weighting,
            #                                        sigma),
            #                                  Dfun=_nlls_jacobian_func)
        #else:
        this_tensor, status = opt.leastsq(_nlls_err_func, start_params[:8],
                                          args=(design_matrix,
                                                flat_data[vox],
                                                Diso,
                                                weighting,
                                                sigma))

        # The parameters are the evals and the evecs:
        try:
            evals, evecs = decompose_tensor(
                               from_lower_triangular(this_tensor[:6]))
            fw_params[vox, :3] = evals
            fw_params[vox, 3:12] = evecs.ravel()
            fw_params[vox, 12] = this_tensor[7]
            fw_params[vox, 13] = np.exp(-this_tensor[6])
        # If leastsq failed to converge and produced nans, we'll resort to the
        # OLS solution in this voxel:
        except np.linalg.LinAlgError:
            evals, evecs = decompose_tensor(
                              from_lower_triangular(start_params[:6]))
            fw_params[vox, :3] = evals
            fw_params[vox, 3:] = evecs.ravel()
            fw_params[vox, 12] = start_params[7]
            fw_params[vox, 13] = np.exp(-start_params[6])

    fw_params.shape = data.shape[:-1] + (14,)
    return fw_params


common_fit_methods = {'WLLS': wls_fit_tensor,
                      'WLS': wls_fit_tensor,
                      'NLLS': nlls_fit_tensor,
                      'NLS': nlls_fit_tensor,
                      }
