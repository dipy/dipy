""" Classes and functions for fitting tensors without free water
contamination """

import warnings

import numpy as np

import scipy.optimize as opt

from dipy.reconst.base import ReconstModel

from dipy.reconst.dti import (TensorFit, design_matrix, decompose_tensor,
                              _decompose_tensor_nan, from_lower_triangular,
                              lower_triangular,  eig_from_lo_tri,
                              MIN_POSITIVE_SIGNAL, ols_fit_tensor,
                              fractional_anisotropy, mean_diffusivity)

from dipy.reconst.dki import _positive_evals

from dipy.reconst.vec_val_sum import vec_val_vect
from dipy.core.ndindex import ndindex
from dipy.core.gradients import check_multi_b
from dipy.reconst.multi_voxel import multi_voxel_fit

from dipy.core.onetime import auto_attr

# to plot error evolution of gradient descent (for debugging)
from matplotlib import pyplot as plt


def fwdti_prediction(params, gtab, S0=1, Diso=3.0e-3):
    r""" Signal prediction given the free water DTI model parameters.

    Parameters
    ----------
    params : (..., 13) ndarray
        Model parameters. The last dimension should have the 12 tensor
        parameters (3 eigenvalues, followed by the 3 corresponding
        eigenvectors) and the volume fraction of the free water compartment.
    gtab : a GradientTable class instance
        The gradient table for this prediction
    S0 : float or ndarray
        The non diffusion-weighted signal in every voxel, or across all
        voxels. Default: 1
    Diso : float, optional
        Value of the free water isotropic diffusion. Default is set to 3e-3
        $mm^{2}.s^{-1}$. Please adjust this value if you are assuming different
        units of diffusion.

    Returns
    --------
    S : (..., N) ndarray
        Simulated signal based on the free water DTI model

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
    .. [1] Henriques, R.N., Rokem, A., Garyfallidis, E., St-Jean, S.,
           Peterson E.T., Correia, M.M., 2017. [Re] Optimization of a free
           water elimination two-compartment model for diffusion tensor
           imaging. ReScience volume 3, issue 1, article number 2
    """
    evals = params[..., :3]
    evecs = params[..., 3:-1].reshape(params.shape[:-1] + (3, 3))
    f = params[..., 12]
    qform = vec_val_vect(evecs, evals)
    lower_dt = lower_triangular(qform, S0)
    lower_diso = lower_dt.copy()
    lower_diso[..., 0] = lower_diso[..., 2] = lower_diso[..., 5] = Diso
    lower_diso[..., 1] = lower_diso[..., 3] = lower_diso[..., 4] = 0
    D = design_matrix(gtab)

    pred_sig = np.zeros(f.shape + (gtab.bvals.shape[0],))
    mask = _positive_evals(evals[..., 0], evals[..., 1], evals[..., 2])
    index = ndindex(f.shape)
    for v in index:
        if mask[v]:
            pred_sig[v] = (1 - f[v]) * np.exp(np.dot(lower_dt[v], D.T)) + \
                          f[v] * np.exp(np.dot(lower_diso[v], D.T))

    return pred_sig


class FreeWaterTensorModel(ReconstModel):
    """ Class for the Free Water Elimination Diffusion Tensor Model """
    def __init__(self, gtab, fit_method="NLS", *args, **kwargs):
        """ Free Water Diffusion Tensor Model [1]_.

        Parameters
        ----------
        gtab : GradientTable class instance
        fit_method : str or callable
            str can be one of the following:

            'WLS' for weighted linear least square fit according to [1]_
                :func:`fwdti.wls_iter`
            'NLS' for non-linear least square fit according to [1]_
                :func:`fwdti.nls_iter`

            callable has to have the signature:
              fit_method(design_matrix, data, *args, **kwargs)
        args, kwargs : arguments and key-word arguments passed to the
           fit_method. See fwdti.wls_iter, fwdti.nls_iter for
           details

        References
        ----------
        .. [1] Henriques, R.N., Rokem, A., Garyfallidis, E., St-Jean, S.,
               Peterson E.T., Correia, M.M., 2017. [Re] Optimization of a free
               water elimination two-compartment model for diffusion tensor
               imaging. ReScience volume 3, issue 1, article number 2
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

        # Check if at least three b-values are given
        enough_b = check_multi_b(self.gtab, 3, non_zero=False)
        if not enough_b:
            mes = "fwDTI requires at least 3 b-values (which can include b=0)"
            raise ValueError(mes)

    @multi_voxel_fit
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
        S0 = np.mean(data[self.gtab.b0s_mask])
        fwdti_params = self.fit_method(self.design_matrix, data, S0,
                                       *self.args, **self.kwargs)

        return FreeWaterTensorFit(self, fwdti_params)

    def predict(self, fwdti_params, S0=1):
        """ Predict a signal for this TensorModel class instance given
        parameters.

        Parameters
        ----------
        fwdti_params : (..., 13) ndarray
            The last dimension should have 13 parameters: the 12 tensor
            parameters (3 eigenvalues, followed by the 3 corresponding
            eigenvectors) and the free water volume fraction.
        S0 : float or ndarray
            The non diffusion-weighted signal in every voxel, or across all
            voxels. Default: 1

        Returns
        --------
        S : (..., N) ndarray
            Simulated signal based on the free water DTI model
        """
        return fwdti_prediction(fwdti_params, self.gtab, S0=S0)


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
        model_paramsmodel_params : ndarray (x, y, z, 13) or (n, 13)
            All parameters estimated from the free water tensor model.
            Parameters are ordered as follows:
                1) Three diffusion tensor's eigenvalues
                2) Three lines of the eigenvector matrix each containing the
                   first, second and third coordinates of the eigenvector
                3) The volume fraction of the free water compartment

        References
        ----------
        .. [1] Henriques, R.N., Rokem, A., Garyfallidis, E., St-Jean, S.,
               Peterson E.T., Correia, M.M., 2017. [Re] Optimization of a free
               water elimination two-compartment model for diffusion tensor
               imaging. ReScience volume 3, issue 1, article number 2
        """
        TensorFit.__init__(self, model, model_params)

    @property
    def f(self):
        """ Returns the free water diffusion volume fraction f """
        return self.model_params[..., 12]

    def predict(self, gtab, S0=1):
        r""" Given a free water tensor model fit, predict the signal on the
        vertices of a gradient table

        Parameters
        ----------
        gtab : a GradientTable class instance
            The gradient table for this prediction

        S0 : float array
           The mean non-diffusion weighted signal in each voxel. Default: 1 in
           all voxels.

        Returns
        --------
        S : (..., N) ndarray
            Simulated signal based on the free water DTI model
        """
        return fwdti_prediction(self.model_params, gtab, S0=S0)


def wls_iter(design_matrix, sig, S0, Diso=3e-3, mdreg=2.7e-3,
             min_signal=1.0e-6, piterations=3):
    """ Applies weighted linear least squares fit of the water free elimination
    model to single voxel signals.

    Parameters
    ----------
    design_matrix : array (g, 7)
        Design matrix holding the covariants used to solve for the regression
        coefficients.
    sig : array (g, )
        Diffusion-weighted signal for a single voxel data.
    S0 : float
        Non diffusion weighted signal (i.e. signal for b-value=0).
    Diso : float, optional
        Value of the free water isotropic diffusion. Default is set to 3e-3
        $mm^{2}.s^{-1}$. Please adjust this value if you are assuming different
        units of diffusion.
     mdreg : float, optimal
        DTI's mean diffusivity regularization threshold. If standard DTI
        diffusion tensor's mean diffusivity is almost near the free water
        diffusion value, the diffusion signal is assumed to be only free water
        diffusion (i.e. volume fraction will be set to 1 and tissue's diffusion
        parameters are set to zero). Default md_reg is 2.7e-3 $mm^{2}.s^{-1}$
        (corresponding to 90% of the free water diffusion value).
    min_signal : float
        The minimum signal value. Needs to be a strictly positive
        number. Default: minimal signal in the data provided to `fit`.
    piterations : inter, optional
        Number of iterations used to refine the precision of f. Default is set
        to 3 corresponding to a precision of 0.01.

    Returns
    -------
    All parameters estimated from the free water tensor model.
    Parameters are ordered as follows:
        1) Three diffusion tensor's eigenvalues
        2) Three lines of the eigenvector matrix each containing the
           first, second and third coordinates of the eigenvector
        3) The volume fraction of the free water compartment
    """
    W = design_matrix

    # DTI ordinary linear least square solution
    log_s = np.log(np.maximum(sig, min_signal))

    # Define weights
    S2 = np.diag(sig**2)

    # DTI weighted linear least square solution
    WTS2 = np.dot(W.T, S2)
    inv_WT_S2_W = np.linalg.pinv(np.dot(WTS2, W))
    invWTS2W_WTS2 = np.dot(inv_WT_S2_W, WTS2)
    params = np.dot(invWTS2W_WTS2, log_s)

    md = (params[0] + params[2] + params[5]) / 3
    # Process voxel if it has significant signal from tissue
    if md < mdreg and np.mean(sig) > min_signal and S0 > min_signal:
        # General free-water signal contribution
        fwsig = np.exp(np.dot(design_matrix,
                              np.array([Diso, 0, Diso, 0, 0, Diso, 0])))

        df = 1  # initialize precision
        flow = 0  # lower f evaluated
        fhig = 1  # higher f evaluated
        ns = 9  # initial number of samples per iteration
        for p in range(piterations):
            df = df * 0.1
            fs = np.linspace(flow+df, fhig-df, num=ns)  # sampling f
            SFW = np.array([fwsig, ]*ns)  # repeat contributions for all values
            FS, SI = np.meshgrid(fs, sig)
            SA = SI - FS*S0*SFW.T
            # SA < 0 means that the signal components from the free water
            # component is larger than the total fiber. This cases are present
            # for inappropriate large volume fractions (given the current S0
            # value estimated). To overcome this issue negative SA are replaced
            # by data's min positive signal.
            SA[SA <= 0] = min_signal
            y = np.log(SA / (1-FS))
            all_new_params = np.dot(invWTS2W_WTS2, y)
            # Select params for lower F2
            SIpred = (1-FS)*np.exp(np.dot(W, all_new_params)) + FS*S0*SFW.T
            F2 = np.sum(np.square(SI - SIpred), axis=0)
            Mind = np.argmin(F2)
            params = all_new_params[:, Mind]
            f = fs[Mind]  # Updated f
            flow = f - df  # refining precision
            fhig = f + df
            ns = 19

        evals, evecs = decompose_tensor(from_lower_triangular(params))
        fw_params = np.concatenate((evals, evecs[0], evecs[1], evecs[2],
                                    np.array([f])), axis=0)
    else:
        fw_params = np.zeros(13)
        if md > mdreg:
            fw_params[12] = 1.0

    return fw_params


def wls_fit_tensor(gtab, data, Diso=3e-3, mask=None, min_signal=1.0e-6,
                   piterations=3, mdreg=2.7e-3):
    r""" Computes weighted least squares (WLS) fit to calculate self-diffusion
    tensor using a linear regression model [1]_.

    Parameters
    ----------
    gtab : a GradientTable class instance
        The gradient table containing diffusion acquisition parameters.
    data : ndarray ([X, Y, Z, ...], g)
        Data or response variables holding the data. Note that the last
        dimension should contain the data. It makes no copies of data.
    Diso : float, optional
        Value of the free water isotropic diffusion. Default is set to 3e-3
        $mm^{2}.s^{-1}$. Please adjust this value if you are assuming different
        units of diffusion.
    mask : array, optional
        A boolean array used to mark the coordinates in the data that should
        be analyzed that has the shape data.shape[:-1]
    min_signal : float
        The minimum signal value. Needs to be a strictly positive
        number. Default: 1.0e-6.
    piterations : inter, optional
        Number of iterations used to refine the precision of f. Default is set
        to 3 corresponding to a precision of 0.01.
    mdreg : float, optimal
        DTI's mean diffusivity regularization threshold. If standard DTI
        diffusion tensor's mean diffusivity is almost near the free water
        diffusion value, the diffusion signal is assumed to be only free water
        diffusion (i.e. volume fraction will be set to 1 and tissue's diffusion
        parameters are set to zero). Default md_reg is 2.7e-3 $mm^{2}.s^{-1}$
        (corresponding to 90% of the free water diffusion value).

    Returns
    -------
    fw_params : ndarray (x, y, z, 13)
        Matrix containing in the last dimension the free water model parameters
        in the following order:
            1) Three diffusion tensor's eigenvalues
            2) Three lines of the eigenvector matrix each containing the
               first, second and third coordinates of the eigenvector
            3) The volume fraction of the free water compartment.

    References
    ----------
    .. [1] Henriques, R.N., Rokem, A., Garyfallidis, E., St-Jean, S.,
           Peterson E.T., Correia, M.M., 2017. [Re] Optimization of a free
           water elimination two-compartment model for diffusion tensor
           imaging. ReScience volume 3, issue 1, article number 2
    """
    fw_params = np.zeros(data.shape[:-1] + (13,))
    W = design_matrix(gtab)

    # Prepare mask
    if mask is None:
        mask = np.ones(data.shape[:-1], dtype=bool)
    else:
        if mask.shape != data.shape[:-1]:
            raise ValueError("Mask is not the same shape as data.")
        mask = np.array(mask, dtype=bool, copy=False)

    # Prepare S0
    S0 = np.mean(data[:, :, :, gtab.b0s_mask], axis=-1)

    index = ndindex(mask.shape)
    for v in index:
        if mask[v]:
            params = wls_iter(W, data[v], S0[v], min_signal=min_signal,
                              Diso=3e-3, piterations=piterations, mdreg=mdreg)
            fw_params[v] = params

    return fw_params


def _nls_err_func(tensor_elements, design_matrix, data, Diso=3e-3,
                  weighting=None, sigma=None, cholesky=False,
                  f_transform=False):
    """ Error function for the non-linear least-squares fit of the tensor water
    elimination model.

    Parameters
    ----------
    tensor_elements : array (8, )
        The six independent elements of the diffusion tensor followed by
        -log(S0) and the volume fraction f of the water elimination
        compartment. Note that if cholesky is set to true, tensor elements are
        assumed to be written as Cholesky's decomposition elements. If
        f_transform is true, volume fraction f has to be converted to
        ft = arcsin(2*f - 1) + pi/2
    design_matrix : array
        The design matrix
    data : array
        The voxel signal in all gradient directions
    Diso : float, optional
        Value of the free water isotropic diffusion. Default is set to 3e-3
        $mm^{2}.s^{-1}$. Please adjust this value if you are assuming different
        units of diffusion.
    weighting : str (optional).
         Whether to use the Geman-McClure weighting criterion (see [1]_
         for details)
    sigma : float or float array (optional)
        If 'sigma' weighting is used, we will weight the error function
        according to the background noise estimated either in aggregate over
        all directions (when a float is provided), or to an estimate of the
        noise in each diffusion-weighting direction (if an array is
        provided). If 'gmm', the Geman-Mclure M-estimator is used for
        weighting.
    cholesky : bool, optional
        If true, the diffusion tensor elements were decomposed using Cholesky
        decomposition. See fwdti.nls_fit_tensor
        Default: False
    f_transform : bool, optional
        If true, the water volume fraction was converted to
        ft = arcsin(2*f - 1) + pi/2, insuring f estimates between 0 and 1.
        See fwdti.nls_fit_tensor
        Default: True
    """
    tensor = np.copy(tensor_elements)
    if cholesky:
        tensor[:6] = cholesky_to_lower_triangular(tensor[:6])

    if f_transform:
        f = 0.5 * (1 + np.sin(tensor[7] - np.pi/2))
    else:
        f = tensor[7]

    # This is the predicted signal given the params:
    y = (1-f) * np.exp(np.dot(design_matrix, tensor[:7])) + \
        f * np.exp(np.dot(design_matrix,
                          np.array([Diso, 0, Diso, 0, 0, Diso, tensor[6]])))

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
        # We use the Geman-McClure M-estimator to compute the weights on the
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


def _nls_jacobian_func(tensor_elements, design_matrix, data, Diso=3e-3,
                       weighting=None, sigma=None, cholesky=False,
                       f_transform=False):
    """The Jacobian is the first derivative of the least squares error
    function.

    Parameters
    ----------
    tensor_elements : array (8, )
        The six independent elements of the diffusion tensor followed by
        -log(S0) and the volume fraction f of the water elimination
        compartment. Note that if f_transform is true, volume fraction f is
        converted to ft = arcsin(2*f - 1) + pi/2
    design_matrix : array
        The design matrix
    Diso : float, optional
        Value of the free water isotropic diffusion. Default is set to 3e-3
        $mm^{2}.s^{-1}$. Please adjust this value if you are assuming different
        units of diffusion.
    f_transform : bool, optional
        If true, the water volume fraction was converted to
        ft = arcsin(2*f - 1) + pi/2, insuring f estimates between 0 and 1.
        See fwdti.nls_fit_tensor
        Default: True
    """
    tensor = np.copy(tensor_elements)
    if f_transform:
        f = 0.5 * (1 + np.sin(tensor[7] - np.pi/2))
    else:
        f = tensor[7]

    t = np.exp(np.dot(design_matrix, tensor[:7]))
    s = np.exp(np.dot(design_matrix,
                      np.array([Diso, 0, Diso, 0, 0, Diso, tensor[6]])))
    T = (f-1.0) * t[:, None] * design_matrix
    S = np.zeros(design_matrix.shape)
    S[:, 6] = f * s

    if f_transform:
        df = (t-s) * (0.5*np.cos(tensor[7]-np.pi/2))
    else:
        df = (t-s)
    return np.concatenate((T - S, df[:, None]), axis=1)


def nls_iter(design_matrix, sig, S0, Diso=3e-3, mdreg=2.7e-3,
             min_signal=1.0e-6, cholesky=False, f_transform=True, jac=False,
             weighting=None, sigma=None):
    """ Applies non linear least squares fit of the water free elimination
    model to single voxel signals.

    Parameters
    ----------
    design_matrix : array (g, 7)
        Design matrix holding the covariants used to solve for the regression
        coefficients.
    sig : array (g, )
        Diffusion-weighted signal for a single voxel data.
    S0 : float
        Non diffusion weighted signal (i.e. signal for b-value=0).
    Diso : float, optional
        Value of the free water isotropic diffusion. Default is set to 3e-3
        $mm^{2}.s^{-1}$. Please adjust this value if you are assuming different
        units of diffusion.
    mdreg : float, optimal
        DTI's mean diffusivity regularization threshold. If standard DTI
        diffusion tensor's mean diffusivity is almost near the free water
        diffusion value, the diffusion signal is assumed to be only free water
        diffusion (i.e. volume fraction will be set to 1 and tissue's diffusion
        parameters are set to zero). Default md_reg is 2.7e-3 $mm^{2}.s^{-1}$
        (corresponding to 90% of the free water diffusion value).
    min_signal : float
        The minimum signal value. Needs to be a strictly positive
        number.
    cholesky : bool, optional
        If true it uses Cholesky decomposition to insure that diffusion tensor
        is positive define.
        Default: False
    f_transform : bool, optional
        If true, the water volume fractions is converted during the convergence
        procedure to ft = arcsin(2*f - 1) + pi/2, insuring f estimates between
        0 and 1.
        Default: True
    jac : bool
        Use the Jacobian? Default: False
    weighting: str, optional
        the weighting scheme to use in considering the
        squared-error. Default behavior is to use uniform weighting. Other
        options: 'sigma' 'gmm'
    sigma: float, optional
        If the 'sigma' weighting scheme is used, a value of sigma needs to be
        provided here. According to [Chang2005]_, a good value to use is
        1.5267 * std(background_noise), where background_noise is estimated
        from some part of the image known to contain no signal (only noise).

    Returns
    -------
    All parameters estimated from the free water tensor model.
    Parameters are ordered as follows:
        1) Three diffusion tensor's eigenvalues
        2) Three lines of the eigenvector matrix each containing the
           first, second and third coordinates of the eigenvector
        3) The volume fraction of the free water compartment.
    """
    # Initial guess
    params = wls_iter(design_matrix, sig, S0,
                      min_signal=min_signal, Diso=Diso, mdreg=mdreg)

    # Process voxel if it has significant signal from tissue
    if params[12] < 0.99 and np.mean(sig) > min_signal and S0 > min_signal:
        # converting evals and evecs to diffusion tensor elements
        evals = params[:3]
        evecs = params[3:12].reshape((3, 3))
        dt = lower_triangular(vec_val_vect(evecs, evals))

        # Cholesky decomposition if requested
        if cholesky:
            dt = lower_triangular_to_cholesky(dt)

        # f transformation if requested
        if f_transform:
            f = np.arcsin(2*params[12] - 1) + np.pi/2
        else:
            f = params[12]

        # Use the Levenberg-Marquardt algorithm wrapped in opt.leastsq
        start_params = np.concatenate((dt, [-np.log(S0), f]), axis=0)
        if jac:
            this_tensor, status = opt.leastsq(_nls_err_func, start_params[:8],
                                              args=(design_matrix, sig, Diso,
                                                    weighting, sigma, cholesky,
                                                    f_transform),
                                              Dfun=_nls_jacobian_func)
        else:
            this_tensor, status = opt.leastsq(_nls_err_func, start_params[:8],
                                              args=(design_matrix, sig, Diso,
                                                    weighting, sigma, cholesky,
                                                    f_transform))

        # Process tissue diffusion tensor
        if cholesky:
            this_tensor[:6] = cholesky_to_lower_triangular(this_tensor[:6])

        evals, evecs = _decompose_tensor_nan(
            from_lower_triangular(this_tensor[:6]),
            from_lower_triangular(start_params[:6]))

        # Process water volume fraction f
        f = this_tensor[7]
        if f_transform:
            f = 0.5 * (1 + np.sin(f - np.pi/2))

        params = np.concatenate((evals, evecs[0], evecs[1], evecs[2],
                                 np.array([f])), axis=0)
    return params


def nls_fit_tensor(gtab, data, mask=None, Diso=3e-3, mdreg=2.7e-3,
                   min_signal=1.0e-6, f_transform=True, cholesky=False,
                   jac=False, weighting=None, sigma=None):
    """
    Fit the water elimination tensor model using the non-linear least-squares.

    Parameters
    ----------
    gtab : a GradientTable class instance
        The gradient table containing diffusion acquisition parameters.
    data : ndarray ([X, Y, Z, ...], g)
        Data or response variables holding the data. Note that the last
        dimension should contain the data. It makes no copies of data.
    mask : array, optional
        A boolean array used to mark the coordinates in the data that should
        be analyzed that has the shape data.shape[:-1]
    Diso : float, optional
        Value of the free water isotropic diffusion. Default is set to 3e-3
        $mm^{2}.s^{-1}$. Please adjust this value if you are assuming different
        units of diffusion.
    mdreg : float, optimal
        DTI's mean diffusivity regularization threshold. If standard DTI
        diffusion tensor's mean diffusivity is almost near the free water
        diffusion value, the diffusion signal is assumed to be only free water
        diffusion (i.e. volume fraction will be set to 1 and tissue's diffusion
        parameters are set to zero). Default md_reg is 2.7e-3 $mm^{2}.s^{-1}$
        (corresponding to 90% of the free water diffusion value).
    min_signal : float
        The minimum signal value. Needs to be a strictly positive
        number. Default: 1.0e-6.
    f_transform : bool, optional
        If true, the water volume fractions is converted during the convergence
        procedure to ft = arcsin(2*f - 1) + pi/2, insuring f estimates between
        0 and 1.
        Default: True
    cholesky : bool, optional
        If true it uses Cholesky decomposition to insure that diffusion tensor
        is positive define.
        Default: False
    jac : bool
        Use the Jacobian? Default: False
    weighting: str, optional
        the weighting scheme to use in considering the
        squared-error. Default behavior is to use uniform weighting. Other
        options: 'sigma' 'gmm'
    sigma: float, optional
        If the 'sigma' weighting scheme is used, a value of sigma needs to be
        provided here. According to [Chang2005]_, a good value to use is
        1.5267 * std(background_noise), where background_noise is estimated
        from some part of the image known to contain no signal (only noise).

    Returns
    -------
    fw_params : ndarray (x, y, z, 13)
        Matrix containing in the dimension the free water model parameters in
        the following order:
            1) Three diffusion tensor's eigenvalues
            2) Three lines of the eigenvector matrix each containing the
               first, second and third coordinates of the eigenvector
            3) The volume fraction of the free water compartment
    """
    fw_params = np.zeros(data.shape[:-1] + (13,))
    W = design_matrix(gtab)

    # Prepare mask
    if mask is None:
        mask = np.ones(data.shape[:-1], dtype=bool)
    else:
        if mask.shape != data.shape[:-1]:
            raise ValueError("Mask is not the same shape as data.")
        mask = np.array(mask, dtype=bool, copy=False)

    # Prepare S0
    S0 = np.mean(data[:, :, :, gtab.b0s_mask], axis=-1)

    index = ndindex(mask.shape)
    for v in index:
        if mask[v]:
            params = nls_iter(W, data[v], S0[v], Diso=Diso, mdreg=mdreg,
                              min_signal=min_signal, f_transform=f_transform,
                              cholesky=cholesky, jac=jac, weighting=weighting,
                              sigma=sigma)
            fw_params[v] = params

    return fw_params


def lower_triangular_to_cholesky(tensor_elements):
    """ Performs Cholesky decomposition of the diffusion tensor

    Parameters
    ----------
    tensor_elements : array (6,)
        Array containing the six elements of diffusion tensor's lower
        triangular.

    Returns
    -------
    cholesky_elements : array (6,)
        Array containing the six Cholesky's decomposition elements
        (R0, R1, R2, R3, R4, R5) [1]_.

    References
    ----------
    .. [1] Koay, C.G., Carew, J.D., Alexander, A.L., Basser, P.J.,
           Meyerand, M.E., 2006. Investigation of anomalous estimates of
           tensor-derived quantities in diffusion tensor imaging. Magnetic
           Resonance in Medicine, 55(4), 930-936. doi:10.1002/mrm.20832
    """
    R0 = np.sqrt(tensor_elements[0])
    R3 = tensor_elements[1] / R0
    R1 = np.sqrt(tensor_elements[2] - R3**2)
    R5 = tensor_elements[3] / R0
    R4 = (tensor_elements[4] - R3*R5) / R1
    R2 = np.sqrt(tensor_elements[5] - R4**2 - R5**2)

    return np.array([R0, R1, R2, R3, R4, R5])


def cholesky_to_lower_triangular(R):
    """ Convert Cholesky decompostion elements to the diffusion tensor elements

    Parameters
    ----------
    R : array (6,)
        Array containing the six Cholesky's decomposition elements
        (R0, R1, R2, R3, R4, R5) [1]_.

    Returns
    -------
    tensor_elements : array (6,)
        Array containing the six elements of diffusion tensor's lower
        triangular.

    References
    ----------
    .. [1] Koay, C.G., Carew, J.D., Alexander, A.L., Basser, P.J.,
           Meyerand, M.E., 2006. Investigation of anomalous estimates of
           tensor-derived quantities in diffusion tensor imaging. Magnetic
           Resonance in Medicine, 55(4), 930-936. doi:10.1002/mrm.20832
    """
    Dxx = R[0]**2
    Dxy = R[0]*R[3]
    Dyy = R[1]**2 + R[3]**2
    Dxz = R[0]*R[5]
    Dyz = R[1]*R[4] + R[3]*R[5]
    Dzz = R[2]**2 + R[4]**2 + R[5]**2
    return np.array([Dxx, Dxy, Dyy, Dxz, Dyz, Dzz])


common_fit_methods = {'WLLS': wls_iter,
                      'WLS': wls_iter,
                      'NLLS': nls_iter,
                      'NLS': nls_iter,
                      }

# ---------------------------- single-shell code -------------------------------

MAX_DIFFFUSIVITY = 5
MIN_DIFFUSIVITY = 0.01


class Manifold():
    """Class to perform Laplace-Beltrami regularization on diffusion manifold"""

    def __init__(self, design_matrix, model_params, attenuations, fmin, fmax,
                 Diso=3, beta=1, mask=None, zooms=None):
        """
        Treats diffusion parameters as a manifold on which regularization is
        performed using a euclidean metric tensor [1] 

        Parameters
        ----------
        design_matrix : array (k, 6)
            The design matrix that hold diffusion gradient information
            Note: this design_matrix has no b0 directions and no dummmy column
            of -1 liki dipy's design matrix
        model_params : ndarray (x, y, z, 13)
            All parameters estimated from the free water tensor model.
            Parameters are ordered as follows:
                1) Three diffusion tensor's eigenvalues
                2) Three lines of the eigenvector matrix each containing the
                   first, second and third coordinates of the eigenvector
                3) The volume fraction of the tissue compartment
        attenuations : array (x, y, z, k)
            Normalized observed attenuations
        fmin : array (x, y, z)
            Lower limit for the tissue fraction
        fmax : array (x, y, z)
            Upper limit for the tissue fraction
        Diso : float
            Diffusivity of free diffusing water at body temperature
        beta : float
            The metric ratio that controls how isotropic is the smoothing of
            the diffusion manifold, see [2] for more details
        mask : boolean array (x, y, z)
            Brain mask of voxels that should be processed
        zooms : array (3, )
            Voxel resolution along x, y and z axes
        
        References
        ----------
        .. [1] Pasternak, O., Maier-Hein, K., Baumgartner,
            C., Shenton, M. E., Rathi, Y., & Westin, C. F. (2014).
            The estimation of free-water corrected diffusion tensors.
            In Visualization and Processing of Tensors and Higher Order
            Descriptors for Multi-Valued Data (pp. 249-270). Springer,
            Berlin, Heidelberg.
        .. [2] Gur, Y., & Sochen, N. (2007, October).
            Fast invariant Riemannian DT-MRI regularization.
            In 2007 IEEE 11th International Conference on Computer Vision
            (pp. 1-7). IEEE.
        """
        # Manifold shape
        self.shape = model_params.shape[:-1]

        # Diffusion parameters
        evals = model_params[..., 0:3]
        evecs = model_params[..., 3:12].reshape(self.shape + (3, 3))
        lowtri = lower_triangular(vec_val_vect(evecs, evals), b0=None)

        # Scaled diffusion parameters, see ref [2]
        self.X = np.copy(lowtri)
        self.X[..., [1, 3, 4]] *= np.sqrt(2)

        # Design matrix
        self.design_matrix = design_matrix

        # Scaled design matrix, for computing fidelity term
        self.dH = np.copy(design_matrix)
        self.dH[..., [1, 3, 4]] *= np.sqrt(2)

        # Metric ratio
        self.beta = beta

        # Mask
        if mask is None:
            self.mask = np.ones(model_params.shape[:-1]).astype(bool)
        else:
            self.mask = mask.astype(bool)
        
        # Masks for derivatives,
        # to avoid unstable derivatives at the mask boundary
        nx, ny, nz = self.mask.shape
        shift_fx = np.append(np.arange(1, nx), nx-1)
        shift_fy = np.append(np.arange(1, ny), ny-1)
        shift_fz = np.append(np.arange(1, nz), nz-1)
        shift_bx = np.append(0, np.arange(nx-1))
        shift_by = np.append(0, np.arange(ny-1))
        shift_bz = np.append(0, np.arange(nz-1))
        self.mask_forward_x = self.mask[shift_fx, ...] * self.mask
        self.mask_forward_y = self.mask[:, shift_fy, :] * self.mask
        self.mask_forward_z = self.mask[..., shift_fz] * self.mask
        self.mask_backward_x = self.mask[shift_bx, ...] * self.mask
        self.mask_backward_y = self.mask[:, shift_by, :] * self.mask
        self.mask_backward_z = self.mask[..., shift_bz] * self.mask

        # Voxel resolution
        if zooms is None:
            self.zooms = np.array([1., 1., 1.])
        else:
            self.zooms = zooms / np.min(zooms)

        # flattened free water tensor components
        self.flat_Diso = np.zeros(self.flat_lowtri.shape)
        self.flat_Diso[..., [0, 2, 5]] = Diso

        # flattened attenuations
        self.flat_attenuations = attenuations[self.mask, :]

        # flattened tissue fraction
        self.flat_fraction = model_params[self.mask, 12][..., None]

        # flattened lower and upper limits for tisue fraction
        self.flat_fmin = fmin[self.mask][..., None]
        self.flat_fmax = fmax[self.mask][..., None]

        # Increment matrices
        self.flat_beltrami = np.zeros(self.flat_fraction.shape[:-1] + (6, ))
        self.flat_fidelity = np.zeros(self.flat_fraction.shape[:-1] + (6, ))
        self.flat_df = np.zeros(self.flat_fraction.shape)

        # cost
        self.flat_cost = np.zeros(self.flat_fraction.shape)
        self.flat_g = np.zeros(self.flat_fraction.shape)


    @staticmethod
    def forward_difference(array, d, axis):
        """
        Forward finite differences

        Parameters
        ----------
        array : (x, y, z, 6) array
            The diffusion components array in lower triangular order
        d : float
            Voxel resolution along the desired axis
        axis : int
            The dimension along which to perform the finite difference:
            0 - derivative along x axis,
            1 - derivative along y axis,
            2 - derivative along z axis.

        Returns
        -------
        darray : (x, y, z, 6) array
            The forward difference of array along the chosen axis,
            normalized by voxel size

        References
        ----------
        .. https://mathworld.wolfram.com/ForwardDifference.html
        """
        n = array.shape[axis]
        shift = np.append(np.arange(1, n), n-1)
        if axis == 0:
            return (array[shift, ...] - array) / d
        elif axis == 1:
            return (array[:, shift, ...] - array) / d
        elif axis == 2:
            return (array[..., shift, :] - array) / d


    @staticmethod
    def backward_difference(array, d, axis):
        """
        Backward finite differences

        Parameters
        ----------
        array : (x, y, z, 6) array
            The diffusion components array in lower triangular order
        d : float
            Voxel resolution along the desired axis
        axis : int
            The dimension along which to perform the finite difference:
            0 - derivative along x axis,
            1 - derivative along y axis,
            2 - derivative along z axis.

        Returns
        -------
        darray : (x, y, z, 6) array
            The backward difference of array along the chosen axis,
            normalized by voxel size

        References
        ----------
        .. https://mathworld.wolfram.com/BackwardDifference.html
        """
        n = array.shape[axis]
        shift = np.append(np.arange(1, n), n-1)
        if axis == 0:
            return (array - array[shift, ...]) / d
        elif axis == 1:
            return (array - array[:, shift, ...]) / d
        elif axis == 2:
            return (array - array[..., shift, :]) / d


    @property
    def flat_lowtri(self):
        out = np.copy(self.X[self.mask, :])
        out[..., [1, 3, 4]] *= 1 / np.sqrt(2)
        return out


    def compute_beltrami(self):
        """
        Laplace-Beltrami operator to regularize the diffusion
        parameters, using an euclidean metric [1]

        References
        ----------
        .. [1] Pasternak, O., Maier-Hein, K., Baumgartner,
            C., Shenton, M. E., Rathi, Y., & Westin, C. F. (2014).
            The estimation of free-water corrected diffusion tensors.
            In Visualization and Processing of Tensors and Higher Order
            Descriptors for Multi-Valued Data (pp. 249-270). Springer,
            Berlin, Heidelberg.
        .. [2] Gur, Y., & Sochen, N. (2007, October).
            Fast invariant Riemannian DT-MRI regularization.
            In 2007 IEEE 11th International Conference on Computer Vision
            (pp. 1-7). IEEE.
        """
        # Computing derivatives
        dx, dy, dz = self.zooms
        X_dx = (Manifold.forward_difference(self.X, dx, 0)
                * self.mask_forward_x[..., None])
        X_dy = (Manifold.forward_difference(self.X, dy, 1)
                * self.mask_forward_y[..., None])
        X_dz = (Manifold.forward_difference(self.X, dz, 2)
                * self.mask_forward_z[..., None])

        # Computing the Manifold metric (Euclidean)  
        g11 = np.sum(X_dx * X_dx, axis=-1) * self.beta + 1.
        g12 = np.sum(X_dx * X_dy, axis=-1) * self.beta
        g22 = np.sum(X_dy * X_dy, axis=-1) * self.beta + 1.
        g13 = np.sum(X_dx * X_dz, axis=-1) * self.beta
        g23 = np.sum(X_dy * X_dz, axis=-1) * self.beta
        g33 = np.sum(X_dz * X_dz, axis=-1) * self.beta + 1.

        # Computing inverse metric
        gdet = (g12 * g13 * g23 * 2 + g11 * g22 * g33
                - g22 * g13**2
                - g33 * g12**2
                - g11 * g23**2)
        # # unstable values
        unstable_g = np.logical_or(gdet <= 0, gdet >= 1000) * self.mask
        gdet[unstable_g] = 1
        g11[unstable_g] = 1
        g12[unstable_g] = 0
        g22[unstable_g] = 1
        g13[unstable_g] = 0
        g23[unstable_g] = 0
        g33[unstable_g] = 1
        # the inverse
        ginv11 = (g22 * g33 - g23**2) / gdet
        ginv22 = (g11 * g33 - g13**2) / gdet
        ginv33 = (g11 * g22 - g12**2) / gdet
        ginv12 = (g13 * g23 - g12 * g33) / gdet
        ginv13 = (g12 * g23 - g13 * g22) / gdet
        ginv23 = (g12 * g13 - g11 * g23) / gdet

        # Computing Beltrami increments
        # auxiliary matrices
        g = np.sqrt(gdet)[..., None]
        g11 = ginv11[..., None]
        g12 = ginv12[..., None]
        g22 = ginv22[..., None]
        g13 = ginv13[..., None]
        g23 = ginv23[..., None]
        g33 = ginv33[..., None]
        Ax = g11 * X_dx + g12 * X_dy + g13 * X_dz
        Ay = g12 * X_dx + g22 * X_dy + g23 * X_dz
        Az = g13 * X_dx + g23 * X_dy + g33 * X_dz
        
        beltrami = (Manifold.backward_difference(g * Ax, dx, 0)
                    * self.mask_backward_x[..., None])
        beltrami += (Manifold.backward_difference(g * Ay, dy, 1)
                     * self.mask_backward_y[..., None])
        beltrami += (Manifold.backward_difference(g * Az, dz, 2)
                     * self.mask_backward_z[..., None])
        beltrami *= 1 / g 
        
        self.flat_beltrami[...] = beltrami[self.mask]

        # Save the unstable voxels masks
        self.unstable_mask = unstable_g[self.mask][..., None]

        # Save srt(det(g))
        self.flat_g[..., 0] = g[self.mask, 0]


    def compute_fidelity(self):
        """
        The fidelity term maintains the parameters close to the model that
        explains the observed signal attenuations, i.e. moves the parameters
        towards the minimum of the cost function [1]

        References
        ----------
        .. [1] Pasternak, O., Maier-Hein, K., Baumgartner,
            C., Shenton, M. E., Rathi, Y., & Westin, C. F. (2014).
            The estimation of free-water corrected diffusion tensors.
            In Visualization and Processing of Tensors and Higher Order
            Descriptors for Multi-Valued Data (pp. 249-270). Springer,
            Berlin, Heidelberg.       
        """
        Awater = np.exp(np.einsum('...j,ij->...i', self.flat_Diso,
                                   self.design_matrix))
        Atissue = np.exp(np.einsum('...j,ij->...i', self.flat_lowtri,
                                   self.design_matrix))
        Cwater = (1 - self.flat_fraction) * Awater
        Ctissue = self.flat_fraction * Atissue
        Amodel = Ctissue + Cwater
        Adiff = Amodel - self.flat_attenuations
        # fidelity term for diffusion components:
        np.einsum('...i,ij->...j', -1 * Adiff * Ctissue,
                  self.dH, out=self.flat_fidelity)
        # fidelity term for tissue fraction:
        np.sum(-1 * (Atissue - Awater) * Adiff, axis=-1,
               out=self.flat_df[..., 0])


    def compute_cost(self, alpha):
        """
        The cost / error function avereged by number of acquired directions
        """
        Awater = np.exp(np.einsum('...j,ij->...i', self.flat_Diso,
                                   self.design_matrix))
        Atissue = np.exp(np.einsum('...j,ij->...i', self.flat_lowtri,
                                   self.design_matrix))
        Cwater = (1 - self.flat_fraction) * Awater
        Ctissue = self.flat_fraction * Atissue
        Amodel = Ctissue + Cwater
        k = Amodel.shape[-1]
        self.flat_cost[..., 0] = np.sum((Amodel - self.flat_attenuations)**2,
                                        axis=-1) / k
        self.flat_cost *= 1/2


    @property
    def update_mask(self):
        """
        only stable voxels should be updated
        """
        # Do not voxels with high free water
        # csf_mask = self.flat_fraction <= 0.2
        # return ~np.logical_or(self.unstable_mask, csf_mask)
        return ~self.unstable_mask

    def update(self, dt, alpha):
        """
        The beltrami and fidelity terms are multiplied by the learning rate
        and used to increment de diffusion components, a separate increment
        is used to update the tissue fraction.

        Parameters
        ----------
        dt : float
            Learning rate
        alpha : int 0 or 1
            This weight controls if the regularization (smoothing) effect
            is turned ON/OFF

        Notes
        -----
        As suggested in [1], at half iterations alpha should be turned to 0,
        in order to avoid excessive regularization and divergence of the
        gradient descent procedure.

        References
        ----------
        .. [1] Pasternak, O., Sochen, N., Gur, Y., Intrator, N., & Assaf, Y.
            (2009). Free water elimination and mapping from diffusion MRI.
            Magnetic Resonance in Medicine: An Official Journal of 
            the International Society for Magnetic Resonance in Medicine,
            62(3), 717-730.
        """
        self.compute_beltrami()
        self.compute_fidelity()

        # Only update stable voxels
        self.flat_beltrami *= self.update_mask
        self.flat_fidelity *= self.update_mask
        self.flat_df *= self.update_mask

        # Update parameters
        self.X[self.mask, :] += dt * (self.flat_fidelity +
                                      self.flat_beltrami * alpha)
        self.flat_fraction += dt * self.flat_df

        # constrain the tissue fraction to its lower and upper bounds
        np.clip(self.flat_fraction, self.flat_fmin, self.flat_fmax,
                out=self.flat_fraction)

        # update cost
        self.compute_cost(alpha)


    @auto_attr
    def parameters(self):
        """
        ouptuts the final parameters in a form consistent with dipy's
        convention

        Returns
        -------
        out : ndarray (x, y, z, 13)
            All parameters estimated from the free water tensor model.
            Parameters are ordered as follows:
                1) Three diffusion tensor's eigenvalues
                2) Three lines of the eigenvector matrix each containing the
                   first, second and third coordinates of the eigenvector
                3) The volume fraction of the tissue compartment
        """
        dti_params = eig_from_lo_tri(self.flat_lowtri)
        out = np.zeros(self.shape + (13, ))
        out[self.mask, 0:12] = dti_params
        out[self.mask, 12] = self.flat_fraction[..., 0]
        return out


class BeltramiModel(ReconstModel):
    """
    Model class that fits the FW-DTI model to single-shell data
    """
    def __init__(self, gtab, init_method='MD', **kwargs):
        """
        This procedure uses the Laplace-Beltrami operator to
        regularize the solution and stabilize the fit with gradient descent,
        first proposed in [1], however, here a euclidean metric is used [2].

        This class is used to set the initialization type and parse keyword
        arguments to the chosen init method and gardient descent

        Parameters
        ----------
        gtab : GradientTable class instance of Dipy
            IMPORTANT: the bvals used to construct gtab must be in the units of
            milisecond/micrometer^2 (i.e. a bval of 1000 becomes 1), otherwise,
            the finite differences used in this fit method can lead to
            underflows/overflows

        init_method : str
            str must be one of the following initialization methods:
            1) 'S0' uses the unweighted image to initialze the tissue
                fraction [1]
            2) 'MD' uses the initial MD map to initialize tissue fraction [3, 4]
            3) 'hybrid' interpolates between 'S0' and 'MD' methods [3, 4]
        kwargs : keyword arguments that depend on the chosen init method,
            see 'fraction_init_s0', 'fraction_init_md' and
            'fraction_init_hybrid' for details;
            other keyword arguments serve as input to the
            'gradient_descent' function

        References
        ----------
        .. [1] Pasternak, O., Sochen, N., Gur, Y., Intrator, N., & Assaf, Y.
                (2009). Free water elimination and mapping from diffusion MRI.
                Magnetic Resonance in Medicine: An Official Journal of 
                the International Society for Magnetic Resonance in Medicine,
                62(3), 717-730.
        .. [2] Pasternak, O., Maier-Hein, K., Baumgartner,
                C., Shenton, M. E., Rathi, Y., & Westin, C. F. (2014).
                The estimation of free-water corrected diffusion tensors.
                In Visualization and Processing of Tensors and Higher Order
                Descriptors for Multi-Valued Data (pp. 249-270). Springer,
                Berlin, Heidelberg.
        .. [3] Ismail, A. A. O., Parker, D., Hernandez-Fernandez, M., Brem,
                S., Alexander, S., Pasternak, O., ... & Verma, R.
                (2018, September). Characterizing Peritumoral Tissue Using
                Free Water Elimination in Clinical DTI.
        .. [4] Ismail, A. A. O., Parker, D., Hernandez-Fernandez, M., Wolf, R.,
                Brem, S., Alexander, S., ... & Verma, R. (2019). Freewater
                EstimatoR using iNtErpolated iniTialization (FERNET): Toward
                Accurate Estimation of Free Water in Peritumoral Region Using
                Single-Shell Diffusion MRI Data.
        """
        ReconstModel.__init__(self, gtab)
        if not callable(init_method):
            try:
                init_method = init_methods[init_method]
            except KeyError:
                e_s = '"' + str(init_method) + '" is not a known init '
                e_s += 'method, the init method should either be a '
                e_s += 'function or one of the available init methods'
                raise ValueError(e_s)
        self.init_method = init_method
        self.kwargs = kwargs
        self.design_matrix = design_matrix(self.gtab)
        init_keys = ('Diso', 'Stissue', 'Swater', 'min_tissue_diff',
                     'max_tissue_diff', 'tissue_MD')
        self.init_kwargs = {k:kwargs[k] for k in init_keys if k in kwargs}
        fit_keys = ('iterations', 'learning_rate', 'zooms', 'metric_ratio'
                    'reg_weight', 'Diso')
        self.fit_kwargs = {k:kwargs[k] for k in fit_keys if k in kwargs}


    def predict(self, model_params, S0=1):
        Diso = self.init_kwargs.get('Diso', 3)
        return mfwdti_prediction(model_params, self.gtab, S0=S0, Diso=Diso)


    def fit(self, data, mask=None):
        """
        Initializes FW-DTI parameters and calls the 'gradient_descent' fucntion

        Parameters
        ----------
        data : array (x, y, z, k)
            Raw data, since this fit method depends on spatial derivatives
            to regularize the solution, it only works on multi-voxel data
        mask : array (x, y, z)
            Brain mask of voxels that should be processed
        """
        # TODO make this work with single voxel, by calling only the init method
        if mask is not None:
            if mask.shape != data.shape[:-1]:
                raise ValueError("Mask is not the same shape as data.")
            mask = mask.astype(bool, copy=False)
        else:
            mask = np.ones(data.shape[:-1]).astype(bool, copy=False)

        # Initializing tissue volume fraction
        data = np.maximum(data, MIN_POSITIVE_SIGNAL)
        masked_data = data[mask, :]
        f0 = np.zeros(data.shape[:-1])
        fmin = np.zeros(data.shape[:-1])
        fmax = np.ones(data.shape[:-1])
        f0[mask], fmin[mask], fmax[mask] = self.init_method(masked_data,
                                                            self.gtab,
                                                            **self.init_kwargs)
        np.clip(f0, fmin, fmax, out=f0) 

        # Initializing tissue tensor
        init_params = np.zeros(data.shape[:-1] + (13, ))
        Diso = self.init_kwargs.get('Diso', 3)
        min_tissue_diff = self.init_kwargs.get('min_tissue_diff', 0.001)
        max_tissue_diff = self.init_kwargs.get('max_tissue_diff', 2.5)
        init_params[mask, 0:12] = tensor_init(masked_data, self.gtab, f0[mask],
                                             min_tissue_diff=min_tissue_diff,
                                             max_tissue_diff=max_tissue_diff,
                                             Diso=Diso)
        init_params[mask, 12] = f0[mask]

        # Voxel where tissue MD > 1.5 can lead implausible fits, in these voxel
        # tissue fraction is set to 0 and diffusion eigvals to 0.001
        md_tissue = np.mean(init_params[..., :3], axis=-1)
        init_params[md_tissue >= 1.5, -1] = 0
        init_params[md_tissue >= 1.5, :3] = 0.001
        init_params[md_tissue >= 1.5, 3:-1] = 0

        # Run gradient descent
        atten, gtab = get_attenuations(data, self.gtab)
        D = design_matrix(gtab)
        beltrami_params = gradient_descent(D, init_params,
                                           atten, fmin, fmax, mask,
                                           **self.fit_kwargs)
        
        fit = BeltramiFit(self, beltrami_params)
    
        # Add the initialization parameters to Class instance (for debugging)
        fit.initial_guess = init_params
        fit.finterval = np.stack((fmin, fmax), axis=-1)

        return fit


class BeltramiFit(TensorFit):
    """
    Class that holds the fit results obtained from fitting the FW-DTI model
    to single shell data (from the BeltramiModel class)
    """
    def __init__(self, model, model_params):
        TensorFit.__init__(self, model, model_params, model_S0=None)
        """
        Initialize a BeltramiFit class instance.
        Since the free water tensor model is an extension of DTI, class
        instance is defined as subclass of the TensorFit from dti.py

        Parameters
        ----------
        model : BeltramiModel class instance
            Class instance containing the single-shell free water tensor model
            for the fit
        model_paramsmodel_params : darray (x, y, z, 13)
            All parameters estimated from the free water tensor model.
            Parameters are ordered as follows:
                1) Three diffusion tensor's eigenvalues
                2) Three lines of the eigenvector matrix each containing the
                   first, second and third coordinates of the eigenvector
                3) The volume fraction of the tissue compartment
        """
    @property
    def f(self):
        return self.model_params[..., 12]


    @property
    def fw(self):
        return (1 - self.model_params[..., 12])


    @property
    def fwmin(self):
        return (1 - self.finterval[..., 1])


    @property
    def fwmax(self):
        return (1 - self.finterval[..., 0])


    @property
    def fw0(self):
        return 1 - self.initial_guess[..., 12]


    @property
    def fa0(self):
        return fractional_anisotropy(self.initial_guess[..., 0:3])


    @property
    def md0(self):
        return mean_diffusivity(self.initial_guess[..., 0:3])


    def predict(self, gtab, S0=1):
        Diso = self.model.fit_kwargs.get('Diso', 3)
        return fwdti_prediction(self.model_params, gtab, S0=S0, Diso=Diso)


def get_attenuations(signal, gtab):
    """
    Pre-processing to get normalized attenuations

    Parameters
    ----------
    signal : array (x, y, z, k)
        Raw data
    gtab : GradientTable class instance

    Returns
    -------
    this_Ak : array (x, y, z, k)
        Normalized attenuations, without the volumes acquired with b0;
        The first volume (k = 0) consists of 1's (dummy data)
    this_gtab : GradientTable class instance
        GradientTable without the origianl b0 directions, only a single dummy b0
    """
    # Averaging S0 and getting normalized attenuations
    b0_inds = gtab.b0s_mask
    S0 = np.mean(signal[..., b0_inds], axis=-1)
    Sk = signal[..., ~b0_inds]
    Ak = Sk / S0[..., None]

    # Correcting non realistic attenuations
    bvals = gtab.bvals[~b0_inds]
    bvecs = gtab.bvecs[~b0_inds]
    Amin = np.exp(-bvals * MAX_DIFFFUSIVITY)
    Amin = np.tile(Amin, Ak.shape[:-1] + (1, ))
    Amax = np.exp(-bvals * MIN_DIFFUSIVITY)
    Amax = np.tile(Amax, Ak.shape[:-1] + (1, ))
    np.clip(Ak, Amin, Amax, out=Ak)

    # Adding 'dummy' b0 zero data to attenuations and gtab
    bvals = np.insert(bvals, 0 , 0)
    bvecs = np.insert(bvecs, 0, np.array([0, 0, 0]), axis=0)
    this_gtab = gradient_table(bvals, bvecs) 
    this_Ak = np.ones(Ak.shape[:-1] + (Ak.shape[-1] + 1, ))
    this_gtab = gradient_table(bvals, bvecs, b0_threshold=0)
    this_Ak[..., 1:] = Ak

    return (this_Ak, this_gtab)


def fraction_init_s0(signal, gtab, Diso=3, Stissue=None, Swater=None,
                     min_tissue_diff=0.001, max_tissue_diff=2.5):
    """
    Initializes the tissue fraction based on S0 image [1]

    Parameters
    ----------
    signal : array (x, y, z, k)
        Raw data
    gtab : GradientTable class instance
    Diso : float
        Diffusivity of free water at body temperature
    Stissue : float
        S0 signal intensity representative of tissue, see Note
    Swater : float
        S0 signal intensity representative of water, see Note
    min_tissue_diff : float
        minimum diffusivity expected in tissue
    max_tissue_diff : floar
        maximum diffusivity expected in tissue

    Returns
    -------
    f0 : array (x, y, z)
        The initialized tissue fraction
    fmin : array (x, y, z)
        Lower limit for the tissue fraction
    fmax : array (x, y, z)
        Upper limit for the tissue fraction

    Note
    ----
    The pair (Stissue, Swater) must be chosen from the S0 image,
    by looking at a region of interest of WM or CSF. If no values are provided,
    the algorithm tries to roughly estimate it from the S0 percentiles,
    although this is not recomended.

    References
    ----------
    .. [1] Pasternak, O., Sochen, N., Gur, Y., Intrator, N., & Assaf, Y.
            (2009). Free water elimination and mapping from diffusion MRI.
            Magnetic Resonance in Medicine: An Official Journal of 
            the International Society for Magnetic Resonance in Medicine,
            62(3), 717-730.
    """

    S0 = np.mean(signal[..., gtab.b0s_mask], axis=-1)
    if Stissue is None or Swater is None:
        Stissue = np.percentile(S0, 75) 
        Swater = np.percentile(S0, 95)
        print('No Stissue and Swater were given, values chosen based on S0 percentiles:')
        print('Stissue = ' + str(Stissue))
        print('Swater = ' + str(Swater))

    # Normalized attenuations
    Ak, this_gtab = get_attenuations(signal, gtab)
    Ak = Ak[..., 1:]
    bvals = this_gtab.bvals[1:]  # non zero bvals
    Awater = np.exp(-bvals * Diso)
    Awater = np.tile(Awater, Ak.shape[:-1] + (1, ))

    # Min and Max attenuations expected in tissue
    Atissue_min = np.exp(-bvals * max_tissue_diff)
    Atissue_max = np.exp(-bvals * min_tissue_diff)

    # Initial volume fraction
    f0 = 1 - np.log(S0 / Stissue) / np.log(Swater / Stissue)

    # Min and Max volume fraction
    fmin = np.min(Ak - Awater, axis=-1) / np.max(Atissue_max - Awater, axis=-1)
    fmax = np.max(Ak - Awater, axis=-1) / np.min(Atissue_min - Awater, axis=-1)
    fmin[fmin <= 0] = 0.0001
    fmin[fmin >= 1] = 1 - 0.0001
    fmax[fmax <= 0] = 0.0001
    fmax[fmax >= 1] = 1 - 0.0001

    return (f0, fmin, fmax)


def fraction_init_md(signal, gtab, Diso=3, tissue_MD=0.6):
    """
    Initializes the tissue fraction based on the initial MD [1, 2]

    Parameters
    ----------
    signal : array (x, y, z, k)
        Raw data
    gtab : GradientTable class instance
    Diso : float
        Diffusivity of free water at body temperature
    tissue_MD : float
        The assumed prior for healthy tissue mean diffusivity

    Returns
    -------
    f0 : array (x, y, z)
        The initialized tissue fraction
    fmin : array (x, y, z)
        Lower limit for the tissue fraction
    fmax : array (x, y, z)
        Upper limit for the tissue fraction

    References
    ----------
    .. [1] Ismail, A. A. O., Parker, D., Hernandez-Fernandez, M., Brem,
            S., Alexander, S., Pasternak, O., ... & Verma, R.
            (2018, September). Characterizing Peritumoral Tissue Using
            Free Water Elimination in Clinical DTI.
    .. [2] Ismail, A. A. O., Parker, D., Hernandez-Fernandez, M., Wolf, R.,
            Brem, S., Alexander, S., ... & Verma, R. (2019). Freewater
            EstimatoR using iNtErpolated iniTialization (FERNET): Toward
            Accurate Estimation of Free Water in Peritumoral Region Using
            Single-Shell Diffusion MRI Data.
    """
    # bvals = gtab.bvals[~gtab.b0s_mask]
    bvals = gtab.bvals
    bvecs = gtab.bvecs
    mean_bval = np.max(bvals)
    # print(mean_bval)

    mbvals = bvals[np.logical_or(bvals==0, bvals==mean_bval)]
    mbvecs = bvecs[np.logical_or(bvals==0, bvals==mean_bval), :]
    mgtab = gradient_table(mbvals, mbvecs, b0_threshold=0)
    msignal = signal[..., np.logical_or(bvals==0, bvals==mean_bval)]

    # Conventional DTI
    dti_params = ols_fit_tensor(design_matrix(mgtab), msignal)
    eigvals = dti_params[..., 0:3]
    MD = np.mean(eigvals, axis=-1)  # mean diffusivity

    # Initial volume fraction
    Awater = np.exp(-mean_bval * Diso)
    Atissue = np.exp(-mean_bval * tissue_MD)
    f0 = (np.exp(-mean_bval * MD) - Awater) / (Atissue - Awater)

    # Min and Max volume fractions
    fmin = np.ones(f0.shape) * 0.0001
    fmax = np.ones(f0.shape) * (1 - 0.0001)

    return (f0, fmin, fmax)


def fraction_init_hybrid(signal, gtab, Diso=3, Stissue=None, Swater=None,
                         min_tissue_diff=0.001, max_tissue_diff=2.5,
                         tissue_MD=0.6):
    """
    Initializes the tissue fraction based on hybrid initialization [1, 2]

    Parameters
    ----------
    signal : array (x, y, z, k)
        Raw data
    gtab : GradientTable class instance
    Diso : float
        Diffusivity of free water at body temperature
    Stissue : float
        S0 signal intensity representative of tissue, see Note
    Swater : float
        S0 signal intensity representative of water, see Note
    min_tissue_diff : float
        minimum diffusivity expected in tissue
    max_tissue_diff : floar
        maximum diffusivity expected in tissue
    tissue_MD : float
        The assumed prior for healthy tissue mean diffusivity

    Returns
    -------
    f0 : array (x, y, z)
        The initialized tissue fraction
    fmin : array (x, y, z)
        Lower limit for the tissue fraction
    fmax : array (x, y, z)
        Upper limit for the tissue fraction

    Note
    ----
    This method is an interpolation between the S0 and MD based initializations;
    see functions 'fraction_init_s0' and 'fraction_init_md'

    References
    ----------
    .. [1] Ismail, A. A. O., Parker, D., Hernandez-Fernandez, M., Brem,
            S., Alexander, S., Pasternak, O., ... & Verma, R.
            (2018, September). Characterizing Peritumoral Tissue Using
            Free Water Elimination in Clinical DTI.
    .. [2] Ismail, A. A. O., Parker, D., Hernandez-Fernandez, M., Wolf, R.,
            Brem, S., Alexander, S., ... & Verma, R. (2019). Freewater
            EstimatoR using iNtErpolated iniTialization (FERNET): Toward
            Accurate Estimation of Free Water in Peritumoral Region Using
            Single-Shell Diffusion MRI Data.
    """
    f_S0, fmin, fmax = fraction_init_s0(signal, gtab, Diso=Diso,
                                        Stissue=Stissue, Swater=Swater,
                                        min_tissue_diff=min_tissue_diff,
                                        max_tissue_diff=max_tissue_diff)
    f_MD, _, _ = fraction_init_md(signal, gtab, Diso=Diso,
                                  tissue_MD=tissue_MD)
    # hybrid initialization
    alpha = np.copy(f_S0)
    np.clip(alpha, 0.0001, 0.9999, out=alpha)
    np.clip(f_S0, fmin, fmax, out=f_S0)
    np.clip(f_MD, 0.0001, 0.9999, out=f_MD)
    f0 = f_MD**alpha * f_S0**(1 - alpha)

    return (f0, fmin, fmax)