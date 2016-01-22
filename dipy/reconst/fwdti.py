#!/usr/bin/python
""" Classes and functions for fitting tensors """
from __future__ import division, print_function, absolute_import

from .base import ReconstModel

from dipy.reconst.dti import (TensorFit, design_matrix, _min_positive_signal,
                              decompose_tensor, from_lower_triangular)

import numpy as np

from dipy.utils.arrfuncs import pinv

from dipy.core.ndindex import ndindex

def fwdti_prediction(params, gtab, S0):
    """
    Predict a signal given the parameters of the free water diffusion tensor
    model.

    Parameters
    ----------
    Params : ndarray
        Model parameters. The last dimension should have the 12 tensor
        parameters (3 eigenvalues, followed by the 3 corresponding
        eigenvectors) and the volume fraction of the free water compartment

    gtab : a GradientTable class instance
        The gradient table for this prediction

    S0 : float or ndarray
        The non diffusion-weighted signal in every voxel, or across all
        voxels. Default: 1

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
    .. [1] Pasternak, O., Sochen, N., Gur, Y., Intrator, N., Assaf, Y., 2009.
           Free water elimination and mapping from diffusion MRI. Magn. Reson.
           Med. 62, 717-739. http://dx.doi.org/10.1002/mrm.22055.
    """
    #evals = dti_params[..., :3]
    #evecs = dti_params[..., 3:].reshape(dti_params.shape[:-1] + (3, 3))
    #qform = vec_val_vect(evecs, evals)
    #sphere = Sphere(xyz=gtab.bvecs[~gtab.b0s_mask])
    #adc = apparent_diffusion_coef(qform, sphere)
    #
    #if isinstance(S0, np.ndarray):
    #    # If it's an array, we need to give it one more dimension:
    #    S0 = S0[..., None]
    #
    # # First do the calculation for the diffusion weighted measurements:
    # pre_pred_sig = S0 * np.exp(-gtab.bvals[~gtab.b0s_mask] * adc)
    #
    # Then we need to sort out what goes where:
    # pred_sig = np.zeros(pre_pred_sig.shape[:-1] + (gtab.bvals.shape[0],))
    #
    # These are the diffusion-weighted values
    # pred_sig[..., ~gtab.b0s_mask] = pre_pred_sig
    #
    # For completeness, we predict the mean S0 for the non-diffusion
    # weighted measurements, which is our best guess:
    # pred_sig[..., gtab.b0s_mask] = S0
    # return pred_sig


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

        # Code and comments have still to be addapted
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
        """ Fit method of the free water elinimation DTI model class

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

        return TensorFit(self, fwdti_params)

    def predict(self, fwdti_params, S0=1):
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
        return fwdti_prediction(fwdti_params, self.gtab, S0)


class FreeWaterTensorFit(TensorFit):
    """ Class for fitting the Free Water Tensor Model"""
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

    def predict(self, gtab, S0=1):
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
        return fwdti_prediction(self.model_params, gtab, S0)


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
    fw_params = np.empty((len(data_flat), 13))

    # inverting design matrix and defining minimun diffusion aloud
    min_diffusivity = tol / -design_matrix.min()
    inv_design = np.linalg.pinv(design_matrix)

    # lopping WLS solution on all data voxels
    for vox in range(len(data_flat)):
        fw_params[vox] = _wls_iter(design_matrix, inv_design, data_flat[vox],
                                    min_diffusivity)

    # Reshape data according to the input data shape
    fw_params = fw_params.reshape((data.shape[:-1]) + (13,))

    return fw_params


def _wls_iter(design_matrix, inv_design, sig, min_diffusivity, Diso=3e-3,
              piterations=3, riterations=2):
    """ Helper function used by wls_fit_tensor - Applies WLS fit of the
    water free elimination model to single voxel signals.

    Parameters
    ----------
    design_matrix : array (g, 7)
        Design matrix holding the covariants used to solve for the regression
        coefficients
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
    WT_S2_LS = np.dot(WTS2, log_s)
    params = np.dot(inv_WT_S2_W, WT_S2_LS)
    
    # General free-water signal contribution
    fwsig = np.exp(np.dot(design_matrix, 
                          np.array([Diso, 0, Diso, 0, 0, Diso, -np.log(1.)])))

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
            y = np.log((SI - FS*S0*SFW.T) / (1 - FS))

            # Estimate tissue's tensor from inv(A.T*S2*A)*A.T*S2*y
            S2 = np.diag(np.square(np.dot(W, params)))
            WS2 = np.dot(W.T, S2)
            invWS2W = np.linalg.pinv(np.dot(WS2, W))
            all_new_params = np.dot(np.dot(invWS2W, WS2), y)

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

    evals, evecs = decompose_tensor(from_lower_triangular(params),
                                    min_diffusivity=min_diffusivity)
    fw_params = np.concatenate((evals, evecs[0], evecs[1], evecs[2], 
                                np.array([f])), axis=0)
    return fw_params


common_fit_methods = {'WLLS': wls_fit_tensor,
                      'WLS': wls_fit_tensor
                      }
