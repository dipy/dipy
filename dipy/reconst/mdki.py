#!/usr/bin/python
""" Classes and functions for fitting the mean spherical diffusion kurtosis
model """
from __future__ import division, print_function, absolute_import

import numpy as np

from dipy.core.gradients import (check_multi_b, unique_bvals)
from dipy.reconst.base import ReconstModel
from dipy.reconst.dti import (MIN_POSITIVE_SIGNAL)
from dipy.core.ndindex import ndindex
from dipy.core.onetime import auto_attr


def mean_signal_bvalue(data, gtab, bmag=None):
    """
    Computes the average signal for each unique b-values of the data's gradient
    table

    Parameters
    ----------
    data : ndarray ([X, Y, Z, ...], g)
        ndarray containing the data signals in its last dimension.
    gtab : a GradientTable class instance
        The gradient table containing diffusion acquisition parameters.
    bmag : The order of magnitude that the bvalues have to differ to be
        considered an unique b-value. Default: derive this value from the
        maximal b-value provided: $bmag=log_{10}(max(bvals)) - 1$.

    Returns
    -------
    msignal : ndarray ([X, Y, Z, ..., nub])
        Mean signal along all gradient direction for each unique b-value
        Note that the last dimension should contain the signal means and nub
        is the number of unique b-values.
    ng : ndarray(nub)
        Number of gradient directions used to compute the mean signal for
        all unique b-values
    """
    bvals = gtab.bvals.copy()

    # Compute unique and rounded bvals
    ub, rb = unique_bvals(bvals, bmag=bmag, rbvals=True)

    # Initialize msignal and ng
    nub = ub.size
    ng = np.zeros(nub)
    msigma = np.zeros(data.shape[:-1] + (nub,))
    for bi in range(ub.size):
        msigma[..., bi] = np.mean(data[..., rb == ub[bi]], axis=-1)
        ng[bi] = np.sum(rb == ub[bi])
    return msigma, ng


def mdki_prediction(mdki_params, gtab, S0=1.0):
    """
    Predict the mean signal given the parameters of the mean spherical DKI, an
    GradientTable Object and S0 signal.

    Parameters
    ----------
    params : ndarray ([X, Y, Z, ...], 2)
        Array containing in the last axis the mean spherical diffusivity and
        mean spherical kurtosis
    gtab : a GradientTable class instance
        The gradient table for this prediction
    S0 : float or ndarray (optional)
        The non diffusion-weighted signal in every voxel, or across all
        voxels. Default: 1

    Notes
    -----
    The predicted signal is given by:
        $MS(b) = S_0 * exp(-bMD + 1/6 b^{2} MD^{2} MK)$, where MD is the
        mean spherical diffusivity and mean spherical kurtosis.  
    """
    # Define MDKI design matrix given gtab.bvals
    A = design_matrix(gtab)

    # Flat parameters and initialize pred_sig
    fevals = evals.reshape((-1, evals.shape[-1]))
    fevecs = evecs.reshape((-1,) + evecs.shape[-2:])
    fkt = kt.reshape((-1, kt.shape[-1]))
    pred_sig = np.zeros((len(fevals), len(gtab.bvals)))
    if isinstance(S0, np.ndarray):
        S0_vol = np.reshape(S0, (len(fevals)))
    else:
        S0_vol = S0
    # looping for all voxels
    for v in range(len(pred_sig)):
        DT = np.dot(np.dot(fevecs[v], np.diag(fevals[v])), fevecs[v].T)
        dt = lower_triangular(DT)
        MD = (dt[0] + dt[2] + dt[5]) / 3
        if isinstance(S0_vol, np.ndarray):
            this_S0 = S0_vol[v]
        else:
            this_S0 = S0_vol
        X = np.concatenate((dt, fkt[v] * MD * MD,
                            np.array([np.log(this_S0)])),
                           axis=0)
        pred_sig[v] = np.exp(np.dot(A, X))

    # Reshape data according to the shape of dki_params
    pred_sig = pred_sig.reshape(dki_params.shape[:-1] + (pred_sig.shape[-1],))

    return pred_sig


class MeanDiffusionKurtosisModel(ReconstModel):
    """ Mean spherical Diffusion Kurtosis Model
    """

    def __init__(self, gtab, bmag=None, return_S0_hat=False, *args, **kwargs):
        """ Mean Spherical Diffusion Kurtosis Model [1]_.

        Parameters
        ----------
        gtab : GradientTable class instance

        bmag : int
            The order of magnitude that the bvalues have to differ to be
            considered an unique b-value. Default: derive this value from the
            maximal b-value provided: $bmag=log_{10}(max(bvals)) - 1$.

        return_S0_hat : bool
            Boolean to return (True) or not (False) the S0 values for the fit.

        args, kwargs : arguments and key-word arguments passed to the
           fit_method. See mdti.wls_fit_mdki for details

        min_signal : float
            The minimum signal value. Needs to be a strictly positive
            number. Default: 0.0001.

        References
        ----------
        .. [1] Henriques, R.N., Correia, M.M., 2017. Interpreting age-related
               changes based on the mean signal diffusion kurtosis. 25th Annual
               Meeting of the ISMRM; Honolulu. April 22-28
        """
        ReconstModel.__init__(self, gtab)

        self.return_S0_hat = return_S0_hat
        self.design_matrix = design_matrix(self.gtab)
        self.bmag = bmag
        self.args = args
        self.kwargs = kwargs
        self.min_signal = self.kwargs.pop('min_signal', None)
        if self.min_signal is not None and self.min_signal <= 0:
            e_s = "The `min_signal` key-word argument needs to be strictly"
            e_s += " positive."
            raise ValueError(e_s)

        # Check if at least three b-values are given
        enough_b = check_multi_b(self.gtab, 3, non_zero=False)
        if not enough_b:
            mes = "MDKI requires at least 3 b-values (which can include b=0)"
            raise ValueError(mes)

    def fit(self, data, mask=None):
        """ Fit method of the DTI model class

        Parameters
        ----------
        data : ndarray ([X, Y, Z, ...], g)
            ndarray containing the data signals in its last dimension.

        mask : array
            A boolean array used to mark the coordinates in the data that
            should be analyzed that has the shape data.shape[:-1]

        """
        S0_params = None

        # Compute mean signal for each unique b-value
        mdata, ng = mean_signal_bvalue(data, self.gtab, bmag=self.bmag)

        # Remove mdata zeros
        if self.min_signal is None:
            min_signal = MIN_POSITIVE_SIGNAL
        else:
            min_signal = self.min_signal

        mdata = np.maximum(mdata, min_signal)

        params = wls_fit_mdki(self.design_matrix, mdata, ng, mask=mask,
                              *self.args, **self.kwargs)
        if self.return_S0_hat:
            params, S0_params = params

        return MeanDiffusionKurtosisFit(self, params, model_S0=S0_params)

    def predict(self, dti_params, S0=1.):
        """
        Predict a signal for this TensorModel class instance given parameters.

        Parameters
        ----------
        params : ndarray
            The parameters of the mean spherical diffusion kurtosis model
        S0 : float or ndarray
            The non diffusion-weighted signal in every voxel, or across all
            voxels. Default: 1
        """
        return mdki_prediction(dti_params, self.gtab, S0)


class MeanDiffusionKurtosisFit(object):

    def __init__(self, model, model_params, model_S0=None):
        """ Initialize a MeanDiffusionKurtosisFit class instance.
        """
        self.model = model
        self.model_params = model_params
        self.model_S0 = model_S0

    def __getitem__(self, index):
        model_params = self.model_params
        model_S0 = self.model_S0
        N = model_params.ndim
        if type(index) is not tuple:
            index = (index,)
        elif len(index) >= model_params.ndim:
            raise IndexError("IndexError: invalid index")
        index = index + (slice(None),) * (N - len(index))
        if model_S0 is not None:
            model_S0 = model_S0[index[:-1]]
        return type(self)(self.model, model_params[index], model_S0=model_S0)

    @property
    def S0_hat(self):
        return self.model_S0

    @auto_attr
    def md(self):
        r"""
        Spherical Mean diffusitivity (MD) calculated from the mean spherical
        Diffusion Kurtosis Model.

        Returns
        ---------
        md : ndarray
            Calculated Spherical Mean diffusitivity.

        References
        ----------
        .. [1] Henriques, R.N., Correia, M.M., 2017. Interpreting age-related
               changes based on the mean signal diffusion kurtosis. 25th Annual
               Meeting of the ISMRM; Honolulu. April 22-28
        """
        return self.model_params[..., 0]

    @auto_attr
    def mk(self):
        r"""
        Spherical Mean Kurtosis (MK) calculated from the mean spherical
        Diffusion Kurtosis Model.

        Returns
        ---------
        mk : ndarray
            Calculated Spherical Mean diffusitivity.

        References
        ----------
        .. [1] Henriques, R.N., Correia, M.M., 2017. Interpreting age-related
               changes based on the mean signal diffusion kurtosis. 25th Annual
               Meeting of the ISMRM; Honolulu. April 22-28
        """
        return self.model_params[..., 1]

    def predict(self, gtab, S0=None, step=None):
        r"""
        Given a mean spherical diffusion kurtosis model fit, predict the signal
        on the vertices of a sphere

        Parameters
        ----------
        gtab : a GradientTable class instance
            This encodes the directions for which a prediction is made

        S0 : float array
           The mean non-diffusion weighted signal in each voxel. Default:
           The fitted S0 value in all voxels if it was fitted. Otherwise 1 in
           all voxels.

        step : int
            The chunk size as a number of voxels. Optional parameter with
            default value 10,000.

            In order to increase speed of processing, tensor fitting is done
            simultaneously over many voxels. This parameter sets the number of
            voxels that will be fit at once in each iteration. A larger step
            value should speed things up, but it will also take up more memory.
            It is advisable to keep an eye on memory consumption as this value
            is increased.

        Notes
        -----
        """
        predict = 0

        return predict


def wls_fit_mdki(design_matrix, msignal, ng, mask=None,
                 min_signal=MIN_POSITIVE_SIGNAL, return_S0_hat=False):
    r"""
    Fits the mean spherical diffusion kurtosis imaging based on a weighted
    least square solution [1]_.

    Parameters
    ----------
    design_matrix : array (nub, 3)
        Design matrix holding the covariants used to solve for the regression
        coefficients of the mean spherical diffusion kurtosis model. Note that
        nub is the number of unique b-values
    msignal : ndarray ([X, Y, Z, ..., nub])
        Mean signal along all gradient direction for each unique b-value
        Note that the last dimension should contain the signal means and nub
        is the number of unique b-values.
    mask : array
        A boolean array used to mark the coordinates in the data that
        should be analyzed that has the shape data.shape[:-1]
    min_signal : float, optional
        Voxel with mean signal intensities lower than the min positive signal
        are not processed. Default: 0.0001
    ng : ndarray(nub)
        Number of gradient directions used to compute the mean signal for
        all unique b-values
    return_S0_hat : bool
        Boolean to return (True) or not (False) the S0 values for the fit.

    Returns
    -------
    params : array (..., 2)
        Containing the mean spherical diffusivity and mean spherical kurtosis

    References
    ----------
    .. [1] Henriques, R.N., Correia, M.M., 2017. Interpreting age-related
           changes based on the mean signal diffusion kurtosis. 25th Annual
           Meeting of the ISMRM; Honolulu. April 22-28
    """
    params = np.zeros(msignal.shape[:-1] + (3,))

    # Prepare mask
    if mask is None:
        mask = np.ones(msignal.shape[:-1], dtype=bool)
    else:
        if mask.shape != msignal.shape[:-1]:
            raise ValueError("Mask is not the same shape as data.")
        mask = np.array(mask, dtype=bool, copy=False)

    index = ndindex(mask.shape)
    for v in index:
        # Skip if out of mask
        if not mask[v]:
            continue
        # Skip if no signal is present
        if np.mean(msignal[v]) <= min_signal:
            continue
        # Define weights as diag(sqrt(ng) * yn**2)
        W = np.diag(ng * msignal[v]**2)

        # WLS fitting
        BTW = np.dot(design_matrix.T, W)
        inv_BT_W_B = np.linalg.pinv(np.dot(BTW, design_matrix))
        invBTWB_BTW = np.dot(inv_BT_W_B, BTW)
        p = np.dot(invBTWB_BTW, np.log(msignal[v]))

        # Process parameters
        p[1] = p[1] / (p[0]**2)
        p[2] = np.exp(p[2])
        params[v] = p

    if return_S0_hat:
        return params[..., :2], params[..., 2]
    else:
        return params[..., :2]


def design_matrix(gtab, bmag=None):
    """  Constructs design matrix for the mean spherical diffusion kurtosis
    model

    Parameters
    ----------
    gtab : a GradientTable class instance
        The gradient table containing diffusion acquisition parameters.

    bmag : The order of magnitude that the bvalues have to differ to be
        considered an unique b-value. Default: derive this value from the
        maximal b-value provided: $bmag=log_{10}(max(bvals)) - 1$.

    dtype : string
        Parameter to control the dtype of returned designed matrix

    Returns
    -------
    design_matrix : array (nb, 3)
        Design matrix or B matrix for the mean spherical diffusion kurtosis
        model assuming that parameters are in the following order:
        design_matrix[j, :] = (MD, MK, S0)
    """
    ubvals = unique_bvals(gtab.bvals, bmag=bmag)
    nb = ubvals.shape
    B = np.zeros(nb + (3,))
    B[:, 0] = -ubvals
    B[:, 1] = 1.0/6.0 * ubvals**2
    B[:, 2] = np.ones(nb)
    return B
