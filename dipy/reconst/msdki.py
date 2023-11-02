#!/usr/bin/python
""" Classes and functions for fitting the mean signal diffusion kurtosis
model """

import numpy as np
import scipy.optimize as opt

from dipy.core.gradients import (check_multi_b, unique_bvals_magnitude,
                                 round_bvals)
from dipy.reconst.base import ReconstModel
from dipy.reconst.dti import MIN_POSITIVE_SIGNAL
from dipy.core.ndindex import ndindex
from dipy.core.onetime import auto_attr


def mean_signal_bvalue(data, gtab, bmag=None):
    """
    Computes the average signal across different diffusion directions
    for each unique b-value

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
        Mean signal along all gradient directions for each unique b-value
        Note that the last dimension contains the signal means and nub is the
        number of unique b-values.
    ng : ndarray(nub)
        Number of gradient directions used to compute the mean signal for
        all unique b-values

    Notes
    -----
    This function assumes that directions are evenly sampled on the sphere or
    on the hemisphere
    """
    bvals = gtab.bvals.copy()

    # Compute unique and rounded bvals
    ub, rb = unique_bvals_magnitude(bvals, bmag=bmag, rbvals=True)

    # Initialize msignal and ng
    nub = ub.size
    ng = np.zeros(nub)
    msignal = np.zeros(data.shape[:-1] + (nub,))
    for bi in range(ub.size):
        msignal[..., bi] = np.mean(data[..., rb == ub[bi]], axis=-1)
        ng[bi] = np.sum(rb == ub[bi])
    return msignal, ng


def msk_from_awf(f):
    """
    Computes mean signal kurtosis from axonal water fraction estimates of the
    SMT2 model

    Parameters
    ----------
    f : ndarray ([X, Y, Z, ...])
        ndarray containing the axonal volume fraction estimate.

    Returns
    -------
    msk : ndarray(nub)
        Mean signal kurtosis (msk)

    Notes
    -----
    Computes mean signal kurtosis using equations 17 of [1]_

    References
    ----------
    .. [1] Neto Henriques R, Jespersen SN, Shemesh N (2019). Microscopic
           anisotropy misestimation in spherical‐mean single diffusion
           encoding MRI. Magnetic Resonance in Medicine (In press).
           doi: 10.1002/mrm.27606
    """
    msk_num = 216*f - 504 * f**2 + 504 * f**3 - 180 * f**4
    msk_den = 135 - 360*f + 420 * f**2 - 240 * f**3 + 60 * f**4
    msk = msk_num / msk_den

    return msk


def _msk_from_awf_error(f, msk):
    """ Helper function that calculates the error of a predicted mean signal
    kurtosis from the axonal water fraction of SMT2 model and a measured
    mean signal kurtosis

    Parameters
    ----------
    f : float
        Axonal volume fraction estimate.
    msk : float
        Measured mean signal kurtosis.

    Returns
    -------
    error : float
       Error computed by subtracting msk with fun(f), where fun is the function
       described in equation 17 of [1]_

    Notes
    -----
    This function corresponds to the differential of equations 17 of [1]_
    """
    return msk_from_awf(f) - msk


def _diff_msk_from_awf(f, msk):
    """
    Helper function that calculates differential of function msk_from_awf

    Parameters
    ----------
    f : ndarray ([X, Y, Z, ...])
        ndarray containing the axonal volume fraction estimate.

    Returns
    -------
    dkdf : ndarray(nub)
        Mean signal kurtosis differential
    msk : float
        Measured mean signal kurtosis.

    Notes
    -----
    This function corresponds to the differential of equations 17 of [1]_.
    This function is applicable to both _msk_from_awf and _msk_from_awf_error.

    References
    ----------
    .. [1] Neto Henriques R, Jespersen SN, Shemesh N (2019). Microscopic
           anisotropy misestimation in spherical‐mean single diffusion
           encoding MRI. Magnetic Resonance in Medicine (In press).
           doi: 10.1002/mrm.27606
    """
    F = 216*f - 504 * f**2 + 504 * f**3 - 180 * f**4  # Numerator
    G = 135 - 360*f + 420 * f**2 - 240 * f**3 + 60 * f**4  # Denominator

    dF = 216 - 1008 * f + 1512 * f**2 - 720 * f**3  # Num. differential
    dG = -360 + 840 * f - 720 * f**2 + 240 * f**3  # Den. differential

    return (G * dF - F * dG) / (G ** 2)


def awf_from_msk(msk, mask=None):
    """
    Computes the axonal water fraction from the mean signal kurtosis
    assuming the 2-compartmental spherical mean technique model [1]_, [2]_

    Parameters
    ----------
    msk : ndarray ([X, Y, Z, ...])
        Mean signal kurtosis (msk)
    mask : ndarray, optional
        A boolean array used to mark the coordinates in the data that should be
        analyzed that has the same shape of the msdki parameters

    Returns
    -------
    awf : ndarray ([X, Y, Z, ...])
        ndarray containing the axonal volume fraction estimate.

    Notes
    -----
    Computes the axonal water fraction from the mean signal kurtosis
    MSK using equation 17 of [1]_

    References
    ----------
    .. [1] Neto Henriques R, Jespersen SN, Shemesh N (2019). Microscopic
           anisotropy misestimation in spherical‐mean single diffusion
           encoding MRI. Magnetic Resonance in Medicine (In press).
           doi: 10.1002/mrm.27606
    .. [2] Kaden E, Kelm ND, Carson RP, et al. (2016) Multi‐compartment
           microscopic diffusion imaging. Neuroimage 139:346–359.
    """
    awf = np.zeros(msk.shape)

    # Prepare mask
    if mask is None:
        mask = np.ones(msk.shape, dtype=bool)
    else:
        if mask.shape != msk.shape:
            raise ValueError("Mask is not the same shape as data.")
        mask = np.array(mask, dtype=bool, copy=False)

    # looping voxels
    index = ndindex(mask.shape)
    for v in index:
        # Skip if out of mask
        if not mask[v]:
            continue

        if msk[v] > 2.4:
            awf[v] = 1
        elif msk[v] < 0:
            awf[v] = 0
        else:
            if np.isnan(msk[v]):
                awf[v] = np.nan
            else:
                mski = msk[v]
                fini = mski / 2.4  # Initial guess based on linear assumption
                awf[v] = opt.fsolve(
                    _msk_from_awf_error,
                    fini,
                    args=(mski,),
                    fprime=_diff_msk_from_awf,
                    col_deriv=True
                ).item()

    return awf


def msdki_prediction(msdki_params, gtab, S0=1.0):
    """
    Predict the mean signal given the parameters of the mean signal DKI, an
    GradientTable object and S0 signal.

    Parameters
    ----------
    msdki_params : ndarray ([X, Y, Z, ...], 2)
        Array containing the mean signal diffusivity and mean signal kurtosis
        in its last axis
    gtab : a GradientTable class instance
        The gradient table for this prediction
    S0 : float or ndarray (optional)
        The non diffusion-weighted signal in every voxel, or across all
        voxels. Default: 1

    Notes
    -----
    The predicted signal is given by:
        $MS(b) = S_0 * exp(-bD + 1/6 b^{2} D^{2} K)$, where $D$ and $K$ are the
        mean signal diffusivity and mean signal kurtosis.

    References
    ----------
    .. [1] Henriques, R.N., 2018. Advanced Methods for Diffusion MRI Data
           Analysis and their Application to the Healthy Ageing Brain (Doctoral
           thesis). Downing College, University of Cambridge.
           https://doi.org/10.17863/CAM.29356
    """
    A = design_matrix(round_bvals(gtab.bvals))

    params = msdki_params.copy()
    params[..., 1] = params[..., 1] * params[..., 0] ** 2

    if isinstance(S0, (float, int)):
        pred_sig = S0 * np.exp(np.dot(params, A[:, :2].T))
    elif S0.size == 1:
        pred_sig = S0 * np.exp(np.dot(params, A[:, :2].T))
    else:
        nv = gtab.bvals.size
        S0r = np.zeros(S0.shape + gtab.bvals.shape)
        for vi in range(nv):
            S0r[..., vi] = S0
        pred_sig = S0r * np.exp(np.dot(params, A[:, :2].T))

    return pred_sig


class MeanDiffusionKurtosisModel(ReconstModel):
    """ Mean signal Diffusion Kurtosis Model
    """

    def __init__(self, gtab, bmag=None, return_S0_hat=False, *args, **kwargs):
        """ Mean Signal Diffusion Kurtosis Model [1]_.

        Parameters
        ----------
        gtab : GradientTable class instance

        bmag : int
            The order of magnitude that the bvalues have to differ to be
            considered an unique b-value. Default: derive this value from the
            maximal b-value provided: $bmag=log_{10}(max(bvals)) - 1$.

        return_S0_hat : bool
            If True, also return S0 values for the fit.

        args, kwargs : arguments and keyword arguments passed to the
        fit_method. See msdki.wls_fit_msdki for details

        References
        ----------
        .. [1] Henriques, R.N., 2018. Advanced Methods for Diffusion MRI Data
               Analysis and their Application to the Healthy Ageing Brain
               (Doctoral thesis). Downing College, University of Cambridge.
               https://doi.org/10.17863/CAM.29356

        """
        ReconstModel.__init__(self, gtab)

        self.return_S0_hat = return_S0_hat
        self.ubvals = unique_bvals_magnitude(gtab.bvals, bmag=bmag)
        self.design_matrix = design_matrix(self.ubvals)
        self.bmag = bmag
        self.args = args
        self.kwargs = kwargs
        self.min_signal = self.kwargs.pop('min_signal', MIN_POSITIVE_SIGNAL)
        if self.min_signal is not None and self.min_signal <= 0:
            e_s = "The `min_signal` key-word argument needs to be strictly"
            e_s += " positive."
            raise ValueError(e_s)

        # Check if at least three b-values are given
        enough_b = check_multi_b(self.gtab, 3, non_zero=False, bmag=bmag)
        if not enough_b:
            mes = "MSDKI requires at least 3 b-values (which can include b=0)"
            raise ValueError(mes)

    def fit(self, data, mask=None):
        """ Fit method of the MSDKI model class

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
        mdata = np.maximum(mdata, self.min_signal)

        params = wls_fit_msdki(self.design_matrix, mdata, ng, mask=mask,
                               return_S0_hat=self.return_S0_hat, *self.args,
                               **self.kwargs)
        if self.return_S0_hat:
            params, S0_params = params

        return MeanDiffusionKurtosisFit(self, params, model_S0=S0_params)

    def predict(self, msdki_params, S0=1.):
        """
        Predict a signal for this MeanDiffusionKurtosisModel class instance
        given parameters.

        Parameters
        ----------
        msdki_params : ndarray
            The parameters of the mean signal diffusion kurtosis model
        S0 : float or ndarray
            The non diffusion-weighted signal in every voxel, or across all
            voxels. Default: 1

        Returns
        -------
        S : (..., N) ndarray
            Simulated mean signal based on the mean signal diffusion kurtosis
            model

        Notes
        -----
        The predicted signal is given by:
            $MS(b) = S_0 * exp(-bD + 1/6 b^{2} D^{2} K)$, where $D$ and $K$ are
            the mean signal diffusivity and mean signal kurtosis.

        References
        ----------
        .. [1] Henriques, R.N., 2018. Advanced Methods for Diffusion MRI Data
               Analysis and their Application to the Healthy Ageing Brain
               (Doctoral thesis). Downing College, University of Cambridge.
               https://doi.org/10.17863/CAM.29356
        """
        return msdki_prediction(msdki_params, self.gtab, S0)


class MeanDiffusionKurtosisFit:

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
        return MeanDiffusionKurtosisFit(self.model, model_params[index],
                                        model_S0=model_S0)

    @property
    def S0_hat(self):
        return self.model_S0

    @auto_attr
    def msd(self):
        r"""
        Mean signal diffusitivity (MSD) calculated from the mean signal
        Diffusion Kurtosis Model.

        Returns
        -------
        msd : ndarray
            Calculated signal mean diffusitivity.

        References
        ----------
        .. [1] Henriques, R.N., 2018. Advanced Methods for Diffusion MRI Data
               Analysis and their Application to the Healthy Ageing Brain
               (Doctoral thesis). Downing College, University of Cambridge.
               https://doi.org/10.17863/CAM.29356
        """
        return self.model_params[..., 0]

    @auto_attr
    def msk(self):
        r"""
        Mean signal kurtosis (MSK) calculated from the mean signal
        Diffusion Kurtosis Model.

        Returns
        -------
        msk : ndarray
            Calculated signal mean kurtosis.

        References
        ----------
        .. [1] Henriques, R.N., 2018. Advanced Methods for Diffusion MRI Data
               Analysis and their Application to the Healthy Ageing Brain
               (Doctoral thesis). Downing College, University of Cambridge.
               https://doi.org/10.17863/CAM.29356
        """
        return self.model_params[..., 1]

    @auto_attr
    def smt2f(self):
        r"""
        Computes the axonal water fraction from the mean signal kurtosis
        assuming the 2-compartmental spherical mean technique model [1]_, [2]_

        Returns
        -------
        ndarray
            Axonal volume fraction calculated from MSK.

        Notes
        -----
        Computes the axonal water fraction from the mean signal kurtosis
        MSK using equation 17 of [1]_

        References
        ----------
        .. [1] Neto Henriques R, Jespersen SN, Shemesh N (2019). Microscopic
               anisotropy misestimation in spherical‐mean single diffusion
               encoding MRI. Magnetic Resonance in Medicine (In press).
               doi: 10.1002/mrm.27606
        .. [2] Kaden E, Kelm ND, Carson RP, et al. (2016) Multi‐compartment
               microscopic diffusion imaging. Neuroimage 139:346–359.
        """
        return awf_from_msk(self.msk)

    @auto_attr
    def smt2di(self):
        r"""
        Computes the intrisic diffusivity from the mean signal diffusional
        kurtosis parameters assuming the 2-compartmental spherical mean
        technique model [1]_, [2]_

        Returns
        -------
        smt2di : ndarray
            Intrinsic diffusivity computed by converting MSDKI to SMT2.

        Notes
        -----
        Computes the intrinsic diffusivity using equation 16 of [1]_

        References
        ----------
        .. [1] Neto Henriques R, Jespersen SN, Shemesh N (2019). Microscopic
               anisotropy misestimation in spherical‐mean single diffusion
               encoding MRI. Magnetic Resonance in Medicine (In press).
               doi: 10.1002/mrm.27606
        .. [2] Kaden E, Kelm ND, Carson RP, et al. (2016) Multi‐compartment
               microscopic diffusion imaging. Neuroimage 139:346–359.
        """
        return 3 * self.msd / (1 + 2 * (1 - self.smt2f)**2)

    @auto_attr
    def smt2uFA(self):
        r"""
        Computes the microscopic fractional anisotropy from the mean signal
        diffusional kurtosis parameters assuming the 2-compartmental spherical
        mean technique model [1]_, [2]_

        Returns
        -------
        smt2uFA : ndarray
            Microscopic fractional anisotropy computed by converting MSDKI to
            SMT2.

        Notes
        -----
        Computes the intrinsic diffusivity using equation 10 of [1]_

        References
        ----------
        .. [1] Neto Henriques R, Jespersen SN, Shemesh N (2019). Microscopic
               anisotropy misestimation in spherical‐mean single diffusion
               encoding MRI. Magnetic Resonance in Medicine (In press).
               doi: 10.1002/mrm.27606
        .. [2] Kaden E, Kelm ND, Carson RP, et al. (2016) Multi‐compartment
               microscopic diffusion imaging. Neuroimage 139:346–359.
        """
        fe = (1 - self.smt2f)
        num = 3 * (1 - 2 * fe ** 2 + fe ** 3)
        den = 3 + 2 * fe ** 3 + 4 * fe ** 4
        return np.sqrt(num/den)

    def predict(self, gtab, S0=1.):
        r"""
        Given a mean signal diffusion kurtosis model fit, predict the signal
        on the vertices of a sphere

        Parameters
        ----------
        gtab : a GradientTable class instance
            This encodes the directions for which a prediction is made

        S0 : float array
           The mean non-diffusion weighted signal in each voxel. Default:
           The fitted S0 value in all voxels if it was fitted. Otherwise 1 in
           all voxels.

        Returns
        -------
        S : (..., N) ndarray
            Simulated mean signal based on the mean signal kurtosis model

        Notes
        -----
        The predicted signal is given by:
        $MS(b) = S_0 * exp(-bD + 1/6 b^{2} D^{2} K)$, where $D$ and $k$ are the
        mean signal diffusivity and mean signal kurtosis.

        References
        ----------
        .. [1] Henriques, R.N., 2018. Advanced Methods for Diffusion MRI Data
               Analysis and their Application to the Healthy Ageing Brain
               (Doctoral thesis). Downing College, University of Cambridge.
               https://doi.org/10.17863/CAM.29356
        """
        return msdki_prediction(self.model_params, gtab, S0=S0)


def wls_fit_msdki(design_matrix, msignal, ng, mask=None,
                  min_signal=MIN_POSITIVE_SIGNAL, return_S0_hat=False):
    r"""
    Fits the mean signal diffusion kurtosis imaging based on a weighted
    least square solution [1]_.

    Parameters
    ----------
    design_matrix : array (nub, 3)
        Design matrix holding the covariants used to solve for the regression
        coefficients of the mean signal diffusion kurtosis model. Note that
        nub is the number of unique b-values
    msignal : ndarray ([X, Y, Z, ..., nub])
        Mean signal along all gradient directions for each unique b-value
        Note that the last dimension should contain the signal means and nub
        is the number of unique b-values.
    ng : ndarray(nub)
        Number of gradient directions used to compute the mean signal for
        all unique b-values
    mask : array
        A boolean array used to mark the coordinates in the data that
        should be analyzed that has the shape data.shape[:-1]
    min_signal : float, optional
        Voxel with mean signal intensities lower than the min positive signal
        are not processed. Default: 0.0001
    return_S0_hat : bool
        If True, also return S0 values for the fit.

    Returns
    -------
    params : array (..., 2)
        Containing the mean signal diffusivity and mean signal kurtosis

    References
    ----------
    .. [1] Henriques, R.N., 2018. Advanced Methods for Diffusion MRI Data
           Analysis and their Application to the Healthy Ageing Brain
           (Doctoral thesis). Downing College, University of Cambridge.
           https://doi.org/10.17863/CAM.29356
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
        # Define weights as diag(ng * yn**2)
        W = np.diag(ng * msignal[v]**2)

        # WLS fitting
        BTW = np.dot(design_matrix.T, W)
        inv_BT_W_B = np.linalg.pinv(np.dot(BTW, design_matrix))
        p = np.linalg.multi_dot([inv_BT_W_B, BTW, np.log(msignal[v])])

        # Process parameters
        p[1] = p[1] / (p[0]**2)
        p[2] = np.exp(p[2])
        params[v] = p

    if return_S0_hat:
        return params[..., :2], params[..., 2]
    else:
        return params[..., :2]


def design_matrix(ubvals):
    """  Constructs design matrix for the mean signal diffusion kurtosis model

    Parameters
    ----------
    ubvals : array
        Containing the unique b-values of the data.

    Returns
    -------
    design_matrix : array (nb, 3)
        Design matrix or B matrix for the mean signal diffusion kurtosis
        model assuming that parameters are in the following order:
        design_matrix[j, :] = (msd, msk, S0)
    """
    nb = ubvals.shape
    B = np.zeros(nb + (3,))
    B[:, 0] = -ubvals
    B[:, 1] = 1.0/6.0 * ubvals**2
    B[:, 2] = np.ones(nb)
    return B
