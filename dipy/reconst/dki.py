#!/usr/bin/python
""" Classes and functions for fitting the diffusion kurtosis model """
from __future__ import division, print_function, absolute_import

import warnings

import numpy as np

import scipy.optimize as opt

from dipy.reconst.dti import (TensorFit, fractional_anisotropy,
                              geodesic_anisotropy, mean_diffusivity,
                              axial_diffusivity, radial_diffusivity, trace,
                              color_fa, determinant, isotropic, deviatoric,
                              norm, mode, linearity, planarity, sphericity,
                              apparent_diffusion_coef, from_lower_triangular,
                              lower_triangular, decompose_tensor,
                              _min_positive_signal)

from dipy.reconst.utils import dki_design_matrix as design_matrix
from dipy.utils.six.moves import range
from dipy.core.geometry import (sphere2cart, cart2sphere)
from dipy.data import get_sphere
from ..core.gradients import gradient_table
from ..core.sphere import Sphere
from .vec_val_sum import vec_val_vect
from ..core.onetime import auto_attr
from .base import ReconstModel
from dipy.reconst.peaks import (PeaksAndMetrics, peak_directions)
from dipy.core.ndindex import ndindex


def apparent_kurtosis_coef(dki_params, sphere, min_diffusivity=0,
                           min_kurtosis=-1):
    r""" Calculate the apparent kurtosis coefficient (AKC) in each direction
    of a sphere.

    Parameters
    ----------
    dki_params : ndarray (x, y, z, 27) or (n, 27)
        All parameters estimated from the diffusion kurtosis model.
        Parameters are ordered as follow:
            1) Three diffusion tensor's eingenvalues
            2) Three lines of the eigenvector matrix each containing the first,
               second and third coordinates of the eigenvectors respectively
            3) Fifteen elements of the kurtosis tensor
    sphere : a Sphere class instance
        The AKC will be calculated for each of the vertices in the sphere
    min_diffusivity : float (optional)
        Because negative eigenvalues are not physical and small eigenvalues
        cause quite a lot of noise in diffusion based metrics, diffusivity
        values smaller than `min_diffusivity` are replaced with
        `min_diffusivity`. defaut = 0
    min_kurtosis : float (optional)
        Because high amplitude negative values of kurtosis are not physicaly
        and biologicaly pluasible, and these causes huge artefacts in kurtosis
        based measures, directional kurtosis values than `min_kurtosis` are
        replaced with `min_kurtosis`. defaut = -1

    Returns
    --------
    AKC : ndarray (x, y, z, g) or (n, g)
        Apparent kurtosis coefficient (AKC) for all g directions of a sphere.

    Notes
    -----
    For each sphere direction with coordinates $(n_{1}, n_{2}, n_{3})$, the
    calculation of AKC is done using formula:

    .. math ::
        AKC(n)=\frac{MD^{2}}{ADC(n)^{2}}\sum_{i=1}^{3}\sum_{j=1}^{3}
        \sum_{k=1}^{3}\sum_{l=1}^{3}n_{i}n_{j}n_{k}n_{l}W_{ijkl}

    where $W_{ijkl}$ are the elements of the kurtosis tensor, MD the mean
    diffusivity and ADC the apparent diffusion coefficent computed as:

    .. math ::
        ADC(n)=\sum_{i=1}^{3}\sum_{j=1}^{3}n_{i}n_{j}D_{ij}

    where $D_{ij}$ are the elements of the diffusion tensor.
    """

    # Flat parameters
    outshape = dki_params.shape[:-1]
    dki_params = dki_params.reshape((-1, dki_params.shape[-1]))

    # Split data
    evals, evecs, kt = split_dki_param(dki_params)

    # Compute MD
    MD = mean_diffusivity(evals)

    # Initialize AKC matrix
    V = sphere.vertices
    AKC = np.zeros((len(kt), len(V)))

    # loop over all voxels
    for vox in range(len(kt)):
        R = evecs[vox]
        dt = lower_triangular(np.dot(np.dot(R, np.diag(evals[vox])), R.T))
        AKC[vox] = _directional_kurtosis(dt, MD[vox], kt[vox], V,
                                         min_diffusivity=min_diffusivity,
                                         min_kurtosis=min_kurtosis)

    # reshape data according to input data
    AKC = AKC.reshape((outshape + (len(V),)))

    return AKC


def _directional_kurtosis(dt, MD, kt, V, min_diffusivity=0, min_kurtosis=-1):
    r""" Helper function that calculate the apparent kurtosis coefficient (AKC)
    in each direction of a sphere for a single voxel.

    Parameters
    ----------
    dt : array (6,)
        elements of the diffusion tensor of the voxel.
    MD : float
        mean diffusivity of the voxel
    kt : array (15,)
        elements of the kurtosis tensor of the voxel.
    V : array (g, 3)
        g directions of a Sphere in Cartesian coordinates
    min_diffusivity : float (optional)
        Because negative eigenvalues are not physical and small eigenvalues
        cause quite a lot of noise in diffusion based metrics, diffusivity
        values smaller than `min_diffusivity` are replaced with
        `min_diffusivity`. defaut = 0
    min_kurtosis : float (optional)
        Because high amplitude negative values of kurtosis are not physicaly
        and biologicaly pluasible, and these causes huge artefacts in kurtosis
        based measures, directional kurtosis values than `min_kurtosis` are
        replaced with `min_kurtosis`. defaut = -1

    Returns
    --------
    AKC : ndarray (g,)
        Apparent kurtosis coefficient (AKC) in all g directions of a sphere for
        a single voxel.

    See Also
    --------
    apparent_kurtosis_coef
    """
    ADC = \
        V[:, 0] * V[:, 0] * dt[0] + \
        2 * V[:, 0] * V[:, 1] * dt[1] + \
        V[:, 1] * V[:, 1] * dt[2] + \
        2 * V[:, 0] * V[:, 2] * dt[3] + \
        2 * V[:, 1] * V[:, 2] * dt[4] + \
        V[:, 2] * V[:, 2] * dt[5]

    if min_diffusivity is not None:
        ADC = ADC.clip(min=min_diffusivity)

    AKC = \
        V[:, 0] * V[:, 0] * V[:, 0] * V[:, 0] * kt[0] + \
        V[:, 1] * V[:, 1] * V[:, 1] * V[:, 1] * kt[1] + \
        V[:, 2] * V[:, 2] * V[:, 2] * V[:, 2] * kt[2] + \
        4 * V[:, 0] * V[:, 0] * V[:, 0] * V[:, 1] * kt[3] + \
        4 * V[:, 0] * V[:, 0] * V[:, 0] * V[:, 2] * kt[4] + \
        4 * V[:, 0] * V[:, 1] * V[:, 1] * V[:, 1] * kt[5] + \
        4 * V[:, 1] * V[:, 1] * V[:, 1] * V[:, 2] * kt[6] + \
        4 * V[:, 0] * V[:, 2] * V[:, 2] * V[:, 2] * kt[7] + \
        4 * V[:, 1] * V[:, 2] * V[:, 2] * V[:, 2] * kt[8] + \
        6 * V[:, 0] * V[:, 0] * V[:, 1] * V[:, 1] * kt[9] + \
        6 * V[:, 0] * V[:, 0] * V[:, 2] * V[:, 2] * kt[10] + \
        6 * V[:, 1] * V[:, 1] * V[:, 2] * V[:, 2] * kt[11] + \
        12 * V[:, 0] * V[:, 0] * V[:, 1] * V[:, 2] * kt[12] + \
        12 * V[:, 0] * V[:, 1] * V[:, 1] * V[:, 2] * kt[13] + \
        12 * V[:, 0] * V[:, 1] * V[:, 2] * V[:, 2] * kt[14]

    if min_kurtosis is not None:
        AKC = AKC.clip(min=min_kurtosis)

    return (MD/ADC) ** 2 * AKC


def dki_prediction(dki_params, gtab, S0=150):
    """ Predict a signal given diffusion kurtosis imaging parameters.

    Parameters
    ----------
    dki_params : ndarray (x, y, z, 27) or (n, 27)
        All parameters estimated from the diffusion kurtosis model.
        Parameters are ordered as follow:
            1) Three diffusion tensor's eingenvalues
            2) Three lines of the eigenvector matrix each containing the first,
               second and third coordinates of the eigenvector
            3) Fifteen elements of the kurtosis tensor
    gtab : a GradientTable class instance
        The gradient table for this prediction
    S0 : float or ndarray (optional)
        The non diffusion-weighted signal in every voxel, or across all
        voxels. Default: 150

    Returns
    --------
    S : (..., N) ndarray
        Simulated signal based on the DKI model:

    .. math::

        S=S_{0}e^{-bD+\frac{1}{6}b^{2}D^{2}K}
    """
    evals, evecs, kt = split_dki_param(dki_params)

    # Define DKI design matrix according to given gtab
    A = design_matrix(gtab)

    # Flat parameters and initialize pred_sig
    fevals = evals.reshape((-1, evals.shape[-1]))
    fevecs = evecs.reshape((-1,) + evecs.shape[-2:])
    fkt = kt.reshape((-1, kt.shape[-1]))
    pred_sig = np.zeros((len(fevals), len(gtab.bvals)))

    # lopping for all voxels
    for v in range(len(pred_sig)):
        DT = np.dot(np.dot(fevecs[v], np.diag(fevals[v])), fevecs[v].T)
        dt = lower_triangular(DT)
        MD = (dt[0] + dt[2] + dt[5]) / 3
        X = np.concatenate((dt, fkt[v]*MD*MD, np.array([np.log(S0)])), axis=0)
        pred_sig[v] = np.exp(np.dot(A, X))

    # Reshape data according to the shape of dki_params
    pred_sig = pred_sig.reshape(dki_params.shape[:-1] + (pred_sig.shape[-1],))

    return pred_sig


class DiffusionKurtosisModel(ReconstModel):
    """ Class for the Diffusion Kurtosis Model
    """
    def __init__(self, gtab, fit_method="OLS", *args, **kwargs):
        """ Diffusion Kurtosis Tensor Model [1]

        Parameters
        ----------
        gtab : GradientTable class instance

        fit_method : str or callable
            str can be one of the following:
            'OLS' or 'ULLS' for ordinary least squares
                dki.ols_fit_dki
            'WLS' or 'UWLLS' for weighted ordinary least squares
                dki.wls_fit_dki

            callable has to have the signature:
                fit_method(design_matrix, data, *args, **kwargs)

        args, kwargs : arguments and key-word arguments passed to the
           fit_method. See dki.ols_fit_dki, dki.wls_fit_dki for details

        References
        ----------
           [1] Tabesh, A., Jensen, J.H., Ardekani, B.A., Helpern, J.A., 2011.
           Estimation of tensors and tensor-derived measures in diffusional
           kurtosis imaging. Magn Reson Med. 65(3), 823-836
        """
        ReconstModel.__init__(self, gtab)

        if not callable(fit_method):
            try:
                self.fit_method = common_fit_methods[fit_method]
            except KeyError:
                raise ValueError('"' + str(fit_method) + '" is not a known '
                                 'fit method, the fit method should either be '
                                 'a function or one of the common fit methods')

        self.design_matrix = design_matrix(self.gtab)
        self.args = args
        self.kwargs = kwargs
        self.min_signal = self.kwargs.pop('min_signal', None)
        if self.min_signal is not None and self.min_signal <= 0:
            e_s = "The `min_signal` key-word argument needs to be strictly"
            e_s += " positive."
            raise ValueError(e_s)

    def fit(self, data, mask=None):
        """ Fit method of the DKI model class

        Parameters
        ----------
        data : array
            The measured signal from one voxel.

        mask : array
            A boolean array used to mark the coordinates in the data that
            should be analyzed that has the shape data.shape[-1]
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
            dki_params = params_in_mask.reshape(out_shape)
        else:
            dki_params = np.zeros(data.shape[:-1] + (27,))
            dki_params[mask, :] = params_in_mask

        return DiffusionKurtosisFit(self, dki_params)

    def predict(self, dki_params, S0=1):
        """ Predict a signal for this DKI model class instance given
        parameters.

        Parameters
        ----------
        dki_params : ndarray (x, y, z, 27) or (n, 27)
            All parameters estimated from the diffusion kurtosis model.
            Parameters are ordered as follow:
                1) Three diffusion tensor's eingenvalues
                2) Three lines of the eigenvector matrix each containing the
                   first, second and third coordinates of the eigenvector
                3) Fifteen elements of the kurtosis tensor
        S0 : float or ndarray (optional)
            The non diffusion-weighted signal in every voxel, or across all
            voxels. Default: 1
        """
        return dki_prediction(dki_params, self.gtab, S0)


class DiffusionKurtosisFit(TensorFit):
    """ Class for fitting the Diffusion Kurtosis Model"""
    def __init__(self, model, model_params):
        """ Initialize a DiffusionKurtosisFit class instance.

        Since DKI is an extension of DTI, class instance is defined as subclass
        of the TensorFit from dti.py

        Parameters
        ----------
        model : DiffusionKurtosisModel Class instance
            Class instance containing the Diffusion Kurtosis Model for the fit
        model_params : ndarray (x, y, z, 27) or (n, 27)
            All parameters estimated from the diffusion kurtosis model.
            Parameters are ordered as follow:
                1) Three diffusion tensor's eingenvalues
                2) Three lines of the eigenvector matrix each containing the
                   first, second and third coordinates of the eigenvector
                3) Fifteen elements of the kurtosis tensor
        """
        TensorFit.__init__(self, model, model_params)

    @property
    def kt(self):
        """
        Returns the 15 independent elements of the kurtosis tensor as an array
        """
        return self.model_params[..., 12:]

    def akc(self, sphere):
        r""" Calculate the apparent kurtosis coefficient (AKC) in each
        direction on the sphere for each voxel in the data

        Parameters
        ----------
        sphere : Sphere class instance

        Returns
        -------
        akc : ndarray
           The estimates of the apparent kurtosis coefficient in every
           direction on the input sphere

        Notes
        -----
        For each sphere direction with coordinates $(n_{1}, n_{2}, n_{3})$, the
        calculation of AKC is done using formula:

        .. math ::
            AKC(n)=\frac{MD^{2}}{ADC(n)^{2}}\sum_{i=1}^{3}\sum_{j=1}^{3}
            \sum_{k=1}^{3}\sum_{l=1}^{3}n_{i}n_{j}n_{k}n_{l}W_{ijkl}

        where $W_{ijkl}$ are the elements of the kurtosis tensor, MD the mean
        diffusivity and ADC the apparent diffusion coefficent computed as:

        .. math ::
            ADC(n)=\sum_{i=1}^{3}\sum_{j=1}^{3}n_{i}n_{j}D_{ij}

        where $D_{ij}$ are the elements of the diffusion tensor.
        """
        return apparent_kurtosis_coef(self.model_params, sphere)

    def dki_odf(self, sphere, alpha=4):
        """
        The diffusion kurtosis estimate of the orientation distribution
        function (DKI-ODF). This estimates the DKI-ODF in every direction of
        the given sphere in every voxel of the input data.

        Parameters
        ----------
        sphere : Sphere class instance.
            The DKI-ODF is calculated in the vertices of this input.
        alpha : float, optional
            radial weighting power. Default is 4 according to [Jen2014]_ and
            [Raf2015]

        Returns
        -------
        dki_odf : ndarray
            The DKI-ODF sampled in every direction of the sphere in every
            voxel of the input data.

        Notes
        -----
        This is based on the DKI-ODF estimate proposed by [Jen2014]_. 
        More information about this method can be found in [Raf2015]_.

        .. [Jen2014] Jensen, J.H., Helpern, J.A., Tabesh, A., (2014). Leading
            non-Gaussian corrections for diffusion orientation distribution
            function. NMR Biomed. 27, 202-211.
            http://dx.doi.org/10.1002/nbm.3053.

        .. [Raf2015] Neto Henriques, R., Correia, M.M., Nunes, R.G., Ferreira,
            H.A. (2015). Exploring the 3D geometry of the diffusion kurtosis
            tensor - Impact on the development of robust tractography procedures
            and novel biomarkers. NeuroImage 111: 85-99.
            doi:10.1016/j.neuroimage.2015.02.004
        """
        return diffusion_kurtosis_odf(self.model_params, sphere, alpha)

    def dki_directions(self, sphere, alpha=4, relative_peak_threshold=0.1,
                       min_separation_angle=20, mask=None, return_odf=False,
                       normalize_peaks=False, npeaks=3, gtol=1e-5):
        """ Estimation of fiber directions based on diffusion kurtosis imaging
        (DKI). Fiber directions are estimated as the maxima of the DKI
        orientation distribution function [Jen2014]. This function is based on 
        the work done by [Raf2015]_.

        Parameters
        ----------
        sphere : Sphere class instance, optional
            The sphere providing sample directions for initial evaluation of
            the DKI-ODF.
        alpha : float, optional
            Radial weighting power of the orientation distribution function.
            Default is 4 according to [Jen2014]_ and [Raf2015]_.
        relative_peak_threshold : float, optional
            Only return peaks greater than ``relative_peak_threshold * m``
            where m is the largest peak.
        min_separation_angle : float in [0, 90], optinal
            The minimum distance between directions. If two peaks are too close
            only the larger of the two is returned.
        mask : array, optional
            If `mask` is provided, voxels that are False in `mask` are skipped
            and no peaks are returned.
        return_odf : bool, optional
            If True, the odfs sampled on sphere directions are returned.
        normalize_peaks : bool, optional
            If true, all peak values are calculated relative to `max(odf)`.
        npeaks : int, optional
            Maximum number of peaks found (default 3 peaks).
        gtol : float, optional
            Degree gradient must be less than gtol before succesful
            termination. If gtol is None, fiber direction is directly taken
            from the initial sampled directions of the given sphere object.

        Returns
        -------
        pam : PeaksAndMetrics
            An object with ``peak_directions``, ``peak_values``,
            ``peak_indices``, ``odf`` as attributes

        .. [Jen2014] Jensen, J.H., Helpern, J.A., Tabesh, A., (2014). Leading
            non-Gaussian corrections for diffusion orientation distribution
            function. NMR Biomed. 27, 202-211.
            http://dx.doi.org/10.1002/nbm.3053.

        .. [Raf2015] Neto Henriques, R., Correia, M.M., Nunes, R.G., Ferreira,
            H.A. (2015). Exploring the 3D geometry of the diffusion kurtosis
            tensor - Impact on the development of robust tractography
            procedures and novel biomarkers. NeuroImage 111: 85-99.
            doi:10.1016/j.neuroimage.2015.02.004
        """
        return dki_directions(self.model_params, sphere, alpha=4,
                              relative_peak_threshold=0.1,
                              min_separation_angle=20, mask=None,
                              return_odf=False, normalize_peaks=False,
                              npeaks=3, gtol=1e-5)

    def predict(self, gtab, S0=1):
        r""" Given a DKI model fit, predict the signal on the vertices of a
        gradient table

        Parameters
        ----------
        dki_params : ndarray (x, y, z, 27) or (n, 27)
            All parameters estimated from the diffusion kurtosis model.
            Parameters are ordered as follow:
                1) Three diffusion tensor's eingenvalues
                2) Three lines of the eigenvector matrix each containing the
                   first, second and third coordinates of the eigenvector
                3) Fifteen elements of the kurtosis tensor

        gtab : a GradientTable class instance
            The gradient table for this prediction

        S0 : float or ndarray (optional)
            The non diffusion-weighted signal in every voxel, or across all
            voxels. Default: 1

        Notes
        -----
        The predicted signal is given by:

        .. math::

            S(n,b)=S_{0}e^{-bD(n)+\frac{1}{6}b^{2}D(n)^{2}K(n)}

        $\mathbf{D(n)}$ and $\mathbf{K(n)}$ can be computed from the DT and KT
        using the following equations:

        .. math::

            D(n)=\sum_{i=1}^{3}\sum_{j=1}^{3}n_{i}n_{j}D_{ij}

        and

        .. math::

            K(n)=\frac{MD^{2}}{D(n)^{2}}\sum_{i=1}^{3}\sum_{j=1}^{3}
            \sum_{k=1}^{3}\sum_{l=1}^{3}n_{i}n_{j}n_{k}n_{l}W_{ijkl}

        where $D_{ij}$ and $W_{ijkl}$ are the elements of the second-order DT
        and the fourth-order KT tensors, respectively, and $MD$ is the mean
        diffusivity.
        """
        return dki_prediction(self.model_params, self.gtab, S0)


def ols_fit_dki(design_matrix, data):
    r""" Computes ordinary least squares (OLS) fit to calculate the diffusion
    tensor and kurtosis tensor using a linear regression diffusion kurtosis
    model [1]_.

    Parameters
    ----------
    design_matrix : array (g, 22)
        Design matrix holding the covariants used to solve for the regression
        coefficients.
    data : array (N, g)
        Data or response variables holding the data. Note that the last
        dimension should contain the data. It makes no copies of data.

    Returns
    -------
    dki_params : array (N, 27)
        All parameters estimated from the diffusion kurtosis model.
        Parameters are ordered as follow:
            1) Three diffusion tensor's eingenvalues
            2) Three lines of the eigenvector matrix each containing the first,
               second and third coordinates of the eigenvector
            3) Fifteen elements of the kurtosis tensor

    See Also
    --------
    wls_fit_dki

    References
    ----------
       [1] Tabesh, A., Jensen, J.H., Ardekani, B.A., Helpern, J.A., 2011.
           Estimation of tensors and tensor-derived measures in diffusional
           kurtosis imaging. Magn Reson Med. 65(3), 823-836
    """
    tol = 1e-6

    # preparing data and initializing parameters
    data = np.asarray(data)
    data_flat = data.reshape((-1, data.shape[-1]))
    dki_params = np.empty((len(data_flat), 27))

    # inverting design matrix and defining minimun diffusion aloud
    min_diffusivity = tol / -design_matrix.min()
    inv_design = np.linalg.pinv(design_matrix)

    # lopping OLS solution on all data voxels
    for vox in range(len(data_flat)):
        dki_params[vox] = _ols_iter(inv_design, data_flat[vox],
                                    min_diffusivity)

    # Reshape data according to the input data shape
    dki_params = dki_params.reshape((data.shape[:-1]) + (27,))

    return dki_params


def _ols_iter(inv_design, sig, min_diffusivity):
    """ Helper function used by ols_fit_dki - Applies OLS fit of the diffusion
    kurtosis model to single voxel signals.

    Parameters
    ----------
    inv_design : array (g, 22)
        Inverse of the design matrix holding the covariants used to solve for
        the regression coefficients.
    sig : array (g,)
        Diffusion-weighted signal for a single voxel data.
    min_diffusivity : float
        Because negative eigenvalues are not physical and small eigenvalues,
        much smaller than the diffusion weighting, cause quite a lot of noise
        in metrics such as fa, diffusivity values smaller than
        `min_diffusivity` are replaced with `min_diffusivity`.

    Returns
    -------
    dki_params : array (27,)
        All parameters estimated from the diffusion kurtosis model.
        Parameters are ordered as follow:
            1) Three diffusion tensor's eingenvalues
            2) Three lines of the eigenvector matrix each containing the first,
               second and third coordinates of the eigenvector
            3) Fifteen elements of the kurtosis tensor
    """
    # DKI ordinary linear least square solution
    log_s = np.log(sig)
    result = np.dot(inv_design, log_s)

    # Extracting the diffusion tensor parameters from solution
    DT_elements = result[:6]
    evals, evecs = decompose_tensor(from_lower_triangular(DT_elements),
                                    min_diffusivity=min_diffusivity)

    # Extracting kurtosis tensor parameters from solution
    MD_square = (evals.mean(0))**2
    KT_elements = result[6:21] / MD_square

    # Write output
    dki_params = np.concatenate((evals, evecs[0], evecs[1], evecs[2],
                                 KT_elements), axis=0)

    return dki_params


def wls_fit_dki(design_matrix, data):
    r""" Computes weighted linear least squares (WLS) fit to calculate
    the diffusion tensor and kurtosis tensor using a weighted linear
    regression diffusion kurtosis model [1]_.

    Parameters
    ----------
    design_matrix : array (g, 22)
        Design matrix holding the covariants used to solve for the regression
        coefficients.
    data : array (N, g)
        Data or response variables holding the data. Note that the last
        dimension should contain the data. It makes no copies of data.
    min_signal : default = 1
        All values below min_signal are repalced with min_signal. This is done
        in order to avoid taking log(0) durring the tensor fitting.

    Returns
    -------
    dki_params : array (N, 27)
        All parameters estimated from the diffusion kurtosis model for all N
        voxels.
        Parameters are ordered as follow:
            1) Three diffusion tensor's eingenvalues
            2) Three lines of the eigenvector matrix each containing the first
               second and third coordinates of the eigenvector
            3) Fifteen elements of the kurtosis tensor

    References
    ----------
       [1] Veraart, J., Sijbers, J., Sunaert, S., Leemans, A., Jeurissen, B.,
           2013. Weighted linear least squares estimation of diffusion MRI
           parameters: Strengths, limitations, and pitfalls. Magn Reson Med 81,
           335-346.
    """

    tol = 1e-6

    # preparing data and initializing parametres
    data = np.asarray(data)
    data_flat = data.reshape((-1, data.shape[-1]))
    dki_params = np.empty((len(data_flat), 27))

    # inverting design matrix and defining minimun diffusion aloud
    min_diffusivity = tol / -design_matrix.min()
    inv_design = np.linalg.pinv(design_matrix)

    # lopping WLS solution on all data voxels
    for vox in range(len(data_flat)):
        dki_params[vox] = _wls_iter(design_matrix, inv_design, data_flat[vox],
                                    min_diffusivity)

    # Reshape data according to the input data shape
    dki_params = dki_params.reshape((data.shape[:-1]) + (27,))

    return dki_params


def _wls_iter(design_matrix, inv_design, sig, min_diffusivity):
    """ Helper function used by wls_fit_dki - Applies WLS fit of the diffusion
    kurtosis model to single voxel signals.

    Parameters
    ----------
    design_matrix : array (g, 22)
        Design matrix holding the covariants used to solve for the regression
        coefficients
    inv_design : array (g, 22)
        Inverse of the design matrix.
    sig : array (g, )
        Diffusion-weighted signal for a single voxel data.
    min_diffusivity : float
        Because negative eigenvalues are not physical and small eigenvalues,
        much smaller than the diffusion weighting, cause quite a lot of noise
        in metrics such as fa, diffusivity values smaller than
        `min_diffusivity` are replaced with `min_diffusivity`.

    Returns
    -------
    dki_params : array (27, )
        All parameters estimated from the diffusion kurtosis model.
        Parameters are ordered as follow:
            1) Three diffusion tensor's eingenvalues
            2) Three lines of the eigenvector matrix each containing the first,
               second and third coordinates of the eigenvector
            3) Fifteen elements of the kurtosis tensor
    """
    A = design_matrix

    # DKI ordinary linear least square solution
    log_s = np.log(sig)
    ols_result = np.dot(inv_design, log_s)

    # Define weights as diag(yn**2)
    W = np.diag(np.exp(2 * np.dot(A, ols_result)))

    # DKI weighted linear least square solution
    inv_AT_W_A = np.linalg.pinv(np.dot(np.dot(A.T, W), A))
    AT_W_LS = np.dot(np.dot(A.T, W), log_s)
    wls_result = np.dot(inv_AT_W_A, AT_W_LS)

    # Extracting the diffusion tensor parameters from solution
    DT_elements = wls_result[:6]
    evals, evecs = decompose_tensor(from_lower_triangular(DT_elements),
                                    min_diffusivity=min_diffusivity)

    # Extracting kurtosis tensor parameters from solution
    MD_square = (evals.mean(0))**2
    KT_elements = wls_result[6:21] / MD_square

    # Write output
    dki_params = np.concatenate((evals, evecs[0], evecs[1], evecs[2],
                                 KT_elements), axis=0)

    return dki_params


def split_dki_param(dki_params):
    r""" Extract the diffusion tensor eigenvalues, the diffusion tensor
    eigenvector matrix, and the 15 independent elements of the kurtosis tensor
    from the model parameters estimated from the DKI model

    Parameters
    ----------
    dki_params : ndarray (x, y, z, 27) or (n, 27)
        All parameters estimated from the diffusion kurtosis model.
        Parameters are ordered as follow:
            1) Three diffusion tensor's eingenvalues
            2) Three lines of the eigenvector matrix each containing the first,
               second and third coordinates of the eigenvector
            3) Fifteen elements of the kurtosis tensor

    Returns
    --------
    eigvals : array (x, y, z, 3) or (n, 3)
        Eigenvalues from eigen decomposition of the tensor.
    eigvecs : array (x, y, z, 3, 3) or (n, 3, 3)
        Associated eigenvectors from eigen decomposition of the tensor.
        Eigenvectors are columnar (e.g. eigvecs[:,j] is associated with
        eigvals[j])
    kt : array (x, y, z, 15) or (n, 15)
        Fifteen elements of the kurtosis tensor
    """
    evals = dki_params[..., :3]
    evecs = dki_params[..., 3:12].reshape(dki_params.shape[:-1] + (3, 3))
    kt = dki_params[..., 12:]

    return evals, evecs, kt


def diffusion_kurtosis_odf(dki_params, sphere, alpha=4):
    """ The diffusion kurtosis estimate of the orientation distribution
    function (DKI-ODF). This estimates the DKI-ODF in every direction of the
    given sphere in every voxel of the input data.

    Parameters
    ----------
    dki_params : ndarray (x, y, z, 27) or (n, 27)
        All parameters estimated from the diffusion kurtosis model.
        Parameters are ordered as follow:
            1) Three diffusion tensor's eingenvalues
            2) Three lines of the eigenvector matrix each containing the first,
               second and third coordinates of the eigenvector
            3) Fifteen elements of the kurtosis tensor
    sphere : Sphere class instance.
        The DKI-ODF is calculated in the vertices of this input.
    alpha : float, optional
        Radial weighting power. Default is 4 according to [Jen2014]_ and
        [Raf2015]_.

    Returns
    -------
    kODF : ndarray (x, y, z, g) or (n, g)
        The DKI-ODF sampled for every voxel of the input data and every g
        directions of the sphere.

    Notes
    -----
    This is based on the DKI-ODF estimate proposed by [Jen2014]_. More
    information about this method can be found in [Raf2015]_.

    .. [Jen2014] Jensen, J.H., Helpern, J.A., Tabesh, A., (2014). Leading
        non-Gaussian corrections for diffusion orientation distribution
        function. NMR Biomed. 27, 202-211. http://dx.doi.org/10.1002/nbm.3053.

    .. [Raf2015] Neto Henriques, R., Correia, M.M., Nunes, R.G., Ferreira,
        H.A. (2015). Exploring the 3D geometry of the diffusion kurtosis
        tensor - Impact on the development of robust tractography procedures
        and novel biomarkers. NeuroImage 111: 85-99.
        doi:10.1016/j.neuroimage.2015.02.004
    """
    # flat data
    outshape = dki_params.shape[:-1]
    dki_params = dki_params.reshape((-1, dki_params.shape[-1]))

    # Split data
    evals, evecs, kt = split_dki_param(dki_params)

    # Initialize AKC matrix
    V = sphere.vertices
    kODF = np.zeros((len(kt), len(V)))

    # select non-zero voxels
    # (when #677 is merged replace this code lines by function _positive_evals)
    # rel_i = _positive_evals(evals[..., 0], evals[..., 1], evals[..., 2])    
    er = np.finfo(evals.ravel()[0]).eps * 1e3
    rel_i = np.logical_and(evals[..., 0] > er,
                           np.logical_and(evals[..., 1] > er,
                                          evals[..., 2] > er))
    kt = kt[rel_i]
    evecs = evecs[rel_i]
    evals = evals[rel_i]
    kODFi = kODF[rel_i]

    # loop over all relevant voxels
    for vox in range(len(kt)):
        # Compute dimensionless tensor U
        R = evecs[vox]
        dt = np.dot(np.dot(R, np.diag(evals[vox])), R.T)
        MD = (dt[0, 0] + dt[1, 1] + dt[2, 2]) / 3
        U = np.linalg.pinv(dt) * MD

        # loop over all directions
        sampledODF = np.zeros(len(V))
        for i in range(len(V)):
             sampledODF[i] = _dki_odf_core(V[i], kt[vox], U, alpha)

        kODFi[vox] = sampledODF

    # reshape data according to input data
    kODF[rel_i] = kODFi
    kODF = kODF.reshape((outshape + (len(V),)))
    return kODF


def _dki_odf_core(n, kt, U, alpha=4):
    """ Helper function thar compute the DKI based ODF estimation of a voxel
    along the given directions n
    
    Parameters
    ----------
    n : array (3,)
        arrays containing the three cartesian coordinates of direction n
    kt : array (15,)
        elements of the kurtosis tensor of the voxel.
    U : array (3, 3)
        dimensionless tensor defined by $U = \overline{D} \times DT^{-1}$,
        where \overline{D} is the mean diffusivity and DT the diffusion tensor
    alpha : float, optional
        Radial weighting power. Default is 4.
    
    Returns
    -------
    dki_odf : float
        The DKI-ODF value for direction n.
    """
    # Compute elements of matrix V
    Un = np.dot(U, n)
    nUn = np.dot(n, Un)
    V00 = Un[0]**2 / nUn
    V11 = Un[1]**2 / nUn
    V22 = Un[2]**2 / nUn
    V01 = Un[0]*Un[1] / nUn
    V02 = Un[0]*Un[2] / nUn
    V12 = Un[1]*Un[2] / nUn

    # diffusion ODF
    ODFg = (1./nUn) ** ((alpha+1.)/2.)

    # Estimate ODF
    ODF = \
        ODFg * (1. + 1/24. * \
        (kt[0] * (3*U[0, 0]*U[0, 0] - 6*(alpha+1)*U[0, 0]*V00 + \
                  (alpha+1)*(alpha+3)*V00*V00) + \
         kt[1] * (3*U[1, 1]*U[1, 1] - 6*(alpha+1)*U[1, 1]*V11 + \
                  (alpha+1) * (alpha + 3)*V11*V11) + \
         kt[2] * (3*U[2, 2]*U[2, 2] - 6*(alpha+1)*U[2, 2]*V22 + \
                  (alpha+1)*(alpha+3)*V22*V22) + \
         kt[3] * (12*U[0, 0]*U[0, 1] - 12*(alpha+1)*U[0, 0]*V01 - \
                  12*(alpha+1)*U[0, 1]*V00 + 4*(alpha+1)*(alpha+3)*V00*V01) + \
         kt[4] * (12*U[0, 0]*U[0, 2] - 12*(alpha+1)*U[0, 0]*V02 - \
                  12*(alpha+1)*U[0, 2]*V00 + 4*(alpha+1)*(alpha+3)*V00*V02) + \
         kt[5] * (12*U[0, 1]*U[1, 1] - 12*(alpha+1)*U[0, 1]*V11 - \
                  12*(alpha+1)*U[1, 1]*V01 + 4*(alpha+1)*(alpha+3)*V01*V11) + \
         kt[6] * (12*U[1, 1]*U[1, 2] - 12*(alpha+1)*U[1, 1]*V12 - \
                  12*(alpha+1)*U[1, 2]*V11 + 4*(alpha+1)*(alpha+3)*V11*V12) + \
         kt[7] * (12*U[0, 2]*U[2, 2] - 12*(alpha+1)*U[0, 2]*V22 - \
                  12*(alpha+1)*U[2, 2]*V02 + 4*(alpha+1)*(alpha+3)*V02*V22) + \
         kt[8] * (12*U[1, 2]*U[2, 2] - 12*(alpha+1)*U[1,2]*V22 - \
                  12*(alpha+1)*U[2, 2]*V12 + 4*(alpha+1)*(alpha+3)*V12*V22) + \
         kt[9] * (6*U[0, 0]*U[1, 1] + 12*U[0, 1]*U[0, 1] - \
                  6*(alpha+1)*U[0, 0]*V11 - 6*(alpha+1)*U[1, 1]*V00 - \
                  24*(alpha+1)*U[0, 1]*V01 + 2*(alpha+1)*(alpha+3)*V00*V11 + \
                  4*(alpha+1)*(alpha+3)*V01*V01) + \
         kt[10] * (6*U[0, 0]*U[2, 2] + 12*U[0, 2]*U[0, 2] - \
                   6*(alpha+1)*U[0, 0]*V22 - 6*(alpha+1)*U[2, 2]*V00 - \
                   24*(alpha+1)*U[0,2]*V02 + 2*(alpha+1)*(alpha+3)*V00*V22 + \
                   4*(alpha+1)*(alpha+3)*V02*V02) + \
         kt[11] * (6*U[1, 1]*U[2, 2] + 12*U[1, 2]*U[1, 2] - \
                   6*(alpha+1)*U[1,1]*V22 - 6*(alpha+1)*U[2, 2]*V11 - \
                   24*(alpha+1)*U[1,2]*V12 + 2*(alpha+1)*(alpha+3)*V11*V22 + \
                   4*(alpha+1)*(alpha+3)*V12*V12) + \
         kt[12] * (12*U[0, 0]*U[1, 2] + 24*U[0, 1]*U[0, 2] - \
                   12*(alpha+1)*U[0, 0]*V12 - 12*(alpha+1)*U[1, 2]*V00 - \
                   24*(alpha+1)*U[0, 1]*V02 - 24*(alpha+1)*U[0, 2]*V01 + \
                   4*(alpha+1)*(alpha+3)*V00*V12 + \
                   8*(alpha+1)*(alpha+3)*V01*V02) + \
         kt[13] * (12*U[1, 1]*U[0, 2] + 24*U[1, 0]*U[1, 2] - \
                   12*(alpha+1)*U[1, 1]*V02 - 12*(alpha+1)*U[0, 2]*V11 - \
                   24*(alpha+1)*U[0, 1]*V12 - 24*(alpha+1)*U[1, 2]*V01 + \
                   4*(alpha+1)*(alpha+3)*V11*V02 + \
                   8*(alpha+1)*(alpha+3)*V01*V12) + \
         kt[14] * (12*U[2, 2]*U[0, 1] + 24*U[2, 0]*U[2, 1] - \
                   12*(alpha+1)*U[2, 2]*V01 - 12*(alpha+1)*U[0, 1]*V22 - \
                   24*(alpha+1)*U[0, 2]*V12 - 24*(alpha+1)*U[1, 2]*V02 + \
                   4*(alpha+1)*(alpha+3)*V22*V01 + \
                   8*(alpha+1)*(alpha+3)*V02*V12)))

    return ODF


common_fit_methods = {'WLS': wls_fit_dki,
                      'OLS': ols_fit_dki,
                      'UWLLS': wls_fit_dki,
                      'ULLS': ols_fit_dki,
                      }

ind_ele = {1: 0, 16: 1, 81: 2, 2: 3, 3: 4, 8: 5, 24: 6, 27: 7, 54: 8, 4: 9,
           9: 10, 36: 11, 6: 12, 12: 13, 18: 14}


def Wcons(k_elements):
    r""" Construct the full 4D kurtosis tensors from its 15 independent
    elements
    
    Parameters
    ----------
    k_elements : (15,)
        elements of the kurtosis tensor in the following order:
        
        .. math::
            
    \begin{matrix} ( & W_{xxxx} & W_{yyyy} & W_{zzzz} & W_{xxxy} & W_{xxxz}
                     & ... \\
                     & W_{xyyy} & W_{yyyz} & W_{xzzz} & W_{yzzz} & W_{xxyy}
                     & ... \\
                     & W_{xxzz} & W_{yyzz} & W_{xxyz} & W_{xyyz} & W_{xyzz}
                     & & )\end{matrix}
    Returns
    -------
    W : array(3, 3, 3, 3)
        Full 4D kurtosis tensor
    """

    W = np.zeros((3, 3, 3, 3))

    xyz = [0, 1, 2]
    for ind_i in xyz:
        for ind_j in xyz:
            for ind_k in xyz:
                for ind_l in xyz:
                    key = (ind_i+1) * (ind_j+1) * (ind_k+1) * (ind_l+1)
                    W[ind_i][ind_j][ind_k][ind_l] = k_elements[ind_ele[key]]

    return W


def dki_directions(dki_params, sphere, alpha=4, relative_peak_threshold=0.1,
                     min_separation_angle=20, mask=None, return_odf=False,
                     normalize_peaks=False, npeaks=3, gtol=1e-5):
    """ Estimation of fiber direction based on diffusion kurtosis imaging
    (DKI). Fiber directions are estimated as the maxima of the orientation
    distribution function [Jen2014]. This function is based on the work done by
    [Raf2015]_.

    Parameters
    ----------
    dki_params : ndarray (x, y, z, 27) or (n, 27)
        All parameters estimated from the diffusion kurtosis model.
        Parameters are ordered as follow:
            1) Three diffusion tensor's eingenvalues
            2) Three lines of the eigenvector matrix each containing the first,
               second and third coordinates of the eigenvector
            3) Fifteen elements of the kurtosis tensor
    sphere : Sphere class instance, optional
        The sphere providing sample directions for initial evaluation of the
        DKI-ODF.
    alpha : float, optional
        Radial weighting power of the orientation distribution function.
        Default is 4 according to [Jen2014]_ and [Raf2015]_.
    relative_peak_threshold : float, optional
        Only return peaks greater than ``relative_peak_threshold * m`` where m
        is the largest peak.
    min_separation_angle : float in [0, 90], optinal
        The minimum distance between directions. If two peaks are too close
        only the larger of the two is returned.
    mask : array, optional
        If `mask` is provided, voxels that are False in `mask` are skipped and
        no peaks are returned.
    return_odf : bool, optional
        If True, the odfs sampled on sphere directions are returned.
    normalize_peaks : bool, optional
        If true, all peak values are calculated relative to `max(odf)`.
    npeaks : int, optional
        Maximum number of peaks found (default 3 peaks).
    gtol : float, optional
        Degree gradient must be less than gtol before succesful termination. If
        gtol is None, fiber direction is directly taken from the initial
        sampled directions of the given sphere object

    Returns
    -------
    pam : PeaksAndMetrics
        An object with ``peak_directions``, ``peak_values``, ``peak_indices``,
        ``odf`` as attributes

    .. [Jen2014] Jensen, J.H., Helpern, J.A., Tabesh, A., (2014). Leading
        non-Gaussian corrections for diffusion orientation distribution
        function. NMR Biomed. 27, 202-211. http://dx.doi.org/10.1002/nbm.3053.

    .. [Raf2015] Neto Henriques, R., Correia, M.M., Nunes, R.G., Ferreira,
        H.A. (2015). Exploring the 3D geometry of the diffusion kurtosis
        tensor - Impact on the development of robust tractography procedures
        and novel biomarkers. NeuroImage 111: 85-99.
        doi:10.1016/j.neuroimage.2015.02.004
    """
    shape = dki_params.shape[:-1]

    # select voxels where to find fiber directions
    if mask is None:
        mask = np.ones(shape, dtype='bool')
    else:
        if mask.shape != shape:
            raise ValueError("Mask is not the same shape as data.")
    evals, evecs, kt = split_dki_param(dki_params)
    # select non-zero voxels
    # (when #677 is merged replace this code lines by function _positive_evals)
    # pos_evals = _positive_evals(evals[..., 0], evals[..., 1], evals[..., 2])
    er = np.finfo(evals.ravel()[0]).eps * 1e3
    pos_evals = np.logical_and(evals[..., 0] > er,
                               np.logical_and(evals[..., 1] > er,
                                              evals[..., 2] > er))
    mask = np.logical_and(mask, pos_evals)

    # initialize parameters for pam
    qa_array = np.zeros((shape + (npeaks,)))
    peak_dirs = np.zeros((shape + (npeaks, 3)))
    peak_values = np.zeros((shape + (npeaks,)))
    peak_indices = np.zeros((shape + (npeaks,)), dtype='int')
    if return_odf:
        odf_array = np.zeros((shape + (len(sphere.vertices),)))

    # for all voxels:
    global_max = -np.inf
    for idx in ndindex(shape):
        if not mask[idx]:
            continue
        
        # Compute dimensionless tensor U
        dt = np.dot(np.dot(evecs[idx], np.diag(evals[idx])), evecs[idx].T)
        MD = (dt[0, 0] + dt[1, 1] + dt[2, 2]) / 3
        U = np.linalg.pinv(dt) * MD
        
        # First sample of the DKI-ODF
        odf = np.zeros(len(sphere.vertices))
        for i in range(len(sphere.vertices)):
             odf[i] = _dki_odf_core(sphere.vertices[i], kt[idx], U, alpha)
        if return_odf:
            odf_array[idx] = odf

        # First estimate of the fiber direction from sphere vertices
        di, pk, ind = peak_directions(odf, sphere, relative_peak_threshold,
                                      min_separation_angle)

        if pk.shape[0] != 0:
            n = min(npeaks, pk.shape[0])
            # Direction convergence
            if gtol is not None:
                for p in range(n):
                    r, theta, phi = cart2sphere(di[p, 0], di[p, 1], di[p, 2])
                    ang = np.array([theta, phi])
                    ang[:] = opt.fmin_bfgs(_dki_odf_converge, ang,
                                           args=(kt[idx], U, alpha),
                                           gtol=gtol, disp=False,
                                           retall=False)
                    di[p] = np.array(sphere2cart(1., ang[0], ang[1]))
                    pk[p] = _dki_odf_core(di[p], kt[idx], U, alpha) 

            # Saving directions
            global_max = max(global_max, pk[0])
            qa_array[idx][:n] = pk[:n] - odf.min()
            peak_dirs[idx][:n] = di[:n]
            peak_indices[idx][:n] = ind[:n]
            peak_values[idx][:n] = pk[:n]

            if normalize_peaks:
                peak_values[idx][:n] /= pk[0]
                peak_dirs[idx] *= peak_values[idx][:, None]
        else:
            global_max = max(global_max, odf.max())

    qa_array /= global_max
    pam = PeaksAndMetrics()
    pam.sphere = sphere
    pam.peak_dirs = peak_dirs
    pam.peak_values = peak_values
    pam.peak_indices = peak_indices
    pam.qa = qa_array
    if return_odf:
        pam.odf = odf_array

    return pam


def _dki_odf_converge(ang, kt, U, alpha=4):
    """ Helper function that computes the negate of the DKI based ODF
    estimation of a voxel along a given directions in polar coordinates.

    Parameters
    ----------
    ang : array (2,)
        arrays containing the two polar angles
    kt : array (15,)
        elements of the kurtosis tensor of the voxel.
    U : array (3, 3)
        dimensionless tensor defined by $U = \overline{D} \times DT^{-1}$,
        where \overline{D} is the mean diffusivity and DT the diffusion tensor
    alpha : float, optional
        Radial weighting power. Default is 4.

    Returns
    -------
    dki_odf : float
        The negated DKI-ODF value for the given direction.
    
    Notes
    -----
    This function is used to refine the ODF maxima directions beyond the
    precision of the directions samples in a given sphere object
    
    See also
    --------
    dipy.reconst.dki.dki_directions
    """
    n = np.array(sphere2cart(1, ang[0], ang[1]))
    return - _dki_odf_core(n, kt, U, alpha=alpha)
