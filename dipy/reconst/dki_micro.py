#!/usr/bin/python
""" Classes and functions for fitting the diffusion kurtosis model """
from __future__ import division, print_function, absolute_import

import numpy as np
from dipy.reconst.dti import (TensorFit, apparent_diffusion_coef,
                              lower_triangular, eig_from_lo_tri,
                              MIN_POSITIVE_SIGNAL)

from dipy.reconst.dki import (_positive_evals, split_dki_param,
                              apparent_kurtosis_coef, kurtosis_maximum,
                              common_fit_methods)
from dipy.reconst.utils import dki_design_matrix as design_matrix
from dipy.reconst.base import ReconstModel
from dipy.core.ndindex import ndindex
from dipy.reconst.vec_val_sum import vec_val_vect


def axonal_water_fraction(dki_params, sphere, mask=None, gtol=1e-5):
    """ Computes the DKI based axonal water fraction [1]_.

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
        The sphere providing sample directions for the initial search of the
        maximal value of kurtosis.
    gtol : float, optional
        This input is to refine kurtosis maxima under the precision of the
        directions sampled on the sphere class instance. The gradient of the
        convergence procedure must be less than gtol before successful
        termination. If gtol is None, fiber direction is directly taken from
        the initial sampled directions of the given sphere object
    mask : ndarray
        A boolean array used to mark the coordinates in the data that should be
        analyzed that has the shape dki_params.shape[:-1]

    Returns
    --------
    AWF : ndarray (x, y, z) or (n)
        Axonal Water Fraction
    Da : ndarray (x, y, z) or (n)
        Axonal Diffusivity
    ADe : ndarray (x, y, z) or (n)
        Axial Diffusivity of extra-cellular compartment
    RDe : ndarray (x, y, z) or (n)
        Radial Diffusivity of extra-cellular compartment
    tort : ndarray (x, y, z) or (n)
        Tortuosity

    References
    ----------
    .. [1] Fieremans E, Jensen JH, Helpern JA, 2011. White matter
           characterization with diffusional kurtosis imaging.
           Neuroimage 58(1):177-88. doi: 10.1016/j.neuroimage.2011.06.006
    """
    shape = dki_params.shape[:-1]

    # select voxels where to find fiber directions
    if mask is None:
        mask = np.ones(shape, dtype='bool')
    else:
        if mask.shape != shape:
            raise ValueError("Mask is not the same shape as dki_params.")

    evals, evecs, kt = split_dki_param(dki_params)

    # select non-zero voxels
    pos_evals = _positive_evals(evals[..., 0], evals[..., 1], evals[..., 2])
    mask = np.logical_and(mask, pos_evals)

    kt_max = np.zeros(mask.shape)

    for idx in ndindex(shape):
        if not mask[idx]:
            continue
        DT = np.dot(np.dot(evecs[idx], np.diag(evals[idx])), evecs[idx].T)
        dt = lower_triangular(DT)
        kt_max[idx], da = kurtosis_maximum(dt, np.mean(evals[idx]), kt[idx],
                                           sphere, gtol=1e-5)

    AWF = kt_max / (kt_max + 3)

    return AWF


def diffusion_components(dki_params, sphere, awf=None, mask=None):
    """ Extracts the intra and extra-cellular diffusion tensors of well aligned
    fibers from diffusion kurtosis imaging parameters [1]_.

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
        The sphere providing sample directions to sample the intra and extra
        cellular diffusion tensors. For more details see Fieremans et al.,
        2011.
    awk : ndarray (optional)
        Array containing values of the axonal water fraction that has the shape
        dki_params.shape[:-1]. If not given this will be automatically computed
        using function axonal_water_fraction with function's default precision.
    mask : ndarray (optional)
        A boolean array used to mark the coordinates in the data that should be
        analyzed that has the shape dki_params.shape[:-1]

    Returns
    --------
    EDT : ndarray (x, y, z, 12) or (n, 12)
        Parameters of the extra-cellular diffusion tensor.
    IDT : ndarray (x, y, z, 12) or (n, 12)
        Parameters of the intra-cellular diffusion tensor.

    Notes
    -----
    The parameters of both extra-cellular and intra-cellular diffusion tensors
    are order as follow:
        1) Three diffusion tensor's eingenvalues
        2) Three lines of the eigenvector matrix each containing the first,
        second and third coordinates of the eigenvector

    References
    ----------
    .. [1] Fieremans E, Jensen JH, Helpern JA, 2011. White matter
           characterization with diffusional kurtosis imaging.
           Neuroimage 58(1):177-88. doi: 10.1016/j.neuroimage.2011.06.006
    """
    shape = dki_params.shape[:-1]

    # select voxels where to applied the single fiber model
    if mask is None:
        mask = np.ones(shape, dtype='bool')
    else:
        if mask.shape != shape:
            raise ValueError("Mask is not the same shape as dki_params.")
        else:
            mask = np.array(mask, dtype=bool, copy=False)

    # check or compute awf values
    if awf is None:
        awf = axonal_water_fraction(dki_params, sphere, mask=mask)
    else:
        if awf.shape != shape:
            raise ValueError("awf array is not the same shape as dki_params.")

    # Initialize hindered and restricted diffusion tensors
    EDT = np.zeros(shape + (12,))
    IDT = np.zeros(shape + (12,))

    # Generate matrix that converts apparant diffusion coefficients to tensors
    B = np.zeros((sphere.x.size, 6))
    B[:, 0] = sphere.x * sphere.x  # Bxx
    B[:, 1] = sphere.x * sphere.y * 2.  # Bxy
    B[:, 2] = sphere.y * sphere.y   # Byy
    B[:, 3] = sphere.x * sphere.z * 2.  # Bxz
    B[:, 4] = sphere.y * sphere.z * 2.  # Byz
    B[:, 5] = sphere.z * sphere.z  # Bzz
    pinvB = np.linalg.pinv(B)

    # Compute hindered and restricted diffusion tensors for all voxels
    evals, evecs, kt = split_dki_param(dki_params)
    for idx in ndindex(shape):
        if not mask[idx]:
            continue
        # sample apparent diffusion and kurtosis values
        di = apparent_diffusion_coef(vec_val_vect(evecs[idx], evals[idx]),
                                     sphere)
        ki = apparent_kurtosis_coef(dki_params[idx], sphere)

        # Convert apparent diffusion and kurtosis values to apparent diffusion
        # values of the hindered and restricted diffusion
        edi = di * (1 + np.sqrt(ki*awf[idx] / (3.0-3.0*awf[idx])))
        idi = di * (1 - np.sqrt(ki * (1.0-awf[idx]) / (3.0*awf[idx])))

        # generate hindered and restricted diffusion tensors
        edt = eig_from_lo_tri(np.dot(pinvB, edi))
        idt = eig_from_lo_tri(np.dot(pinvB, idi))
        EDT[idx] = edt
        IDT[idx] = idt

    return EDT, IDT


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
        .. [1] Tabesh, A., Jensen, J.H., Ardekani, B.A., Helpern, J.A., 2011.
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
            min_signal = MIN_POSITIVE_SIGNAL
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

    def predict(self, dki_params, S0=1.):
        """ Predict a signal for this DKI model class instance given
        parameters.

        Parameters
        ----------
        dki_params : ndarray (x, y, z, 27) or (n, 27)
            All parameters estimated from the diffusion kurtosis model.
            Parameters are ordered as follows:
                1) Three diffusion tensor's eigenvalues
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
            Parameters are ordered as follows:
                1) Three diffusion tensor's eigenvalues
                2) Three lines of the eigenvector matrix each containing the
                   first, second and third coordinates of the eigenvector
                3) Fifteen elements of the kurtosis tensor
        """
        TensorFit.__init__(self, model, model_params)

    def predict(self, gtab, S0=1.):
        r""" Given a DKI model fit, predict the signal on the vertices of a
        gradient table

        Parameters
        ----------
        dki_params : ndarray (x, y, z, 27) or (n, 27)
            All parameters estimated from the diffusion kurtosis model.
            Parameters are ordered as follows:
                1) Three diffusion tensor's eigenvalues
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
        return dki_prediction(self.model_params, gtab, S0)
