#!/usr/bin/python
""" Classes and functions for fitting the diffusion kurtosis model """
from __future__ import division, print_function, absolute_import

import numpy as np
import dipy.reconst.dki as dki
from dipy.reconst.dti import (TensorFit, apparent_diffusion_coef,
                              _positive_evals, eig_from_lo_tri,
                              _decompose_tensor_nan, trace,
                              radial_diffusivity, axial_diffusivity,
                              MIN_POSITIVE_SIGNAL)

from dipy.reconst.dki import (split_dki_param, apparent_kurtosis_coef,
                              kurtosis_maximum, common_fit_methods,
                              DiffusionKurtosisModel, DiffusionKurtosisFit)
from dipy.reconst.utils import dki_design_matrix as design_matrix
from dipy.reconst.dti import design_matrix as dti_design_matrix
from dipy.reconst.base import ReconstModel
from dipy.core.ndindex import ndindex
from dipy.reconst.vec_val_sum import vec_val_vect


def axonal_water_fraction(dki_params, sphere='symmetric362', gtol=1e-5,
                          mask='None'):
    """ Computes the axonal water fraction from DKI [1]_.

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

    References
    ----------
    .. [1] Fieremans E, Jensen JH, Helpern JA, 2011. White matter
           characterization with diffusional kurtosis imaging.
           Neuroimage 58(1):177-88. doi: 10.1016/j.neuroimage.2011.06.006
    """
    kt_max = kurtosis_maximum(dki_params, sphere=sphere, gtol=gtol, mask=mask)

    AWF = kt_max / (kt_max + 3)

    return AWF


def diffusion_components(dki_params, sphere='symmetric362', awf=None,
                         mask=None):
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


def dkimicro_prediction(params, gtab, S0=1):
    r""" Signal prediction given the free water DTI model parameters.

    Parameters
    ----------
    params : ndarray (x, y, z, 40) or (n, 40)
    All parameters estimated from the diffusion kurtosis microstructural model.
        Parameters are ordered as follows:
            1) Three diffusion tensor's eigenvalues
            2) Three lines of the eigenvector matrix each containing the
               first, second and third coordinates of the eigenvector
            3) Fifteen elements of the kurtosis tensor
            4) Six elements of the hindered diffusion tensor
            5) Six elements of the restricted diffusion tensor
            6) Axonal water fraction
    gtab : a GradientTable class instance
        The gradient table for this prediction
    S0 : float or ndarray
        The non diffusion-weighted signal in every voxel, or across all
        voxels. Default: 1

    Returns
    --------
    S : (..., N) ndarray
        Simulated signal based on the free water DTI model

    Notes
    -----
    The predicted signal is given by:
    $S(\theta, b) = S_0 * [f * e^{-b ADC_{r}} + (1-f) * e^{-b ADC_{h}]$, where
    $ ADC_{r} and ADC_{h} are the apparent diffusion coefficients of the
    diffusion hindered and restricted compartment for a given direction
    $\theta$, $b$ is the b value provided in the GradientTable input for that
    direction, $f$ is the volume fraction of the restricted diffusion
    compartment (also known as the axonal water fraction).
    """

    # Initialize pred_sig
    pred_sig = np.zeros(f.shape + (gtab.bvals.shape[0],))

    # Define dti design matrix and region to process
    D = dti_design_matrix(gtab)
    evals = params[..., :3]
    mask = _positive_evals(evals[..., 0], evals[..., 1], evals[..., 2])

    # Prepare parameters
    f = params[..., 39]
    adce = params[..., 27:33]
    adce = params[..., 33:39]

    # Process pred_sig for all data voxels
    index = ndindex(evals.shape[:-1])
    for v in index:
        if mask[v]:
            pred_sig[v] = (1 - f[v]) * np.exp(np.dot(adce[v], D.T)) + \
                          f[v] * np.exp(np.dot(adci[v], D.T))

    return pred_sig


class KurtosisMicrostructuralModel(DiffusionKurtosisModel):
    """ Class for the Diffusion Kurtosis Microstructural Model
    """

    def __init__(self,  gtab, fit_method="WLS", *args, **kwargs):
        """ Initialize a KurtosisMicrostruturalModel class instance [1]_.

        Parameters
        ----------
        gtab : GradientTable class instance

        fit_method : str or callable
            str can be one of the following:
            'OLS' or 'ULLS' to fit the diffusion tensor and kurtosis tensor
            using the ordinary linear least squares solution
                dki.ols_fit_dki
            'WLS' or 'UWLLS' to fit the diffusion tensor and kurtosis tensor
            using the ordinary linear least squares solution
                dki.wls_fit_dki

            callable has to have the signature:
                fit_method(design_matrix, data, *args, **kwargs)

        args, kwargs : arguments and key-word arguments passed to the
           fit_method. See dki.ols_fit_dki, dki.wls_fit_dki for details

        Notes
        -----
        1) Since this model is an extension of DKI, class instance is defined
           as subclass DiffusionKurtosisModel from dki.py
        2) The first step of the DKI based microstructural model requires
           diffusion tensor and kurtosis tensor fit. This fit is performed
           using the DKI model solution specified by users by input parameter
           fit_method.

        References
        ----------
        .. [1] Fieremans, E., Jensen, J.H., Helpern, J.A., 2011. White Matter
               Characterization with Diffusion Kurtosis Imaging. Neuroimage
               58(1): 177-188. doi:10.1016/j.neuroimage.2011.06.006
        """
        DiffusionKurtosisModel.__init__(self, gtab, fit_method="WLS", *args,
                                        **kwargs)

        ReconstModel.__init__(self, gtab)

    def fit(self, data, mask=None, sphere, gtol=1e-5, awf_only=False):
        """ Fit method of the Diffusion Kurtosis Microstructural Model

        Parameters
        ----------
        data : array
            The measured signal from one voxel.

        mask : array
            A boolean array used to mark the coordinates in the data that
            should be analyzed that has the shape data.shape[-1]

        sphere : Sphere class instance, optional
            The sphere providing sample directions for the initial search of
            the maximal value of kurtosis.

        gtol : float, optional
            This input is to refine kurtosis maxima under the precision of the
            directions sampled on the sphere class instance. The gradient of
            the convergence procedure must be less than gtol before successful
            termination. If gtol is None, fiber direction is directly taken
            from the initial sampled directions of the given sphere object

        awf_only : bool, optiomal
            If set to true only the axonal volume fraction is computed from
            the kurtosis tensor. Default = False
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

        # DKI fit
        params_in_mask = self.fit_method(self.design_matrix, data_in_mask,
                                         *self.args, **self.kwargs)

        # Computing AWF
        awf = axonal_water_fraction(params_in_mask, sphere=sphere, gtol=1e-5)

        if awf_only:
            params = np.concatenate((dki_params, np.array([f])), axis=0)
        else:

            # Computing the hindered and restricted diffusion tensors
            edt, idt = diffusion_components(dki_params, sphere=sphere, awf=awf)

            params = np.concatenate((dki_params,  edt, idt, np.array([f])),
                                    axis=0)

        if mask is None:
            out_shape = data.shape[:-1] + (-1, )
            params = params_in_mask.reshape(out_shape)
        else:
            params = np.zeros(data.shape[:-1] + params.shape[-1])
            params[mask, :] = params_in_mask

        return KurtosisMicrostructuralFit(self, params)

    def predict(self, params, S0=1.):
        """ Predict a signal for the DKI microstructural model class instance
        given parameters.

        Parameters
        ----------
        params : ndarray (x, y, z, 40) or (n, 40)
            All parameters estimated from the diffusion kurtosis
            microstructural model.
            Parameters are ordered as follows:
                1) Three diffusion tensor's eigenvalues
                2) Three lines of the eigenvector matrix each containing the
                   first, second and third coordinates of the eigenvector
                3) Fifteen elements of the kurtosis tensor
                4) Six elements of the hindered diffusion tensor
                5) Six elements of the restricted diffusion tensor
                6) Axonal water fraction
        S0 : float or ndarray (optional)
            The non diffusion-weighted signal in every voxel, or across all
            voxels. Default: 1
        """
        return dkimicro_prediction(params, self.gtab, S0)


class KurtosisMicrostructuralFit(DiffusionKurtosisFit):
    """ Class for fitting the Diffusion Kurtosis Microstructural Model """

    def __init__(self, model, model_params):
        """ Initialize a KurtosisMicrostructural Fit class instance.

        Parameters
        ----------
        model : DiffusionKurtosisModel Class instance
            Class instance containing the Diffusion Kurtosis Model for the fit
        model_params : ndarray (x, y, z, 40) or (n, 40)
            All parameters estimated from the diffusion kurtosis
            microstructural model.
            Parameters are ordered as follows:
                1) Three diffusion tensor's eigenvalues
                2) Three lines of the eigenvector matrix each containing the
                   first, second and third coordinates of the eigenvector
                3) Fifteen elements of the kurtosis tensor
                4) Six elements of the hindered diffusion tensor
                5) Six elements of the restricted diffusion tensor
                6) Axonal water fraction

        Notes
        -----
        Since this model is an extension of DKI, class instance is defined
        as subclass of the DiffusionKurtosisFit from dki.py
        """
        DiffusionKurtosisFit.__init__(self, model, model_params)

    @property
    def awf(self):
        """ Returns the volume fraction of the restricted diffusion compartment
        also known as axonal water fraction.
        """
        return self.model_params[..., 39]

    @property
    def restricted_evals(self):
        """ Returns the eigenvalues of the restricted diffusion compartment.
        """
        evals, evecs = _decompose_tensor_nan(self.model_params[..., 33:39])
        return evals

    @property
    def hindered_evals(self):
        """ Returns the eigenvalues of the restricted diffusion compartment.
        """
        evals, evecs = _decompose_tensor_nan(self.model_params[..., 27:33])
        return evals

    @property
    def axonal_diffusivity(self):
        """ Returns the axonal diffusivity defined as the restricted diffusion
        tensor trace [1]_.

        References
        ----------
        .. [1] Fieremans, E., Jensen, J.H., Helpern, J.A., 2011. White Matter
               Characterization with Diffusion Kurtosis Imaging. Neuroimage
               58(1): 177-188. doi:10.1016/j.neuroimage.2011.06.006
        """
        return trace(self.restricted_evals)

    @property
    def hindered_ad(self):
        """ Returns the axial diffusivity of the hindered compartment.
        """
        return axial_diffusivity(self.hindered_evals)

    @property
    def hindered_rd(self):
        """ Returns the radial diffusivity of the hindered compartment.
        """
        return radial_diffusivity(self.hindered_evals)

    @property
    def tortuosity(self):
        """ Returns the tortuosity of the hindered diffusion which is defined
        by ADe / RDe, where ADe and RDe are the axial and radial diffusivities
        of the hindered compartment [1]_.

        References
        ----------
        .. [1] Fieremans, E., Jensen, J.H., Helpern, J.A., 2011. White Matter
               Characterization with Diffusion Kurtosis Imaging. Neuroimage
               58(1): 177-188. doi:10.1016/j.neuroimage.2011.06.006
        """
        rd = self.hindered_rd
        ad = self.hindered_ad
        tortuosity = np.zeros(rd.shape)

        # mask to avoid divisions by zero
        mask = self.hindered_rd > 0

        tortuosity[mask] = ad[mask] / rd[mask]
        return tortuosity

    def predict(self, gtab, S0=1.):
        r""" Given a DKI microstructural model fit, predict the signal on the
        vertices of a gradient table

        gtab : a GradientTable class instance
            The gradient table for this prediction

        S0 : float or ndarray (optional)
            The non diffusion-weighted signal in every voxel, or across all
            voxels. Default: 1

        Notes
        -----
        The predicted signal is given by:

        $S(\theta, b) = S_0 * [f * e^{-b ADC_{r}} + (1-f) * e^{-b ADC_{h}]$,
        where $ ADC_{r} and ADC_{h} are the apparent diffusion coefficients of
        the diffusion hindered and restricted compartment for a given direction
        $\theta$, $b$ is the b value provided in the GradientTable input for
        that direction, $f$ is the volume fraction of the restricted diffusion
        compartment (also known as the axonal water fraction).
        """
        return dki_prediction(self.model_params, gtab, S0)
