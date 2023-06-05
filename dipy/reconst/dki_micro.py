#!/usr/bin/python
""" Classes and functions for fitting the DKI-based microstructural model """

import numpy as np
from dipy.reconst.dti import (lower_triangular, from_lower_triangular,
                              decompose_tensor, trace, mean_diffusivity,
                              radial_diffusivity, axial_diffusivity,
                              MIN_POSITIVE_SIGNAL)

from dipy.reconst.dki import (split_dki_param, _positive_evals,
                              directional_kurtosis,
                              directional_diffusion, kurtosis_maximum,
                              DiffusionKurtosisModel, DiffusionKurtosisFit)
from dipy.reconst.dti import design_matrix as dti_design_matrix
from dipy.core.ndindex import ndindex
from dipy.reconst.vec_val_sum import vec_val_vect
from dipy.data import get_sphere
import dipy.core.sphere as dps


def axonal_water_fraction(dki_params, sphere='repulsion100', gtol=1e-2,
                          mask=None):
    """ Computes the axonal water fraction from DKI [1]_.

    Parameters
    ----------
    dki_params : ndarray (x, y, z, 27) or (n, 27)
        All parameters estimated from the diffusion kurtosis model.
        Parameters are ordered as follows:
            1) Three diffusion tensor's eigenvalues
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
    -------
    awf : ndarray (x, y, z) or (n)
        Axonal Water Fraction

    References
    ----------
    .. [1] Fieremans E, Jensen JH, Helpern JA, 2011. White matter
           characterization with diffusional kurtosis imaging.
           Neuroimage 58(1):177-88. doi: 10.1016/j.neuroimage.2011.06.006
    """
    kt_max = kurtosis_maximum(dki_params, sphere=sphere, gtol=gtol, mask=mask)

    awf = kt_max / (kt_max + 3)

    return awf


def diffusion_components(dki_params, sphere='repulsion100', awf=None,
                         mask=None):
    """ Extracts the restricted and hindered diffusion tensors of well aligned
    fibers from diffusion kurtosis imaging parameters [1]_.

    Parameters
    ----------
    dki_params : ndarray (x, y, z, 27) or (n, 27)
        All parameters estimated from the diffusion kurtosis model.
        Parameters are ordered as follows:
            1) Three diffusion tensor's eigenvalues
            2) Three lines of the eigenvector matrix each containing the first,
               second and third coordinates of the eigenvector
            3) Fifteen elements of the kurtosis tensor
    sphere : Sphere class instance, optional
        The sphere providing sample directions to sample the restricted and
        hindered cellular diffusion tensors. For more details see Fieremans
        et al., 2011.
    awf : ndarray (optional)
        Array containing values of the axonal water fraction that has the shape
        dki_params.shape[:-1]. If not given this will be automatically computed
        using :func:`axonal_water_fraction`" with function's default precision.
    mask : ndarray (optional)
        A boolean array used to mark the coordinates in the data that should be
        analyzed that has the shape dki_params.shape[:-1]

    Returns
    -------
    edt : ndarray (x, y, z, 6) or (n, 6)
        Parameters of the hindered diffusion tensor.
    idt : ndarray (x, y, z, 6) or (n, 6)
        Parameters of the restricted diffusion tensor.

    Notes
    -----
    In the original article of DKI microstructural model [1]_, the hindered and
    restricted tensors were defined as the intra-cellular and extra-cellular
    diffusion compartments respectively.

    References
    ----------
    .. [1] Fieremans E, Jensen JH, Helpern JA, 2011. White matter
           characterization with diffusional kurtosis imaging.
           Neuroimage 58(1):177-88. doi: 10.1016/j.neuroimage.2011.06.006
    """
    shape = dki_params.shape[:-1]

    # load gradient directions
    if not isinstance(sphere, dps.Sphere):
        sphere = get_sphere(sphere)

    # select voxels where to apply the single fiber model
    if mask is None:
        mask = np.ones(shape, dtype='bool')
    else:
        if mask.shape != shape:
            raise ValueError("Mask is not the same shape as dki_params.")
        else:
            mask = np.array(mask, dtype=bool, copy=False)

    # check or compute awf values
    if awf is None:
        awf = axonal_water_fraction(dki_params, sphere=sphere, mask=mask)
    else:
        if awf.shape != shape:
            raise ValueError("awf array is not the same shape as dki_params.")

    # Initialize hindered and restricted diffusion tensors
    edt_all = np.zeros(shape + (6,))
    idt_all = np.zeros(shape + (6,))

    # Generate matrix that converts apparent diffusion coefficients to tensors
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
    dt = lower_triangular(vec_val_vect(evecs, evals))
    md = mean_diffusivity(evals)

    index = ndindex(mask.shape)
    for idx in index:
        if not mask[idx]:
            continue
        # sample apparent diffusion and kurtosis values
        di = directional_diffusion(dt[idx], sphere.vertices)
        ki = directional_kurtosis(dt[idx], md[idx], kt[idx], sphere.vertices,
                                  adc=di, min_kurtosis=0)
        edi = di * (1 + np.sqrt(ki * awf[idx] / (3.0 - 3.0 * awf[idx])))
        edt = np.dot(pinvB, edi)
        edt_all[idx] = edt

        # We only move on if there is an axonal water fraction.
        # Otherwise, remaining params are already zero, so move on
        if awf[idx] == 0:
            continue
        # Convert apparent diffusion and kurtosis values to apparent diffusion
        # values of the hindered and restricted diffusion
        idi = di * (1 - np.sqrt(ki * (1.0 - awf[idx]) / (3.0 * awf[idx])))
        # generate hindered and restricted diffusion tensors
        idt = np.dot(pinvB, idi)
        idt_all[idx] = idt

    return edt_all, idt_all


def dkimicro_prediction(params, gtab, S0=1):
    r""" Signal prediction given the DKI microstructure model parameters.

    Parameters
    ----------
    params : ndarray (x, y, z, 40) or (n, 40)
    All parameters estimated from the diffusion kurtosis microstructure model.
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
    -------
    S : (..., N) ndarray
        Simulated signal based on the DKI microstructure model

    Notes
    -----
    1) The predicted signal is given by:
    $S(\theta, b) = S_0 * [f * e^{-b ADC_{r}} + (1-f) * e^{-b ADC_{h}]$, where
    $ ADC_{r} and ADC_{h} are the apparent diffusion coefficients of the
    diffusion hindered and restricted compartment for a given direction
    $\theta$, $b$ is the b value provided in the GradientTable input for that
    direction, $f$ is the volume fraction of the restricted diffusion
    compartment (also known as the axonal water fraction).

    2) In the original article of DKI microstructural model [1]_, the hindered
    and restricted tensors were defined as the intra-cellular and
    extra-cellular diffusion compartments respectively.
    """

    # Initialize pred_sig
    pred_sig = np.zeros(params.shape[:-1] + (gtab.bvals.shape[0],))

    # Define dti design matrix and region to process
    D = dti_design_matrix(gtab)
    evals = params[..., :3]
    mask = _positive_evals(evals[..., 0], evals[..., 1], evals[..., 2])

    # Prepare parameters
    f = params[..., 27]
    adce = params[..., 28:34]
    adci = params[..., 34:40]

    if isinstance(S0, np.ndarray):
        S0_vol = S0 * np.ones(params.shape[:-1])
    else:
        S0_vol = S0

    # Process pred_sig for all data voxels
    index = ndindex(evals.shape[:-1])
    for v in index:
        if mask[v]:
            pred_sig[v] = (1. - f[v]) * np.exp(np.dot(D[:, :6], adce[v])) + \
                f[v] * np.exp(np.dot(D[:, :6], adci[v]))

    return pred_sig * S0_vol


def tortuosity(hindered_ad, hindered_rd):
    """ Computes the tortuosity of the hindered diffusion compartment given
    its axial and radial diffusivities

    Parameters
    ----------
    hindered_ad: ndarray
        Array containing the values of the hindered axial diffusivity.
    hindered_rd: ndarray
        Array containing the values of the hindered radial diffusivity.

    Returns
    -------
    Tortuosity of the hindered diffusion compartment
    """
    if not isinstance(hindered_rd, np.ndarray):
        hindered_rd = np.array(hindered_rd)
    if not isinstance(hindered_ad, np.ndarray):
        hindered_ad = np.array(hindered_ad)

    tortuosity = np.zeros(hindered_rd.shape)

    # mask to avoid divisions by zero
    mask = hindered_rd > 0

    # Check single voxel cases. For numpy versions more recent than 1.7,
    # this if else condition is not required since single voxel can be
    # processed using the same line of code of multi-voxel
    if hindered_rd.size == 1:
        if mask:
                tortuosity = hindered_ad / hindered_rd
    else:
        tortuosity[mask] = hindered_ad[mask] / hindered_rd[mask]

    return tortuosity


def _compartments_eigenvalues(cdt):
    """ Helper function that computes the eigenvalues of a tissue sub
    compartment given its individual diffusion tensor

    Parameters
    ----------
    cdt : ndarray (..., 6)
        Diffusion tensors elements of the tissue compartment stored in lower
        triangular order.

    Returns
    -------
    eval : ndarry (..., 3)
        Eigenvalues of the tissue compartment
    """
    evals, evecs = decompose_tensor(from_lower_triangular(cdt))
    return evals


class KurtosisMicrostructureModel(DiffusionKurtosisModel):
    """ Class for the Diffusion Kurtosis Microstructural Model
    """

    def __init__(self,  gtab, fit_method="WLS", *args, **kwargs):
        """ Initialize a KurtosisMicrostrutureModel class instance [1]_.

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

        References
        ----------
        .. [1] Fieremans, E., Jensen, J.H., Helpern, J.A., 2011. White Matter
               Characterization with Diffusion Kurtosis Imaging. Neuroimage
               58(1): 177-188. doi:10.1016/j.neuroimage.2011.06.006
        """
        DiffusionKurtosisModel.__init__(self, gtab, fit_method="WLS", *args,
                                        **kwargs)

    def fit(self, data, mask=None, sphere='repulsion100', gtol=1e-2,
            awf_only=False):
        """ Fit method of the Diffusion Kurtosis Microstructural Model

        Parameters
        ----------
        data : array
            An 4D matrix containing the diffusion-weighted data.

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
        if mask is not None:
            # Check for valid shape of the mask
            if mask.shape != data.shape[:-1]:
                raise ValueError("Mask is not the same shape as data.")
            mask = np.array(mask, dtype=bool, copy=False)
        data_in_mask = np.reshape(data[mask], (-1, data.shape[-1]))

        if self.min_signal is None:
            self.min_signal = MIN_POSITIVE_SIGNAL

        data_in_mask = np.maximum(data_in_mask, self.min_signal)

        # DKI fit
        dki_params = super().fit(data_in_mask).model_params

        # Computing awf
        awf = axonal_water_fraction(dki_params, sphere=sphere, gtol=gtol)

        if awf_only:
            params_all_mask = np.concatenate((dki_params, np.array([awf]).T),
                                             axis=-1)
        else:
            # Computing the hindered and restricted diffusion tensors
            hdt, rdt = diffusion_components(dki_params, sphere=sphere,
                                            awf=awf)
            params_all_mask = np.concatenate((dki_params, np.array([awf]).T,
                                              hdt, rdt), axis=-1)

        if mask is None:
            out_shape = data.shape[:-1] + (-1,)
            params = params_all_mask.reshape(out_shape)
            #if extra is not None:
            #    self.extra = extra.reshape(data.shape)
        else:
            params = np.zeros(data.shape[:-1] + (params_all_mask.shape[-1],))
            params[mask, :] = params_all_mask
            #if extra is not None:
            #    self.extra = np.zeros(data.shape)
            #    self.extra[mask, :] = extra

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

        Notes
        -----
        In the original article of DKI microstructural model [1]_, the hindered
        and restricted tensors were defined as the intra-cellular and
        extra-cellular diffusion compartments respectively.

        References
        ----------
        .. [1] Fieremans, E., Jensen, J.H., Helpern, J.A., 2011. White Matter
               Characterization with Diffusion Kurtosis Imaging. Neuroimage
               58(1): 177-188. doi:10.1016/j.neuroimage.2011.06.006
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
        In the original article of DKI microstructural model [1]_, the hindered
        and restricted tensors were defined as the intra-cellular and
        extra-cellular diffusion compartments respectively.

        References
        ----------
        .. [1] Fieremans, E., Jensen, J.H., Helpern, J.A., 2011. White Matter
               Characterization with Diffusion Kurtosis Imaging. Neuroimage
               58(1): 177-188. doi:10.1016/j.neuroimage.2011.06.006
        """
        DiffusionKurtosisFit.__init__(self, model, model_params)

    @property
    def awf(self):
        """ Returns the volume fraction of the restricted diffusion compartment
        also known as axonal water fraction.

        Notes
        -----
        The volume fraction of the restricted diffusion compartment can be seem
        as the volume fraction of the intra-cellular compartment [1]_.

        References
        ----------
        .. [1] Fieremans, E., Jensen, J.H., Helpern, J.A., 2011. White Matter
               Characterization with Diffusion Kurtosis Imaging. Neuroimage
               58(1): 177-188. doi:10.1016/j.neuroimage.2011.06.006
        """
        return self.model_params[..., 27]

    @property
    def restricted_evals(self):
        """ Returns the eigenvalues of the restricted diffusion compartment.

        Notes
        -----
        The restricted diffusion tensor can be seem as the tissue's
        intra-cellular diffusion compartment [1]_.

        References
        ----------
        .. [1] Fieremans, E., Jensen, J.H., Helpern, J.A., 2011. White Matter
               Characterization with Diffusion Kurtosis Imaging. Neuroimage
               58(1): 177-188. doi:10.1016/j.neuroimage.2011.06.006
        """
        self._is_awfonly()
        return _compartments_eigenvalues(self.model_params[..., 34:40])

    @property
    def hindered_evals(self):
        """ Returns the eigenvalues of the hindered diffusion compartment.

        Notes
        -----
        The hindered diffusion tensor can be seem as the tissue's
        extra-cellular diffusion compartment [1]_.

        References
        ----------
        .. [1] Fieremans, E., Jensen, J.H., Helpern, J.A., 2011. White Matter
               Characterization with Diffusion Kurtosis Imaging. Neuroimage
               58(1): 177-188. doi:10.1016/j.neuroimage.2011.06.006
        """
        self._is_awfonly()
        return _compartments_eigenvalues(self.model_params[..., 28:34])

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

        Notes
        -----
        The hindered diffusion tensor can be seem as the tissue's
        extra-cellular diffusion compartment [1]_.

        References
        ----------
        .. [1] Fieremans, E., Jensen, J.H., Helpern, J.A., 2011. White Matter
               Characterization with Diffusion Kurtosis Imaging. Neuroimage
               58(1): 177-188. doi:10.1016/j.neuroimage.2011.06.006
        """
        return axial_diffusivity(self.hindered_evals)

    @property
    def hindered_rd(self):
        """ Returns the radial diffusivity of the hindered compartment.

        Notes
        -----
        The hindered diffusion tensor can be seem as the tissue's
        extra-cellular diffusion compartment [1]_.

        References
        ----------
        .. [1] Fieremans, E., Jensen, J.H., Helpern, J.A., 2011. White Matter
               Characterization with Diffusion Kurtosis Imaging. Neuroimage
               58(1): 177-188. doi:10.1016/j.neuroimage.2011.06.006
        """
        return radial_diffusivity(self.hindered_evals)

    @property
    def tortuosity(self):
        """ Returns the tortuosity of the hindered diffusion which is defined
        by ADe / RDe, where ADe and RDe are the axial and radial diffusivities
        of the hindered compartment [1]_.

        Notes
        -----
        The hindered diffusion tensor can be seem as the tissue's
        extra-cellular diffusion compartment [1]_.

        References
        ----------
        .. [1] Fieremans, E., Jensen, J.H., Helpern, J.A., 2011. White Matter
               Characterization with Diffusion Kurtosis Imaging. Neuroimage
               58(1): 177-188. doi:10.1016/j.neuroimage.2011.06.006
        """
        return tortuosity(self.hindered_ad, self.hindered_rd)

    def _is_awfonly(self):
        """ To raise error if only the axonal water fraction was computed """
        if self.model_params.shape[-1] < 39:
            raise ValueError('Only the awf was processed! Rerun model fit '
                             'with input parameter awf_only set to False')

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
        where $ADC_{r}$ and $ADC_{h}$ are the apparent diffusion coefficients
        of the diffusion hindered and restricted compartment for a given
        direction $\theta$, $b$ is the b value provided in the GradientTable
        input for that direction, $f$ is the volume fraction of the restricted
        diffusion compartment (also known as the axonal water fraction).
        """
        self._is_awfonly()
        return dkimicro_prediction(self.model_params, gtab, S0)
