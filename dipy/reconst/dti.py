#!/usr/bin/python
""" Classes and functions for fitting tensors """
from __future__ import division, print_function, absolute_import

import warnings

import functools

import numpy as np

import scipy.optimize as opt

from dipy.utils.six.moves import range
from dipy.utils.arrfuncs import pinv, eigh
from dipy.data import get_sphere
from dipy.core.gradients import gradient_table
from dipy.core.geometry import vector_norm
from dipy.reconst.vec_val_sum import vec_val_vect
from dipy.core.onetime import auto_attr
from dipy.reconst.base import ReconstModel


MIN_POSITIVE_SIGNAL = 0.0001


def _roll_evals(evals, axis=-1):
    """
    Helper function to check that the evals provided to functions calculating
    tensor statistics have the right shape

    Parameters
    ----------
    evals : array-like
        Eigenvalues of a diffusion tensor. shape should be (...,3).

    axis : int
        The axis of the array which contains the 3 eigenvals. Default: -1

    Returns
    -------
    evals : array-like
        Eigenvalues of a diffusion tensor, rolled so that the 3 eigenvals are
        the last axis.
    """
    if evals.shape[-1] != 3:
        msg = "Expecting 3 eigenvalues, got {}".format(evals.shape[-1])
        raise ValueError(msg)

    evals = np.rollaxis(evals, axis)

    return evals


def fractional_anisotropy(evals, axis=-1):
    """
    Fractional anisotropy (FA) of a diffusion tensor.

    Parameters
    ----------
    evals : array-like
        Eigenvalues of a diffusion tensor.
    axis : int
        Axis of `evals` which contains 3 eigenvalues.

    Returns
    -------
    fa : array
        Calculated FA. Range is 0 <= FA <= 1.

    Notes
    -----
    FA is calculated using the following equation:

    .. math::

        FA = \sqrt{\frac{1}{2}\frac{(\lambda_1-\lambda_2)^2+(\lambda_1-
                    \lambda_3)^2+(\lambda_2-\lambda_3)^2}{\lambda_1^2+
                    \lambda_2^2+\lambda_3^2}}

    """
    evals = _roll_evals(evals, axis)
    # Make sure not to get nans
    all_zero = (evals == 0).all(axis=0)
    ev1, ev2, ev3 = evals
    fa = np.sqrt(0.5 * ((ev1 - ev2) ** 2 +
                        (ev2 - ev3) ** 2 +
                        (ev3 - ev1) ** 2) /
                 ((evals * evals).sum(0) + all_zero))

    return fa


def geodesic_anisotropy(evals, axis=-1):
    """
    Geodesic anisotropy (GA) of a diffusion tensor.

    Parameters
    ----------
    evals : array-like
        Eigenvalues of a diffusion tensor.
    axis : int
        Axis of `evals` which contains 3 eigenvalues.

    Returns
    -------
    ga : array
        Calculated GA. In the range 0 to +infinity

    Notes
    -----
    GA is calculated using the following equation given in [1]_:

    .. math::

        GA = \sqrt{\sum_{i=1}^3
        \log^2{\left ( \lambda_i/<\mathbf{D}> \right )}},
        \quad \textrm{where} \quad <\mathbf{D}> =
        (\lambda_1\lambda_2\lambda_3)^{1/3}

    Note that the notation, $<D>$, is often used as the mean diffusivity (MD)
    of the diffusion tensor and can lead to confusions in the literature
    (see [1]_ versus [2]_ versus [3]_ for example). Reference [2]_ defines
    geodesic anisotropy (GA) with $<D>$ as the MD in the denominator of the
    sum. This is wrong. The original paper [1]_ defines GA with
    $<D> = det(D)^{1/3}$, as the isotropic part of the distance. This might be
    an explanation for the confusion. The isotropic part of the diffusion
    tensor in Euclidean space is the MD whereas the isotropic part of the
    tensor in log-Euclidean space is $det(D)^{1/3}$. The Appendix of [1]_ and
    log-Euclidean derivations from [3]_ are clear on this. Hence, all that to
    say that $<D> = det(D)^{1/3}$ here for the GA definition and not MD.

    References
    ----------

    .. [1] P. G. Batchelor, M. Moakher, D. Atkinson, F. Calamante,
        A. Connelly, "A rigorous framework for diffusion tensor calculus",
        Magnetic Resonance in Medicine, vol. 53, pp. 221-225, 2005.

    .. [2] M. M. Correia, V. F. Newcombe, G.B. Williams.
        "Contrast-to-noise ratios for indices of anisotropy obtained from
        diffusion MRI: a study with standard clinical b-values at 3T".
        NeuroImage, vol. 57, pp. 1103-1115, 2011.

    .. [3] A. D. Lee, etal, P. M. Thompson.
        "Comparison of fractional and geodesic anisotropy in diffusion tensor
        images of 90 monozygotic and dizygotic twins". 5th IEEE International
        Symposium on Biomedical Imaging (ISBI), pp. 943-946, May 2008.

    .. [4] V. Arsigny, P. Fillard, X. Pennec, N. Ayache.
        "Log-Euclidean metrics for fast and simple calculus on diffusion
        tensors." Magnetic Resonance in Medecine, vol 56, pp. 411-421, 2006.

    """

    evals = _roll_evals(evals, axis)
    ev1, ev2, ev3 = evals

    log1 = np.zeros(ev1.shape)
    log2 = np.zeros(ev1.shape)
    log3 = np.zeros(ev1.shape)
    idx = np.nonzero(ev1)

    # this is the definition in [1]_
    detD = np.power(ev1 * ev2 * ev3, 1 / 3.)
    log1[idx] = np.log(ev1[idx] / detD[idx])
    log2[idx] = np.log(ev2[idx] / detD[idx])
    log3[idx] = np.log(ev3[idx] / detD[idx])

    ga = np.sqrt(log1 ** 2 + log2 ** 2 + log3 ** 2)

    return ga


def mean_diffusivity(evals, axis=-1):
    """
    Mean Diffusivity (MD) of a diffusion tensor.

    Parameters
    ----------
    evals : array-like
        Eigenvalues of a diffusion tensor.
    axis : int
        Axis of `evals` which contains 3 eigenvalues.

    Returns
    -------
    md : array
        Calculated MD.

    Notes
    -----
    MD is calculated with the following equation:

    .. math::

        MD = \frac{\lambda_1 + \lambda_2 + \lambda_3}{3}

    """
    evals = _roll_evals(evals, axis)
    return evals.mean(0)


def axial_diffusivity(evals, axis=-1):
    """
    Axial Diffusivity (AD) of a diffusion tensor.
    Also called parallel diffusivity.

    Parameters
    ----------
    evals : array-like
        Eigenvalues of a diffusion tensor, must be sorted in descending order
        along `axis`.
    axis : int
        Axis of `evals` which contains 3 eigenvalues.

    Returns
    -------
    ad : array
        Calculated AD.

    Notes
    -----
    AD is calculated with the following equation:

    .. math::

        AD = \lambda_1

    """
    evals = _roll_evals(evals, axis)
    ev1, ev2, ev3 = evals
    return ev1


def radial_diffusivity(evals, axis=-1):
    """
    Radial Diffusivity (RD) of a diffusion tensor.
    Also called perpendicular diffusivity.

    Parameters
    ----------
    evals : array-like
        Eigenvalues of a diffusion tensor, must be sorted in descending order
        along `axis`.
    axis : int
        Axis of `evals` which contains 3 eigenvalues.

    Returns
    -------
    rd : array
        Calculated RD.

    Notes
    -----
    RD is calculated with the following equation:

    .. math::

        RD = \frac{\lambda_2 + \lambda_3}{2}

    """
    evals = _roll_evals(evals, axis)
    return evals[1:].mean(0)


def trace(evals, axis=-1):
    """
    Trace of a diffusion tensor.

    Parameters
    ----------
    evals : array-like
        Eigenvalues of a diffusion tensor.
    axis : int
        Axis of `evals` which contains 3 eigenvalues.

    Returns
    -------
    trace : array
        Calculated trace of the diffusion tensor.

    Notes
    -----
    Trace is calculated with the following equation:

    .. math::

        Trace = \lambda_1 + \lambda_2 + \lambda_3

    """
    evals = _roll_evals(evals, axis)
    return evals.sum(0)


def color_fa(fa, evecs):
    """ Color fractional anisotropy of diffusion tensor

    Parameters
    ----------
    fa : array-like
        Array of the fractional anisotropy (can be 1D, 2D or 3D)

    evecs : array-like
        eigen vectors from the tensor model

    Returns
    -------
    rgb : Array with 3 channels for each color as the last dimension.
        Colormap of the FA with red for the x value, y for the green
        value and z for the blue value.

    Notes
    -----

    It is computed from the clipped FA between 0 and 1 using the following
    formula

    .. math::

        rgb = abs(max(\vec{e})) \times fa
    """

    if (fa.shape != evecs[..., 0, 0].shape) or ((3, 3) != evecs.shape[-2:]):
        raise ValueError("Wrong number of dimensions for evecs")

    return np.abs(evecs[..., 0]) * np.clip(fa, 0, 1)[..., None]


# The following are used to calculate the tensor mode:
def determinant(q_form):
    """
    The determinant of a tensor, given in quadratic form

    Parameters
    ----------
    q_form : ndarray
        The quadratic form of a tensor, or an array with quadratic forms of
        tensors. Should be of shape (x, y, z, 3, 3) or (n, 3, 3) or (3, 3).

    Returns
    -------
    det : array
        The determinant of the tensor in each spatial coordinate
    """

    # Following the conventions used here:
    # http://en.wikipedia.org/wiki/Determinant
    aei = q_form[..., 0, 0] * q_form[..., 1, 1] * q_form[..., 2, 2]
    bfg = q_form[..., 0, 1] * q_form[..., 1, 2] * q_form[..., 2, 0]
    cdh = q_form[..., 0, 2] * q_form[..., 1, 0] * q_form[..., 2, 1]
    ceg = q_form[..., 0, 2] * q_form[..., 1, 1] * q_form[..., 2, 0]
    bdi = q_form[..., 0, 1] * q_form[..., 1, 0] * q_form[..., 2, 2]
    afh = q_form[..., 0, 0] * q_form[..., 1, 2] * q_form[..., 2, 1]
    return aei + bfg + cdh - ceg - bdi - afh


def isotropic(q_form):
    """
    Calculate the isotropic part of the tensor [1]_.

    Parameters
    ----------
    q_form : ndarray
        The quadratic form of a tensor, or an array with quadratic forms of
        tensors. Should be of shape (x,y,z,3,3) or (n, 3, 3) or (3,3).

    Returns
    -------
    A_hat: ndarray
        The isotropic part of the tensor in each spatial coordinate

    Notes
    -----
    The isotropic part of a tensor is defined as (equations 3-5 of [1]_):

    .. math ::
        \bar{A} = \frac{1}{2} tr(A) I

    References
    ----------
    .. [1] Daniel B. Ennis and G. Kindlmann, "Orthogonal Tensor
        Invariants and the Analysis of Diffusion Tensor Magnetic Resonance
        Images", Magnetic Resonance in Medicine, vol. 55, no. 1, pp. 136-146,
        2006.
    """
    tr_A = q_form[..., 0, 0] + q_form[..., 1, 1] + q_form[..., 2, 2]
    my_I = np.eye(3)
    tr_AI = (tr_A.reshape(tr_A.shape + (1, 1)) * my_I)
    return (1 / 3.0) * tr_AI


def deviatoric(q_form):
    """
    Calculate the deviatoric (anisotropic) part of the tensor [1]_.

    Parameters
    ----------
    q_form : ndarray
        The quadratic form of a tensor, or an array with quadratic forms of
        tensors. Should be of shape (x,y,z,3,3) or (n, 3, 3) or (3,3).

    Returns
    -------
    A_squiggle : ndarray
        The deviatoric part of the tensor in each spatial coordinate.

    Notes
    -----
    The deviatoric part of the tensor is defined as (equations 3-5 in [1]_):

    .. math ::
         \widetilde{A} = A - \bar{A}

    Where $A$ is the tensor quadratic form and $\bar{A}$ is the anisotropic
    part of the tensor.

    References
    ----------
    .. [1] Daniel B. Ennis and G. Kindlmann, "Orthogonal Tensor
        Invariants and the Analysis of Diffusion Tensor Magnetic Resonance
        Images", Magnetic Resonance in Medicine, vol. 55, no. 1, pp. 136-146,
        2006.
    """
    A_squiggle = q_form - isotropic(q_form)
    return A_squiggle


def norm(q_form):
    """
    Calculate the Frobenius norm of a tensor quadratic form

    Parameters
    ----------
    q_form: ndarray
        The quadratic form of a tensor, or an array with quadratic forms of
        tensors. Should be of shape (x,y,z,3,3) or (n, 3, 3) or (3,3).

    Returns
    -------
    norm : ndarray
        The Frobenius norm of the 3,3 tensor q_form in each spatial
        coordinate.

    Notes
    -----
    The Frobenius norm is defined as:

    :math:
        ||A||_F = [\sum_{i,j} abs(a_{i,j})^2]^{1/2}

    See also
    --------
    np.linalg.norm
    """
    return np.sqrt(np.sum(np.sum(np.abs(q_form ** 2), -1), -1))


def mode(q_form):
    """
    Mode (MO) of a diffusion tensor [1]_.

    Parameters
    ----------
    q_form : ndarray
        The quadratic form of a tensor, or an array with quadratic forms of
        tensors. Should be of shape (x, y, z, 3, 3) or (n, 3, 3) or (3, 3).

    Returns
    -------
    mode : array
        Calculated tensor mode in each spatial coordinate.

    Notes
    -----
    Mode ranges between -1 (planar anisotropy) and +1 (linear anisotropy)
    with 0 representing orthotropy. Mode is calculated with the
    following equation (equation 9 in [1]_):

    .. math::

        Mode = 3*\sqrt{6}*det(\widetilde{A}/norm(\widetilde{A}))

    Where $\widetilde{A}$ is the deviatoric part of the tensor quadratic form.

    References
    ----------

    .. [1] Daniel B. Ennis and G. Kindlmann, "Orthogonal Tensor
        Invariants and the Analysis of Diffusion Tensor Magnetic Resonance
        Images", Magnetic Resonance in Medicine, vol. 55, no. 1, pp. 136-146,
        2006.
    """

    A_squiggle = deviatoric(q_form)
    A_s_norm = norm(A_squiggle)
    # Add two dims for the (3,3), so that it can broadcast on A_squiggle:
    A_s_norm = A_s_norm.reshape(A_s_norm.shape + (1, 1))

    return 3 * np.sqrt(6) * determinant((A_squiggle / A_s_norm))


def linearity(evals, axis=-1):
    """
    The linearity of the tensor [1]_

    Parameters
    ----------
    evals : array-like
        Eigenvalues of a diffusion tensor.
    axis : int
        Axis of `evals` which contains 3 eigenvalues.

    Returns
    -------
    linearity : array
        Calculated linearity of the diffusion tensor.

    Notes
    -----
    Linearity is calculated with the following equation:

    .. math::

        Linearity = \frac{\lambda_1-\lambda_2}{\lambda_1+\lambda_2+\lambda_3}

    References
    ----------
    [1] Westin C.-F., Peled S., Gubjartsson H., Kikinis R., Jolesz F.,
        "Geometrical diffusion measures for MRI from tensor basis analysis" in
        Proc. 5th Annual ISMRM, 1997.
    """
    evals = _roll_evals(evals, axis)
    ev1, ev2, ev3 = evals
    return (ev1 - ev2) / evals.sum(0)


def planarity(evals, axis=-1):
    """
    The planarity of the tensor [1]_

    Parameters
    ----------
    evals : array-like
        Eigenvalues of a diffusion tensor.
    axis : int
        Axis of `evals` which contains 3 eigenvalues.

    Returns
    -------
    linearity : array
        Calculated linearity of the diffusion tensor.

    Notes
    -----
    Planarity is calculated with the following equation:

    .. math::

        Planarity =
        \frac{2 (\lambda_2-\lambda_3)}{\lambda_1+\lambda_2+\lambda_3}

    References
    ----------
    [1] Westin C.-F., Peled S., Gubjartsson H., Kikinis R., Jolesz F.,
        "Geometrical diffusion measures for MRI from tensor basis analysis" in
        Proc. 5th Annual ISMRM, 1997.
    """
    evals = _roll_evals(evals, axis)
    ev1, ev2, ev3 = evals
    return (2 * (ev2 - ev3) / evals.sum(0))


def sphericity(evals, axis=-1):
    """
    The sphericity of the tensor [1]_

    Parameters
    ----------
    evals : array-like
        Eigenvalues of a diffusion tensor.
    axis : int
        Axis of `evals` which contains 3 eigenvalues.

    Returns
    -------
    sphericity : array
        Calculated sphericity of the diffusion tensor.

    Notes
    -----
    Sphericity is calculated with the following equation:

    .. math::

        Sphericity = \frac{3 \lambda_3)}{\lambda_1+\lambda_2+\lambda_3}

    References
    ----------
    [1] Westin C.-F., Peled S., Gubjartsson H., Kikinis R., Jolesz F.,
        "Geometrical diffusion measures for MRI from tensor basis analysis" in
        Proc. 5th Annual ISMRM, 1997.
    """
    evals = _roll_evals(evals, axis)
    ev1, ev2, ev3 = evals
    return (3 * ev3) / evals.sum(0)


def apparent_diffusion_coef(q_form, sphere):
    """
    Calculate the apparent diffusion coefficient (ADC) in each direction of a
    sphere.

    Parameters
    ----------
    q_form : ndarray
        The quadratic form of a tensor, or an array with quadratic forms of
        tensors. Should be of shape (..., 3, 3)

    sphere : a Sphere class instance
        The ADC will be calculated for each of the vertices in the sphere

    Notes
    -----
    The calculation of ADC, relies on the following relationship:

    .. math ::
            ADC = \vec{b} Q \vec{b}^T

    Where Q is the quadratic form of the tensor.

    """
    bvecs = sphere.vertices
    bvals = np.ones(bvecs.shape[0])
    gtab = gradient_table(bvals, bvecs)
    D = design_matrix(gtab)[:, :6]
    return -np.dot(lower_triangular(q_form), D.T)


def tensor_prediction(dti_params, gtab, S0):
    """
    Predict a signal given tensor parameters.

    Parameters
    ----------
    dti_params : ndarray
        Tensor parameters. The last dimension should have 12 tensor
        parameters: 3 eigenvalues, followed by the 3 corresponding
        eigenvectors.

    gtab : a GradientTable class instance
        The gradient table for this prediction

    S0 : float or ndarray
        The non diffusion-weighted signal in every voxel, or across all
        voxels. Default: 1

    Notes
    -----
    The predicted signal is given by: $S(\theta, b) = S_0 * e^{-b ADC}$, where
    $ADC = \theta Q \theta^T$, $\theta$ is a unit vector pointing at any
    direction on the sphere for which a signal is to be predicted, $b$ is the b
    value provided in the GradientTable input for that direction, $Q$ is the
    quadratic form of the tensor determined by the input parameters.
    """
    evals = dti_params[..., :3]
    evecs = dti_params[..., 3:].reshape(dti_params.shape[:-1] + (3, 3))
    qform = vec_val_vect(evecs, evals)
    del evals, evecs
    lower_tri = lower_triangular(qform, S0)
    del qform

    D = design_matrix(gtab)
    return np.exp(np.dot(lower_tri, D.T))


class TensorModel(ReconstModel):
    """ Diffusion Tensor
    """

    def __init__(self, gtab, fit_method="WLS", return_S0_hat=False, *args,
                 **kwargs):
        """ A Diffusion Tensor Model [1]_, [2]_.

        Parameters
        ----------
        gtab : GradientTable class instance

        fit_method : str or callable
            str can be one of the following:

            'WLS' for weighted least squares
                :func:`dti.wls_fit_tensor`
            'LS' or 'OLS' for ordinary least squares
                :func:`dti.ols_fit_tensor`
            'NLLS' for non-linear least-squares
                :func:`dti.nlls_fit_tensor`
            'RT' or 'restore' or 'RESTORE' for RESTORE robust tensor
                fitting [3]_
                :func:`dti.restore_fit_tensor`

            callable has to have the signature:
              fit_method(design_matrix, data, *args, **kwargs)

        return_S0_hat : bool
            Boolean to return (True) or not (False) the S0 values for the fit.

        args, kwargs : arguments and key-word arguments passed to the
           fit_method. See dti.wls_fit_tensor, dti.ols_fit_tensor for details

        min_signal : float
            The minimum signal value. Needs to be a strictly positive
            number. Default: minimal signal in the data provided to `fit`.

        Notes
        -----
        In order to increase speed of processing, tensor fitting is done
        simultaneously over many voxels. Many fit_methods use the 'step'
        parameter to set the number of voxels that will be fit at once in each
        iteration. This is the chunk size as a number of voxels. A larger step
        value should speed things up, but it will also take up more memory. It
        is advisable to keep an eye on memory consumption as this value is
        increased.

        Example : In :func:`iter_fit_tensor` we have a default step value of
        1e4

        References
        ----------
        .. [1] Basser, P.J., Mattiello, J., LeBihan, D., 1994. Estimation of
           the effective self-diffusion tensor from the NMR spin echo. J Magn
           Reson B 103, 247-254.
        .. [2] Basser, P., Pierpaoli, C., 1996. Microstructural and
           physiological features of tissues elucidated by quantitative
           diffusion-tensor MRI.  Journal of Magnetic Resonance 111, 209-219.
        .. [3] Lin-Ching C., Jones D.K., Pierpaoli, C. 2005. RESTORE: Robust
           estimation of tensors by outlier rejection. MRM 53: 1088-1095

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
        self.return_S0_hat = return_S0_hat
        self.design_matrix = design_matrix(self.gtab)
        self.args = args
        self.kwargs = kwargs
        self.min_signal = self.kwargs.pop('min_signal', None)
        if self.min_signal is not None and self.min_signal <= 0:
            e_s = "The `min_signal` key-word argument needs to be strictly"
            e_s += " positive."
            raise ValueError(e_s)

    def fit(self, data, mask=None):
        """ Fit method of the DTI model class

        Parameters
        ----------
        data : array
            The measured signal from one voxel.

        mask : array
            A boolean array used to mark the coordinates in the data that
            should be analyzed that has the shape data.shape[:-1]

        """
        S0_params = None

        if mask is not None:
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

        params_in_mask = self.fit_method(
                self.design_matrix,
                data_in_mask,
                return_S0_hat=self.return_S0_hat,
                *self.args,
                **self.kwargs)
        if self.return_S0_hat:
            params_in_mask, model_S0 = params_in_mask

        if mask is None:
            out_shape = data.shape[:-1] + (-1, )
            dti_params = params_in_mask.reshape(out_shape)
            if self.return_S0_hat:
                S0_params = model_S0.reshape(out_shape[:-1])
        else:
            dti_params = np.zeros(data.shape[:-1] + (12,))
            dti_params[mask, :] = params_in_mask
            if self.return_S0_hat:
                S0_params = np.zeros(data.shape[:-1])
                S0_params[mask] = model_S0

        return TensorFit(self, dti_params, model_S0=S0_params)

    def predict(self, dti_params, S0=1.):
        """
        Predict a signal for this TensorModel class instance given parameters.

        Parameters
        ----------
        dti_params : ndarray
            The last dimension should have 12 tensor parameters: 3
            eigenvalues, followed by the 3 eigenvectors

        S0 : float or ndarray
            The non diffusion-weighted signal in every voxel, or across all
            voxels. Default: 1
        """
        return tensor_prediction(dti_params, self.gtab, S0)


class TensorFit(object):

    def __init__(self, model, model_params, model_S0=None):
        """ Initialize a TensorFit class instance.
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

    @property
    def shape(self):
        return self.model_params.shape[:-1]

    @property
    def directions(self):
        """
        For tracking - return the primary direction in each voxel
        """
        return self.evecs[..., None, :, 0]

    @property
    def evals(self):
        """
        Returns the eigenvalues of the tensor as an array
        """
        return self.model_params[..., :3]

    @property
    def evecs(self):
        """
        Returns the eigenvectors of the tensor as an array, columnwise
        """
        evecs = self.model_params[..., 3:12]
        return evecs.reshape(self.shape + (3, 3))

    @property
    def quadratic_form(self):
        """Calculates the 3x3 diffusion tensor for each voxel"""
        # do `evecs * evals * evecs.T` where * is matrix multiply
        # einsum does this with:
        # np.einsum('...ij,...j,...kj->...ik', evecs, evals, evecs)
        return vec_val_vect(self.evecs, self.evals)

    def lower_triangular(self, b0=None):
        return lower_triangular(self.quadratic_form, b0)

    @auto_attr
    def fa(self):
        """Fractional anisotropy (FA) calculated from cached eigenvalues."""
        return fractional_anisotropy(self.evals)

    @auto_attr
    def color_fa(self):
        """Color fractional anisotropy of diffusion tensor"""
        return color_fa(self.fa, self.evecs)

    @auto_attr
    def ga(self):
        """Geodesic anisotropy (GA) calculated from cached eigenvalues."""
        return geodesic_anisotropy(self.evals)

    @auto_attr
    def mode(self):
        """
        Tensor mode calculated from cached eigenvalues.
        """
        return mode(self.quadratic_form)

    @auto_attr
    def md(self):
        """
        Mean diffusivity (MD) calculated from cached eigenvalues.

        Returns
        -------
        md : array (V, 1)
            Calculated MD.

        Notes
        -----
        MD is calculated with the following equation:

        .. math::

            MD = \frac{\lambda_1+\lambda_2+\lambda_3}{3}

        """
        return self.trace / 3.0

    @auto_attr
    def rd(self):
        """
        Radial diffusivity (RD) calculated from cached eigenvalues.

        Returns
        -------
        rd : array (V, 1)
            Calculated RD.

        Notes
        -----
        RD is calculated with the following equation:

        .. math::

          RD = \frac{\lambda_2 + \lambda_3}{2}


        """
        return radial_diffusivity(self.evals)

    @auto_attr
    def ad(self):
        """
        Axial diffusivity (AD) calculated from cached eigenvalues.

        Returns
        -------
        ad : array (V, 1)
            Calculated AD.

        Notes
        -----
        RD is calculated with the following equation:

        .. math::

          AD = \lambda_1


        """
        return axial_diffusivity(self.evals)

    @auto_attr
    def trace(self):
        """
        Trace of the tensor calculated from cached eigenvalues.

        Returns
        -------
        trace : array (V, 1)
            Calculated trace.

        Notes
        -----
        The trace is calculated with the following equation:

        .. math::

          trace = \lambda_1 + \lambda_2 + \lambda_3
        """
        return trace(self.evals)

    @auto_attr
    def planarity(self):
        """
        Returns
        -------
        sphericity : array
            Calculated sphericity of the diffusion tensor [1]_.

        Notes
        -----
        Sphericity is calculated with the following equation:

        .. math::

            Sphericity =
            \frac{2 (\lambda_2 - \lambda_3)}{\lambda_1+\lambda_2+\lambda_3}

        References
        ----------
        [1] Westin C.-F., Peled S., Gubjartsson H., Kikinis R., Jolesz
            F., "Geometrical diffusion measures for MRI from tensor basis
            analysis" in Proc. 5th Annual ISMRM, 1997.

        """
        return planarity(self.evals)

    @auto_attr
    def linearity(self):
        """
        Returns
        -------
        linearity : array
            Calculated linearity of the diffusion tensor [1]_.

        Notes
        -----
        Linearity is calculated with the following equation:

        .. math::

            Linearity =
            \frac{\lambda_1-\lambda_2}{\lambda_1+\lambda_2+\lambda_3}

        References
        ----------
        [1] Westin C.-F., Peled S., Gubjartsson H., Kikinis R., Jolesz
            F., "Geometrical diffusion measures for MRI from tensor basis
            analysis" in Proc. 5th Annual ISMRM, 1997.

        """
        return linearity(self.evals)

    @auto_attr
    def sphericity(self):
        """
        Returns
        -------
        sphericity : array
            Calculated sphericity of the diffusion tensor [1]_.

        Notes
        -----
        Sphericity is calculated with the following equation:

        .. math::

            Sphericity = \frac{3 \lambda_3}{\lambda_1+\lambda_2+\lambda_3}

        References
        ----------
        [1] Westin C.-F., Peled S., Gubjartsson H., Kikinis R., Jolesz
            F., "Geometrical diffusion measures for MRI from tensor basis
            analysis" in Proc. 5th Annual ISMRM, 1997.

        """
        return sphericity(self.evals)

    def odf(self, sphere):
        """
        The diffusion orientation distribution function (dODF). This is an
        estimate of the diffusion distance in each direction

        Parameters
        ----------
        sphere : Sphere class instance.
            The dODF is calculated in the vertices of this input.

        Returns
        -------
        odf : ndarray
            The diffusion distance in every direction of the sphere in every
            voxel in the input data.

        Notes
        -----
        This is based on equation 3 in [Aganj2010]_. To re-derive it from
        scratch, follow steps in [Descoteaux2008]_, Section 7.9 Equation
        7.24 but with an $r^2$ term in the integral.

        References
        ----------
        .. [Aganj2010] Aganj, I., Lenglet, C., Sapiro, G., Yacoub, E., Ugurbil,
            K., & Harel, N. (2010). Reconstruction of the orientation
            distribution function in single- and multiple-shell q-ball imaging
            within constant solid angle. Magnetic Resonance in Medicine, 64(2),
            554-566. doi:DOI: 10.1002/mrm.22365

        .. [Descoteaux2008] Descoteaux, M. (2008). PhD Thesis: High Angular
           Resolution Diffusion MRI: from Local Estimation to Segmentation and
           Tractography. ftp://ftp-sop.inria.fr/athena/Publications/PhDs/descoteaux_thesis.pdf
        """
        odf = np.zeros((self.evals.shape[:-1] + (sphere.vertices.shape[0],)))
        if len(self.evals.shape) > 1:
            mask = np.where((self.evals[..., 0] > 0) &
                            (self.evals[..., 1] > 0) &
                            (self.evals[..., 2] > 0))
            evals = self.evals[mask]
            evecs = self.evecs[mask]
        else:
            evals = self.evals
            evecs = self.evecs
        lower = 4 * np.pi * np.sqrt(np.prod(evals, -1))
        projection = np.dot(sphere.vertices, evecs)
        projection /= np.sqrt(evals)
        result = ((vector_norm(projection) ** -3) / lower).T
        if len(self.evals.shape) > 1:
            odf[mask] = result
        else:
            odf = result
        return odf

    def adc(self, sphere):
        """
        Calculate the apparent diffusion coefficient (ADC) in each direction on
        the sphere for each voxel in the data

        Parameters
        ----------
        sphere : Sphere class instance

        Returns
        -------
        adc : ndarray
           The estimates of the apparent diffusion coefficient in every
           direction on the input sphere

        Notes
        -----
        The calculation of ADC, relies on the following relationship:

        .. math ::

            ADC = \vec{b} Q \vec{b}^T

        Where Q is the quadratic form of the tensor.
        """
        return apparent_diffusion_coef(self.quadratic_form, sphere)

    def predict(self, gtab, S0=None, step=None):
        """
        Given a model fit, predict the signal on the vertices of a sphere

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
        The predicted signal is given by:

        .. math ::

            S(\theta, b) = S_0 * e^{-b ADC}

        Where:
        .. math ::
            ADC = \theta Q \theta^T

        $\theta$ is a unit vector pointing at any direction on the sphere for
        which a signal is to be predicted and $b$ is the b value provided in
        the GradientTable input for that direction
        """
        if S0 is None:
            S0 = self.model_S0
            if S0 is None:  # if we didn't input or estimate S0 just use 1
                S0 = 1.
        shape = self.model_params.shape[:-1]
        size = np.prod(shape)
        if step is None:
            step = self.model.kwargs.get('step', size)
        if step >= size:
            return tensor_prediction(self.model_params[..., 0:12], gtab, S0=S0)
        params = np.reshape(self.model_params,
                            (-1, self.model_params.shape[-1]))
        predict = np.empty((size, gtab.bvals.shape[0]))
        if isinstance(S0, np.ndarray):
            S0 = S0.ravel()
        for i in range(0, size, step):
            if isinstance(S0, np.ndarray):
                this_S0 = S0[i:i + step]
            else:
                this_S0 = S0
            predict[i:i + step] = tensor_prediction(params[i:i + step], gtab,
                                                    S0=this_S0)
        return predict.reshape(shape + (gtab.bvals.shape[0], ))


def iter_fit_tensor(step=1e4):
    """Wrap a fit_tensor func and iterate over chunks of data with given length

    Splits data into a number of chunks of specified size and iterates the
    decorated fit_tensor function over them. This is useful to counteract the
    temporary but significant memory usage increase in fit_tensor functions
    that use vectorized operations and need to store large temporary arrays for
    their vectorized operations.

    Parameters
    ----------
    step : int
        The chunk size as a number of voxels. Optional parameter with default
        value 10,000.

        In order to increase speed of processing, tensor fitting is done
        simultaneously over many voxels. This parameter sets the number of
        voxels that will be fit at once in each iteration. A larger step value
        should speed things up, but it will also take up more memory. It is
        advisable to keep an eye on memory consumption as this value is
        increased.
    """

    def iter_decorator(fit_tensor):
        """Actual iter decorator returned by iter_fit_tensor dec factory

        Parameters
        ----------
        fit_tensor : callable
            A tensor fitting callable (most likely a function). The callable
            has to have the signature:
              fit_method(design_matrix, data, *args, **kwargs)
        """

        @functools.wraps(fit_tensor)
        def wrapped_fit_tensor(design_matrix, data, return_S0_hat=False,
                               step=step, *args, **kwargs):
            """Iterate fit_tensor function over the data chunks

            Parameters
            ----------
            design_matrix : array (g, 7)
                Design matrix holding the covariants used to solve for the
                regression coefficients.
            data : array ([X, Y, Z, ...], g)
                Data or response variables holding the data. Note that the last
                dimension should contain the data. It makes no copies of data.
            return_S0_hat : bool
                Boolean to return (True) or not (False) the S0 values for the
                fit.
            step : int
                The chunk size as a number of voxels. Overrides `step` value
                of `iter_fit_tensor`.
            args : {list,tuple}
                Any extra optional positional arguments passed to `fit_tensor`.
            kwargs : dict
                Any extra optional keyword arguments passed to `fit_tensor`.
            """
            shape = data.shape[:-1]
            size = np.prod(shape)
            step = int(step) or size
            if step >= size:
                return fit_tensor(design_matrix, data,
                                  return_S0_hat=return_S0_hat,
                                  *args, **kwargs)
            data = data.reshape(-1, data.shape[-1])
            dtiparams = np.empty((size, 12), dtype=np.float64)
            if return_S0_hat:
                S0params = np.empty(size, dtype=np.float64)
            for i in range(0, size, step):
                if return_S0_hat:
                    dtiparams[i:i + step], S0params[i:i + step] \
                        = fit_tensor(design_matrix,
                                     data[i:i + step],
                                     return_S0_hat=return_S0_hat,
                                     *args, **kwargs)
                else:
                    dtiparams[i:i + step] = fit_tensor(design_matrix,
                                                       data[i:i + step],
                                                       *args, **kwargs)
            if return_S0_hat:
                return (dtiparams.reshape(shape + (12, )),
                        S0params.reshape(shape + (1, )))
            else:
                return dtiparams.reshape(shape + (12, ))

        return wrapped_fit_tensor

    return iter_decorator


@iter_fit_tensor()
def wls_fit_tensor(design_matrix, data, return_S0_hat=False):
    """
    Computes weighted least squares (WLS) fit to calculate self-diffusion
    tensor using a linear regression model [1]_.

    Parameters
    ----------
    design_matrix : array (g, 7)
        Design matrix holding the covariants used to solve for the regression
        coefficients.
    data : array ([X, Y, Z, ...], g)
        Data or response variables holding the data. Note that the last
        dimension should contain the data. It makes no copies of data.
    return_S0_hat : bool
        Boolean to return (True) or not (False) the S0 values for the fit.

    Returns
    -------
    eigvals : array (..., 3)
        Eigenvalues from eigen decomposition of the tensor.
    eigvecs : array (..., 3, 3)
        Associated eigenvectors from eigen decomposition of the tensor.
        Eigenvectors are columnar (e.g. eigvecs[:,j] is associated with
        eigvals[j])


    See Also
    --------
    decompose_tensor

    Notes
    -----
    In Chung, et al. 2006, the regression of the WLS fit needed an unbiased
    preliminary estimate of the weights and therefore the ordinary least
    squares (OLS) estimates were used. A "two pass" method was implemented:

        1. calculate OLS estimates of the data
        2. apply the OLS estimates as weights to the WLS fit of the data

    This ensured heteroscedasticity could be properly modeled for various
    types of bootstrap resampling (namely residual bootstrap).

    .. math::

        y = \mathrm{data} \\
        X = \mathrm{design matrix} \\
        \hat{\beta}_\mathrm{WLS} =
        \mathrm{desired regression coefficients (e.g. tensor)}\\
        \\
        \hat{\beta}_\mathrm{WLS} = (X^T W X)^{-1} X^T W y \\
        \\
        W = \mathrm{diag}((X \hat{\beta}_\mathrm{OLS})^2),
        \mathrm{where} \hat{\beta}_\mathrm{OLS} = (X^T X)^{-1} X^T y

    References
    ----------
    .. [1] Chung, SW., Lu, Y., Henry, R.G., 2006. Comparison of bootstrap
       approaches for estimation of uncertainties of DTI parameters.
       NeuroImage 33, 531-541.
    """
    tol = 1e-6
    data = np.asarray(data)
    ols_fit = _ols_fit_matrix(design_matrix)
    log_s = np.log(data)
    w = np.exp(np.einsum('...ij,...j', ols_fit, log_s))
    fit_result = np.einsum('...ij,...j',
                           pinv(design_matrix * w[..., None]),
                           w * log_s)
    if return_S0_hat:
        return (eig_from_lo_tri(fit_result,
                                min_diffusivity=tol / -design_matrix.min()),
                np.exp(-fit_result[:, -1]))
    else:
        return eig_from_lo_tri(fit_result,
                               min_diffusivity=tol / -design_matrix.min())


@iter_fit_tensor()
def ols_fit_tensor(design_matrix, data, return_S0_hat=False):
    """
    Computes ordinary least squares (OLS) fit to calculate self-diffusion
    tensor using a linear regression model [1]_.

    Parameters
    ----------
    design_matrix : array (g, 7)
        Design matrix holding the covariants used to solve for the regression
        coefficients.
    data : array ([X, Y, Z, ...], g)
        Data or response variables holding the data. Note that the last
        dimension should contain the data. It makes no copies of data.
    return_S0_hat : bool
        Boolean to return (True) or not (False) the S0 values for the fit.

    Returns
    -------
    eigvals : array (..., 3)
        Eigenvalues from eigen decomposition of the tensor.
    eigvecs : array (..., 3, 3)
        Associated eigenvectors from eigen decomposition of the tensor.
        Eigenvectors are columnar (e.g. eigvecs[:,j] is associated with
        eigvals[j])


    See Also
    --------
    WLS_fit_tensor, decompose_tensor, design_matrix

    Notes
    -----
    .. math::

        y = \mathrm{data} \\
        X = \mathrm{design matrix} \\

        \hat{\beta}_\mathrm{OLS} = (X^T X)^{-1} X^T y

    References
    ----------
    ..  [1] Chung, SW., Lu, Y., Henry, R.G., 2006. Comparison of bootstrap
        approaches for estimation of uncertainties of DTI parameters.
        NeuroImage 33, 531-541.
    """
    tol = 1e-6
    data = np.asarray(data)
    fit_result = np.einsum('...ij,...j', np.linalg.pinv(design_matrix),
                           np.log(data))
    if return_S0_hat:
        return (eig_from_lo_tri(fit_result,
                                min_diffusivity=tol / -design_matrix.min()),
                np.exp(-fit_result[:, -1]))
    else:
        return eig_from_lo_tri(fit_result,
                               min_diffusivity=tol / -design_matrix.min())


def _ols_fit_matrix(design_matrix):
    """
    Helper function to calculate the ordinary least squares (OLS)
    fit as a matrix multiplication. Mainly used to calculate WLS weights. Can
    be used to calculate regression coefficients in OLS but not recommended.

    See Also:
    ---------
    wls_fit_tensor, ols_fit_tensor

    Example:
    --------
    ols_fit = _ols_fit_matrix(design_mat)
    ols_data = np.dot(ols_fit, data)
    """

    U, S, V = np.linalg.svd(design_matrix, False)
    return np.dot(U, U.T)


def _nlls_err_func(tensor, design_matrix, data, weighting=None,
                   sigma=None):
    """
    Error function for the non-linear least-squares fit of the tensor.

    Parameters
    ----------
    tensor : array (3,3)
        The 3-by-3 tensor matrix

    design_matrix : array
        The design matrix

    data : array
        The voxel signal in all gradient directions

    weighting : str (optional).
         Whether to use the Geman-McClure weighting criterion (see [1]_
         for details)

    sigma : float or float array (optional)
        If 'sigma' weighting is used, we will weight the error function
        according to the background noise estimated either in aggregate over
        all directions (when a float is provided), or to an estimate of the
        noise in each diffusion-weighting direction (if an array is
        provided). If 'gmm', the Geman-Mclure M-estimator is used for
        weighting (see below).

    Notes
    -----
    The Geman-McClure M-estimator is described as follows [1]_ (page 1089): "The
    scale factor C affects the shape of the GMM [Geman-McClure M-estimator]
    weighting function and represents the expected spread of the residuals
    (i.e., the SD of the residuals) due to Gaussian distributed noise. The
    scale factor C can be estimated by many robust scale estimators. We used
    the median absolute deviation (MAD) estimator because it is very robust to
    outliers having a 50% breakdown point (6,7). The explicit formula for C
    using the MAD estimator is:

    .. math ::

            C = 1.4826 x MAD = 1.4826 x median{|r1-\hat{r}|,... |r_n-\hat{r}|}

    where $\hat{r} = median{r_1, r_2, ..., r_3}$ and n is the number of data
    points. The multiplicative constant 1.4826 makes this an approximately
    unbiased estimate of scale when the error model is Gaussian."


    References
    ----------
    [1] Chang, L-C, Jones, DK and Pierpaoli, C (2005). RESTORE: robust
    estimation of tensors by outlier rejection. MRM, 53: 1088-95.
    """
    # This is the predicted signal given the params:
    y = np.exp(np.dot(design_matrix, tensor))

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
        w = 1 / (sigma**2)

    elif weighting == 'gmm':
        # We use the Geman-McClure M-estimator to compute the weights on the
        # residuals:
        C = 1.4826 * np.median(np.abs(residuals - np.median(residuals)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = 1 / (se + C**2)
            # The weights are normalized to the mean weight (see p. 1089):
            w = w / np.mean(w)

    # Return the weighted residuals:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.sqrt(w * se)


def _nlls_jacobian_func(tensor, design_matrix, data, *arg, **kwargs):
    """The Jacobian is the first derivative of the error function [1]_.

    Notes
    -----
    This is an implementation of equation 14 in [1]_.

    References
    ----------
    [1] Koay, CG, Chang, L-C, Carew, JD, Pierpaoli, C, Basser PJ (2006).
        A unifying theoretical and algorithmic framework for least squares
        methods of estimation in diffusion tensor imaging. MRM 182, 115-25.

    """
    pred = np.exp(np.dot(design_matrix, tensor))
    return -pred[:, None] * design_matrix


def _decompose_tensor_nan(tensor, tensor_alternative, min_diffusivity=0):
    """ Helper function that expands the function decompose_tensor to deal
    with tensor with nan elements.

    Computes tensor eigen decomposition to calculate eigenvalues and
    eigenvectors (Basser et al., 1994a). Some fit approaches can produce nan
    tensor elements in background voxels (particularly non-linear approaches).
    This function avoids the eigen decomposition errors of nan tensor elements
    by replacing tensor with nan elements by a given alternative tensor
    estimate.

    Parameters
    ----------
    tensor : array (3, 3)
        Hermitian matrix representing a diffusion tensor.
    tensor_alternative : array (3, 3)
        Hermitian matrix representing a diffusion tensor obtain from an
        approach that does not produce nan tensor elements
    min_diffusivity : float
        Because negative eigenvalues are not physical and small eigenvalues,
        much smaller than the diffusion weighting, cause quite a lot of noise
        in metrics such as fa, diffusivity values smaller than
        `min_diffusivity` are replaced with `min_diffusivity`.

    Returns
    -------
    eigvals : array (3)
        Eigenvalues from eigen decomposition of the tensor. Negative
        eigenvalues are replaced by zero. Sorted from largest to smallest.
    eigvecs : array (3, 3)
        Associated eigenvectors from eigen decomposition of the tensor.
        Eigenvectors are columnar (e.g. eigvecs[..., :, j] is associated with
        eigvals[..., j])

    """
    try:
        evals, evecs = decompose_tensor(tensor[:6],
                                        min_diffusivity=min_diffusivity)

    except np.linalg.LinAlgError:
        evals, evecs = decompose_tensor(tensor_alternative[:6],
                                        min_diffusivity=min_diffusivity)
    return evals, evecs


def nlls_fit_tensor(design_matrix, data, weighting=None,
                    sigma=None, jac=True, return_S0_hat=False):
    """
    Fit the tensor params using non-linear least-squares.

    Parameters
    ----------
    design_matrix : array (g, 7)
        Design matrix holding the covariants used to solve for the regression
        coefficients.

    data : array ([X, Y, Z, ...], g)
        Data or response variables holding the data. Note that the last
        dimension should contain the data. It makes no copies of data.

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
        Use the Jacobian? Default: True

    return_S0_hat : bool
        Boolean to return (True) or not (False) the S0 values for the fit.

    Returns
    -------
    nlls_params: the eigen-values and eigen-vectors of the tensor in each
        voxel.

    """
    # Flatten for the iteration over voxels:
    flat_data = data.reshape((-1, data.shape[-1]))
    # Use the OLS method parameters as the starting point for the optimization:
    inv_design = np.linalg.pinv(design_matrix)
    log_s = np.log(flat_data)
    D = np.dot(inv_design, log_s.T).T

    # Flatten for the iteration over voxels:
    ols_params = np.reshape(D, (-1, D.shape[-1]))
    # 12 parameters per voxel (evals + evecs):
    dti_params = np.empty((flat_data.shape[0], 12))
    if return_S0_hat:
        model_S0 = np.empty((flat_data.shape[0], 1))
    for vox in range(flat_data.shape[0]):
        if np.all(flat_data[vox] == 0):
            raise ValueError("The data in this voxel contains only zeros")

        start_params = ols_params[vox]
        # Do the optimization in this voxel:
        if jac:
            this_tensor, status = opt.leastsq(_nlls_err_func, start_params,
                                              args=(design_matrix,
                                                    flat_data[vox],
                                                    weighting,
                                                    sigma),
                                              Dfun=_nlls_jacobian_func)
        else:
            this_tensor, status = opt.leastsq(_nlls_err_func, start_params,
                                              args=(design_matrix,
                                                    flat_data[vox],
                                                    weighting,
                                                    sigma))

        # The parameters are the evals and the evecs:
        try:
            evals, evecs = decompose_tensor(
                from_lower_triangular(this_tensor[:6]))
            dti_params[vox, :3] = evals
            dti_params[vox, 3:] = evecs.ravel()
            if return_S0_hat:
                model_S0[vox] = np.exp(-this_tensor[6])
        # If leastsq failed to converge and produced nans, we'll resort to the
        # OLS solution in this voxel:
        except np.linalg.LinAlgError:
            evals, evecs = decompose_tensor(
                from_lower_triangular(start_params[:6]))
            dti_params[vox, :3] = evals
            dti_params[vox, 3:] = evecs.ravel()
            if return_S0_hat:
                model_S0[vox] = np.exp(-start_params[6])

    dti_params.shape = data.shape[:-1] + (12,)
    if return_S0_hat:
        model_S0.shape = data.shape[:-1] + (1,)
        return (dti_params, model_S0)
    else:
        return dti_params


def restore_fit_tensor(design_matrix, data, sigma=None, jac=True,
                       return_S0_hat=False):
    """
    Use the RESTORE algorithm [Chang2005]_ to calculate a robust tensor fit

    Parameters
    ----------

    design_matrix : array of shape (g, 7)
        Design matrix holding the covariants used to solve for the regression
        coefficients.

    data : array of shape ([X, Y, Z, n_directions], g)
        Data or response variables holding the data. Note that the last
        dimension should contain the data. It makes no copies of data.

    sigma : float
        An estimate of the variance. [Chang2005]_ recommend to use
        1.5267 * std(background_noise), where background_noise is estimated
        from some part of the image known to contain no signal (only noise).

    jac : bool, optional
        Whether to use the Jacobian of the tensor to speed the non-linear
        optimization procedure used to fit the tensor parameters (see also
        :func:`nlls_fit_tensor`). Default: True

    return_S0_hat : bool
        Boolean to return (True) or not (False) the S0 values for the fit.


    Returns
    -------
    restore_params : an estimate of the tensor parameters in each voxel.

    References
    ----------
    Chang, L-C, Jones, DK and Pierpaoli, C (2005). RESTORE: robust estimation
    of tensors by outlier rejection. MRM, 53: 1088-95.

    """
    # Flatten for the iteration over voxels:
    flat_data = data.reshape((-1, data.shape[-1]))
    # Use the OLS method parameters as the starting point for the optimization:
    inv_design = np.linalg.pinv(design_matrix)
    log_s = np.log(flat_data)
    D = np.dot(inv_design, log_s.T).T
    ols_params = np.reshape(D, (-1, D.shape[-1]))
    # 12 parameters per voxel (evals + evecs):
    dti_params = np.empty((flat_data.shape[0], 12))
    if return_S0_hat:
        model_S0 = np.empty((flat_data.shape[0], 1))
    for vox in range(flat_data.shape[0]):
        if np.all(flat_data[vox] == 0):
            raise ValueError("The data in this voxel contains only zeros")

        start_params = ols_params[vox]
        # Do nlls using sigma weighting in this voxel:
        if jac:
            this_tensor, status = opt.leastsq(_nlls_err_func, start_params,
                                              args=(design_matrix,
                                                    flat_data[vox],
                                                    'sigma',
                                                    sigma),
                                              Dfun=_nlls_jacobian_func)
        else:
            this_tensor, status = opt.leastsq(_nlls_err_func, start_params,
                                              args=(design_matrix,
                                                    flat_data[vox],
                                                    'sigma',
                                                    sigma))

        # Get the residuals:
        pred_sig = np.exp(np.dot(design_matrix, this_tensor))
        residuals = flat_data[vox] - pred_sig
        # If any of the residuals are outliers (using 3 sigma as a criterion
        # following Chang et al., e.g page 1089):
        if np.any(np.abs(residuals) > 3 * sigma):
            # Do nlls with GMM-weighting:
            if jac:
                this_tensor, status = opt.leastsq(_nlls_err_func,
                                                  start_params,
                                                  args=(design_matrix,
                                                        flat_data[vox],
                                                        'gmm'),
                                                  Dfun=_nlls_jacobian_func)
            else:
                this_tensor, status = opt.leastsq(_nlls_err_func,
                                                  start_params,
                                                  args=(design_matrix,
                                                        flat_data[vox],
                                                        'gmm'))

            # How are you doin' on those residuals?
            pred_sig = np.exp(np.dot(design_matrix, this_tensor))
            residuals = flat_data[vox] - pred_sig
            if np.any(np.abs(residuals) > 3 * sigma):
                # If you still have outliers, refit without those outliers:
                non_outlier_idx = np.where(np.abs(residuals) <= 3 * sigma)
                clean_design = design_matrix[non_outlier_idx]
                clean_sig = flat_data[vox][non_outlier_idx]
                if np.iterable(sigma):
                    this_sigma = sigma[non_outlier_idx]
                else:
                    this_sigma = sigma

                if jac:
                    this_tensor, status = opt.leastsq(_nlls_err_func,
                                                      start_params,
                                                      args=(clean_design,
                                                            clean_sig,
                                                            'sigma',
                                                            this_sigma),
                                                      Dfun=_nlls_jacobian_func)
                else:
                    this_tensor, status = opt.leastsq(_nlls_err_func,
                                                      start_params,
                                                      args=(clean_design,
                                                            clean_sig,
                                                            'sigma',
                                                            this_sigma))

        # The parameters are the evals and the evecs:
        evals, evecs = _decompose_tensor_nan(
            from_lower_triangular(this_tensor[:6]),
            from_lower_triangular(start_params[:6]))
        dti_params[vox, :3] = evals
        dti_params[vox, 3:] = evecs.ravel()
        if return_S0_hat:
            model_S0[vox] = np.exp(-this_tensor[6])

    dti_params.shape = data.shape[:-1] + (12,)
    restore_params = dti_params
    if return_S0_hat:
        model_S0.shape = data.shape[:-1] + (1,)
        return (restore_params, model_S0)
    else:
        return restore_params


_lt_indices = np.array([[0, 1, 3],
                        [1, 2, 4],
                        [3, 4, 5]])


def from_lower_triangular(D):
    """ Returns a tensor given the six unique tensor elements

    Given the six unique tensor elements (in the order: Dxx, Dxy, Dyy, Dxz,
    Dyz, Dzz) returns a 3 by 3 tensor. All elements after the sixth are
    ignored.

    Parameters
    -----------
    D : array_like, (..., >6)
        Unique elements of the tensors

    Returns
    -------
    tensor : ndarray (..., 3, 3)
        3 by 3 tensors

    """
    return D[..., _lt_indices]


_lt_rows = np.array([0, 1, 1, 2, 2, 2])
_lt_cols = np.array([0, 0, 1, 0, 1, 2])


def lower_triangular(tensor, b0=None):
    """
    Returns the six lower triangular values of the tensor and a dummy variable
    if b0 is not None

    Parameters
    ----------
    tensor : array_like (..., 3, 3)
        a collection of 3, 3 diffusion tensors
    b0 : float
        if b0 is not none log(b0) is returned as the dummy variable

    Returns
    -------
    D : ndarray
        If b0 is none, then the shape will be (..., 6) otherwise (..., 7)

    """
    if tensor.shape[-2:] != (3, 3):
        raise ValueError("Diffusion tensors should be (..., 3, 3)")
    if b0 is None:
        return tensor[..., _lt_rows, _lt_cols]
    else:
        D = np.empty(tensor.shape[:-2] + (7,), dtype=tensor.dtype)
        D[..., 6] = -np.log(b0)
        D[..., :6] = tensor[..., _lt_rows, _lt_cols]
        return D


def decompose_tensor(tensor, min_diffusivity=0):
    """ Returns eigenvalues and eigenvectors given a diffusion tensor

    Computes tensor eigen decomposition to calculate eigenvalues and
    eigenvectors (Basser et al., 1994a).

    Parameters
    ----------
    tensor : array (..., 3, 3)
        Hermitian matrix representing a diffusion tensor.
    min_diffusivity : float
        Because negative eigenvalues are not physical and small eigenvalues,
        much smaller than the diffusion weighting, cause quite a lot of noise
        in metrics such as fa, diffusivity values smaller than
        `min_diffusivity` are replaced with `min_diffusivity`.

    Returns
    -------
    eigvals : array (..., 3)
        Eigenvalues from eigen decomposition of the tensor. Negative
        eigenvalues are replaced by zero. Sorted from largest to smallest.
    eigvecs : array (..., 3, 3)
        Associated eigenvectors from eigen decomposition of the tensor.
        Eigenvectors are columnar (e.g. eigvecs[..., :, j] is associated with
        eigvals[..., j])

    """
    # outputs multiplicity as well so need to unique
    eigenvals, eigenvecs = eigh(tensor)

    # need to sort the eigenvalues and associated eigenvectors
    if eigenvals.ndim == 1:
        # this is a lot faster when dealing with a single voxel
        order = eigenvals.argsort()[::-1]
        eigenvecs = eigenvecs[:, order]
        eigenvals = eigenvals[order]
    else:
        # temporarily flatten eigenvals and eigenvecs to make sorting easier
        shape = eigenvals.shape[:-1]
        eigenvals = eigenvals.reshape(-1, 3)
        eigenvecs = eigenvecs.reshape(-1, 3, 3)
        size = eigenvals.shape[0]
        order = eigenvals.argsort()[:, ::-1]
        xi, yi = np.ogrid[:size, :3, :3][:2]
        eigenvecs = eigenvecs[xi, yi, order[:, None, :]]
        xi = np.ogrid[:size, :3][0]
        eigenvals = eigenvals[xi, order]
        eigenvecs = eigenvecs.reshape(shape + (3, 3))
        eigenvals = eigenvals.reshape(shape + (3, ))
    eigenvals = eigenvals.clip(min=min_diffusivity)
    # eigenvecs: each vector is columnar

    return eigenvals, eigenvecs


def design_matrix(gtab, dtype=None):
    """  Constructs design matrix for DTI weighted least squares or
    least squares fitting. (Basser et al., 1994a)

    Parameters
    ----------
    gtab : A GradientTable class instance

    dtype : string
        Parameter to control the dtype of returned designed matrix

    Returns
    -------
    design_matrix : array (g,7)
        Design matrix or B matrix assuming Gaussian distributed tensor model
        design_matrix[j, :] = (Bxx, Byy, Bzz, Bxy, Bxz, Byz, dummy)
    """
    B = np.zeros((gtab.gradients.shape[0], 7))
    B[:, 0] = gtab.bvecs[:, 0] * gtab.bvecs[:, 0] * 1. * gtab.bvals   # Bxx
    B[:, 1] = gtab.bvecs[:, 0] * gtab.bvecs[:, 1] * 2. * gtab.bvals   # Bxy
    B[:, 2] = gtab.bvecs[:, 1] * gtab.bvecs[:, 1] * 1. * gtab.bvals   # Byy
    B[:, 3] = gtab.bvecs[:, 0] * gtab.bvecs[:, 2] * 2. * gtab.bvals   # Bxz
    B[:, 4] = gtab.bvecs[:, 1] * gtab.bvecs[:, 2] * 2. * gtab.bvals   # Byz
    B[:, 5] = gtab.bvecs[:, 2] * gtab.bvecs[:, 2] * 1. * gtab.bvals   # Bzz
    B[:, 6] = np.ones(gtab.gradients.shape[0])

    return -B


def quantize_evecs(evecs, odf_vertices=None):
    """ Find the closest orientation of an evenly distributed sphere

    Parameters
    ----------
    evecs : ndarray
    odf_vertices : None or ndarray
        If None, then set vertices from symmetric362 sphere.  Otherwise use
        passed ndarray as vertices

    Returns
    -------
    IN : ndarray
    """
    max_evecs = evecs[..., :, 0]
    if odf_vertices is None:
        odf_vertices = get_sphere('symmetric362').vertices
    tup = max_evecs.shape[:-1]
    mec = max_evecs.reshape(np.prod(np.array(tup)), 3)
    IN = np.array([np.argmin(np.dot(odf_vertices, m)) for m in mec])
    IN = IN.reshape(tup)
    return IN


def eig_from_lo_tri(data, min_diffusivity=0):
    """
    Calculates tensor eigenvalues/eigenvectors from an array containing the
    lower diagonal form of the six unique tensor elements.

    Parameters
    ----------
    data : array_like (..., 6)
        diffusion tensors elements stored in lower triangular order
    min_diffusivity : float
        See decompose_tensor()

    Returns
    -------
    dti_params : array (..., 12)
        Eigen-values and eigen-vectors of the same array.
    """
    data = np.asarray(data)
    evals, evecs = decompose_tensor(from_lower_triangular(data),
                                    min_diffusivity=min_diffusivity)
    dti_params = np.concatenate((evals[..., None, :], evecs), axis=-2)
    return dti_params.reshape(data.shape[:-1] + (12, ))


common_fit_methods = {'WLS': wls_fit_tensor,
                      'LS': ols_fit_tensor,
                      'OLS': ols_fit_tensor,
                      'NLLS': nlls_fit_tensor,
                      'RT': restore_fit_tensor,
                      'restore': restore_fit_tensor,
                      'RESTORE': restore_fit_tensor
                      }
