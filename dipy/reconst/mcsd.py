import numbers
import warnings

import numpy as np

from dipy.core import geometry as geo
from dipy.core.gradients import (GradientTable, gradient_table,
                                 unique_bvals_tolerance, get_bval_indices)
from dipy.data import default_sphere
from dipy.reconst import shm
from dipy.reconst.csdeconv import response_from_mask_ssst
from dipy.reconst.dti import (TensorModel, fractional_anisotropy,
                              mean_diffusivity)
from dipy.reconst.multi_voxel import multi_voxel_fit
from dipy.reconst.utils import _roi_in_volume, _mask_from_roi
from dipy.sims.voxel import single_tensor
from dipy.utils.deprecator import deprecated_params

from dipy.utils.optpkg import optional_package
cvxpy, have_cvxpy, _ = optional_package("cvxpy", min_version="1.4.1")

SH_CONST = .5 / np.sqrt(np.pi)

@deprecated_params('sh_order', 'sh_order_max', since='1.9', until='2.0')
def multi_tissue_basis(gtab, sh_order_max, iso_comp):
    """
    Builds a basis for multi-shell multi-tissue CSD model.

    Parameters
    ----------
    gtab : GradientTable
    sh_order_max : int
        Maximal spherical harmonics order (l).
    iso_comp: int
        Number of tissue compartments for running the MSMT-CSD. Minimum
        number of compartments required is 2.

    Returns
    -------
    B : ndarray
        Matrix of the spherical harmonics model used to fit the data
    m_values : int ``|m_value| <= l_value``
        The phase factor (m) of the harmonic.
    l_values : int ``l_value >= 0``
        The order (l) of the harmonic.
    """
    if iso_comp < 2:
        msg = "Multi-tissue CSD requires at least 2 tissue compartments"
        raise ValueError(msg)
    r, theta, phi = geo.cart2sphere(*gtab.gradients.T)
    m_values, l_values = shm.sph_harm_ind_list(sh_order_max)
    B = shm.real_sh_descoteaux_from_index(m_values, l_values,
                                          theta[:, None], phi[:, None])
    B[np.ix_(gtab.b0s_mask, l_values > 0)] = 0.

    iso = np.empty([B.shape[0], iso_comp])
    iso[:] = SH_CONST

    B = np.concatenate([iso, B], axis=1)
    return B, m_values, l_values


class MultiShellResponse:
    @deprecated_params('sh_order', 'sh_order_max', since='1.9', until='2.0')
    def __init__(self, response, sh_order_max, shells, S0=None):
        """ Estimate Multi Shell response function for multiple tissues and
        multiple shells.

        The method `multi_shell_fiber_response` allows to create a multi-shell
        fiber response with the right format, for a three compartments model.
        It can be referred to in order to understand the inputs of this class.

        Parameters
        ----------
        response : ndarray
            Multi-shell fiber response. The ordering of the responses should
            follow the same logic as S0.
        sh_order_max : int
            Maximal spherical harmonics order (l).
        shells : int
            Number of shells in the data
        S0 : array (3,)
            Signal with no diffusion weighting for each tissue compartments, in
            the same tissue order as `response`. This S0 can be used for
            predicting from a fit model later on.
        """
        self.S0 = S0
        self.response = response
        self.sh_order_max = sh_order_max
        self.l_values = np.arange(0, sh_order_max + 1, 2)
        self.m_values = np.zeros_like(self.l_values)
        self.shells = shells
        if self.iso < 1:
            raise ValueError("sh_order_max and shape of response do not agree")

    @property
    def iso(self):
        return self.response.shape[1] - (self.sh_order_max // 2) - 1

@deprecated_params('sh_order', 'sh_order_max', since='1.9', until='2.0')
def _inflate_response(response, gtab, sh_order_max, delta):
    """Used to inflate the response for the `multiplier_matrix` in the
    `MultiShellDeconvModel`.
    Parameters
    ----------
    response : MultiShellResponse object
    gtab : GradientTable
    sh_order_max : int ``>= 0``
        The maximal order (l) of the harmonic.
    delta : Delta generated from `_basic_delta`
    """
    if any((sh_order_max % 2) != 0) or \
        (sh_order_max.max() // 2) >= response.sh_order_max:
        raise ValueError("Response and n do not match")

    iso = response.iso
    n_idx = np.empty(len(sh_order_max) + iso, dtype=int)
    n_idx[:iso] = np.arange(0, iso)
    n_idx[iso:] = sh_order_max // 2 + iso
    diff = abs(response.shells[:, None] - gtab.bvals)
    b_idx = np.argmin(diff, axis=0)
    kernel = response.response / delta

    return kernel[np.ix_(b_idx, n_idx)]


def _basic_delta(iso, m_value, l_value, theta, phi):
    """Simple delta function
    Parameters
    ----------
    iso: int
        Number of tissue compartments for running the MSMT-CSD. Minimum
        number of compartments required is 2.
        Default: 2
    m_value : int ``|m| <= l``
        The phase factor (m) of the harmonic.
    l_value : int ``>= 0``
        The order (l) of the harmonic.
    theta : array_like
       inclination or polar angle
    phi : array_like
       azimuth angle
    """
    wm_d = shm.gen_dirac(m_value, l_value, theta, phi)
    iso_d = [SH_CONST] * iso
    return np.concatenate([iso_d, wm_d])


class MultiShellDeconvModel(shm.SphHarmModel):
    @deprecated_params('sh_order', 'sh_order_max', since='1.9', until='2.0')
    def __init__(self, gtab, response, reg_sphere=default_sphere,
                 sh_order_max=8, iso=2, tol=20):
        r"""
        Multi-Shell Multi-Tissue Constrained Spherical Deconvolution
        (MSMT-CSD) [1]_. This method extends the CSD model proposed in [2]_ by
        the estimation of multiple response functions as a function of multiple
        b-values and multiple tissue types.

        Spherical deconvolution computes a fiber orientation distribution
        (FOD), also called fiber ODF (fODF) [2]_. The fODF is derived from
        different tissue types and thus overcomes the overestimation of WM in
        GM and CSF areas.

        The response function is based on the different tissue types
        and is provided as input to the MultiShellDeconvModel.
        It will be used as deconvolution kernel, as described in [2]_.

        Parameters
        ----------
        gtab : GradientTable
        response : ndarray or MultiShellResponse object
            Pre-computed multi-shell fiber response function in the form of a
            MultiShellResponse object, or simple response function as a ndarray.
            The later must be of shape (3, len(bvals)-1, 4), because it will be
            converted into a MultiShellResponse object via the
            `multi_shell_fiber_response` method (important note: the function
            `unique_bvals_tolerance` is used here to select unique bvalues from
            gtab as input). Each column (3,) has two elements. The first is the
            eigen-values as a (3,) ndarray and the second is the signal value
            for the response function without diffusion weighting (S0). Note
            that in order to use more than three compartments, one must create
            a MultiShellResponse object on the side.
        reg_sphere : Sphere (optional)
            sphere used to build the regularization B matrix.
            Default: 'symmetric362'.
        sh_order_max : int (optional)
            Maximal spherical harmonics order (l). Default: 8
        iso: int (optional)
            Number of tissue compartments for running the MSMT-CSD. Minimum
            number of compartments required is 2.
            Default: 2
        tol : int, optional
            Tolerance gap for b-values clustering.

        References
        ----------
        .. [1] Jeurissen, B., et al. NeuroImage 2014. Multi-tissue constrained
               spherical deconvolution for improved analysis of multi-shell
               diffusion MRI data
        .. [2] Tournier, J.D., et al. NeuroImage 2007. Robust determination of
               the fibre orientation distribution in diffusion MRI:
               Non-negativity constrained super-resolved spherical
               deconvolution
        .. [3] Tournier, J.D, et al. Imaging Systems and Technology
               2012. MRtrix: Diffusion Tractography in Crossing Fiber Regions
        """
        if not iso >= 2:
            msg = "Multi-tissue CSD requires at least 2 tissue compartments"
            raise ValueError(msg)

        super(MultiShellDeconvModel, self).__init__(gtab)

        if not isinstance(response, MultiShellResponse):
            bvals = unique_bvals_tolerance(gtab.bvals, tol=tol)
            if iso > 2:
                msg = """Too many compartments for this kind of response
                input. It must be two tissue compartments."""
                raise ValueError(msg)
            if response.shape != (3, len(bvals)-1, 4):
                msg = """Response must be of shape (3, len(bvals)-1, 4) or be a
                MultiShellResponse object."""
                raise ValueError(msg)
            response = multi_shell_fiber_response(sh_order_max,
                                                  bvals=bvals,
                                                  wm_rf=response[0],
                                                  gm_rf=response[1],
                                                  csf_rf=response[2])

        B, m_values, l_values = multi_tissue_basis(gtab, sh_order_max, iso)

        delta = _basic_delta(response.iso, response.m_values,
                             response.l_values, 0., 0.)
        self.delta = delta
        multiplier_matrix = _inflate_response(response, gtab, l_values, delta)

        r, theta, phi = geo.cart2sphere(*reg_sphere.vertices.T)
        odf_reg, _, _ = shm.real_sh_descoteaux(sh_order_max, theta, phi)
        reg = np.zeros([i + iso for i in odf_reg.shape])
        reg[:iso, :iso] = np.eye(iso)
        reg[iso:, iso:] = odf_reg

        X = B * multiplier_matrix

        self.fitter = QpFitter(X, reg)
        self.sh_order_max = sh_order_max
        self._X = X
        self.sphere = reg_sphere
        self.gtab = gtab
        self.B_dwi = B
        self.m_values = m_values
        self.l_values = l_values
        self.response = response

    def predict(self, params, gtab=None, S0=None):
        """Compute a signal prediction given spherical harmonic coefficients
        for the provided GradientTable class instance.

        Parameters
        ----------
        params : ndarray
            The spherical harmonic representation of the FOD from which to make
            the signal prediction.
        gtab : GradientTable
            The gradients for which the signal will be predicted. Use the
            model's gradient table by default.
        S0 : ndarray or float
            The non diffusion-weighted signal value.
        """
        if gtab is None or gtab is self.gtab:
            X = self._X
        else:
            iso = self.response.iso
            B, m_values, l_values = multi_tissue_basis(gtab,
                                                       self.sh_order_max,
                                                       iso)
            multiplier_matrix = _inflate_response(self.response,
                                                  gtab,
                                                  l_values,
                                                  self.delta)
            X = B * multiplier_matrix

        scaling = 1.
        if S0 and S0 != 1.:     # The S0=1. case comes from fit.predict().
            raise NotImplementedError
            # This case is not implemented yet because it would require to have
            # access to volume fractions (vf) from the fit. The following code
            # gives an idea of how to use this with S0 and vf. It could also be
            # calculated externally and used as scaling = S0.
            # response_scaling = np.ndarray(params.shape[0:3])
            # response_scaling[...] = (vf[..., 0] * self.response.S0[0]
            #                          + vf[..., 1] * self.response.S0[1]
            #                          + vf[..., 2] * self.response.S0[2])
            # scaling = np.where(response_scaling > 1, S0 / response_scaling, 0)
            # scaling = np.expand_dims(scaling, 3)
            # scaling = np.repeat(scaling, len(gtab.bvals), axis=3)

        pred_sig = scaling * np.dot(params, X.T)
        return pred_sig

    @multi_voxel_fit
    def fit(self, data, verbose=True):
        """Fits the model to diffusion data and returns the model fit.

        Sometimes the solving process of some voxels can end in a SolverError
        from cvxpy. This might be attributed to the response functions not
        being tuned properly, as the solving process is very sensitive to it.
        The method will fill the problematic voxels with a NaN value, so that
        it is traceable. The user should check for the number of NaN values and
        could then fill the problematic voxels with zeros, for example.
        Running a fit again only on those problematic voxels can also work.

        Parameters
        ----------
        data : ndarray
            The diffusion data to fit the model on.
        verbose : bool (optional)
            Whether to show warnings when a SolverError appears or not.
            Default: True
        """
        coeff = self.fitter(data)
        if verbose:
            if np.isnan(coeff[..., 0]):
                msg = """Voxel could not be solved properly and ended up with a
                SolverError. Proceeding to fill it with NaN values.
                """
                warnings.warn(msg, UserWarning)

        return MSDeconvFit(self, coeff, None)


class MSDeconvFit(shm.SphHarmFit):

    def __init__(self, model, coeff, mask):
        """
        Abstract class which holds the fit result of MultiShellDeconvModel.
        Inherits the SphHarmFit which fits the diffusion data to a spherical
        harmonic model.

        Parameters
        ----------
        model: object
            MultiShellDeconvModel
        coeff : array
            Spherical harmonic coefficients for the ODF.
        mask: ndarray
            Mask for fitting
        """
        self._shm_coef = coeff
        self.mask = mask
        self.model = model

    @property
    def shm_coeff(self):
        return self._shm_coef[..., self.model.response.iso:]

    @property
    def all_shm_coeff(self):
        return self._shm_coef

    @property
    def volume_fractions(self):
        tissue_classes = self.model.response.iso + 1
        return self._shm_coef[..., :tissue_classes] / SH_CONST


def solve_qp(P, Q, G, H):
    r"""
    Helper function to set up and solve the Quadratic Program (QP) in CVXPY.
    A QP problem has the following form:
    minimize      1/2 x' P x + Q' x
    subject to    G x <= H

    Here the QP solver is based on CVXPY and uses OSQP.

    Parameters
    ----------
    P : ndarray
        n x n matrix for the primal QP objective function.
    Q : ndarray
        n x 1 matrix for the primal QP objective function.
    G : ndarray
        m x n matrix for the inequality constraint.
    H : ndarray
        m x 1 matrix for the inequality constraint.

    Returns
    -------
    x : array
        Optimal solution to the QP problem.
    """
    x = cvxpy.Variable(Q.shape[0])
    P = cvxpy.Constant(P)
    objective = cvxpy.Minimize(0.5 * cvxpy.quad_form(x, P, True) + Q @ x)
    constraints = [G @ x <= H]

    # setting up the problem
    prob = cvxpy.Problem(objective, constraints)
    try:
        prob.solve()
        opt = np.array(x.value).reshape((Q.shape[0],))
    except cvxpy.error.SolverError:
        opt = np.empty((Q.shape[0],))
        opt[:] = np.NaN

    return opt


class QpFitter:

    def __init__(self, X, reg):
        r"""
        Makes use of the quadratic programming solver `solve_qp` to fit the
        model. The initialization for the model is done using the warm-start by
        default in `CVXPY`.

        Parameters
        ----------
        X : ndarray
            Matrix to be fit by the QP solver calculated in
            `MultiShellDeconvModel`
        reg : ndarray
            the regularization B matrix calculated in `MultiShellDeconvModel`
        """
        self._P = P = np.dot(X.T, X)
        self._X = X

        self._reg = reg
        self._P_mat = np.array(P)
        self._reg_mat = np.array(-reg)
        self._h_mat = np.array([0])

    def __call__(self, signal):
        Q = np.dot(self._X.T, signal)
        Q_mat = np.array(-Q)
        fodf_sh = solve_qp(self._P_mat, Q_mat, self._reg_mat, self._h_mat)
        return fodf_sh

@deprecated_params('sh_order', 'sh_order_max', since='1.9', until='2.0')
def multi_shell_fiber_response(sh_order_max, bvals, wm_rf, gm_rf, csf_rf,
                               sphere=None, tol=20, btens=None):
    """Fiber response function estimation for multi-shell data.

    Parameters
    ----------
    sh_order_max : int
         Maximum spherical harmonics order (l).
    bvals : ndarray
        Array containing the b-values. Must be unique b-values, like outputted
        by `dipy.core.gradients.unique_bvals_tolerance`.
    wm_rf : (N-1, 4) ndarray
        Response function of the WM tissue, for each bvals,
        where N is the number of unique b-values including the b0.
    gm_rf : (N-1, 4) ndarray
        Response function of the GM tissue, for each bvals.
    csf_rf : (N-1, 4) ndarray
        Response function of the CSF tissue, for each bvals.
    sphere : `dipy.core.Sphere` instance, optional
        Sphere where the signal will be evaluated.
    tol : int, optional
        Tolerance gap for b-values clustering.
    btens : can be any of two options, optional

        1. an array of strings of shape (N,) specifying
           encoding tensor shape associated with all unique b-values
           separately. N corresponds to the number of unique b-values,
           including the b0. Options for elements in array: 'LTE',
           'PTE', 'STE', 'CTE' corresponding to linear, planar, spherical, and
           "cigar-shaped" tensor encoding.
        2. an array of shape (N,3,3) specifying the b-tensor of each unique
           b-values exactly. N corresponds to the number of unique b-values,
           including the b0.

    Returns
    -------
    MultiShellResponse
        MultiShellResponse object.

    """
    bvals = np.array(bvals, copy=True)
    if btens is None:
        btens = np.repeat(["LTE"], len(bvals))
    elif len(btens) != len(bvals):
        msg = """bvals and btens parameters must have the same dimension."""
        raise ValueError(msg)
    evecs = np.zeros((3, 3))
    z = np.array([0, 0, 1.])
    evecs[:, 0] = z
    evecs[:2, 1:] = np.eye(2)

    l_values = np.arange(0, sh_order_max + 1, 2)
    m_values = np.zeros_like(l_values)

    if sphere is None:
        sphere = default_sphere

    big_sphere = sphere.subdivide()
    theta, phi = big_sphere.theta, big_sphere.phi

    B = shm.real_sh_descoteaux_from_index(m_values, l_values,
                                          theta[:, None], phi[:, None])
    A = shm.real_sh_descoteaux_from_index(0, 0, 0, 0)

    response = np.empty([len(bvals), len(l_values) + 2])

    if bvals[0] < tol:
        gtab = GradientTable(big_sphere.vertices * 0, btens=btens[0])
        wm_response = single_tensor(gtab, wm_rf[0, 3], wm_rf[0, :3], evecs,
                                    snr=None)
        response[0, 2:] = np.linalg.lstsq(B, wm_response, rcond=None)[0]

        response[0, 1] = gm_rf[0, 3] / A
        response[0, 0] = csf_rf[0, 3] / A

        for i, bvalue in enumerate(bvals[1:]):
            gtab = GradientTable(big_sphere.vertices * bvalue,
                                 btens=btens[i + 1])
            wm_response = single_tensor(gtab, wm_rf[i, 3], wm_rf[i, :3], evecs,
                                        snr=None)
            response[i+1, 2:] = np.linalg.lstsq(B, wm_response,
                                                rcond=None)[0]

            response[i+1, 1] = gm_rf[i, 3] * np.exp(-bvalue * gm_rf[i, 0]) / A
            response[i+1, 0] = csf_rf[i, 3] * np.exp(-bvalue * csf_rf[i, 0]) / A

        S0 = [csf_rf[0, 3], gm_rf[0, 3], wm_rf[0, 3]]

    else:
        warnings.warn("""No b0 given. Proceeding either way.""", UserWarning)
        for i, bvalue in enumerate(bvals):
            gtab = GradientTable(big_sphere.vertices * bvalue,
                                 btens=btens[i])
            wm_response = single_tensor(gtab, wm_rf[i, 3], wm_rf[i, :3], evecs,
                                        snr=None)
            response[i, 2:] = np.linalg.lstsq(B, wm_response,
                                              rcond=None)[0]

            response[i, 1] = gm_rf[i, 3] * np.exp(-bvalue * gm_rf[i, 0]) / A
            response[i, 0] = csf_rf[i, 3] * np.exp(-bvalue * csf_rf[i, 0]) / A

        S0 = [csf_rf[0, 3], gm_rf[0, 3], wm_rf[0, 3]]

    return MultiShellResponse(response, sh_order_max, bvals, S0=S0)


def mask_for_response_msmt(gtab, data, roi_center=None, roi_radii=10,
                           wm_fa_thr=0.7, gm_fa_thr=0.2, csf_fa_thr=0.1,
                           gm_md_thr=0.0007, csf_md_thr=0.002):
    """ Computation of masks for multi-shell multi-tissue (msmt) response
        function using FA and MD.

    Parameters
    ----------
    gtab : GradientTable
    data : ndarray
        diffusion data (4D)
    roi_center : array-like, (3,)
        Center of ROI in data. If center is None, it is assumed that it is
        the center of the volume with shape `data.shape[:3]`.
    roi_radii : int or array-like, (3,)
        radii of cuboid ROI
    wm_fa_thr : float
        FA threshold for WM.
    gm_fa_thr : float
        FA threshold for GM.
    csf_fa_thr : float
        FA threshold for CSF.
    gm_md_thr : float
        MD threshold for GM.
    csf_md_thr : float
        MD threshold for CSF.

    Returns
    -------
    mask_wm : ndarray
        Mask of voxels within the ROI and with FA above the FA threshold
        for WM.
    mask_gm : ndarray
        Mask of voxels within the ROI and with FA below the FA threshold
        for GM and with MD below the MD threshold for GM.
    mask_csf : ndarray
        Mask of voxels within the ROI and with FA below the FA threshold
        for CSF and with MD below the MD threshold for CSF.

    Notes
    -----
    In msmt-CSD there is an important pre-processing step: the estimation of
    every tissue's response function. In order to do this, we look for voxels
    corresponding to WM, GM and CSF. This function aims to accomplish that by
    returning a mask of voxels within a ROI and who respect some threshold
    constraints, for each tissue. More precisely, the WM mask must have a FA
    value above a given threshold. The GM mask and CSF mask must have a FA
    below given thresholds and a MD below other thresholds. To get the FA and
    MD, we need to fit a Tensor model to the datasets.
    """

    if len(data.shape) < 4:
        msg = """Data must be 4D (3D image + directions). To use a 2D image,
        please reshape it into a (N, N, 1, ndirs) array."""
        raise ValueError(msg)

    if isinstance(roi_radii, numbers.Number):
        roi_radii = (roi_radii, roi_radii, roi_radii)

    if roi_center is None:
        roi_center = np.array(data.shape[:3]) // 2

    roi_radii = _roi_in_volume(data.shape, np.asarray(roi_center),
                               np.asarray(roi_radii))

    roi_mask = _mask_from_roi(data.shape[:3], roi_center, roi_radii)

    list_bvals = unique_bvals_tolerance(gtab.bvals)
    if not np.all(list_bvals <= 1200):
        msg_bvals = """Some b-values are higher than 1200.
        The DTI fit might be affected."""
        warnings.warn(msg_bvals, UserWarning)

    ten = TensorModel(gtab)
    tenfit = ten.fit(data, mask=roi_mask)
    fa = fractional_anisotropy(tenfit.evals)
    fa[np.isnan(fa)] = 0
    md = mean_diffusivity(tenfit.evals)
    md[np.isnan(md)] = 0

    mask_wm = np.zeros(fa.shape, dtype=np.int64)
    mask_wm[fa > wm_fa_thr] = 1
    mask_wm *= roi_mask

    md_mask_gm = np.zeros(md.shape, dtype=np.int64)
    md_mask_gm[(md < gm_md_thr)] = 1

    fa_mask_gm = np.zeros(fa.shape, dtype=np.int64)
    fa_mask_gm[(fa < gm_fa_thr) & (fa > 0)] = 1

    mask_gm = md_mask_gm * fa_mask_gm
    mask_gm *= roi_mask

    md_mask_csf = np.zeros(md.shape, dtype=np.int64)
    md_mask_csf[(md < csf_md_thr) & (md > 0)] = 1

    fa_mask_csf = np.zeros(fa.shape, dtype=np.int64)
    fa_mask_csf[(fa < csf_fa_thr) & (fa > 0)] = 1

    mask_csf = md_mask_csf * fa_mask_csf
    mask_csf *= roi_mask

    msg = """No voxel with a {0} than {1} were found.
    Try a larger roi or a {2} threshold for {3}."""

    if np.sum(mask_wm) == 0:
        msg_fa = msg.format('FA higher', str(wm_fa_thr), 'lower FA', 'WM')
        warnings.warn(msg_fa, UserWarning)

    if np.sum(mask_gm) == 0:
        msg_fa = msg.format('FA lower', str(gm_fa_thr), 'higher FA', 'GM')
        msg_md = msg.format('MD lower', str(gm_md_thr), 'higher MD', 'GM')
        warnings.warn(msg_fa, UserWarning)
        warnings.warn(msg_md, UserWarning)

    if np.sum(mask_csf) == 0:
        msg_fa = msg.format('FA lower', str(csf_fa_thr), 'higher FA', 'CSF')
        msg_md = msg.format('MD lower', str(csf_md_thr), 'higher MD', 'CSF')
        warnings.warn(msg_fa, UserWarning)
        warnings.warn(msg_md, UserWarning)

    return mask_wm, mask_gm, mask_csf


def response_from_mask_msmt(gtab, data, mask_wm, mask_gm, mask_csf, tol=20):
    """ Computation of multi-shell multi-tissue (msmt) response
        functions from given tissues masks.

    Parameters
    ----------
    gtab : GradientTable
    data : ndarray
        diffusion data
    mask_wm : ndarray
        mask from where to compute the WM response function.
    mask_gm : ndarray
        mask from where to compute the GM response function.
    mask_csf : ndarray
        mask from where to compute the CSF response function.
    tol : int
        tolerance gap for b-values clustering. (Default = 20)

    Returns
    -------
    response_wm : ndarray, (len(unique_bvals_tolerance(gtab.bvals))-1, 4)
        (`evals`, `S0`) for WM for each unique bvalues (except b0).
    response_gm : ndarray, (len(unique_bvals_tolerance(gtab.bvals))-1, 4)
        (`evals`, `S0`) for GM for each unique bvalues (except b0).
    response_csf : ndarray, (len(unique_bvals_tolerance(gtab.bvals))-1, 4)
        (`evals`, `S0`) for CSF for each unique bvalues (except b0).

    Notes
    -----
    In msmt-CSD there is an important pre-processing step: the estimation of
    every tissue's response function. In order to do this, we look for voxels
    corresponding to WM, GM and CSF. This information can be obtained by using
    mcsd.mask_for_response_msmt() through masks of selected voxels. The present
    function uses such masks to compute the msmt response functions.

    For the responses, we base our approach on the function
    csdeconv.response_from_mask_ssst(), with the added layers of multishell and
    multi-tissue (see the ssst function for more information about the
    computation of the ssst response function). This means that for each tissue
    we use the previously found masks and loop on them. For each mask, we loop
    on the b-values (clustered using the tolerance gap) to get many responses
    and then average them to get one response per tissue.
    """

    bvals = gtab.bvals
    bvecs = gtab.bvecs
    btens = gtab.btens

    list_bvals = unique_bvals_tolerance(bvals, tol)

    b0_indices = get_bval_indices(bvals, list_bvals[0], tol)
    b0_map = np.mean(data[..., b0_indices], axis=-1)[..., np.newaxis]

    masks = [mask_wm, mask_gm, mask_csf]
    tissue_responses = []
    for mask in masks:
        responses = []
        for bval in list_bvals[1:]:
            indices = get_bval_indices(bvals, bval, tol)

            bvecs_sub = np.concatenate([[bvecs[b0_indices[0]]],
                                       bvecs[indices]])
            bvals_sub = np.concatenate([[0], bvals[indices]])
            if btens is not None:
                btens_b0 = btens[b0_indices[0]].reshape((1, 3, 3))
                btens_sub = np.concatenate([btens_b0, btens[indices]])
            else:
                btens_sub = None

            data_conc = np.concatenate([b0_map, data[..., indices]], axis=3)

            gtab = gradient_table(bvals_sub, bvecs_sub, btens=btens_sub)
            response, _ = response_from_mask_ssst(gtab, data_conc, mask)

            responses.append(list(np.concatenate([response[0], [response[1]]])))

        tissue_responses.append(list(responses))

    wm_response = np.asarray(tissue_responses[0])
    gm_response = np.asarray(tissue_responses[1])
    csf_response = np.asarray(tissue_responses[2])
    return wm_response, gm_response, csf_response


def auto_response_msmt(gtab, data, tol=20, roi_center=None, roi_radii=10,
                       wm_fa_thr=0.7, gm_fa_thr=0.3, csf_fa_thr=0.15,
                       gm_md_thr=0.001, csf_md_thr=0.0032):
    """ Automatic estimation of multi-shell multi-tissue (msmt) response
        functions using FA and MD.

    Parameters
    ----------
    gtab : GradientTable
    data : ndarray
        diffusion data
    tol : int, optional
        Tolerance gap for b-values clustering.
    roi_center : array-like, (3,)
        Center of ROI in data. If center is None, it is assumed that it is
        the center of the volume with shape `data.shape[:3]`.
    roi_radii : int or array-like, (3,)
        radii of cuboid ROI
    wm_fa_thr : float
        FA threshold for WM.
    gm_fa_thr : float
        FA threshold for GM.
    csf_fa_thr : float
        FA threshold for CSF.
    gm_md_thr : float
        MD threshold for GM.
    csf_md_thr : float
        MD threshold for CSF.

    Returns
    -------
    response_wm : ndarray, (len(unique_bvals_tolerance(gtab.bvals))-1, 4)
        (`evals`, `S0`) for WM for each unique bvalues (except b0).
    response_gm : ndarray, (len(unique_bvals_tolerance(gtab.bvals))-1, 4)
        (`evals`, `S0`) for GM for each unique bvalues (except b0).
    response_csf : ndarray, (len(unique_bvals_tolerance(gtab.bvals))-1, 4)
        (`evals`, `S0`) for CSF for each unique bvalues (except b0).

    Notes
    -----
    In msmt-CSD there is an important pre-processing step: the estimation of
    every tissue's response function. In order to do this, we look for voxels
    corresponding to WM, GM and CSF. We get this information from
    mcsd.mask_for_response_msmt(), which returns masks of selected voxels
    (more details are available in the description of the function).

    With the masks, we compute the response functions by using
    mcsd.response_from_mask_msmt(), which returns the `response` for each
    tissue (more details are available in the description of the function).
    """

    list_bvals = unique_bvals_tolerance(gtab.bvals)
    if not np.all(list_bvals <= 1200):
        msg_bvals = """Some b-values are higher than 1200.
        The DTI fit might be affected. It is advised to use
        mask_for_response_msmt with bvalues lower than 1200, followed by
        response_from_mask_msmt with all bvalues to overcome this."""
        warnings.warn(msg_bvals, UserWarning)
    mask_wm, mask_gm, mask_csf = mask_for_response_msmt(gtab, data,
                                                        roi_center,
                                                        roi_radii,
                                                        wm_fa_thr,
                                                        gm_fa_thr,
                                                        csf_fa_thr,
                                                        gm_md_thr,
                                                        csf_md_thr)
    response_wm, response_gm, response_csf = response_from_mask_msmt(
                                                        gtab, data,
                                                        mask_wm, mask_gm,
                                                        mask_csf, tol)

    return response_wm, response_gm, response_csf
