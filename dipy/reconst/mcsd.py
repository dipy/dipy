import numpy as np
from dipy.core import geometry as geo
from dipy.core.gradients import GradientTable
from dipy.data import default_sphere
from dipy.reconst import shm
from dipy.reconst.multi_voxel import multi_voxel_fit

from dipy.utils.optpkg import optional_package
cvx, have_cvxpy, _ = optional_package("cvxpy")

SH_CONST = .5 / np.sqrt(np.pi)


def multi_tissue_basis(gtab, sh_order, iso_comp):
    """
    Builds a basis for multi-shell multi-tissue CSD model.

    Parameters
    ----------
    gtab : GradientTable
    sh_order : int
    iso_comp: int
        Number of tissue compartments for running the MSMT-CSD. Minimum
        number of compartments required is 2.

    Returns
    -------
    B : ndarray
        Matrix of the spherical harmonics model used to fit the data
    m : int ``|m| <= n``
        The order of the harmonic.
    n : int ``>= 0``
        The degree of the harmonic.
    """
    if iso_comp < 2:
        msg = ("Multi-tissue CSD requires at least 2 tissue compartments")
        raise ValueError(msg)
    r, theta, phi = geo.cart2sphere(*gtab.gradients.T)
    m, n = shm.sph_harm_ind_list(sh_order)
    B = shm.real_sph_harm(m, n, theta[:, None], phi[:, None])
    B[np.ix_(gtab.b0s_mask, n > 0)] = 0.

    iso = np.empty([B.shape[0], iso_comp])
    iso[:] = SH_CONST

    B = np.concatenate([iso, B], axis=1)
    return B, m, n


class MultiShellResponse(object):

    def __init__(self, response, sh_order, shells):
        """ Estimate Multi Shell response function for multiple tissues and
        multiple shells.

        Parameters
        ----------
        response : tuple or AxSymShResponse object
            A tuple with two elements. The first is the eigen-values as an (3,)
            ndarray and the second is the signal value for the response
            function without diffusion weighting.  This is to be able to
            generate a single fiber synthetic signal.
        sh_order : int
        shells : int
            Number of shells in the data
        """
        self.response = response
        self.sh_order = sh_order
        self.n = np.arange(0, sh_order + 1, 2)
        self.m = np.zeros_like(self.n)
        self.shells = shells
        if self.iso < 1:
            raise ValueError("sh_order and shape of response do not agree")

    @property
    def iso(self):
        return self.response.shape[1] - (self.sh_order // 2) - 1


def _inflate_response(response, gtab, n, delta):
    """Used to inflate the response for the `multiplier_matrix` in the
    `MultiShellDeconvModel`.
    Parameters
    ----------
    response : tuple or AxSymShResponse object
        A tuple with two elements. The first is the eigen-values as an (3,)
        ndarray and the second is the signal value for the response
        function without diffusion weighting.  This is to be able to
        generate a single fiber synthetic signal. The response function
        will be used as deconvolution kernel ([1]_)
    gtab : GradientTable
    n : int ``>= 0``
        The degree of the harmonic.
    delta : Delta generated from `_basic_delta`
    """
    if any((n % 2) != 0) or (n.max() // 2) >= response.sh_order:
        raise ValueError("Response and n do not match")

    iso = response.iso
    n_idx = np.empty(len(n) + iso, dtype=int)
    n_idx[:iso] = np.arange(0, iso)
    n_idx[iso:] = n // 2 + iso
    diff = abs(response.shells[:, None] - gtab.bvals)
    b_idx = np.argmin(diff, axis=0)
    kernal = response.response / delta

    return kernal[np.ix_(b_idx, n_idx)]


def _basic_delta(iso, m, n, theta, phi):
    """Simple delta function
    Parameters
    ----------
    iso: int
        Number of tissue compartments for running the MSMT-CSD. Minimum
        number of compartments required is 2.
        Default: 2
    m : int ``|m| <= n``
        The order of the harmonic.
    n : int ``>= 0``
        The degree of the harmonic.
    theta : array_like
       inclination or polar angle
    phi : array_like
       azimuth angle
    """
    wm_d = shm.gen_dirac(m, n, theta, phi)
    iso_d = [SH_CONST] * iso
    return np.concatenate([iso_d, wm_d])


class MultiShellDeconvModel(shm.SphHarmModel):
    def __init__(self, gtab, response, reg_sphere=default_sphere, iso=2):
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
        response : tuple or AxSymShResponse object
            A tuple with two elements. The first is the eigen-values as an (3,)
            ndarray and the second is the signal value for the response
            function without diffusion weighting.  This is to be able to
            generate a single fiber synthetic signal. The response function
            will be used as deconvolution kernel ([1]_)
        reg_sphere : Sphere (optional)
            sphere used to build the regularization B matrix.
            Default: 'symmetric362'.
        iso: int (optional)
            Number of tissue compartments for running the MSMT-CSD. Minimum
            number of compartments required is 2.
            Default: 2

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
            msg = ("Multi-tissue CSD requires at least 2 tissue compartments")
            raise ValueError(msg)

        sh_order = response.sh_order
        super(MultiShellDeconvModel, self).__init__(gtab)
        B, m, n = multi_tissue_basis(gtab, sh_order, iso)

        delta = _basic_delta(response.iso, response.m, response.n, 0., 0.)
        self.delta = delta
        multiplier_matrix = _inflate_response(response, gtab, n, delta)

        r, theta, phi = geo.cart2sphere(*reg_sphere.vertices.T)
        odf_reg, _, _ = shm.real_sym_sh_basis(sh_order, theta, phi)
        reg = np.zeros([i + iso for i in odf_reg.shape])
        reg[:iso, :iso] = np.eye(iso)
        reg[iso:, iso:] = odf_reg

        X = B * multiplier_matrix

        self.fitter = QpFitter(X, reg)
        self.sh_order = sh_order
        self._X = X
        self.sphere = reg_sphere
        self.response = response
        self.B_dwi = B
        self.m = m
        self.n = n

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
            Default : None
        """
        if gtab is None:
            X = self._X
        else:
            iso = self.response.iso
            B, m, n = multi_tissue_basis(gtab, self.sh_order, iso)
            multiplier_matrix = _inflate_response(self.response, gtab, n,
                                                  self.delta)
            X = B * multiplier_matrix
        return np.dot(params, X.T)

    @multi_voxel_fit
    def fit(self, data):
        coeff = self.fitter(data)
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
    x = cvx.Variable(Q.shape[0])
    P = cvx.Constant(P)
    objective = cvx.Minimize(0.5 * cvx.quad_form(x, P) + Q * x)
    constraints = [G*x <= H]

    # setting up the problem
    prob = cvx.Problem(objective, constraints)
    prob.solve()
    opt = np.array(x.value).reshape((Q.shape[0],))
    return opt


class QpFitter(object):

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
