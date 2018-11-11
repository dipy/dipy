import numpy as np
import numpy.linalg as la
import cvxpy as cvx

from dipy.core import geometry as geo
from dipy.data import default_sphere
from dipy.reconst import shm
from dipy.reconst import csdeconv as csd
from dipy.reconst.multi_voxel import multi_voxel_fit

from dipy.utils.optpkg import optional_package

have_cvxpy = optional_package("cvxpy")

sh_const = .5 / np.sqrt(np.pi)


def multi_tissue_basis(gtab, sh_order, iso_comp):
    """Builds a basis for multi-shell CSD model"""
    if iso_comp < 1:
        msg = ("Multi-tissue CSD requires at least 2 tissue compartments")
        raise ValueError(msg)
    r, theta, phi = geo.cart2sphere(*gtab.gradients.T)
    m, n = shm.sph_harm_ind_list(sh_order)
    B = shm.real_sph_harm(m, n, theta[:, None], phi[:, None])
    B[np.ix_(gtab.b0s_mask, n > 0)] = 0.

    iso = np.empty([B.shape[0], iso_comp])
    iso[:] = sh_const

    B = np.concatenate([iso, B], axis=1)
    return B, m, n


class MultiShellResponse(object):

    def __init__(self, response, sh_order, shells):
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


def closest(haystack, needle):
    diff = abs(haystack[:, None] - needle)
    return diff.argmin(axis=0)


def _inflate_response(response, gtab, n, delta):
    if any((n % 2) != 0) or (n.max() // 2) >= response.sh_order:
        raise ValueError("Response and n do not match")

    iso = response.iso
    n_idx = np.empty(len(n) + iso, dtype=int)
    n_idx[:iso] = np.arange(0, iso)
    n_idx[iso:] = n // 2 + iso

    b_idx = closest(response.shells, gtab.bvals)
    kernal = response.response / delta

    return kernal[np.ix_(b_idx, n_idx)]


def _basic_delta(iso, m, n, theta, phi):
    """Simple delta function"""
    wm_d = shm.gen_dirac(m, n, theta, phi)
    iso_d = [sh_const] * iso
    return np.concatenate([iso_d, wm_d])


def _pos_constrained_delta(iso, m, n, theta, phi, reg_sphere=default_sphere):
    """Delta function optimized to avoid negative lobes."""

    x, y, z = geo.sphere2cart(1., theta, phi)

    # Realign reg_sphere so that the first vertex is aligned with delta
    # orientation (theta, phi).
    M = geo.vec2vec_rotmat(reg_sphere.vertices[0], [x, y, z])
    new_vertices = np.dot(reg_sphere.vertices, M.T)
    _, t, p = geo.cart2sphere(*new_vertices.T)

    B = shm.real_sph_harm(m, n, t[:, None], p[:, None])
    G_ = np.ascontiguousarray(B[:, n != 0])
    # c_ samples the delta function at the delta orientation.
    c_ = G_[0][:, None]
    print("G", G_.shape)
    print("c", c_.shape)
    a_, b_ = G_.shape

    c_int = cvx.Parameter((c_.shape[0], 1))
    c_int.value = -c_
    G = cvx.Parameter((G_.shape[0], 4))
    G.value = -G_
    h_ = cvx.Parameter((a_, 1))
    h_int = np.full((a_, 1), sh_const ** 2)
    h_.value = h_int
    print("h", h_int.shape)

    # n == 0 is set to sh_const to ensure a normalized delta function.
    # n > 0 values are optimized so that delta > 0 on all points of the sphere
    # and delta(theta, phi) is maximized.
#    r = cvx.solvers.lp(c, G, h_)
    lp_prob = cvx.Problem(cvx.Maximize(cvx.sum(c_)), [G, h_])
    r = lp_prob.solve(solver=cvx.GLPK)  # solver = cvx.GLPK_MI
    x = np.asarray(r['x'])[:, 0]
    out = np.zeros(B.shape[1])
    out[n == 0] = sh_const
    out[n != 0] = x

    iso_d = [sh_const] * iso
    return np.concatenate([iso_d, out])


delta_functions = {"basic": _basic_delta,
                   "positivity_constrained": _pos_constrained_delta}


class MultiShellDeconvModel(shm.SphHarmModel):

    def __init__(self, gtab, response, reg_sphere=default_sphere, iso=2,
                 delta_form='basic'):
        """
        """
        sh_order = response.sh_order
        super(MultiShellDeconvModel, self).__init__(gtab)
        B, m, n = multi_tissue_basis(gtab, sh_order, iso)

        delta_f = delta_functions[delta_form]
        delta = delta_f(response.iso, response.m, response.n, 0., 0.)
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
        self._shm_coef = coeff
        self.mask = mask
        self.model = model

    @property
    def shm_coeff(self):
        return self._shm_coef[..., self.model.response.iso:]

    @property
    def volume_fractions(self):
        tissue_classes = self.model.response.iso + 1
        return self._shm_coef[..., :tissue_classes] / sh_const


def _rank(A, tol=1e-8):
    s = la.svd(A, False, False)
    threshold = (s[0] * tol)
    rnk = (s > threshold).sum()
    return rnk


class QpFitter(object):

    def _lstsq_initial(self, z):
        fodf_sh = csd._solve_cholesky(self._P, z)
        s = np.dot(self._reg, fodf_sh)
        fodf_sh_ = cvx.Variable(fodf_sh.shape)
        fodf_sh_.value = fodf_sh
        s_ = cvx.Variable(s.shape)
        s_.value = np.clip(s, 1e-10, None)
        init = {'x': fodf_sh_,
                's': s_}
        return init

    def __init__(self, X, reg):
        self._P = P = np.dot(X.T, X)
        self._X = X

        # No super res for now.
        assert _rank(P) == P.shape[0]

        self._reg = reg
        # self._P_init = np.dot(X[:, :N].T, X[:, :N])

        # Make cvxpy matrix types for later re-use.
        self._P_mat = cvx.Parameter(P.shape)
        self._P_mat.value = P
        self._x_mat = cvx.Variable(P.shape[0])
        self._reg_mat = cvx.Parameter(reg.shape)
        self._reg_mat.value = -reg
        self._h_mat = cvx.Parameter(reg.shape)
        self._h_mat.value = np.zeros(reg.shape)

    def __call__(self, signal):
        z = np.dot(self._X.T, signal)
        init = self._lstsq_initial(z)
        z_mat = cvx.Variable(z.shape)
#        qp_obj = cvx.quad_form(self._x_mat, self._P_mat)  # needs change
#        objMin = cvx.Minimize(qp_obj + (z_mat.T * self._x_mat))

        objMin = cvx.quad_form(self._x_mat, self._P_mat) + \
                              (z_mat.T * self._x_mat)

        constraints = [self._reg_mat * self._x_mat >= 0]
        prob = cvx.Problem(cvx.Minimize(objMin, constraints))
        r = prob.solve(solver=cvx.GUROBI, initvals=init)
        fodf_sh = r['x']
        fodf_sh = np.array(fodf_sh)[:, 0]
        return fodf_sh
