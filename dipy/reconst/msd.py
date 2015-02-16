import numpy as np
import numpy.linalg as la

from dipy.core.gradients import GradientTable
from dipy.sims.voxel import single_tensor

from dipy.core.geometry import cart2sphere
from dipy.data import default_sphere
from dipy.reconst import shm
from dipy.reconst import csdeconv as csd
from dipy.reconst.multi_voxel import multi_voxel_fit


from cvxopt import matrix
from cvxopt import solvers
from cvxopt.solvers import qp
solvers.options['show_progress'] = False

csf_md=3e-3
gm_md=.76e-3
evals_d = np.array([.992, .254, .254]) * 1e-3

def sim_response(sh_order, bvals, evals=evals_d, csf_md=3e-3, gm_md=.76e-3):
    bvals = np.array(bvals, copy=True)
    evecs = np.zeros((3, 3))
    z = np.array([0, 0, 1.])
    evecs[:, 0] = z
    evecs[:2, 1:] = np.eye(2)

    n = np.arange(0, sh_order + 1, 2)
    m = np.zeros_like(n)

    big_sphere = default_sphere.subdivide()
    theta, phi = big_sphere.theta, big_sphere.phi

    B = shm.real_sph_harm(m, n, theta[:, None], phi[:, None])
    A = shm.real_sph_harm(0, 0, 0, 0)

    response = np.empty([len(bvals), len(n) + 2])
    for i, bvalue in enumerate(bvals):
        gtab = GradientTable(big_sphere.vertices * bvalue)
        wm_response = single_tensor(gtab, 1., evals, evecs, snr=None)
        response[i, 2:] = np.linalg.lstsq(B, wm_response)[0]

        response[i, 0] = np.exp(-bvalue * csf_md) / A
        response[i, 1] = np.exp(-bvalue * gm_md) / A

    return MultiShellResponse(response, sh_order, bvals)

class MultiShellResponse(object):

    def __init__(self, response, sh_order, shells):
        self.response = response
        self.sh_order = sh_order
        self.n = np.arange(0, sh_order + 1, 2)
        self.m = np.zeros_like(self.n)
        self.shells = shells
        self.n_isotripic = response.shape[1] - len(self.n)
        if self.n_isotripic < 1:
            raise ValueError("sh_order and shape of response do not agree")

    @property
    def iso(self):
        return self.response.shape[1] - (self.sh_order // 2) - 1


def closest(haystack, needle):
    diff = abs(haystack[:, None] - needle)
    return diff.argmin(axis=0)


def _inflate_response(response, gtab, n):
    if any((n % 2) != 0) or (n.max() // 2) >= response.sh_order:
        raise ValueError("Response and n do not match")

    iso = response.iso
    n_idx = np.empty(len(n) + iso, dtype=int)
    n_idx[:iso] = np.arange(0, iso)
    n_idx[iso:] = n // 2 + iso

    b_idx = closest(response.shells, gtab.bvals)

    return response.response[np.ix_(b_idx, n_idx)]

class MultiShellDeconvModel(shm.SphHarmModel):

    def __init__(self, gtab, response, reg_sphere=default_sphere, iso=2):
        """
        """
        sh_order = response.sh_order
        super(MultiShellDeconvModel, self).__init__(gtab)
        B, m, n = csd.multi_tissue_basis(gtab, sh_order, iso)
        multiplier_matrix = _inflate_response(response, gtab, n)

        r, theta, phi = cart2sphere(reg_sphere.x, reg_sphere.y, reg_sphere.z)
        odf_reg, _, _ = shm.real_sym_sh_basis(sh_order, theta, phi)
        reg = np.zeros([i + iso for i in odf_reg.shape])
        reg[:iso, :iso] = np.eye(iso) * 1000.
        reg[iso:, iso:] = odf_reg

        X = B * multiplier_matrix

        self.fitter = QpFitter(X, reg)
        self.sh_order = sh_order
        self._X = X
        self.sphere = reg_sphere
        self.response = response
        self.B_dwi = B

    def predict(self, params, gtab=None, S0=None):
        if gtab is None:
            X = self._X
        else:
            iso = self.response.iso
            B, m, n = csd.multi_tissue_basis(gtab, self.sh_order, iso)
            multiplier_matrix = _inflate_response(self.response, gtab, n)
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
        return self._shm_coef[..., :tissue_classes]


def _rank(A, tol=1e-8):
    s = la.svd(A, False, False)
    threshold = (s[0] * tol)
    rnk = (s > threshold).sum()
    return rnk


class QpFitter(object):

    def _lstsq_initial(self, z):
        fodf_sh = csd._solve_cholesky(self._P, z)
        s = np.dot(self._reg, fodf_sh)
        init = {'x':matrix(fodf_sh),
                's':matrix(s.clip(1e-10))}
        return init

    def __init__(self, X, reg):
        self._P = P = np.dot(X.T, X)
        self._X = X

        # No super res for now.
        assert _rank(P) == P.shape[0]

        self._reg = reg
        # self._P_init = np.dot(X[:, :N].T, X[:, :N])

        # Make cvxopt matrix types for later re-use.
        self._P_mat = matrix(P)
        self._reg_mat = matrix(-reg)
        self._h_mat = matrix(0., (reg.shape[0], 1))

    def __call__(self, signal):
        z = np.dot(self._X.T, signal)
        init = self._lstsq_initial(z)

        z_mat = matrix(-z)
        r = qp(self._P_mat, z_mat, self._reg_mat, self._h_mat, initvals=init)
        fodf_sh = r['x']
        fodf_sh = np.array(fodf_sh)[:, 0]
        return fodf_sh

