import numpy as np
from dipy.reconst.cache import Cache
from dipy.reconst.multi_voxel import multi_voxel_fit
from scipy.special import hermite, gamma
from scipy.misc import factorial, factorial2
import dipy.reconst.dti as dti
from ..utils.optpkg import optional_package

cvxopt, have_cvxopt, _ = optional_package("cvxopt")


class MapmriModel(Cache):

    def __init__(self, gtab, radial_order=4, lambd=0, eap_cons = False, anisotropic_scaling = True):

        self.bvals = gtab.bvals
        self.bvecs = gtab.bvecs
        self.gtab = gtab
        self.radial_order = radial_order
        self.lambd = lambd
        self.eap_cons = eap_cons
        self.anisotropic_scaling=anisotropic_scaling
        if (gtab.big_delta is None) or (gtab.small_delta is None):
            self.tau = 1 / (4 * np.pi ** 2)
        else:
            self.tau = gtab.big_delta - gtab.small_delta / 3.0


    @multi_voxel_fit
    def fit(self, data):
        ind=self.gtab.bvals<=2000
        gtab2 = gradient_table(self.gtab.bvals[ind], self.gtab.bvecs[ind,:])
        tenmodel=dti.TensorModel(gtab2)
        tenfit=tenmodel.fit(data[...,ind])
        
        evals = tenfit.evals
        R=tenfit.evecs

        ind_evals = np.argsort(evals)[::-1]
        evals = evals[ind_evals]
        R = R[ind_evals,:]
        
        evals = np.clip(evals,1e-04,evals.max())
        
        if self.anisotropic_scaling:
            mu = np.sqrt(evals*2*self.tau)
            
        else:
            mumean=np.sqrt(evals.mean()*2*self.tau)
            mu=np.array([mumean,mumean,mumean])
        
        qvals=np.sqrt(self.gtab.bvals/self.tau) / (2 * np.pi)
        qvecs=np.dot(self.gtab.bvecs,R)
        q=qvecs*qvals[:,None]

        M = mapmri_phi_matrix(self.radial_order, mu, q.T)

        ind_mat = mapmri_index_matrix(self.radial_order)

        if self.eap_cons:
            # rmax is linear in mu with rmax \aprox 0.3 for mu = 1/(2*pi*sqrt(700))
            rmax = 0.35 * self.mu * (2 * np.pi * np.sqrt(700))
            rgrad = gen_rgrid(rmax = rmax, Nstep = 10)
            K = mapmri_psi_matrix(self.radial_order,  mu, rgrad, self.tau)

        Q = matrix(np.dot(M.T,M))
        p = matrix(-1*np.dot(M.T,data))
        
        if self.eap_cons:
            G = matrix(-1*K)
            h = matrix(np.zeros((K.shape[0])),(K.shape[0],1))
        else:
            G = None
            h = None

        A = None
        b = None

        solvers.options['show_progress'] = False
        sol = solvers.qp(Q, p, G, h, A, b)

        coef = np.array(sol['x'])[:,0]

        return MapmriFit(self, coef)

class MapmriFit():

    def __init__(self, model, mapmri_coef):
        """ Calculates diffusion properties for a single voxel

        Parameters
        ----------
        model : object,
            AnalyticalModel
        mapmri_coef : 1d ndarray,
            mapmri coefficients
        """

        self.model = model
        self._mapmri_coef = mapmri_coef
        self.gtab = model.gtab
        self.radial_order = model.radial_order
        self.mu = model.mu

    @property
    def mapmri_coeff(self):
        """The SHORE coefficients
        """
        return self._mapmri_coef

    def odf(self, sphere, smoment=0):
        r""" Calculates the real analytical odf for a given discrete sphere.

        Eq.32
        """
        I_s = self.model.cache_get('mapmri_odf_matrix', key=sphere)
        if I_s is None:
            I_s = mapmri_odf_matrix(self.radial_order,
                                   self.mu, smoment, sphere.vertices)
            self.model.cache_set('mapmri_odf_matrix', sphere, I_s)

        odf = np.dot(I_s, self._mapmri_coef)
        print(odf.shape)
        return odf


def mapmri_index_matrix(radial_order):

    index_matrix = []
    for n in range(0, radial_order + 1, 2):
        for i in range(0, n + 1):
            for j in range(0, n - i + 1):
                index_matrix.append([i, j, n - i - j])

    return np.array(index_matrix)


def mapmri_phi_1d(n, q, mu):
    """
    Eq. 4
    """

    qn = 2 * np.pi * mu * q
    H = hermite(n)(qn)
    i = np.complex(0, 1)
    f = factorial(n)

    k = i ** (-n) / np.sqrt(2 ** (n) * f)
    phi = k * np.exp(- qn ** 2 / 2) * H

    return phi


def mapmri_phi_3d(n, q, mu):
    """
    Eq. 23
    """

    if isinstance(mu, float):
        mu = (mu,  mu, mu)

    n1, n2, n3 = n
    qx, qy, qz = q
    mux, muy, muz = mu

    phi = mapmri_phi_1d
    return np.real(phi(n1, qx, mux) * phi(n2, qy, muy) * phi(n3, qz, muz))


def mapmri_psi_1d(n, x, mu):
    """
    Eq. 10
    """

    H = hermite(n)(x / mu)
    f = factorial(n)

    k = 1 / (np.sqrt(2 ** (n + 1) * np.pi * f) * mu)
    psi = k * np.exp(- x ** 2 / (2 * mu ** 2)) * H

    return psi


def mapmri_psi_3d(n, r, mu):
    """
    Eq. 22
    """

    if isinstance(mu, float):
        mu = (mu,  mu, mu)

    n1, n2, n3 = n
    x, y, z = r
    mux, muy, muz = mu

    psi = mapmri_psi_1d
    return psi(n1, x, mux) * psi(n2, y, muy) * psi(n3, z, muz)


def mapmri_phi_matrix(radial_order, mu, q_gradients):

    ind_mat = mapmri_index_matrix(radial_order)

    n_elem = ind_mat.shape[0]

    n_qgrad = q_gradients.shape[1]

    M = np.zeros((n_qgrad, n_elem))

    for j in range(n_elem):
        M[:, j] = mapmri_phi_3d(ind_mat[j], q_gradients, mu)

    return M

def mapmri_psi_matrix(radial_order, mu, rgrad, tau):

    ind_mat = mapmri_index_matrix(radial_order)

    n_elem = ind_mat.shape[0]

    n_rgrad = rgrad.shape[0]

    K = np.zeros((n_rgrad, n_elem))

    for i in range(n_rgrad):
        for j in range(n_elem):
            K[i, j] = mapmri_psi_3d(ind_mat[j], rgrad[i], mu)

    return K


def mapmri_odf_matrix(radial_order, mu, smoment, vertices):
    """
    Eq. 33 (choose ux=uy=uz)
    """

    ind_mat = mapmri_index_matrix(radial_order)

    n_vert = vertices.shape[0]

    n_elem = ind_mat.shape[0]

    odf_mat = np.zeros((n_vert, n_elem))

    rho = mu

    for i in range(n_vert):

        vx, vy, vz = vertices[i]

        for j in range(n_elem):

            n1, n2, n3 = ind_mat[j]
            f = np.sqrt(factorial(n1) * factorial(n2) * factorial(n3))

            k = mu ** (smoment) / np.sqrt(2 ** (2 - smoment) * np.pi ** 3)

            odf_mat[i, j] = k * f * _odf_cfunc(n1, n2, n3, vx, vy, vz, smoment)

    return odf_mat


def _odf_cfunc(n1, n2, n3, vx, vy, vz, smoment):
    """
    Eq. 34
    """

    f = factorial
    f2 = factorial2

    sumc = 0
    for i in range(0, n1 + 1, 2):
        for j in range(0, n2 + 1, 2):
            for k in range(0, n3 + 1, 2):

                nn = n1 + n2 + n3 - i - j - k

                gam = (-1) ** ((i + j + k) / 2) * gamma((3 + smoment + nn) / 2)

                num1 = vx ** (n1 - i)

                num2 = vy ** (n2 - j)

                num3 = vz ** (n3 - k)

                num = 2 ** (nn) * num1 * num2 * num3

                denom = f(n1 - i) * f(n2 - j) * f(
                    n3 - k) * f2(i) * f2(j) * f2(k)

                sumc += (gam * num) / denom

    return sumc

def gen_rgrid(rmax, Nstep = 10):
    rgrad = []
    # Build a regular grid of Nstep**3 points in (R^2 X R+)
    gridmax = rmax / np.sqrt(3)
    for xx in np.linspace(-gridmax,gridmax,Nstep):
        for yy in np.linspace(-gridmax,gridmax,Nstep):
            for zz in np.linspace(0,gridmax,Nstep):
                rgrad.append([xx, yy, zz])
    return np.array(rgrad)


def mapmri_e0(radial_order, coeff):

    ind_mat = mapmri_index_matrix(radial_order)

    n_elem = ind_mat.shape[0]

    s0 = 0

    for n in range(n_elem):

        n1, n2, n3 = ind_mat[n]

        if (n1 % 2 == 0) and (n2 % 2 == 0) and (n3 % 2 == 0):

            num = (np.sqrt(factorial(n1) * factorial(n2) * factorial(n3)))
        
            den = factorial2(n1) *  factorial2(n2) * factorial2(n3)
        
            s0 += (num / np.float(den))  * coeff[n]
    
    return s0 


def mapmri_evaluate_E(radial_order, coeff, qlist, mu):

    ind_mat = mapmri_index_matrix(radial_order)

    n_elem = ind_mat.shape[0]

    n_qgrad = qlist.shape[0]

    data_out = np.zeros(n_qgrad)

    for i in range(n_qgrad):
        for j in range(n_elem):
            data_out[i] += coeff[j] * mapmri_phi_3d(ind_mat[j], qlist[i], mu)

    return data_out

def mapmri_evaluate_EAP(radial_order, coeff, rlist, mu):

    ind_mat = mapmri_index_matrix(radial_order)
    
    n_elem = ind_mat.shape[0]

    n_rgrad = rlist.shape[0]

    data_out = np.zeros(n_rgrad)

    for i in range(n_rgrad):
        for j in range(n_elem):
            data_out[i] += coeff[j] * mapmri_psi_3d(ind_mat[j], rlist[i], mu)

    return data_out

