import numpy as np
from dipy.reconst.multi_voxel import multi_voxel_fit
from scipy.special import hermite, gamma
from scipy.misc import factorial, factorial2
import dipy.reconst.dti as dti
from warnings import warn
from dipy.core.gradients import gradient_table
from ..utils.optpkg import optional_package

cvxopt, have_cvxopt, _ = optional_package("cvxopt")


class MapmriModel():

    def __init__(self, gtab, radial_order=4, lambd=0,  eap_cons = False, anisotropic_scaling = True):

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

        # This is a simple empirical regularization, to be replaced
        I = np.diag(ind_mat.sum(1)**2)

        if self.eap_cons:
            if not have_cvxopt:
                raise ValueError(
                    'CVXOPT package needed to enforce constraints')
            import cvxopt.solvers
            rmax = 2* np.sqrt(10 * evals.max()*self.tau)
            r_index, r_grad = create_rspace(11, rmax)
            K = mapmri_psi_matrix(self.radial_order,  mu, r_grad[0:len(r_grad)/2,:], self.tau)

            Q = cvxopt.matrix(np.dot(M.T,M)+ self.lambd * I)
            p = cvxopt.matrix(-1*np.dot(M.T,data))
            G = cvxopt.matrix(-1*K)
            h = cvxopt.matrix(np.zeros((K.shape[0])),(K.shape[0],1))
            cvxopt.solvers.options['show_progress'] = False
            sol = cvxopt.solvers.qp(Q, p, G, h)
            if sol['status'] != 'optimal':
                warn('Optimization did not find a solution')

            coef = np.array(sol['x'])[:,0]
        else:
            pseudoInv = np.dot(np.linalg.inv(np.dot(M.T, M) + self.lambd * I), M.T)
            coef = np.dot(pseudoInv, data)

        Bm=Bmat(ind_mat)

        E0 = 0
        for i in range(ind_mat.shape[0]):
            E0 = E0 + coef[i] * Bm[i]
        coef = coef / E0

        return MapmriFit(self, coef, mu, R, ind_mat)

class MapmriFit():

    def __init__(self, model, mapmri_coef, mu, R, ind_mat):
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
        self.mu = mu
        self.R = R
        self.ind_mat = ind_mat

    @property
    def mapmri_mu(self):
        """The SHORE coefficients
        """
        return self.mu
    @property
    def mapmri_R(self):
        """The SHORE coefficients
        """
        return self.R    
    @property
    def mapmri_coeff(self):
        """The SHORE coefficients
        """
        return self._mapmri_coef

    def odf(self, sphere, s=0):
        r""" Calculates the real analytical odf for a given discrete sphere.

        Eq.32
        """
        v_ = sphere.vertices
        v = np.dot(v_,self.R)
        I_s = mapmri_odf_matrix(self.radial_order,self.mu, s, v)
        odf = np.dot(I_s, self._mapmri_coef)
        return np.clip(odf,0,odf.max())



def mapmri_index_matrix(radial_order):

    index_matrix = []
    for n in range(0, radial_order + 1, 2):
        for i in range(0, n + 1):
            for j in range(0, n - i + 1):
                index_matrix.append([n - i - j, j, i])

    return np.array(index_matrix)

def Bmat(ind_mat):
    B = np.zeros(ind_mat.shape[0])
    for i in range (ind_mat.shape[0]):
        n1, n2, n3 = ind_mat[i]
        K = int(not(n1%2) and not(n2%2) and not(n3%2))
        B[i] = K * np.sqrt(factorial(n1)*factorial(n2)*factorial(n3)) /(factorial2(n1)*factorial2(n2)*factorial2(n3))
        
    return B

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

    n1, n2, n3 = n
    qx, qy, qz = q
    mux, muy, muz = mu

    phi = mapmri_phi_1d
    return np.real(phi(n1, qx, mux) * phi(n2, qy, muy) * phi(n3, qz, muz))

def mapmri_phi_matrix(radial_order, mu, q_gradients):

    ind_mat = mapmri_index_matrix(radial_order)

    n_elem = ind_mat.shape[0]

    n_qgrad = q_gradients.shape[1]

    M = np.zeros((n_qgrad, n_elem))

    for j in range(n_elem):
        M[:, j] = mapmri_phi_3d(ind_mat[j], q_gradients, mu)

    return M

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

    n1, n2, n3 = n
    x, y, z = r.T
    mux, muy, muz = mu

    psi = mapmri_psi_1d
    return psi(n1, x, mux) * psi(n2, y, muy) * psi(n3, z, muz)


def mapmri_psi_matrix(radial_order, mu, rgrad, tau):

    ind_mat = mapmri_index_matrix(radial_order)

    n_elem = ind_mat.shape[0]

    n_rgrad = rgrad.shape[0]

    K = np.zeros((n_rgrad, n_elem))

    for j in range(n_elem):
        K[:, j] = mapmri_psi_3d(ind_mat[j], rgrad, mu)

    return K


def mapmri_odf_matrix(radial_order, mu, s, vertices):
    """
    Eq. 33 (choose ux=uy=uz)
    """

    ind_mat = mapmri_index_matrix(radial_order)

    n_vert = vertices.shape[0]

    n_elem = ind_mat.shape[0]

    odf_mat = np.zeros((n_vert, n_elem))

    rho = mu
    mux,muy,muz = mu
    rho=1.0/np.sqrt((vertices[:,0]/mux)**2 + (vertices[:,1]/muy)**2 + (vertices[:,2]/muz)**2)
    alpha = 2 * rho * (vertices[:,0]/mux)
    beta = 2 * rho * (vertices[:,1]/muy)
    gamma = 2 * rho * (vertices[:,2]/muz)
    const= rho ** (3 + s) / np.sqrt(2 ** (2 - s) * np.pi ** 3 * (mux ** 2 * muy ** 2 * muz ** 2))

    for j in range(n_elem):
        n1, n2, n3 = ind_mat[j]
        f = np.sqrt(factorial(n1) * factorial(n2) * factorial(n3))
        odf_mat[:, j] = const * f * _odf_cfunc(n1, n2, n3, alpha, beta, gamma, s)

    return odf_mat


def _odf_cfunc(n1, n2, n3, a, b, g, s):
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

                gam = (-1) ** ((i + j + k) / 2.0) * gamma((3 + s + nn) / 2.0)

                num1 = a ** (n1 - i)

                num2 = b ** (n2 - j)

                num3 = g ** (n3 - k)

                num = gam * num1 * num2 * num3

                denom = f(n1 - i) * f(n2 - j) * f(
                    n3 - k) * f2(i) * f2(j) * f2(k)

                sumc += num / denom

    return sumc



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

def create_rspace(gridsize, radius_max):
    """ Create the real space table, that contains the points in which
        to compute the pdf.

    Parameters
    ----------
    gridsize : unsigned int
        dimension of the propagator grid
    radius_max : float
        maximal radius in which compute the propagator

    Returns
    -------
    vecs : array, shape (N,3)
        positions of the pdf points in a 3D matrix

    tab : array, shape (N,3)
        real space points in which calculates the pdf
    """

    radius = gridsize // 2
    vecs = []
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            for k in range(-radius, radius + 1):
                vecs.append([i, j, k])

    vecs = np.array(vecs, dtype=np.float32)
    tab = vecs / radius
    tab = tab * radius_max
    vecs = vecs + radius

    return vecs, tab
