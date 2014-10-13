import numpy as np
from dipy.reconst.cache import Cache
from dipy.reconst.multi_voxel import multi_voxel_fit
from scipy.special import hermite, gamma
from scipy.misc import factorial, factorial2


class ShoreCartModel(Cache):

    def __init__(self, gtab, radial_order=6, mu=1, lambd=None):

        self.bvals = gtab.bvals
        self.bvecs = gtab.bvecs
        self.gtab = gtab
        self.radial_order = radial_order
        self.mu = mu
        self.lambd = lambd

        if (gtab.big_delta is None) or (gtab.small_delta is None):
            self.tau = 1 / (4 * np.pi ** 2)
        else:
            self.tau = gtab.big_delta - gtab.small_delta / 3.0

    @multi_voxel_fit
    def fit(self, data):

        # Generate the SHORE basis
        M = self.cache_get('shore_phi_matrix', key=self.gtab)
        if M is None:
            M = shore_phi_matrix(
                self.radial_order,  self.mu, self.gtab, self.tau)
            self.cache_set('shore_phi_matrix', self.gtab, M)

        ind_mat = self.cache_get('shore_index_matrix', key=self.gtab)
        if ind_mat is None:
            ind_mat = shore_index_matrix(self.radial_order)
            self.cache_set('shore_index_matrix', self.gtab, ind_mat)

        pseudo_inv = np.linalg.pinv(M)
        coef = np.dot(pseudo_inv, data)
        return ShoreCartFit(self, coef)


class ShoreCartFit():

    def __init__(self, model, shore_coef):
        """ Calculates diffusion properties for a single voxel

        Parameters
        ----------
        model : object,
            AnalyticalModel
        shore_coef : 1d ndarray,
            shore coefficients
        """

        self.model = model
        self._shore_coef = shore_coef
        self.gtab = model.gtab
        self.radial_order = model.radial_order
        self.mu = model.mu

    @property
    def shore_coeff(self):
        """The SHORE coefficients
        """
        return self._shore_coef

    def odf(self, sphere, smoment=0):
        r""" Calculates the real analytical odf for a given discrete sphere.

        Eq.32
        """
        I_s = self.model.cache_get('shore_odf_matrix', key=sphere)
        if I_s is None:
            I_s = shore_odf_matrix(self.radial_order,
                                       self.mu, smoment, sphere.vertices)
            self.model.cache_set('shore_odf_matrix', sphere, I_s)

        odf = np.dot(I_s, self._shore_coef)
        return odf


def shore_index_matrix(radial_order):

    index_matrix = []
    for n in range(0, radial_order + 1, 2):
        for i in range(0, n + 1):
            for j in range(0, n - i + 1):
                index_matrix.append([i, j, n - i - j])

    return np.array(index_matrix)


def shore_phi_1d(n, q, mu):
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


def shore_phi_3d(n, q, mu):
    """
    Eq. 23
    """

    if isinstance(mu, float):
        mu = (mu,  mu, mu)

    n1, n2, n3 = n
    qx, qy, qz = q
    mux, muy, muz = mu

    phi = shore_phi_1d
    return np.real(phi(n1, qx, mux) * phi(n2, qy, muy) * phi(n3, qz, muz))


def shore_psi_1d(n, x, mu):
    """
    Eq. 10
    """

    H = hermite(n)(x / mu)
    f = factorial(n)

    k = 1 / (np.sqrt(2 ** (n + 1) * np.pi * f) * mu)
    psi = k * np.exp(- x ** 2 / (2 * mu ** 2)) * H

    return psi


def shore_psi_3d(n, r, mu):
    """
    Eq. 22
    """

    if isinstance(mu, float):
        mu = (mu,  mu, mu)

    n1, n2, n3 = n
    x, y, z = r
    mux, muy, muz = mu

    psi = shore_psi_1d
    return psi(n1, x, mux) * psi(n2, y, muy) * psi(n3, z, muz)


def shore_phi_matrix(radial_order, mu, gtab, tau):

    ind_mat = shore_index_matrix(radial_order)

    qvals = np.sqrt(gtab.bvals / (4 * np.pi ** 2 * tau))
    bvecs = gtab.bvecs

    qgradients = qvals[:, None] * bvecs

    np.savetxt('qgradients.txt', qgradients)

    n_elem = ind_mat.shape[0]

    n_qgrad = qgradients.shape[0]

    M = np.zeros((n_qgrad, n_elem))

    for i in range(n_qgrad):
        for j in range(n_elem):
            M[i, j] = shore_phi_3d(ind_mat[j], qgradients[i], mu)

    return M


def shore_odf_matrix(radial_order, mu, smoment, vertices):
    """
    Eq. 33 (choose ux=uy=uz)
    """

    ind_mat = shore_index_matrix(radial_order)

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

                num1 =  vx  ** (n1 - i)

                num2 = vy  ** (n2 - j)

                num3 = vz  ** (n3 - k)

                num = 2 ** (nn) * num1 * num2 * num3

                denom = f(n1 - i) * f(n2 - j) * f(n3 - k) * f2(i) * f2(j) * f2(k)

                sumc += (gam * num) / denom

    return sumc


def laplacian_regularization(radial_order, mu):
    pass
