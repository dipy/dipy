import numpy as np
from dipy.reconst.cache import Cache
from dipy.reconst.multi_voxel import multi_voxel_fit
from scipy.special import hermite, gamma
from scipy.misc import factorial, factorial2
from cvxopt import matrix, solvers


class ShoreCartModel(Cache):

    def __init__(self, gtab, radial_order=6, mu=1, lambd=0):

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

        LR = self.cache_get('shore_laplace_matrix', key=self.gtab)
        if LR is None:
            LR = shore_laplace_reg_matrix(self.radial_order, self.mu)
            self.cache_set('shore_laplace_matrix', self.gtab, LR)


        pseudo_inv = np.dot(np.linalg.inv(np.dot(M.T, M) + self.lambd * LR),
                            M.T)

        coef = np.dot(pseudo_inv, data)

        return ShoreCartFit(self, coef)


    @multi_voxel_fit
    def fit_cvx(self, data):

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

        LR = self.cache_get('shore_laplace_matrix', key=self.gtab)
        if LR is None:
            LR = shore_laplace_reg_matrix(self.radial_order, self.mu)
            self.cache_set('shore_laplace_matrix', self.gtab, LR)

        # K = self.cache_get('shore_psi_matrix', key=self.OTHER_gtab)
        # if K is None:
        #     K = shore_psi_matrix(
        #         self.radial_order,  self.mu, self.OTHER_gtab, self.tau)
        #     self.cache_set('shore_psi_matrix', self.OTHER_gtab, K)


        """
        K: shore_psi_matrix for some N q-points

        min_coef 0.5*||M*coef-data||_2^2 + 0.5*lambd*||LR*coef||_2^2
        
        s.t.

        K*coef >= 0
        M["line of q=0"]*coef = 1


        recasting as QP


        min_coef 0.5 coef' * [M'*M + lambd * LR'*LR] * coef + [- M' * data]' coef

        s.t.

        -K*coef <= 0
        M["line of q=0"]*coef = 1
        """

        Q = matrix(np.dot(M.T,M) + self.lambd * np.dot(LR.T,LR))
        p = matrix(-1*np.dot(M.T,data))
        # G = matrix(-1*K)
        G = None
        # h = matrix(np.zeros((N)),(N,1))
        h = None
        A = matrix(M[0],(1,M.shape[1])) #line of M corresponding to q=0
        b = matrix(1.0)

        sol = solvers.qp(Q, p, G, h, A, b)

        coef = np.array(sol['x'])[:,0]

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
        print(odf.shape)
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

                num1 = vx ** (n1 - i)

                num2 = vy ** (n2 - j)

                num3 = vz ** (n3 - k)

                num = 2 ** (nn) * num1 * num2 * num3

                denom = f(n1 - i) * f(n2 - j) * f(
                    n3 - k) * f2(i) * f2(j) * f2(k)

                sumc += (gam * num) / denom

    return sumc


def delta(n, m):
    if n == m:
        return 1
    return 0


def shore_laplace_s(n, m, mu):
    """ S(n, m)
    """

    return (-1) ** n * delta(n, m) / (2 * np.sqrt(np.pi) * mu)


def shore_laplace_l(n, m, mu):
    """ L(m, n)
    """

    a = np.sqrt((m - 1) * m) * delta(m - 2, n)

    b = np.sqrt((n - 1) * n) * delta(n - 2, m)

    c = (2 * n + 1) * delta(m, n)

    return np.pi ** (3 / 2.) * (-1) ** (n + 1) * mu * (a + b + c)


def shore_laplace_r(n, m, mu):

    k = 2 * np.pi ** (7 / 2.) * (-1) ** (n) * mu ** 3

    a0 = 3 * (2 * n ** 2 + 2 * n + 1) * delta(n, m)

    sqmn = np.sqrt(gamma(m + 1) / gamma(n + 1))

    sqnm = 1 / sqmn

    an2 = 2 * (2 * n + 3) * sqmn * delta(m, n + 2)

    an4 = sqmn * delta(m, n + 4)

    am2 = 2 * (2 * m + 3) * sqnm * delta(m + 2, n)

    am4 = sqnm * delta(m + 4, n)

    return k * (a0 + an2 + an4 + am2 + am4)


def shore_laplace_delta(indn, indm, mu):

    n1, n2, n3 = indn
    m1, m2, m3 = indm

    L = shore_laplace_l
    R = shore_laplace_r
    S = shore_laplace_s


    delta = 0

    delta1 = (L(n2, m2, mu) * L(m3, n3, mu) + L(m2, n2, mu) * L(n3, m3, mu)) * S(n1, m1, mu)

    delta2 = (L(n1, m1, mu) * L(m3, n3, mu) + L(m1, n1, mu) * L(n3, m3, mu)) * S(n2, m2, mu)

    delta3 = (L(n1, m1, mu) * L(m2, n2, mu) + L(m1, n1, mu) * L(n2, m2, mu)) * S(n3, m3, mu)

    delta += delta1 + delta2 + delta3

    delta += S(n1, m1, mu) * S(n2, m2, mu) * R(n3, m3, mu)

    delta += S(n1, m1, mu) * R(n2, m2, mu) * S(n3, m3, mu)

    delta += R(n1, m1, mu) * S(n2, m2, mu) * S(n3, m3, mu)

    return delta


def shore_laplace_reg_matrix(radial_order, mu):

    ind_mat = shore_index_matrix(radial_order)

    n_elem = ind_mat.shape[0]

    LR = np.zeros((n_elem, n_elem))

    for i in range(n_elem):
        for j in range(n_elem):

            LR[i, j] = shore_laplace_delta(ind_mat[i], ind_mat[j], mu)

    return LR


def shore_e0(radial_order, coeff):

    ind_mat = shore_index_matrix(radial_order)

    n_elem = ind_mat.shape[0]

    s0 = 0

    for n in range(n_elem):

        n1, n2, n3 = ind_mat[n]

        if (n1 % 2 == 0) and (n2 % 2 == 0) and (n3 % 2 == 0):

            num = (np.sqrt(factorial(n1) * factorial(n2) * factorial(n3)))
        
            den = factorial2(n1) *  factorial2(n2) * factorial2(n3)
        
            s0 += (num / np.float(den))  * coeff[n]
    
    return s0 


def shore_evaluate_E(radial_order, coeff, qlist, mu):

    ind_mat = shore_index_matrix(radial_order)

    n_elem = ind_mat.shape[0]

    n_qgrad = qlist.shape[0]

    data_out = np.zeros(n_qgrad)

    for i in range(n_qgrad):
        for j in range(n_elem):
            data_out[i] += coeff[j] * shore_phi_3d(ind_mat[j], qlist[i], mu)

    return data_out

