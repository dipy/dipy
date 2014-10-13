import numpy as np
from dipy.reconst.cache import Cache
from dipy.reconst.multi_voxel import multi_voxel_fit
from scipy.special import hermite
from scipy.misc import factorial


class ShoreCartModel(Cache):

    def __init__(self, gtab, radial_order=6, mu=1, lamd=None):

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
        M = self.cache_get('shore_matrix', key=self.gtab)
        if M is None:
            M = shore_matrix(self.radial_order,  self.mu, self.gtab, self.tau)
            self.cache_set('shore_matrix', self.gtab, M)


def shore_index_matrix(radial_order):

    index_matrix = []
    for n in range(0, radial_order + 1, 2):
        for i in range(0, n + 1):
            for j in range(0, n - i + 1):
                #print(i, j, n - i - j)
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

    k = 1/ (np.sqrt(2 ** (n+1) * np.pi * f) * mu)
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



def shore_matrix(radial_order, mu, gtab, tau):

    pass
