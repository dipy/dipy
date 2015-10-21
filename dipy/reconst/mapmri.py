import numpy as np
from dipy.reconst.multi_voxel import multi_voxel_fit
from dipy.reconst.base import ReconstModel, ReconstFit
from scipy.special import hermite, gamma
from scipy.misc import factorial, factorial2
import dipy.reconst.dti as dti
from warnings import warn
from dipy.core.gradients import gradient_table
from ..utils.optpkg import optional_package

cvxopt, have_cvxopt, _ = optional_package("cvxopt")


class MapmriModel(ReconstModel):

    r"""Mean Apparent Propagator MRI (MAPMRI) [1]_ of the diffusion signal.

    The main idea is to model the diffusion signal as a linear combination of
    the continuous functions presented in [2]_ but extending it in three
    dimensions.
    The main difference with the SHORE proposed in [3]_ is that MAPMRI 3D
    extension is provided using a set of three basis functions for the radial
    part, one for the signal along x, one for y and one for z, while [3]_ 
    uses one basis function to model the radial part and real Spherical
    Harmonics to model the angular part.
    From the MAPMRI coefficients is possible to use the analytical formulae 
    to estimate the ODF.

    References
    ----------
    .. [1] Ozarslan E. et. al, "Mean apparent propagator (MAP) MRI: A novel
           diffusion imaging method for mapping tissue microstructure",
           NeuroImage, 2013.

    .. [2] Ozarslan E. et. al, "Simple harmonic oscillator based reconstruction
           and estimation for one-dimensional q-space magnetic resonance
           1D-SHORE)", eapoc Intl Soc Mag Reson Med, vol. 16, p. 35., 2008.

    .. [3] Merlet S. et. al, "Continuous diffusion signal, EAP and ODF
           estimation via Compressive Sensing in diffusion MRI", Medical
           Image Analysis, 2013.
    """

    def __init__(self,
                 gtab,
                 radial_order=4,
                 lambd=1e-16,
                 eap_cons=False,
                 anisotropic_scaling=True,
                 eigenvalue_threshold=1e-04,
                 bmax_threshold=2000):
        r""" Analytical and continuous modeling of the diffusion signal with
        respect to the MAPMRI basis [1]_.

        The main idea is to model the diffusion signal as a linear combination of
        the continuous functions presented in [2]_ but extending it in three
        dimensions.

        The main difference with the SHORE proposed in [3]_ is that MAPMRI 3D
        extension is provided using a set of three basis functions for the radial
        part, one for the signal along x, one for y and one for z, while [3]_ 
        uses one basis function to model the radial part and real Spherical
        Harmonics to model the angular part.
        From the MAPMRI coefficients is possible to use the analytical formulae 
        to estimate the ODF.


        Parameters
        ----------
        gtab : GradientTable,
            gradient directions and bvalues container class
        radial_order : unsigned int,
            an even integer that represent the order of the basis
        lambd : float,
            radial regularisation constant
        eap_cons : bool,
            Constrain the propagator to be positive.
        anisotropic_scaling : bool,
            If false, force the basis function to be identical in the three
            dimensions (SHORE like).
        eigenvalue_threshold : float,
            set the minimum of the tensor eigenvalues in order to avoid 
            stability problem
        bmax_threshold : float,
            set the maximum b-value for the tensor estimation

        References
        ----------
        .. [1] Ozarslan E. et. al, "Mean apparent propagator (MAP) MRI: A novel
               diffusion imaging method for mapping tissue microstructure",
               NeuroImage, 2013.

        .. [2] Ozarslan E. et. al, "Simple harmonic oscillator based reconstruction
               and estimation for one-dimensional q-space magnetic resonance
               1D-SHORE)", eapoc Intl Soc Mag Reson Med, vol. 16, p. 35., 2008.

        .. [3] Ozarslan E. et. al, "Simple harmonic oscillator based reconstruction
               and estimation for three-dimensional q-space mri", ISMRM 2009.

        Examples
        --------
        In this example, where the data, gradient table and sphere tessellation
        used for reconstruction are provided, we model the diffusion signal
        with respect to the MAPMRI model and compute the analytical ODF.

        >>> from dipy.core.gradients import gradient_table
        >>> from dipy.data import dsi_voxels, get_sphere
        >>> data, gtab = dsi_voxels()
        >>> sphere = get_sphere('symmetric724')
        >>> from dipy.sims.voxel import SticksAndBall
        >>> data, golden_directions = SticksAndBall(gtab, d=0.0015, S0=1, angles=[(0, 0), (90, 0)], fractions=[50, 50], snr=None)
        >>> from dipy.reconst.mapmri import MapmriModel
        >>> radial_order = 4
        >>> map_model = MapmriModel(gtab, radial_order=radial_order)
        >>> mapfit = map_model.fit(data)
        >>> odf= mapfit.odf(sphere)
        """

        self.bvals = gtab.bvals
        self.bvecs = gtab.bvecs
        self.gtab = gtab
        self.radial_order = radial_order
        self.lambd = lambd
        self.eap_cons = eap_cons
        if self.eap_cons:
            if not have_cvxopt:
                raise ValueError(
                    'CVXOPT package needed to enforce constraints')
            import cvxopt.solvers

        self.anisotropic_scaling = anisotropic_scaling
        if (gtab.big_delta is None) or (gtab.small_delta is None):
            self.tau = 1 / (4 * np.pi ** 2)
        else:
            self.tau = gtab.big_delta - gtab.small_delta / 3.0
        self.eigenvalue_threshold = eigenvalue_threshold

        self.ind = self.gtab.bvals <= bmax_threshold
        gtab_dti = gradient_table(
            self.gtab.bvals[self.ind], self.gtab.bvecs[self.ind, :])
        self.tenmodel = dti.TensorModel(gtab_dti)
        self.ind_mat = mapmri_index_matrix(self.radial_order)
        self.Bm = b_mat(self.ind_mat)

    @multi_voxel_fit
    def fit(self, data):

        tenfit = self.tenmodel.fit(data[self.ind])
        evals = tenfit.evals
        R = tenfit.evecs
        evals = np.clip(evals, self.eigenvalue_threshold, evals.max())
        if self.anisotropic_scaling:
            mu = np.sqrt(evals * 2 * self.tau)

        else:
            mumean = np.sqrt(evals.mean() * 2 * self.tau)
            mu = np.array([mumean, mumean, mumean])

        qvals = np.sqrt(self.gtab.bvals / self.tau) / (2 * np.pi)
        qvecs = np.dot(self.gtab.bvecs, R)
        q = qvecs * qvals[:, None]
        M = mapmri_phi_matrix(self.radial_order, mu, q.T)
        # This is a simple empirical regularization, to be replaced
        I = np.diag(self.ind_mat.sum(1) ** 2)
        if self.eap_cons:
            if not have_cvxopt:
                raise ValueError(
                    'CVXOPT package needed to enforce constraints')
            w_s = "The implementation of MAPMRI depends on CVXOPT "
            w_s += " (http://cvxopt.org/). This software is licensed "
            w_s += "under the GPL (see: http://cvxopt.org/copyright.html) "
            w_s += " and you may be subject to this license when using MAPMRI."
            warn(w_s)
            import cvxopt.solvers
            rmax = 2 * np.sqrt(10 * evals.max() * self.tau)
            r_index, r_grad = create_rspace(11, rmax)
            K = mapmri_psi_matrix(
                self.radial_order,  mu, r_grad[0:len(r_grad) / 2, :])

            Q = cvxopt.matrix(np.dot(M.T, M) + self.lambd * I)
            p = cvxopt.matrix(-1 * np.dot(M.T, data))
            G = cvxopt.matrix(-1 * K)
            h = cvxopt.matrix(np.zeros((K.shape[0])), (K.shape[0], 1))
            cvxopt.solvers.options['show_progress'] = False
            sol = cvxopt.solvers.qp(Q, p, G, h)
            if sol['status'] != 'optimal':
                warn('Optimization did not find a solution')

            coef = np.array(sol['x'])[:, 0]
        else:
            pseudoInv = np.dot(
                np.linalg.inv(np.dot(M.T, M) + self.lambd * I), M.T)
            coef = np.dot(pseudoInv, data)

        E0 = 0
        for i in range(self.ind_mat.shape[0]):
            E0 = E0 + coef[i] * self.Bm[i]
        coef = coef / E0

        return MapmriFit(self, coef, mu, R, self.ind_mat)


class MapmriFit(ReconstFit):

    def __init__(self, model, mapmri_coef, mu, R, ind_mat):
        """ Calculates diffusion properties for a single voxel

        Parameters
        ----------
        model : object,
            AnalyticalModel
        mapmri_coef : 1d ndarray,
            mapmri coefficients
        mu : array, shape (3,)
            scale parameters vector for x, y and z
        R : array, shape (3,3)
            rotation matrix
        ind_mat : array, shape (N,3)
            indices of the basis for x, y and z
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
        """The MAPMRI scale factors
        """
        return self.mu

    @property
    def mapmri_R(self):
        """The MAPMRI rotation matrix
        """
        return self.R

    @property
    def mapmri_coeff(self):
        """The MAPMRI coefficients
        """
        return self._mapmri_coef

    def odf(self, sphere, s=0):
        r""" Calculates the analytical Orientation Distribution Function (ODF)
        from the signal [1]_ Eq. 32.

        Parameters
        ----------
        s : unsigned int
            radial moment of the ODF

        References
        ----------
        .. [1] Ozarslan E. et. al, "Mean apparent propagator (MAP) MRI: A novel
        diffusion imaging method for mapping tissue microstructure",
        NeuroImage, 2013.
        """

        v_ = sphere.vertices
        v = np.dot(v_, self.R)
        I_s = mapmri_odf_matrix(self.radial_order, self.mu, s, v)
        odf = np.dot(I_s, self._mapmri_coef)
        return odf

    def rtpp(self):
        r""" Calculates the analytical return to the plane probability (RTPP)
        [1]_.

        References
        ----------
        .. [1] Ozarslan E. et. al, "Mean apparent propagator (MAP) MRI: A novel
        diffusion imaging method for mapping tissue microstructure",
        NeuroImage, 2013.
        """
        Bm = self.model.Bm
        rtpp = 0
        const = 1 / (np.sqrt(2 * np.pi) * self.mu[0])
        for i in range(self.ind_mat.shape[0]):
            if Bm[i] > 0.0:
                rtpp += (-1.0) ** (self.ind_mat[i, 0] /
                                   2.0) * self._mapmri_coef[i] * Bm[i]
        return const * rtpp

    def rtap(self):
        r""" Calculates the analytical return to the axis probability (RTAP)
        [1]_.

        References
        ----------
        .. [1] Ozarslan E. et. al, "Mean apparent propagator (MAP) MRI: A novel
        diffusion imaging method for mapping tissue microstructure",
        NeuroImage, 2013.
        """
        Bm = self.model.Bm
        rtap = 0
        const = 1 / (2 * np.pi * self.mu[1] * self.mu[2])
        for i in range(self.ind_mat.shape[0]):
            if Bm[i] > 0.0:
                rtap += (-1.0) ** (
                    (self.ind_mat[i, 1] + self.ind_mat[i, 2]) / 2.0) * self._mapmri_coef[i] * Bm[i]
        return const * rtap

    def rtop(self):
        r""" Calculates the analytical return to the origin probability (RTOP)
        [1]_.

        References
        ----------
        .. [1] Ozarslan E. et. al, "Mean apparent propagator (MAP) MRI: A novel
        diffusion imaging method for mapping tissue microstructure",
        NeuroImage, 2013.
        """
        Bm = self.model.Bm
        rtop = 0
        const = 1 / \
            np.sqrt(
                8 * np.pi ** 3 * (self.mu[0] ** 2 * self.mu[1] ** 2 * self.mu[2] ** 2))
        for i in range(self.ind_mat.shape[0]):
            if Bm[i] > 0.0:
                rtop += (-1.0) ** ((self.ind_mat[i, 0] + self.ind_mat[i, 1] + self.ind_mat[
                    i, 2]) / 2.0) * self._mapmri_coef[i] * Bm[i]
        return const * rtop

    def predict(self, gtab, S0=1.0):
        """
        Predict a signal for this MapmriModel class instance given a gradient
        table.

        Parameters
        ----------
        gtab : GradientTable,
            gradient directions and bvalues container class

        S0 : float or ndarray
            The non diffusion-weighted signal in every voxel, or across all
            voxels. Default: 1
        """
        if (gtab.big_delta is None) or (gtab.small_delta is None):
            tau = 1 / (4 * np.pi ** 2)
        else:
            tau = gtab.big_delta - gtab.small_delta / 3.0

        qvals = np.sqrt(gtab.bvals / tau) / (2 * np.pi)
        qvecs = np.dot(gtab.bvecs, self.R)
        q = qvecs * qvals[:, None]
        s_mat = mapmri_phi_matrix(self.radial_order, self.mu, q.T)
        S_reconst = S0 * np.dot(s_mat, self._mapmri_coef)

        return S_reconst


def mapmri_index_matrix(radial_order):
    r""" Calculates the indices for the MAPMRI [1]_ basis in x, y and z.

    Parameters
    ----------
    radial_order : unsigned int
        radial order of MAPMRI basis

    Returns
    -------
    index_matrix : array, shape (N,3)
        ordering of the basis in x, y, z

    References
    ----------
    .. [1] Ozarslan E. et. al, "Mean apparent propagator (MAP) MRI: A novel
    diffusion imaging method for mapping tissue microstructure",
    NeuroImage, 2013.
    """
    index_matrix = []
    for n in range(0, radial_order + 1, 2):
        for i in range(0, n + 1):
            for j in range(0, n - i + 1):
                index_matrix.append([n - i - j, j, i])

    return np.array(index_matrix)


def b_mat(ind_mat):
    r""" Calculates the B coefficients from [1]_ Eq. 27.

    Parameters
    ----------
    index_matrix : array, shape (N,3)
        ordering of the basis in x, y, z

    Returns
    -------
    B : array, shape (N,)
        B coefficients for the basis

    References
    ----------
    .. [1] Ozarslan E. et. al, "Mean apparent propagator (MAP) MRI: A novel
    diffusion imaging method for mapping tissue microstructure",
    NeuroImage, 2013.
    """

    B = np.zeros(ind_mat.shape[0])
    for i in range(ind_mat.shape[0]):
        n1, n2, n3 = ind_mat[i]
        K = int(not(n1 % 2) and not(n2 % 2) and not(n3 % 2))
        B[i] = K * np.sqrt(factorial(n1) * factorial(n2) * factorial(n3)
                           ) / (factorial2(n1) * factorial2(n2) * factorial2(n3))

    return B


def mapmri_phi_1d(n, q, mu):
    r""" One dimensional MAPMRI basis function from [1]_ Eq. 4.

    Parameters
    -------
    n : unsigned int
        order of the basis
    q : array, shape (N,)
        points in the q-space in which evaluate the basis
    mu : float
        scale factor of the basis

    References
    ----------
    .. [1] Ozarslan E. et. al, "Mean apparent propagator (MAP) MRI: A novel
    diffusion imaging method for mapping tissue microstructure",
    NeuroImage, 2013.
    """

    qn = 2 * np.pi * mu * q
    H = hermite(n)(qn)
    i = np.complex(0, 1)
    f = factorial(n)

    k = i ** (-n) / np.sqrt(2 ** (n) * f)
    phi = k * np.exp(- qn ** 2 / 2) * H

    return phi


def mapmri_phi_3d(n, q, mu):
    r""" Three dimensional MAPMRI basis function from [1]_ Eq. 23.

    Parameters
    ----------
    n : array, shape (3,)
        order of the basis function for x, y, z
    q : array, shape (N,3)
        points in the q-space in which evaluate the basis
    mu : array, shape (3,)
        scale factors of the basis for x, y, z

    References
    ----------
    .. [1] Ozarslan E. et. al, "Mean apparent propagator (MAP) MRI: A novel
    diffusion imaging method for mapping tissue microstructure",
    NeuroImage, 2013.
    """

    n1, n2, n3 = n
    qx, qy, qz = q
    mux, muy, muz = mu
    phi = mapmri_phi_1d
    return np.real(phi(n1, qx, mux) * phi(n2, qy, muy) * phi(n3, qz, muz))


def mapmri_phi_matrix(radial_order, mu, q_gradients):
    r"""Compute the MAPMRI phi matrix for the signal [1]_

    Parameters
    ----------
    radial_order : unsigned int,
        an even integer that represent the order of the basis
    mu : array, shape (3,)
        scale factors of the basis for x, y, z
    q_gradients : array, shape (N,3)
        points in the q-space in which evaluate the basis

    References
    ----------
    .. [1] Ozarslan E. et. al, "Mean apparent propagator (MAP) MRI: A novel
    diffusion imaging method for mapping tissue microstructure",
    NeuroImage, 2013.
    """

    ind_mat = mapmri_index_matrix(radial_order)
    n_elem = ind_mat.shape[0]
    n_qgrad = q_gradients.shape[1]
    M = np.zeros((n_qgrad, n_elem))
    for j in range(n_elem):
        M[:, j] = mapmri_phi_3d(ind_mat[j], q_gradients, mu)

    return M


def mapmri_psi_1d(n, x, mu):
    r""" One dimensional MAPMRI propagator basis function from [1]_ Eq. 10.

    Parameters
    ----------
    n : unsigned int
        order of the basis
    x : array, shape (N,)
        points in the r-space in which evaluate the basis
    mu : float
        scale factor of the basis

    References
    ----------
    .. [1] Ozarslan E. et. al, "Mean apparent propagator (MAP) MRI: A novel
    diffusion imaging method for mapping tissue microstructure",
    NeuroImage, 2013.
    """

    H = hermite(n)(x / mu)
    f = factorial(n)
    k = 1 / (np.sqrt(2 ** (n + 1) * np.pi * f) * mu)
    psi = k * np.exp(- x ** 2 / (2 * mu ** 2)) * H

    return psi


def mapmri_psi_3d(n, r, mu):
    r""" Three dimensional MAPMRI propagator basis function from [1]_ Eq. 22.

    Parameters
    ----------
    n : array, shape (3,)
        order of the basis function for x, y, z
    q : array, shape (N,3)
        points in the q-space in which evaluate the basis
    mu : array, shape (3,)
        scale factors of the basis for x, y, z

    References
    ----------
    .. [1] Ozarslan E. et. al, "Mean apparent propagator (MAP) MRI: A novel
    diffusion imaging method for mapping tissue microstructure",
    NeuroImage, 2013.
    """
    n1, n2, n3 = n
    x, y, z = r.T
    mux, muy, muz = mu
    psi = mapmri_psi_1d
    return psi(n1, x, mux) * psi(n2, y, muy) * psi(n3, z, muz)


def mapmri_psi_matrix(radial_order, mu, rgrad):
    r"""Compute the MAPMRI psi matrix for the propagator [1]_

    Parameters
    ----------
    radial_order : unsigned int,
        an even integer that represent the order of the basis
    mu : array, shape (3,)
        scale factors of the basis for x, y, z
    rgrad : array, shape (N,3)
        points in the r-space in which evaluate the EAP

    References
    ----------
    .. [1] Ozarslan E. et. al, "Mean apparent propagator (MAP) MRI: A novel
    diffusion imaging method for mapping tissue microstructure",
    NeuroImage, 2013.
    """

    ind_mat = mapmri_index_matrix(radial_order)
    n_elem = ind_mat.shape[0]
    n_rgrad = rgrad.shape[0]
    K = np.zeros((n_rgrad, n_elem))
    for j in range(n_elem):
        K[:, j] = mapmri_psi_3d(ind_mat[j], rgrad, mu)

    return K


def mapmri_odf_matrix(radial_order, mu, s, vertices):
    r"""Compute the MAPMRI ODF matrix [1]_  Eq. 33.

    Parameters
    ----------
    radial_order : unsigned int,
        an even integer that represent the order of the basis
    mu : array, shape (3,)
        scale factors of the basis for x, y, z
    s : unsigned int
        radial moment of the ODF
    vertices : array, shape (N,3)
        points of the sphere shell in the r-space in which evaluate the ODF


    References
    ----------
    .. [1] Ozarslan E. et. al, "Mean apparent propagator (MAP) MRI: A novel
    diffusion imaging method for mapping tissue microstructure",
    NeuroImage, 2013.
    """

    ind_mat = mapmri_index_matrix(radial_order)
    n_vert = vertices.shape[0]
    n_elem = ind_mat.shape[0]
    odf_mat = np.zeros((n_vert, n_elem))
    mux, muy, muz = mu
    # Eq, 35a
    rho = 1.0 / np.sqrt((vertices[:, 0] / mux) ** 2 +
                        (vertices[:, 1] / muy) ** 2 + (vertices[:, 2] / muz) ** 2)
    # Eq, 35b
    alpha = 2 * rho * (vertices[:, 0] / mux)
    # Eq, 35c
    beta = 2 * rho * (vertices[:, 1] / muy)
    # Eq, 35d
    gamma = 2 * rho * (vertices[:, 2] / muz)
    const = rho ** (3 + s) / np.sqrt(2 ** (2 - s) * np.pi **
                                     3 * (mux ** 2 * muy ** 2 * muz ** 2))
    for j in range(n_elem):
        n1, n2, n3 = ind_mat[j]
        f = np.sqrt(factorial(n1) * factorial(n2) * factorial(n3))
        odf_mat[:, j] = const * f * \
            _odf_cfunc(n1, n2, n3, alpha, beta, gamma, s)

    return odf_mat


def _odf_cfunc(n1, n2, n3, a, b, g, s):
    r"""Compute the MAPMRI ODF function from [1]_  Eq. 34.

    References
    ----------
    .. [1] Ozarslan E. et. al, "Mean apparent propagator (MAP) MRI: A novel
    diffusion imaging method for mapping tissue microstructure",
    NeuroImage, 2013.
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


def mapmri_EAP(r_list, radial_order, coeff, mu, R):
    r""" Evaluate the MAPMRI propagator in a set of points of the r-space.

    Parameters
    ----------
    r_list : array, shape (N,3)
        points of the r-space in which evaluate the EAP
    radial_order : unsigned int,
        an even integer that represent the order of the basis
    coeff : array, shape (N,)
        the MAPMRI coefficients
    mu : array, shape (3,)
        scale factors of the basis for x, y, z
    R : array, shape (3,3)
        MAPMRI rotation matrix
    """

    r_list = np.dot(r_list, R)
    ind_mat = mapmri_index_matrix(radial_order)
    n_elem = ind_mat.shape[0]
    n_rgrad = r_list.shape[0]
    data_out = np.zeros(n_rgrad)
    for j in range(n_elem):
        data_out[:] += coeff[j] * mapmri_psi_3d(ind_mat[j], r_list, mu)

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
