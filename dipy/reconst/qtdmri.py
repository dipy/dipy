import numpy as np
from dipy.reconst.cache import Cache
from dipy.core.geometry import cart2sphere
from dipy.reconst.multi_voxel import multi_voxel_fit
from scipy.special import genlaguerre, gamma
from scipy import special
from dipy.reconst import mapmri
try:  # preferred scipy >= 0.14, required scipy >= 1.0
    from scipy.special import factorial, factorial2
except ImportError:
    from scipy.misc import factorial, factorial2
from scipy.optimize import fmin_l_bfgs_b
from dipy.reconst.shm import real_sph_harm
import dipy.reconst.dti as dti
import cvxpy
from dipy.utils.optpkg import optional_package
import random

cvxopt, have_cvxopt, _ = optional_package("cvxopt")


class MaptimeModel(Cache):
    r""" Analytical and continuous modeling of the diffusion signal using
        the diffusion time extended MAP-MRI basis [1].
        This implementation is based on the recent IPMI publication [2]

        The main idea is to model the diffusion signal over time and space as
        a linear combination of continuous functions $\phi_i$,

        ..math::
            :nowrap:
                \begin{equation}
                    E(\mathbf{q},\tau)= \sum_{i=0}^I  c_{i}
                    \S_{i}(\mathbf{q})T_{i}(\tau).
                \end{equation}

        where $\mathbf{q}$ is the wavector which corresponds to different
        gradient directions.

        From the $c_i$ coefficients, there exists an analytical formula to
        estimate the ODF, RTOP, RTAP, RTPP and MSD, for any diffusion time.


        Parameters
        ----------
        gtab : GradientTable,
            gradient directions and bvalues container class. The bvalues
            should be in the normal s/mm^2. big_delta and small_delta need to
            given in seconds.
        radial_order : unsigned int,
            an even integer that represent the order of the basis.
        time_order : unsigned int,

        laplacian_regularization: bool,
            Regularize using the Laplacian of the SHORE basis.
        laplacian_weighting: string or scalar,
            The string 'GCV' makes it use generalized cross-validation to find
            the regularization weight [3]. A scalar sets the regularization
            weight to that value.
        tau : float,
            diffusion time. Defined as $\Delta-\delta/3$ in seconds.
            Default value makes q equal to the square root of the b-value.

        References
        ----------
        .. [1] Ozarslan E. et al., "Mean apparent propagator (MAP) MRI: A novel
           diffusion imaging method for mapping tissue microstructure",
           NeuroImage, 2013.
           [2] Fick et al., "A unifying framework for spatial and temporal
           diffusion", IPMI, 2015.
           [3] Craven et al. "Smoothing Noisy Data with Spline Functions."
           NUMER MATH 31.4 (1978): 377-403.
        """

    def __init__(self,
                 gtab,
                 radial_order=4,
                 time_order=3,
                 cartesian=True,
                 anisotropic_scaling=True,
                 normalization=False,
                 laplacian_regularization=False,
                 laplacian_weighting=0.2,
                 l1_regularization=False,
                 l1_weighting=0.1,
                 elastic_net=False,
                 constrain_q0=True,
                 bval_threshold=np.inf
                 ):

        self.gtab = gtab
        self.constrain_q0 = constrain_q0
        self.bval_threshold = bval_threshold
        self.laplacian_regularization = laplacian_regularization
        self.laplacian_weighting = laplacian_weighting
        self.anisotropic_scaling = anisotropic_scaling
        self.cartesian = cartesian
        self.normalization = normalization
        if radial_order % 2 or radial_order < 0:
            msg = "radial_order must be a non-zero even positive number."
            raise ValueError(msg)
        self.radial_order = radial_order
        if time_order < 0:
            msg = "time_order must be a positive number."
            raise ValueError(msg)
        self.time_order = time_order
        if self.anisotropic_scaling:
            self.ind_mat = maptime_index_matrix(radial_order, time_order)
        else:
            self.ind_mat = maptime_isotropic_index_matrix(radial_order,
                                                          time_order)

        self.S_mat, self.T_mat, self.U_mat = mapmri.mapmri_STU_reg_matrices(
                radial_order
        )
        self.part4_reg_mat_tau = part4_reg_matrix_tau(self.ind_mat, 1.)
        self.part23_reg_mat_tau = part23_reg_matrix_tau(self.ind_mat, 1.)
        self.part1_reg_mat_tau = part1_reg_matrix_tau(self.ind_mat, 1.)
        self.l1_regularization = l1_regularization
        self.l1_weighting = l1_weighting
        self.elastic_net = elastic_net
        self.tenmodel = dti.TensorModel(gtab)

    @multi_voxel_fit
    def fit(self, data):
        bval_mask = self.gtab.bvals < self.bval_threshold
        data_norm = data / data[self.gtab.b0s_mask].mean()
        tau = self.gtab.tau
        bvecs = self.gtab.bvecs
        qvals = self.gtab.qvals
        b0s_mask = self.gtab.b0s_mask

        if self.cartesian:
            if self.anisotropic_scaling:
                us, ut, R = maptime_anisotropic_scaling(data_norm[bval_mask],
                                                        qvals[bval_mask],
                                                        bvecs[bval_mask],
                                                        tau[bval_mask])
                tau_scaling = ut / us.mean()
                tau_scaled = tau * tau_scaling
                us, ut, R = maptime_anisotropic_scaling(data_norm[bval_mask],
                                                        qvals[bval_mask],
                                                        bvecs[bval_mask],
                                                        tau_scaled[bval_mask])
                us = np.clip(us, 1e-4, np.inf)
                q = np.dot(bvecs, R) * qvals[:, None]
                M = maptime_signal_matrix_(
                    self.radial_order, self.time_order, us, ut, q, tau_scaled,
                    self.normalization
                )
            else:
                us, ut = maptime_isotropic_scaling(data_norm, qvals, tau)
                tau_scaling = ut / us
                tau_scaled = tau * tau_scaling
                us, ut = maptime_isotropic_scaling(data_norm, qvals,
                                                   tau_scaled)
                R = np.eye(3)
                us = np.tile(us, 3)
                q = bvecs * qvals[:, None]
                M = maptime_signal_matrix_(
                    self.radial_order, self.time_order, us, ut, q, tau_scaled,
                    self.normalization
                )
        else:
            us, ut = maptime_isotropic_scaling(data_norm, qvals, tau)
            tau_scaling = ut / us
            tau_scaled = tau * tau_scaling
            us, ut = maptime_isotropic_scaling(data_norm, qvals, tau_scaled)
            R = np.eye(3)
            us = np.tile(us, 3)
            q = bvecs * qvals[:, None]
            M = maptime_isotropic_signal_matrix_(
                self.radial_order, self.time_order, us[0], ut, q, tau_scaled
            )

        b0_indices = np.arange(self.gtab.tau.shape[0])[self.gtab.b0s_mask]
        tau0_ordered = self.gtab.tau[b0_indices]
        unique_taus = np.unique(self.gtab.tau)
        first_tau_pos = []
        for unique_tau in unique_taus:
            first_tau_pos.append(np.where(tau0_ordered == unique_tau)[0][0])
        M0 = M[b0_indices[first_tau_pos]]

        lopt = 0.
        alpha = 0.
        if self.laplacian_regularization:
            if self.cartesian:
                laplacian_matrix = maptime_laplacian_reg_matrix(
                    self.ind_mat, us, ut, self.S_mat, self.T_mat, self.U_mat,
                    self.part1_reg_mat_tau,
                    self.part23_reg_mat_tau,
                    self.part4_reg_mat_tau
                )
            else:
                laplacian_matrix = maptime_isotropic_laplacian_reg_matrix(
                    self.ind_mat, self.us, self.ut
                )
            if self.laplacian_weighting == 'GCV':
                try:
                    lopt = generalized_crossvalidation(data_norm, M,
                                                       laplacian_matrix)
                except:
                    lopt = 3e-4
            elif np.isscalar(self.laplacian_weighting):
                lopt = self.laplacian_weighting
            elif type(self.laplacian_weighting) == np.ndarray:
                lopt = generalized_crossvalidation(data, M, laplacian_matrix,
                                                   self.laplacian_weighting)
            c = cvxpy.Variable(M.shape[1])
            design_matrix = cvxpy.Constant(M)
            objective = cvxpy.Minimize(
                cvxpy.sum_squares(design_matrix * c - data_norm) +
                lopt * cvxpy.quad_form(c, laplacian_matrix)
            )
            if self.constrain_q0:
                # just constraint first and last, otherwise the solver fails
                constraints = [M0[0] * c == 1,
                               M0[-1] * c == 1]
            else:
                constraints = []
            prob = cvxpy.Problem(objective, constraints)
            try:
                prob.solve(solver="ECOS", verbose=False)
                maptime_coef = np.asarray(c.value).squeeze()
            except:
                maptime_coef = np.zeros(M.shape[1])
        elif self.l1_regularization:
            if self.l1_weighting == 'CV':
                alpha = l1_crossvalidation(b0s_mask, data_norm, M)
            elif np.isscalar(self.l1_weighting):
                alpha = self.l1_weighting
            c = cvxpy.Variable(M.shape[1])
            design_matrix = cvxpy.Constant(M)
            objective = cvxpy.Minimize(
                cvxpy.sum_squares(design_matrix * c - data_norm) +
                alpha * cvxpy.norm1(c)
            )
            if self.constrain_q0:
                # just constraint first and last, otherwise the solver fails
                constraints = [M0[0] * c == 1,
                               M0[-1] * c == 1]
            else:
                constraints = []
            prob = cvxpy.Problem(objective, constraints)
            try:
                prob.solve(solver="ECOS", verbose=False)
                maptime_coef = np.asarray(c.value).squeeze()
            except:
                maptime_coef = np.zeros(M.shape[1])
        elif self.elastic_net:
            if self.cartesian:
                laplacian_matrix = maptime_laplacian_reg_matrix(
                    self.ind_mat, us, ut, self.S_mat, self.T_mat, self.U_mat,
                    self.part1_reg_mat_tau,
                    self.part23_reg_mat_tau,
                    self.part4_reg_mat_tau
                )
            else:
                laplacian_matrix = maptime_isotropic_laplacian_reg_matrix(
                    self.ind_mat, self.us, self.ut
                )
            if self.laplacian_weighting == 'GCV':
                lopt = generalized_crossvalidation(data_norm, M,
                                                   laplacian_matrix)
            elif np.isscalar(self.laplacian_weighting):
                lopt = self.laplacian_weighting
            elif type(self.laplacian_weighting) == np.ndarray:
                lopt = generalized_crossvalidation(data, M, laplacian_matrix,
                                                   self.laplacian_weighting)
            if self.l1_weighting == 'CV':
                alpha = elastic_crossvalidation(b0s_mask, data_norm, M,
                                                laplacian_matrix, lopt)
            elif np.isscalar(self.l1_weighting):
                alpha = self.l1_weighting
            c = cvxpy.Variable(M.shape[1])
            design_matrix = cvxpy.Constant(M)
            objective = cvxpy.Minimize(
                cvxpy.sum_squares(design_matrix * c - data_norm) +
                alpha * cvxpy.norm1(c) +
                lopt * cvxpy.quad_form(c, laplacian_matrix)
            )
            if self.constrain_q0:
                # just constraint first and last, otherwise the solver fails
                constraints = [M0[0] * c == 1,
                               M0[-1] * c == 1]
            else:
                constraints = []
            prob = cvxpy.Problem(objective, constraints)
            try:
                prob.solve(solver="ECOS", verbose=False)
                maptime_coef = np.asarray(c.value).squeeze()
            except:
                maptime_coef = np.zeros(M.shape[1])
        else:
            pseudoInv = np.linalg.pinv(M)
            maptime_coef = np.dot(pseudoInv, data_norm)

        return MaptimeFit(
            self, maptime_coef, us, ut, tau_scaling, R, lopt, alpha
        )


class MaptimeFit():

    def __init__(self, model, maptime_coef, us, ut, tau_scaling, R, lopt,
                 alpha):
        """ Calculates diffusion properties for a single voxel

        Parameters
        ----------
        model : object,
            AnalyticalModel
        maptime_coef : 1d ndarray,
            maptime coefficients
        us : array, 3 x 1
            spatial scaling factors
        ut : float
            temporal scaling factor
        R : 3x3 numpy array,
            tensor eigenvectors
        lopt : float,
            laplacian regularization weight
        """

        self.model = model
        self._maptime_coef = maptime_coef
        self.us = us
        self.ut = ut
        self.tau_scaling = tau_scaling
        self.R = R
        self.lopt = lopt
        self.alpha = alpha

    @property
    def maptime_coeff(self):
        """The MAPTIME coefficients
        """
        return self._maptime_coef

    def sparsity_abs(self, threshold=0.99):
        total_weight = np.sum(abs(self._maptime_coef))
        absolute_normalized_coef_array = (
            np.sort(abs(self._maptime_coef))[::-1] / total_weight)
        current_weight = 0.
        counter = 0
        while current_weight < threshold:
            current_weight += absolute_normalized_coef_array[counter]
            counter += 1
        return counter

    def sparsity_density(self, threshold=0.99):
        total_weight = np.sum(self._maptime_coef ** 2)
        squared_normalized_coef_array = (
            np.sort(self._maptime_coef ** 2)[::-1] / total_weight)
        current_weight = 0.
        counter = 0
        while current_weight < threshold:
            current_weight += squared_normalized_coef_array[counter]
            counter += 1
        return counter

    def fitted_signal(self, gtab=None):
        """ Recovers the fitted signal. If no gtab is given it recovers
        the signal for the gtab of the data.
        """
        if gtab is None:
            E = self.predict(self.model.gtab)
        else:
            E = self.predict(gtab)
        return E

    def predict(self, qvals_or_gtab, S0=1.):
        r'''Recovers the reconstructed signal for any qvalue array or
        gradient table. We precompute the mu independent part of the
        design matrix Q to speed up the computation.
        '''
        tau_scaling = self.tau_scaling
        if isinstance(qvals_or_gtab, np.ndarray):
            q = qvals_or_gtab[:, :3]
            tau = qvals_or_gtab[:, 3] * tau_scaling
        else:
            gtab = qvals_or_gtab
            qvals = gtab.qvals
            tau = gtab.tau * tau_scaling
            q = qvals[:, None] * gtab.bvecs

        if self.model.cartesian:
            if self.model.anisotropic_scaling:
                q_rot = np.dot(q, self.R)
                M = maptime_signal_matrix_(self.model.radial_order,
                                           self.model.time_order,
                                           self.us, self.ut, q_rot, tau,
                                           self.model.normalization)
            else:
                M = maptime_signal_matrix_(self.model.radial_order,
                                           self.model.time_order,
                                           self.us, self.ut, q, tau,
                                           self.model.normalization)
        else:
            M = maptime_isotropic_signal_matrix_(self.model.radial_order,
                                                 self.model.time_order,
                                                 self.us[0], self.ut, q, tau)
        E = S0 * np.dot(M, self._maptime_coef)
        return E

    def norm_of_laplacian_signal(self):
        if self.model.anisotropic_scaling:
            lap_matrix = maptime_laplacian_reg_matrix(self.model.ind_mat,
                                                      self.us, self.ut,
                                                      self.model.S_mat,
                                                      self.model.T_mat,
                                                      self.model.U_mat)
        else:
            lap_matrix = maptime_isotropic_laplacian_reg_matrix(
                self.model.ind_mat, self.us, self.ut
            )
        norm_laplacian = np.dot(self._maptime_coef,
                                np.dot(self._maptime_coef, lap_matrix))
        return norm_laplacian

    def pdf(self, rt_points):
        """ Diffusion propagator on a given set of real points.
            if the array r_points is non writeable, then intermediate
            results are cached for faster recalculation
        """
        tau_scaling = self.tau_scaling
        rt_points_ = rt_points * np.r_[1, 1, 1, tau_scaling]
        if self.model.anisotropic_scaling:
            K = maptime_eap_matrix_(self.model.radial_order,
                                    self.model.time_order,
                                    self.us, self.ut, rt_points_,
                                    self.model.normalization)
        else:
            K = maptime_isotropic_eap_matrix_(self.model.radial_order,
                                              self.model.time_order,
                                              self.us[0], self.ut, rt_points_)
        eap = np.dot(K, self._maptime_coef)
        return eap

    def maptime_to_mapmri_coef(self, tau):
        if self.model.anisotropic_scaling:
            I = self.model.cache_get('maptime_to_mapmri_matrix',
                                     key=(tau))
            if I is None:
                I = maptime_to_mapmri_matrix(self.model.radial_order,
                                             self.model.time_order, self.ut,
                                             self.tau_scaling * tau)
                self.model.cache_set('maptime_to_mapmri_matrix',
                                     (tau), I)
        else:
            I = self.model.cache_get('maptime_isotropic_to_mapmri_matrix',
                                     key=(tau))
            if I is None:
                I = maptime_isotropic_to_mapmri_matrix(self.model.radial_order,
                                                       self.model.time_order,
                                                       self.ut,
                                                       self.tau_scaling * tau)
                self.model.cache_set('maptime_isotropic_to_mapmri_matrix',
                                     (tau), I)

        mapmri_coef = np.dot(I, self._maptime_coef)
        return mapmri_coef

    def msd(self, tau):
        ind_mat = maptime_index_matrix(self.model.radial_order,
                                       self.model.time_order)
        mu = self.us
        max_o = ind_mat[:, 3].max()
        small_temporal_storage = np.zeros(max_o + 1)
        for o in range(max_o + 1):
            small_temporal_storage[o] = temporal_basis(o, self.ut,
                                                       tau * self.tau_scaling)
        msd = 0.
        for i in range(ind_mat.shape[0]):
            nx, ny, nz = ind_mat[i, :3]
            if not(nx % 2) and not(ny % 2) and not(nz % 2):
                msd += (
                    self._maptime_coef[i] * (-1) ** (0.5 * (- nx - ny - nz)) *
                    np.pi ** (3/2.0) *
                    ((1 + 2 * nx) * mu[0] ** 2 + (1 + 2 * ny) * mu[1] ** 2 +
                     (1 + 2 * nz) * mu[2] ** 2) /
                    (np.sqrt(2 ** (-nx - ny - nz) *
                             factorial(nx) * factorial(ny) * factorial(nz)) *
                     gamma(0.5 - 0.5 * nx) * gamma(0.5 - 0.5 * ny) *
                     gamma(0.5 - 0.5 * nz)) *
                    small_temporal_storage[ind_mat[i, 3]]
                )
        return msd

    def rtop(self, tau):
        mapmri_coef = self.maptime_to_mapmri_coef(tau)
        ind_mat = mapmri.mapmri_index_matrix(self.model.radial_order)
        B_mat = mapmri.b_mat(ind_mat)
        mu = self.us

        rtop = 0.
        const = 1. / np.sqrt(
            8 * np.pi ** 3 * (mu[0] ** 2 * mu[1] ** 2 * mu[2] ** 2)
        )
        for i in range(ind_mat.shape[0]):
            nx, ny, nz = ind_mat[i]
            if B_mat[i] > 0.:
                rtop += (
                    const * (-1.0) ** ((nx + ny + nz) / 2.0) * mapmri_coef[i] *
                    B_mat[i]
                )
        return rtop

    def rtap(self, tau):
        mapmri_coef = self.maptime_to_mapmri_coef(tau)
        ind_mat = mapmri.mapmri_index_matrix(self.model.radial_order)
        B_mat = mapmri.b_mat(ind_mat)
        mu = self.us

        # if self.model.anisotropic_scaling:
        sel = B_mat > 0.  # select only relevant coefficients
        const = 1 / (2 * np.pi * np.prod(mu[1:]))
        ind_sum = (-1.0) ** ((np.sum(ind_mat[sel, 1:], axis=1) / 2.0))
        rtap_vec = const * B_mat[sel] * ind_sum * mapmri_coef[sel]
        rtap = np.sum(rtap_vec)
        return rtap

    def rtpp(self, tau):
        mapmri_coef = self.maptime_to_mapmri_coef(tau)
        ind_mat = mapmri.mapmri_index_matrix(self.model.radial_order)
        B_mat = mapmri.b_mat(ind_mat)
        mu = self.us

        # if self.model.anisotropic_scaling:
        sel = B_mat > 0.  # select only relevant coefficients
        const = 1 / (np.sqrt(2 * np.pi) * mu[0])
        ind_sum = (-1.0) ** (ind_mat[sel, 0] / 2.0)
        rtpp_vec = const * B_mat[sel] * ind_sum * mapmri_coef[sel]
        rtpp = rtpp_vec.sum()
        return rtpp


def maptime_to_mapmri_matrix(radial_order, time_order, ut, tau):
    mapmri_ind_mat = mapmri.mapmri_index_matrix(radial_order)
    n_elem_mapmri = mapmri_ind_mat.shape[0]
    maptime_ind_mat = maptime_index_matrix(radial_order, time_order)
    n_elem_maptime = maptime_ind_mat.shape[0]

    temporal_storage = np.zeros(time_order + 1)
    for o in range(time_order + 1):
        temporal_storage[o] = temporal_basis(o, ut, tau)

    counter = 0
    mapmri_mat = np.zeros((n_elem_mapmri, n_elem_maptime))
    for nxt, nyt, nzt, o in maptime_ind_mat:
        index_overlap = np.all([nxt == mapmri_ind_mat[:, 0],
                                nyt == mapmri_ind_mat[:, 1],
                                nzt == mapmri_ind_mat[:, 2]], 0)
        mapmri_mat[:, counter] = temporal_storage[o] * index_overlap
        counter += 1
    return mapmri_mat


def maptime_isotropic_to_mapmri_matrix(radial_order, time_order, ut, tau):
    mapmri_ind_mat = mapmri.mapmri_isotropic_index_matrix(radial_order)
    n_elem_mapmri = mapmri_ind_mat.shape[0]
    maptime_ind_mat = maptime_isotropic_index_matrix(radial_order, time_order)
    n_elem_maptime = maptime_ind_mat.shape[0]

    temporal_storage = np.zeros(time_order + 1)
    for o in range(time_order + 1):
        temporal_storage[o] = temporal_basis(o, ut, tau)

    counter = 0
    mapmri_isotropic_mat = np.zeros((n_elem_mapmri, n_elem_maptime))
    for j, l, m, o in maptime_ind_mat:
        index_overlap = np.all([j == mapmri_ind_mat[:, 0],
                                l == mapmri_ind_mat[:, 1],
                                m == mapmri_ind_mat[:, 2]], 0)
        mapmri_isotropic_mat[:, counter] = temporal_storage[o] * index_overlap
        counter += 1
    return mapmri_isotropic_mat


def maptime_temporal_normalization(ut):
    return np.sqrt(ut)


def maptime_signal_matrix_(radial_order, time_order, us, ut, q, tau,
                           normalization=False):
    sqrtC = 1.
    sqrtut = 1.
    sqrtCut = 1.
    if normalization:
        sqrtC = mapmri.mapmri_normalization(us)
        sqrtut = maptime_temporal_normalization(ut)
        sqrtCut = sqrtC * sqrtut
    M_tau = (maptime_signal_matrix(radial_order, time_order, us, ut, q, tau) *
             sqrtCut)
    return M_tau


def maptime_signal_matrix(radial_order, time_order, us, ut, q, tau):
    r'''Constructs the design matrix as a product of 3 separated radial,
    angular and temporal design matrices. It precomputes the relevant basis
    orders for each one and finally puts them together according to the index
    matrix
    '''
    ind_mat = maptime_index_matrix(radial_order, time_order)

    n_dat = q.shape[0]
    n_elem = ind_mat.shape[0]
    qx, qy, qz = q.T
    mux, muy, muz = us

    temporal_storage = np.zeros((n_dat, time_order + 1))
    for o in range(time_order + 1):
        temporal_storage[:, o] = temporal_basis(o, ut, tau)

    Qx_storage = np.array(np.zeros((n_dat, radial_order + 1 + 4)),
                          dtype=complex)
    Qy_storage = np.array(np.zeros((n_dat, radial_order + 1 + 4)),
                          dtype=complex)
    Qz_storage = np.array(np.zeros((n_dat, radial_order + 1 + 4)),
                          dtype=complex)
    for n in range(radial_order + 1 + 4):
        Qx_storage[:, n] = mapmri.mapmri_phi_1d(n, qx, mux)
        Qy_storage[:, n] = mapmri.mapmri_phi_1d(n, qy, muy)
        Qz_storage[:, n] = mapmri.mapmri_phi_1d(n, qz, muz)

    counter = 0
    Q = np.zeros((n_dat, n_elem))
    for nx, ny, nz, o in ind_mat:
        Q[:, counter] = (np.real(
            Qx_storage[:, nx] * Qy_storage[:, ny] * Qz_storage[:, nz]) *
            temporal_storage[:, o]
        )
        counter += 1

    return Q


def design_matrix_normalized(radial_order, time_order, us, ut, q, tau):
    sqrtC = mapmri.mapmri_normalization(us)
    sqrtut = maptime_temporal_normalization(ut)
    normalization = sqrtC * sqrtut
    normalized_design_matrix = (
        normalization *
        maptime_signal_matrix(radial_order, time_order, us, ut, q, tau)
    )
    return normalized_design_matrix


def maptime_eap_matrix(radial_order, time_order, us, ut, grid):
    r'''Constructs the design matrix as a product of 3 separated radial,
    angular and temporal design matrices. It precomputes the relevant basis
    orders for each one and finally puts them together according to the index
    matrix
    '''
    ind_mat = maptime_index_matrix(radial_order, time_order)
    rx, ry, rz, tau = grid.T

    n_dat = rx.shape[0]
    n_elem = ind_mat.shape[0]
    mux, muy, muz = us

    temporal_storage = np.zeros((n_dat, time_order + 1))
    for o in range(time_order + 1):
        temporal_storage[:, o] = temporal_basis(o, ut, tau)

    Kx_storage = np.zeros((n_dat, radial_order + 1))
    Ky_storage = np.zeros((n_dat, radial_order + 1))
    Kz_storage = np.zeros((n_dat, radial_order + 1))
    for n in range(radial_order + 1):
        Kx_storage[:, n] = mapmri.mapmri_psi_1d(n, rx, mux)
        Ky_storage[:, n] = mapmri.mapmri_psi_1d(n, ry, muy)
        Kz_storage[:, n] = mapmri.mapmri_psi_1d(n, rz, muz)

    counter = 0
    K = np.zeros((n_dat, n_elem))
    for nx, ny, nz, o in ind_mat:
        K[:, counter] = (
            Kx_storage[:, nx] * Ky_storage[:, ny] * Kz_storage[:, nz] *
            temporal_storage[:, o]
        )
        counter += 1

    return K


def maptime_isotropic_signal_matrix_(radial_order, time_order, us, ut, q, tau):
    M_tau = maptime_isotropic_signal_matrix(
        radial_order, time_order, us, ut, q, tau
    )
    return M_tau


def maptime_isotropic_signal_matrix(radial_order, time_order, us, ut, q, tau):
    ind_mat = maptime_isotropic_index_matrix(radial_order, time_order)
    qvals, theta, phi = cart2sphere(q[:, 0], q[:, 1], q[:, 2])

    n_dat = qvals.shape[0]
    n_elem = ind_mat.shape[0]

    num_j = np.max(ind_mat[:, 0])
    num_o = time_order + 1
    num_l = radial_order / 2 + 1
    num_m = radial_order * 2 + 1

    # Radial Basis
    radial_storage = np.zeros([num_j, num_l, n_dat])
    for j in range(1, num_j + 1):
        for l in range(0, radial_order + 1, 2):
            radial_storage[j-1, l/2, :] = radial_basis_opt(j, l, us, qvals)

    # Angular Basis
    angular_storage = np.zeros([num_l, num_m, n_dat])
    for l in range(0, radial_order + 1, 2):
        for m in range(-l, l+1):
            angular_storage[l / 2, m + l, :] = (
                angular_basis_opt(l, m, qvals, theta, phi)
            )

    # Temporal Basis
    temporal_storage = np.zeros([num_o + 1, n_dat])
    for o in range(0, num_o + 1):
        temporal_storage[o, :] = temporal_basis(o, ut, tau)

    # Construct full design matrix
    M = np.zeros((n_dat, n_elem))
    counter = 0
    for j, l, m, o in ind_mat:
        M[:, counter] = (radial_storage[j-1, l/2, :] *
                         angular_storage[l / 2, m + l, :] *
                         temporal_storage[o, :])
        counter += 1
    return M


def maptime_eap_matrix_(radial_order, time_order, us, ut, grid,
                        normalization=False):
    sqrtC = 1.
    sqrtut = 1.
    sqrtCut = 1.
    if normalization:
        sqrtC = mapmri.mapmri_normalization(us)
        sqrtut = maptime_temporal_normalization(ut)
        sqrtCut = sqrtC * sqrtut
    K_tau = (
        maptime_eap_matrix(radial_order, time_order, us, ut, grid) * sqrtCut
    )
    return K_tau


def maptime_isotropic_eap_matrix_(radial_order, time_order, us, ut, grid):
    K_tau = maptime_isotropic_eap_matrix(
        radial_order, time_order, us, ut, grid
    )
    return K_tau


def maptime_isotropic_eap_matrix(radial_order, time_order, us, ut, grid,
                                 spatial_storage=None):
    r'''Constructs the design matrix as a product of 3 separated radial,
    angular and temporal design matrices. It precomputes the relevant basis
    orders for each one and finally puts them together according to the index
    matrix
    '''

    rx, ry, rz, tau = grid.T
    R, theta, phi = cart2sphere(rx, ry, rz)
    theta[np.isnan(theta)] = 0

    ind_mat = maptime_isotropic_index_matrix(radial_order, time_order)
    n_dat = R.shape[0]
    n_elem = ind_mat.shape[0]

    num_j = np.max(ind_mat[:, 0])
    num_o = time_order + 1
    num_l = radial_order / 2 + 1
    num_m = radial_order * 2 + 1

    # Radial Basis
    radial_storage = np.zeros([num_j, num_l, n_dat])
    for j in range(1, num_j + 1):
        for l in range(0, radial_order + 1, 2):
            radial_storage[j - 1, l / 2, :] = radial_basis_EAP_opt(j, l, us, R)

    # Angular Basis
    angular_storage = np.zeros([num_j, num_l, num_m, n_dat])
    for j in range(1, num_j + 1):
        for l in range(0, radial_order + 1, 2):
            for m in range(-l, l + 1):
                angular_storage[j - 1, l / 2, m + l, :] = (
                    angular_basis_EAP_opt(j, l, m, R, theta, phi)
                )

    # Temporal Basis
    temporal_storage = np.zeros([num_o + 1, n_dat])
    for o in range(0, num_o + 1):
        temporal_storage[o, :] = temporal_basis(o, ut, tau)

    # Construct full design matrix
    M = np.zeros((n_dat, n_elem))
    counter = 0
    for j, l, m, o in ind_mat:
        M[:, counter] = (radial_storage[j-1, l/2, :] *
                         angular_storage[j - 1, l / 2, m + l, :] *
                         temporal_storage[o, :])
        counter += 1
    return M


def radial_basis_opt(j, l, us, q):
    ''' Spatial basis dependent on spatial scaling factor us
    '''
    const = (
        us ** l * np.exp(-2 * np.pi ** 2 * us ** 2 * q ** 2) *
        genlaguerre(j - 1, l + 0.5)(4 * np.pi ** 2 * us ** 2 * q ** 2)
    )
    return const


def angular_basis_opt(l, m, q, theta, phi):
    ''' Angular basis independent of spatial scaling factor us. Though it
    includes q, it is independent of the data and can be precomputed.
    '''
    const = (
        (-1) ** (l / 2) * np.sqrt(4.0 * np.pi) *
        (2 * np.pi ** 2 * q ** 2) ** (l / 2) *
        real_sph_harm(m, l, theta, phi)
    )
    return const


def radial_basis_EAP_opt(j, l, us, r):
    radial_part = (
        (us ** 3) ** (-1) / (us ** 2) ** (l / 2) *
        np.exp(- r ** 2 / (2 * us ** 2)) *
        genlaguerre(j - 1, l + 0.5)(r ** 2 / us ** 2)
    )
    return radial_part


def angular_basis_EAP_opt(j, l, m, r, theta, phi):
    angular_part = (
        (-1) ** (j - 1) * (np.sqrt(2) * np.pi) ** (-1) *
        (r ** 2 / 2) ** (l / 2) * real_sph_harm(m, l, theta, phi)
    )
    return angular_part


def temporal_basis(o, ut, tau):
    ''' Temporal basis dependent on temporal scaling factor ut
    '''
    const = np.exp(-ut * tau / 2.0) * special.laguerre(o)(ut * tau)
    return const


def maptime_index_matrix(radial_order, time_order):
    """Computes the SHORE basis order indices according to [1].
    """
    index_matrix = []
    for n in range(0, radial_order + 1, 2):
        for i in range(0, n + 1):
            for j in range(0, n - i + 1):
                for o in range(0, time_order + 1):
                    index_matrix.append([n - i - j, j, i, o])

    return np.array(index_matrix)


def maptime_isotropic_index_matrix(radial_order, time_order):
    """Computes the SHORE basis order indices according to [1].
    """
    index_matrix = []
    for n in range(0, radial_order + 1, 2):
        for j in range(1, 2 + n / 2):
            l = n + 2 - 2 * j
            for m in range(-l, l + 1):
                for o in range(0, time_order+1):
                    index_matrix.append([j, l, m, o])

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
        n1, n2, n3, _ = ind_mat[i]
        K = int(not(n1 % 2) and not(n2 % 2) and not(n3 % 2))
        B[i] = (
            K * np.sqrt(factorial(n1) * factorial(n2) * factorial(n3)) /
            (factorial2(n1) * factorial2(n2) * factorial2(n3))
            )

    return B


def maptime_laplacian_reg_matrix_normalized(ind_mat, us, ut,
                                            S_mat, T_mat, U_mat):
    sqrtC = mapmri.mapmri_normalization(us)
    sqrtut = maptime_temporal_normalization(ut)
    normalization = sqrtC * sqrtut
    normalized_laplacian_matrix = (
        normalization ** 2 * maptime_laplacian_reg_matrix(ind_mat, us, ut,
                                                          S_mat, T_mat, U_mat)
                                                          )
    return normalized_laplacian_matrix


def maptime_laplacian_reg_matrix(ind_mat, us, ut, S_mat, T_mat, U_mat,
                                 part1_ut_precomp=None,
                                 part23_ut_precomp=None,
                                 part4_ut_precomp=None):
    part1_us = mapmri.mapmri_laplacian_reg_matrix(ind_mat[:, :3], us,
                                                  S_mat, T_mat, U_mat)
    part23_us = part23_reg_matrix_q(ind_mat, U_mat, T_mat, us)
    part4_us = part4_reg_matrix_q(ind_mat, U_mat, us)

    if part1_ut_precomp is None:
        part1_ut = part1_reg_matrix_tau(ind_mat, ut)
    else:
        part1_ut = part1_ut_precomp / ut
    if part23_ut_precomp is None:
        part23_ut = part23_reg_matrix_tau(ind_mat, ut)
    else:
        part23_ut = part23_ut_precomp * ut
    if part4_ut_precomp is None:
        part4_ut = part4_reg_matrix_tau(ind_mat, ut)
    else:
        part4_ut = part4_ut_precomp * ut ** 3

    regularization_matrix = (
        part1_us * part1_ut + part23_us * part23_ut + part4_us * part4_ut
    )
    return regularization_matrix


def maptime_isotropic_laplacian_reg_matrix(ind_mat, us, ut):
    part1_us = mapmri.mapmri_isotropic_laplacian_reg_matrix(ind_mat, us[0])
    part23_us = part23_iso_reg_matrix_q(ind_mat, us[0])
    part4_us = part4_iso_reg_matrix_q(ind_mat, us[0])

    part1_ut = part1_reg_matrix_tau(ind_mat, ut)
    part23_ut = part23_reg_matrix_tau(ind_mat, ut)
    part4_ut = part4_reg_matrix_tau(ind_mat, ut)

    regularization_matrix = (
        part1_us * part1_ut + part23_us * part23_ut + part4_us * part4_ut
    )
    return regularization_matrix


def part23_reg_matrix_q(ind_mat, U_mat, T_mat, us):
    ux, uy, uz = us
    x, y, z, _ = ind_mat.T
    n_elem = ind_mat.shape[0]
    LR = np.zeros((n_elem, n_elem))
    for i in range(n_elem):
        for k in range(i, n_elem):
            val = 0
            if x[i] == x[k] and y[i] == y[k]:
                val += (
                    (uz / (ux * uy)) *
                    U_mat[x[i], x[k]] * U_mat[y[i], y[k]] * T_mat[z[i], z[k]]
                )
            if x[i] == x[k] and z[i] == z[k]:
                val += (
                    (uy / (ux * uz)) *
                    U_mat[x[i], x[k]] * T_mat[y[i], y[k]] * U_mat[z[i], z[k]]
                )
            if y[i] == y[k] and z[i] == z[k]:
                val += (
                    (ux / (uy * uz)) *
                    T_mat[x[i], x[k]] * U_mat[y[i], y[k]] * U_mat[z[i], z[k]]
                )
            LR[i, k] = LR[k, i] = val
    return LR


def part23_iso_reg_matrix_q(ind_mat, us):
    n_elem = ind_mat.shape[0]

    LR = np.zeros((n_elem, n_elem))

    for i in range(n_elem):
        for k in range(i, n_elem):
            if ind_mat[i, 1] == ind_mat[k, 1] and \
               ind_mat[i, 2] == ind_mat[k, 2]:
                ji = ind_mat[i, 0]
                jk = ind_mat[k, 0]
                l = ind_mat[i, 1]
                if ji == (jk + 1):
                    LR[i, k] = LR[k, i] = (
                        2 ** (-l) * -gamma(3 / 2.0 + jk + l) / gamma(jk)
                    )
                elif ji == jk:
                    LR[i, k] = LR[k, i] = 2 ** (-(l+1)) *\
                        (1 - 4 * ji - 2 * l) *\
                        gamma(1 / 2.0 + ji + l) / gamma(ji)
                elif ji == (jk - 1):
                    LR[i, k] = LR[k, i] = 2 ** (-l) *\
                        -gamma(3 / 2.0 + ji + l) / gamma(ji)
    return LR / us


def part4_reg_matrix_q(ind_mat, U_mat, us):
    ux, uy, uz = us
    x, y, z, _ = ind_mat.T
    n_elem = ind_mat.shape[0]
    LR = np.zeros((n_elem, n_elem))
    for i in range(n_elem):
        for k in range(i, n_elem):
            if x[i] == x[k] and \
               y[i] == y[k] and \
               z[i] == z[k]:
                LR[i, k] = LR[k, i] = (
                    (1. / (ux * uy * uz)) * U_mat[x[i], x[k]] *
                    U_mat[y[i], y[k]] * U_mat[z[i], z[k]]
                )
    return LR


def part4_iso_reg_matrix_q(ind_mat, us):
    n_elem = ind_mat.shape[0]
    LR = np.zeros((n_elem, n_elem))
    for i in range(n_elem):
        for k in range(i, n_elem):
            if ind_mat[i, 0] == ind_mat[k, 0] and \
               ind_mat[i, 1] == ind_mat[k, 1] and \
               ind_mat[i, 2] == ind_mat[k, 2]:
                ji = ind_mat[i, 0]
                l = ind_mat[i, 1]
                LR[i, k] = LR[k, i] = (
                    2 ** (-(l + 2)) * gamma(1 / 2.0 + ji + l) /
                    (np.pi ** 2 * gamma(ji))
                )

    return LR / us ** 3


def part1_reg_matrix_tau(ind_mat, ut):
    n_elem = ind_mat.shape[0]
    LD = np.zeros((n_elem, n_elem))
    for i in range(n_elem):
        for k in range(i, n_elem):
            oi = ind_mat[i, 3]
            ok = ind_mat[k, 3]
            if oi == ok:
                LD[i, k] = LD[k, i] = 1. / ut
    return LD


def part23_reg_matrix_tau(ind_mat, ut):
    n_elem = ind_mat.shape[0]
    LD = np.zeros((n_elem, n_elem))
    for i in range(n_elem):
        for k in range(i, n_elem):
            oi = ind_mat[i, 3]
            ok = ind_mat[k, 3]
            if oi == ok:
                LD[i, k] = LD[k, i] = 1/2.
            else:
                LD[i, k] = LD[k, i] = np.abs(oi-ok)
    return ut * LD


def part4_reg_matrix_tau(ind_mat, ut):
    n_elem = ind_mat.shape[0]
    LD = np.zeros((n_elem, n_elem))

    for i in range(n_elem):
            for k in range(i, n_elem):
                oi = ind_mat[i, 3]
                ok = ind_mat[k, 3]

                sum1 = 0
                for p in range(1, min([ok, oi]) + 1 + 1):
                    sum1 += (oi - p) * (ok - p) * H(min([oi, ok]) - p)

                sum2 = 0
                for p in range(0, min(ok - 2, oi - 1) + 1):
                    sum2 += p

                sum3 = 0
                for p in range(0, min(ok - 1, oi - 2) + 1):
                    sum3 += p

                LD[i, k] = LD[k, i] = (
                    0.25 * np.abs(oi - ok) + (1 / 16.) * mapmri.delta(oi, ok) +
                    min([oi, ok]) + sum1 + H(oi - 1) * H(ok - 1) *
                    (oi + ok - 2 + sum2 + sum3 + H(abs(oi - ok) - 1) *
                     (abs(oi - ok) - 1) * min([ok - 1, oi - 1]))
                )
    return LD * ut ** 3


def maptime_laplace_S_tau(oi, ok):
    sum1 = 0
    for p in range(1, min([ok, oi]) + 1 + 1):
        sum1 += (oi - p) * (ok - p) * H(min([oi, ok]) - p)

    sum2 = 0
    for p in range(0, min(ok - 2, oi - 1) + 1):
        sum2 += p

    sum3 = 0
    for p in range(0, min(ok - 1, oi - 2) + 1):
        sum3 += p

    val = (
        (1 / 4.) * np.abs(oi - ok) + (1 / 16.) * mapmri.delta(oi, ok) +
        min([oi, ok]) + sum1 + H(oi - 1) * H(ok - 1) *
        (oi + ok - 2 + sum2 + sum3 + H(abs(oi - ok) - 1) * (abs(oi - ok) - 1) *
         min([ok - 1, oi - 1]))
    )
    return val


def maptime_laplace_T_tau(oi, ok):
    if oi == ok:
        val = 1/2.
    else:
        val = np.abs(oi-ok)
    return val


def maptime_laplace_U_tau(oi, ok):
    if oi == ok:
        val = 1.
    else:
        val = 0.
    return val


def maptime_STU_time_reg_matrices(time_order):
    """ Generates the static portions of the Laplacian regularization matrix
    according to [1]_ eq. (11, 12, 13).

    Parameters
    ----------
    radial_order : unsigned int,
        an even integer that represent the order of the basis

    Returns
    -------
    S, T, U : Matrices, shape (N_coef,N_coef)
        Regularization submatrices

    References
    ----------
    .. [1]_ Fick et al. "MAPL: Tissue Microstructure Estimation Using
    Laplacian-Regularized MAP-MRI and its Application to HCP Data",
    NeuroImage, Under Review.
    """
    S = np.zeros((time_order + 1, time_order + 1))
    for i in range(time_order + 1):
        for j in range(time_order + 1):
            S[i, j] = maptime_laplace_S_tau(i, j)

    T = np.zeros((time_order + 1, time_order + 1))
    for i in range(time_order + 1):
        for j in range(time_order + 1):
            T[i, j] = maptime_laplace_T_tau(i, j)

    U = np.zeros((time_order + 1, time_order + 1))
    for i in range(time_order + 1):
        for j in range(time_order + 1):
            U[i, j] = maptime_laplace_U_tau(i, j)
    return S, T, U


def H(value):
    if value >= 0:
        return 1
    return 0


def generalized_crossvalidation(data, M, LR, startpoint=5e-4):
    """Generalized Cross Validation Function [4]
    """
    startpoint = 1e-4
    MMt = np.dot(M.T, M)
    K = len(data)
    input_stuff = (data, M, MMt, K, LR)

    bounds = ((1e-5, 1),)
    res = fmin_l_bfgs_b(lambda x,
                        input_stuff: GCV_cost_function(x, input_stuff),
                        (startpoint), args=(input_stuff,), approx_grad=True,
                        bounds=bounds, disp=True, pgtol=1e-10, factr=10.)
    return res[0][0]


def GCV_cost_function(weight, input_stuff):
    """The GCV cost function that is iterated [4]
    """
    data, M, MMt, K, LR = input_stuff
    S = np.dot(np.dot(M, np.linalg.pinv(MMt + weight * LR)), M.T)
    trS = np.matrix.trace(S)
    normyytilde = np.linalg.norm(data - np.dot(S, data), 2)
    gcv_value = normyytilde / (K - trS)
    return gcv_value


def maptime_isotropic_scaling(data, q, tau):
    """  Constructs design matrix for fitting an exponential to the
    diffusion time points.
    """
    dataclip = np.clip(data, 1e-05, 1.)
    logE = -np.log(dataclip)
    logE_q = logE / (2 * np.pi ** 2)
    logE_tau = logE * 2

    B_q = np.array([q * q])
    inv_B_q = np.linalg.pinv(B_q)

    B_tau = np.array([tau])
    inv_B_tau = np.linalg.pinv(B_tau)

    us = np.sqrt(np.dot(logE_q, inv_B_q))
    ut = np.dot(logE_tau, inv_B_tau)
    return us, ut


def maptime_anisotropic_scaling(data, q, bvecs, tau):
    """  Constructs design matrix for fitting an exponential to the
    diffusion time points.
    """
    dataclip = np.clip(data, 1e-05, 10e10)
    logE = -np.log(dataclip)
    logE_q = logE / (2 * np.pi ** 2)
    logE_tau = logE * 2

    B_q = design_matrix_spatial(bvecs, q)
    inv_B_q = np.linalg.pinv(B_q)
    A = np.dot(inv_B_q, logE_q)

    evals, R = dti.decompose_tensor(dti.from_lower_triangular(A))
    us = np.sqrt(evals)

    B_tau = np.array([tau])
    inv_B_tau = np.linalg.pinv(B_tau)

    ut = np.dot(logE_tau, inv_B_tau)

    return us, ut, R


def design_matrix_spatial(bvecs, qvals, dtype=None):
    """  Constructs design matrix for DTI weighted least squares or
    least squares fitting. (Basser et al., 1994a)

    Parameters
    ----------
    gtab : A GradientTable class instance

    dtype : string
        Parameter to control the dtype of returned designed matrix

    Returns
    -------
    design_matrix : array (g,7)
        Design matrix or B matrix assuming Gaussian distributed tensor model
        design_matrix[j, :] = (Bxx, Byy, Bzz, Bxy, Bxz, Byz, dummy)
    """
    B = np.zeros((bvecs.shape[0], 6))
    B[:, 0] = bvecs[:, 0] * bvecs[:, 0] * 1. * qvals ** 2  # Bxx
    B[:, 1] = bvecs[:, 0] * bvecs[:, 1] * 2. * qvals ** 2  # Bxy
    B[:, 2] = bvecs[:, 1] * bvecs[:, 1] * 1. * qvals ** 2  # Byy
    B[:, 3] = bvecs[:, 0] * bvecs[:, 2] * 2. * qvals ** 2  # Bxz
    B[:, 4] = bvecs[:, 1] * bvecs[:, 2] * 2. * qvals ** 2  # Byz
    B[:, 5] = bvecs[:, 2] * bvecs[:, 2] * 1. * qvals ** 2  # Bzz
    return B


def create_rt_space_grid(grid_size_r, max_radius_r, grid_size_tau,
                         min_radius_tau, max_radius_tau):
    """ Generates EAP grid (for potential positivity constraint)."""
    tau_list = np.linspace(min_radius_tau, max_radius_tau, grid_size_tau)
    constraint_grid_tau = np.c_[0., 0., 0., 0.]
    for tau in tau_list:
        constraint_grid = mapmri.create_rspace(grid_size_r, max_radius_r)
        constraint_grid_tau = np.vstack(
            [constraint_grid_tau,
             np.c_[constraint_grid, np.zeros(constraint_grid.shape[0]) + tau]]
        )
    return constraint_grid_tau[1:]


def maptime_number_of_coefficients(radial_order, time_order):
    F = np.floor(radial_order / 2.)
    Msym = (F + 1) * (F + 2) * (4 * F + 3) / 6
    M_total = Msym * (time_order + 1)
    return M_total


def l1_crossvalidation(b0s_mask, E, M, weight_array=np.linspace(0, .4, 21)):
    dwi_mask = ~b0s_mask
    b0_mask = b0s_mask
    dwi_indices = np.arange(E.shape[0])[dwi_mask]
    b0_indices = np.arange(E.shape[0])[b0_mask]
    random.shuffle(dwi_indices)

    sub0 = dwi_indices[0::5]
    sub1 = dwi_indices[1::5]
    sub2 = dwi_indices[2::5]
    sub3 = dwi_indices[3::5]
    sub4 = dwi_indices[4::5]

    test0 = np.hstack((b0_indices, sub1, sub2, sub3, sub4))
    test1 = np.hstack((b0_indices, sub0, sub2, sub3, sub4))
    test2 = np.hstack((b0_indices, sub0, sub1, sub3, sub4))
    test3 = np.hstack((b0_indices, sub0, sub1, sub2, sub4))
    test4 = np.hstack((b0_indices, sub0, sub1, sub2, sub3))

    cv_list = (
        (sub0, test0),
        (sub1, test1),
        (sub2, test2),
        (sub3, test3),
        (sub4, test4)
    )

    errorlist = np.zeros((5, 21))
    errorlist[:, 0] = 100.
    optimal_alpha_sub = np.zeros(5)
    for i, (sub, test) in enumerate(cv_list):
        counter = 1
        cv_old = errorlist[i, 0]
        cv_new = errorlist[i, 0]
        while cv_old >= cv_new and counter < weight_array.shape[0]:
            alpha = weight_array[counter]
            c = cvxpy.Variable(M.shape[1])
            design_matrix = cvxpy.Constant(M[test])
            design_matrix_to_recover = cvxpy.Constant(M[sub])
            data = cvxpy.Constant(E[test])
            objective = cvxpy.Minimize(
                cvxpy.sum_squares(design_matrix * c - data) +
                alpha * cvxpy.norm1(c)
            )
            constraints = []
            prob = cvxpy.Problem(objective, constraints)
            prob.solve(solver="ECOS", verbose=False)
            recovered_signal = design_matrix_to_recover * c
            errorlist[i, counter] = np.mean(
                (E[sub] - np.asarray(recovered_signal.value).squeeze()) ** 2)
            cv_old = errorlist[i, counter - 1]
            cv_new = errorlist[i, counter]
            counter += 1
        optimal_alpha_sub[i] = weight_array[counter - 1]
    optimal_alpha = optimal_alpha_sub.mean()
    return optimal_alpha


def elastic_crossvalidation(b0s_mask, E, M, L, lopt,
                            weight_array=np.linspace(0, .2, 21)):
    dwi_mask = ~b0s_mask
    b0_mask = b0s_mask
    dwi_indices = np.arange(E.shape[0])[dwi_mask]
    b0_indices = np.arange(E.shape[0])[b0_mask]
    random.shuffle(dwi_indices)

    sub0 = dwi_indices[0::5]
    sub1 = dwi_indices[1::5]
    sub2 = dwi_indices[2::5]
    sub3 = dwi_indices[3::5]
    sub4 = dwi_indices[4::5]

    test0 = np.hstack((b0_indices, sub1, sub2, sub3, sub4))
    test1 = np.hstack((b0_indices, sub0, sub2, sub3, sub4))
    test2 = np.hstack((b0_indices, sub0, sub1, sub3, sub4))
    test3 = np.hstack((b0_indices, sub0, sub1, sub2, sub4))
    test4 = np.hstack((b0_indices, sub0, sub1, sub2, sub3))

    cv_list = (
        (sub0, test0),
        (sub1, test1),
        (sub2, test2),
        (sub3, test3),
        (sub4, test4)
    )

    errorlist = np.zeros((5, 21))
    errorlist[:, 0] = 100.
    optimal_alpha_sub = np.zeros(5)
    for i, (sub, test) in enumerate(cv_list):
        counter = 1
        cv_old = errorlist[i, 0]
        cv_new = errorlist[i, 0]
        alpha = cvxpy.Parameter(sign="positive")
        c = cvxpy.Variable(M.shape[1])
        design_matrix = cvxpy.Constant(M[test])
        design_matrix_to_recover = cvxpy.Constant(M[sub])
        data = cvxpy.Constant(E[test])
        objective = cvxpy.Minimize(
            cvxpy.sum_squares(design_matrix * c - data) +
            alpha * cvxpy.norm1(c) +
            lopt * cvxpy.quad_form(c, L)
        )
        constraints = []
        prob = cvxpy.Problem(objective, constraints)
        while cv_old >= cv_new and counter < weight_array.shape[0]:
            alpha.value = weight_array[counter]
            prob.solve(solver="ECOS", verbose=False)
            recovered_signal = design_matrix_to_recover * c
            errorlist[i, counter] = np.mean(
                (E[sub] - np.asarray(recovered_signal.value).squeeze()) ** 2)
            cv_old = errorlist[i, counter - 1]
            cv_new = errorlist[i, counter]
            counter += 1
        optimal_alpha_sub[i] = weight_array[counter - 1]
    optimal_alpha = optimal_alpha_sub.mean()
    return optimal_alpha
