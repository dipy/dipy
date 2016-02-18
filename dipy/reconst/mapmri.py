# -*- coding: utf-8 -*-
import numpy as np
from dipy.reconst.multi_voxel import multi_voxel_fit
from dipy.reconst.base import ReconstModel, ReconstFit
from dipy.reconst.cache import Cache
from scipy.special import hermite, gamma, genlaguerre
from scipy.misc import factorial, factorial2
from dipy.core.geometry import cart2sphere
from dipy.reconst.shm import real_sph_harm, sph_harm_ind_list
import dipy.reconst.dti as dti
from warnings import warn
from dipy.core.gradients import gradient_table
from ..utils.optpkg import optional_package

cvxopt, have_cvxopt, _ = optional_package("cvxopt")


class MapmriModel(Cache):

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

    .. [4] Fick et al. "MAPL: Tissue Microstructure Estimation Using
           Laplacian-Regularized MAP-MRI and its Application to HCP Data",
           NeuroImage, Submitted.

    .. [5] Cheng, J., 2014. Estimation and Processing of Ensemble Average
           Propagator and Its Features in Diffusion MRI. Ph.D. Thesis.

    .. [6] Hosseinbor et al. "Bessel fourier orientation reconstruction
           (bfor): An analytical diffusion propagator reconstruction for hybrid
           diffusion imaging and computation of q-space indices". NeuroImage
           64, 2013, 650–670.

    .. [7] Craven et al. "Smoothing Noisy Data with Spline Functions."
           NUMER MATH 31.4 (1978): 377-403.

    .. [8] Avram et al. "Clinical feasibility of using mean apparent
           propagator (MAP) MRI to characterize brain tissue microstructure".
           NeuroImage 2015, in press.
    """

    def __init__(self,
                 gtab,
                 radial_order=6,
                 laplacian_regularization=True,
                 laplacian_weighting=0.2,
                 positivity_constraint=False,
                 anisotropic_scaling=True,
                 eigenvalue_threshold=1e-04,
                 bval_threshold=np.inf,
                 DTI_scale_estimation=True,
                 pos_grid=10,
                 pos_radius=20e-3):
        r""" Analytical and continuous modeling of the diffusion signal with
        respect to the MAPMRI basis [1]_.

        The main idea is to model the diffusion signal as a linear combination
        of the continuous functions presented in [2]_ but extending it in three
        dimensions.

        The main difference with the SHORE proposed in [3]_ is that MAPMRI 3D
        extension is provided using a set of three basis functions for the
        radial part, one for the signal along x, one for y and one for z, while
        [3]_ uses one basis function to model the radial part and real
        Spherical Harmonics to model the angular part.

        From the MAPMRI coefficients it is possible to estimate various
        q-space indices, the PDF and the ODF.

        The fitting procedure can be constrained using the positivity
        constraint proposed in [1]_ and/or the laplacian regularization
        proposed in [4]_.

        For the estimation of q-space indices we recommend using the 'regular'
        anisotropic implementation of MAPMRI. However, it has been shown that
        the ODF estimation in this implementation has a bias which
        'squeezes together' the ODF peaks when there is a crossing at an angle
        smaller than 90 degrees [4]_. When you want to estimate ODFs for
        tractography we therefore recommend using the isotropic implementation
        (which is equivalent to [3]_).

        The switch between isotropic and anisotropic can be easily made through
        the anisotropic_scaling option.


        Parameters
        ----------
        gtab : GradientTable,
            gradient directions and bvalues container class
        radial_order : unsigned int,
            an even integer that represent the order of the basis
        laplacian_regularization: bool,
            Regularize using the Laplacian of the MAP-MRI basis.
        laplacian_weighting: string or scalar or ndarray,
            The string 'GCV' makes it use generalized cross-validation to find
            the regularization weight [3]. A scalar sets the regularization
            weight to that value. A numpy array of values will make the GCV
            function find the optimal value among those values.
        positivity_constraint : bool,
            Constrain the propagator to be positive.
        anisotropic_scaling : bool,
            If false, force the basis function to be identical in the three
            dimensions (SHORE like).
        eigenvalue_threshold : float,
            set the minimum of the tensor eigenvalues in order to avoid
            stability problem
        bval_threshold : float,
            sets the b-value threshold to be used in the scale factor
            estimation. In order for the stimated non-Gaussianity to have
            meaning this value should set to a lower value (b<2000 s/mm2)
            such that the scale factors are estimated on signal points that
            reasonably represent the spins at Gaussian diffusion.
        DTI_scale_estimation : bool,
            Whether or not DTI fitting is used to estimate the isotropic scale
            factor for isotropic MAP-MRI.
            When set to False vastly increases fitting speed by presetting
            the isotropic tissue diffusivity to typical white matter
            diffusivity of D=0.7e-3. Can only be used in combination with
            anisotropic_scaling=False, laplacian_regularization=True and
            regularization_weight set to a float.
            WARNING: This may reduce fitting quality.

        References
        ----------
        .. [1] Ozarslan E. et. al, "Mean apparent propagator (MAP) MRI: A novel
               diffusion imaging method for mapping tissue microstructure",
               NeuroImage, 2013.

        .. [2] Ozarslan E. et. al, "Simple harmonic oscillator based
               reconstruction and estimation for one-dimensional q-space
               magnetic resonance 1D-SHORE)", eapoc Intl Soc Mag Reson Med,
               vol. 16, p. 35., 2008.

        .. [3] Ozarslan E. et. al, "Simple harmonic oscillator based
               reconstruction and estimation for three-dimensional q-space
               mri", ISMRM 2009.

        .. [4] Fick et al. "MAPL: Tissue Microstructure Estimation Using
               Laplacian-Regularized MAP-MRI and its Application to HCP Data",
               NeuroImage, Under Review.

        Examples
        --------
        In this example, where the data, gradient table and sphere tessellation
        used for reconstruction are provided, we model the diffusion signal
        with respect to the SHORE basis and compute the real and analytical
        ODF.

        from dipy.data import get_data,get_sphere
        sphere = get_sphere('symmetric724')
        fimg, fbvals, fbvecs = get_data('ISBI_testing_2shells_table')
        bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
        gtab = gradient_table(bvals, bvecs)
        from dipy.sims.voxel import SticksAndBall
        data, golden_directions = SticksAndBall(gtab, d=0.0015,
                                                S0=1, angles=[(0, 0), (90, 0)],
                                                fractions=[50, 50], snr=None)
        from dipy.reconst.mapmri import MapmriModel
        radial_order = 4
        map_model = MapmriModel(gtab, radial_order=radial_order)
        mapfit = map_model.fit(data)
        odf= mapfit.odf(sphere)
        """

        self.bvals = gtab.bvals
        self.bvecs = gtab.bvecs
        self.gtab = gtab
        self.radial_order = radial_order
        self.bval_threshold = bval_threshold
        self.DTI_scale_estimation = DTI_scale_estimation
        self.pos_grid = pos_grid
        self.pos_radius = pos_radius

        self.laplacian_regularization = laplacian_regularization
        if self.laplacian_regularization:
            msg = "Laplacian Regularization weighting must be 'GCV', "
            msg += "a positive float or an array of positive floats."
            if isinstance(laplacian_weighting, str):
                if laplacian_weighting != 'GCV':
                    raise ValueError(msg)
            elif (
                isinstance(laplacian_weighting, float) or
                isinstance(laplacian_weighting, np.ndarray)
            ):
                if np.sum(laplacian_weighting < 0) > 0:
                    raise ValueError(msg)
            self.laplacian_weighting = laplacian_weighting

        self.positivity_constraint = positivity_constraint
        if self.positivity_constraint:
            if not have_cvxopt:
                raise ValueError(
                    'CVXOPT package needed to enforce constraints')
            constraint_grid = create_rspace(pos_grid, pos_radius)
            self.constraint_grid = constraint_grid
            self.pos_K_independent = mapmri_isotropic_K_mu_independent(
                radial_order, constraint_grid)

        self.anisotropic_scaling = anisotropic_scaling
        if (gtab.big_delta is None) or (gtab.small_delta is None):
            self.tau = 1 / (4 * np.pi ** 2)
        else:
            self.tau = gtab.big_delta - gtab.small_delta / 3.0
        self.eigenvalue_threshold = eigenvalue_threshold

        self.cutoff = gtab.bvals < self.bval_threshold
        gtab_cutoff = gradient_table(bvals=self.gtab.bvals[self.cutoff],
                                     bvecs=self.gtab.bvecs[self.cutoff])
        self.tenmodel = dti.TensorModel(gtab_cutoff)

        self.quick_fit = False
        if self.anisotropic_scaling:
            self.ind_mat = mapmri_index_matrix(self.radial_order)
            self.Bm = b_mat(self.ind_mat)
            self.R_mat, self.L_mat, self.S_mat = mapmri_RLS_reg_matrices(
                radial_order)
        else:
            self.ind_mat = mapmri_isotropic_index_matrix(self.radial_order)
            self.Bm = b_mat_isotropic(self.ind_mat)
            if self.laplacian_regularization:
                self.laplacian_matrix = mapmri_isotropic_laplacian_reg_matrix(
                    radial_order, 1.)

            qvals = np.sqrt(self.gtab.bvals / self.tau) / (2 * np.pi)
            q = gtab.bvecs * qvals[:, None]
            if self.DTI_scale_estimation:
                self.M_mu_independent = mapmri_isotropic_M_mu_independent(
                    self.radial_order, q)
            else:
                D = 0.7e-3  # average choice for white matter diffusivity
                mumean = np.sqrt(2 * D * self.tau)
                self.mu = np.array([mumean, mumean, mumean])
                self.M = mapmri_isotropic_phi_matrix(radial_order, mumean, q)
                if (self.laplacian_regularization and
                   isinstance(laplacian_weighting, float) and
                   not positivity_constraint):
                    MMt = (np.dot(self.M.T, self.M) +
                           laplacian_weighting * mumean
                           * self.laplacian_matrix)
                    self.MMt_inv_Mt = np.dot(np.linalg.pinv(MMt), self.M.T)
                    self.quick_fit = True

    @multi_voxel_fit
    def fit(self, data):

        tenfit = self.tenmodel.fit(data[self.cutoff])
        evals = tenfit.evals
        R = tenfit.evecs
        evals = np.clip(evals, self.eigenvalue_threshold, evals.max())
        qvals = np.sqrt(self.gtab.bvals / self.tau) / (2 * np.pi)
        if self.anisotropic_scaling:
            mu = np.sqrt(evals * 2 * self.tau)
            qvecs = np.dot(self.gtab.bvecs, R)
            q = qvecs * qvals[:, None]
            M = mapmri_phi_matrix(self.radial_order, mu, q.T)
        else:
            if self.quick_fit:
                lopt = self.laplacian_weighting
                coef = np.dot(self.MMt_inv_Mt, data)
                coef = coef / sum(coef * self.Bm)
                return MapmriFit(self, coef, self.mu, R, self.ind_mat, lopt)
            else:
                mumean = np.sqrt(evals.mean() * 2 * self.tau)
                mu = np.array([mumean, mumean, mumean])
                q = self.gtab.bvecs * qvals[:, None]
                M_mu_dependent = mapmri_isotropic_M_mu_dependent(
                    self.radial_order, mu[0], qvals)
                M = M_mu_dependent * self.M_mu_independent

        if self.laplacian_regularization:
            if self.anisotropic_scaling:
                laplacian_matrix = mapmri_laplacian_reg_matrix(
                    self.ind_mat, mu, self.R_mat, self.L_mat, self.S_mat)
            else:
                laplacian_matrix = self.laplacian_matrix * mu[0]
            if self.laplacian_weighting == 'GCV':
                lopt = generalized_crossvalidation(data, M, laplacian_matrix)
            elif np.isscalar(self.laplacian_weighting):
                lopt = self.laplacian_weighting
            elif type(self.laplacian_weighting) == np.ndarray:
                lopt = generalized_crossvalidation(data, M, laplacian_matrix,
                                                   self.laplacian_weighting)
        else:
            lopt = 0.
            laplacian_matrix = np.ones((self.ind_mat.shape[0],
                                        self.ind_mat.shape[0]))

        if self.positivity_constraint:
            w_s = "The MAPMRI positivity constraint depends on CVXOPT "
            w_s += "(http://cvxopt.org/). CVXOPT is licensed "
            w_s += "under the GPL (see: http://cvxopt.org/copyright.html) "
            w_s += "and you may be subject to this license when using the "
            w_s += "positivity constraint."
            warn(w_s)
            constraint_grid = self.constraint_grid
            if self.anisotropic_scaling:
                K = mapmri_psi_matrix(self.radial_order, mu, constraint_grid)
            else:
                K_dependent = mapmri_isotropic_K_mu_dependent(
                    self.radial_order, mu[0], constraint_grid)
                K = K_dependent * self.pos_K_independent
            Q = cvxopt.matrix(np.dot(M.T, M) + lopt * laplacian_matrix)
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
                np.linalg.inv(np.dot(M.T, M) + lopt * laplacian_matrix), M.T)
            coef = np.dot(pseudoInv, data)

        coef = coef / sum(coef * self.Bm)

        return MapmriFit(self, coef, mu, R, self.ind_mat, lopt)


class MapmriFit(ReconstFit):

    def __init__(self, model, mapmri_coef, mu, R, ind_mat, lopt):
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
        self.lopt = lopt

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

    def odf(self, sphere, s=2):
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

        if self.model.anisotropic_scaling:
            v_ = sphere.vertices
            v = np.dot(v_, self.R)
            I_s = mapmri_odf_matrix(self.radial_order, self.mu, s, v)
            odf = np.dot(I_s, self._mapmri_coef)
        else:
            I = self.model.cache_get('ODF_matrix', key=(sphere, s))
            if I is None:
                I = mapmri_isotropic_odf_matrix(self.radial_order, 1,
                                                s, sphere.vertices)
                self.model.cache_set('ODF_matrix', (sphere, s), I)

            odf = self.mu[0] ** s * np.dot(I, self._mapmri_coef)

        return odf

    def odf_sh(self, s=2):
        r""" Calculates the real analytical odf for a given discrete sphere.
        Computes the design matrix of the ODF for the given sphere vertices
        and radial moment. The radial moment s acts as a
        sharpening method [1,2].
        ..math::
            :nowrap:
                \begin{equation}\label{eq:ODF}
                    \textrm{ODF}_s(\textbf{u}_r)=\int r^{2+s}
                    P(r\cdot \textbf{u}_r)dr.
                \end{equation}
        """
        if self.model.anisotropic_scaling:
            msg = 'odf in spherical harmonics not yet implemented for '
            msg += 'anisotropic implementation'
            ValueError(msg)
        I = self.model.cache_get('ODF_sh_matrix', key=(self.radial_order, s))

        if I is None:
            I = mapmri_isotropic_odf_sh_matrix(self.radial_order, 1, s)
            self.model.cache_set('ODF_sh_matrix', (self.radial_order, s), I)

        odf = self.mu[0] ** s * np.dot(I, self._mapmri_coef)

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
        ind_mat = self.ind_mat
        if self.model.anisotropic_scaling:
            rtpp = 0
            const = 1 / (np.sqrt(2 * np.pi) * self.mu[0])
            for i in range(ind_mat.shape[0]):
                if Bm[i] > 0.0:
                    rtpp += ((-1.0) ** (ind_mat[i, 0] / 2.0)
                             * self._mapmri_coef[i] * Bm[i])
            return const * rtpp

        else:
            rtpp_vec = np.zeros((ind_mat.shape[0]))
            count = 0
            for n in range(0, self.model.radial_order + 1, 2):
                    for j in range(1, 2 + n / 2):
                        l = n + 2 - 2 * j
                        const = (-1/2.0) ** (l/2) / np.sqrt(np.pi)
                        matsum = 0
                        for k in range(0, j):
                            matsum += (-1) ** k * \
                                binomialfloat(j + l - 0.5, j - k - 1) *\
                                gamma(l / 2 + k + 1 / 2.0) /\
                                (factorial(k) * 0.5 ** (l / 2 + 1 / 2.0 + k))
                        for m in range(-l, l + 1):
                            rtpp_vec[count] = const * matsum
                            count += 1

            direction = np.array(self.R[:, 0], ndmin=2)
            r, theta, phi = cart2sphere(direction[:, 0], direction[:, 1],
                                        direction[:, 2])

            rtpp = self._mapmri_coef * (1 / self.mu[0]) *\
                rtpp_vec * real_sph_harm(ind_mat[:, 2], ind_mat[:, 1],
                                         theta, phi)

            return rtpp.sum()

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
        ind_mat = self.ind_mat
        if self.model.anisotropic_scaling:
            rtap = 0
            const = 1 / (2 * np.pi * self.mu[1] * self.mu[2])
            for i in range(ind_mat.shape[0]):
                if Bm[i] > 0.0:
                    rtap += ((-1.0) ** ((ind_mat[i, 1] + ind_mat[i, 2]) / 2.0)
                             * self._mapmri_coef[i] * Bm[i])
            return const * rtap
        else:
            rtap_vec = np.zeros((ind_mat.shape[0]))
            count = 0

            for n in range(0, self.model.radial_order + 1, 2):
                for j in range(1, 2 + n / 2):
                    l = n + 2 - 2 * j
                    kappa = ((-1) ** (j - 1) * 2 ** (-(l + 3) / 2.0)) / np.pi
                    matsum = 0
                    for k in range(0, j):
                        matsum += ((-1) ** k *
                                   binomialfloat(j + l - 0.5, j - k - 1) *
                                   gamma((l + 1) / 2.0 + k)) /\
                            (factorial(k) * 0.5 ** ((l + 1) / 2.0 + k))
                    for m in range(-l, l + 1):
                        rtap_vec[count] = kappa * matsum
                        count += 1
            rtap_vec *= 2

            direction = np.array(self.R[:, 0], ndmin=2)
            r, theta, phi = cart2sphere(direction[:, 0],
                                        direction[:, 1], direction[:, 2])
            rtap = self._mapmri_coef * (1 / self.mu[0] ** 2) *\
                rtap_vec * real_sph_harm(ind_mat[:, 2], ind_mat[:, 1],
                                         theta, phi)
            return rtap.sum()

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

        if self.model.anisotropic_scaling:
            rtop = 0
            const = 1 / \
                np.sqrt(
                    8 * np.pi ** 3 * (self.mu[0] ** 2 * self.mu[1] ** 2 *
                                      self.mu[2] ** 2))
            for i in range(self.ind_mat.shape[0]):
                if Bm[i] > 0.0:
                    rtop += (-1.0) ** (
                        (self.ind_mat[i, 0] + self.ind_mat[i, 1] +
                         self.ind_mat[i, 2])
                        / 2.0) * self._mapmri_coef[i] * Bm[i]
            rtop = const * rtop
        else:
            const = 1 / (2 * np.sqrt(2.0) * np.pi ** (3 / 2.0))
            rtop_vec = const * (-1.0) ** (self.model.ind_mat[:, 0]-1) * Bm
            rtop = (1 / self.mu[0] ** 3) * rtop_vec * self._mapmri_coef
            rtop = rtop.sum()
        return rtop

    def msd(self):
        r""" Calculates the analytical Mean Squared Displacement (MSD).
        The analytical formula was derived through the Laplacian of the origin
        of the estimated signal [4].

        References
        ----------
        .. [5] Cheng, J., 2014. Estimation and Processing of Ensemble Average
        Propagator and Its Features in Diffusion MRI. Ph.D. Thesis.
        """

        mu = self.mu
        ind_mat = self.model.ind_mat
        if self.model.anisotropic_scaling:
            msd = 0
            for i in range(ind_mat.shape[0]):
                nx, ny, nz = ind_mat[i]
                if not(nx % 2) and not(ny % 2) and not(nz % 2):
                    msd += (
                        self._mapmri_coef[i] *
                        (-1) ** (0.5 * (- nx - ny - nz)) *
                        np.pi ** (3 / 2.0) *
                        ((1 + 2 * nx) * mu[0] ** 2 + (1 + 2 * ny) *
                         mu[1] ** 2 + (1 + 2 * nz) * mu[2] ** 2) /
                        (np.sqrt(2 ** (-nx - ny - nz) *
                         factorial(nx) * factorial(ny) * factorial(nz)) *
                         gamma(0.5 - 0.5 * nx) * gamma(0.5 - 0.5 * ny) *
                         gamma(0.5 - 0.5 * nz))
                        )
        else:
            Bm = self.model.Bm
            msd_vec = (4 * ind_mat[:, 0] - 1) * Bm
            msd = self.mu[0] ** 2 * msd_vec * self._mapmri_coef
            msd = msd.sum()
        return msd

    def qiv(self):
        r""" Calculates the analytical Q-space Inverse Variance (QIV).
        The analytical formula was derived through the Laplacian of the origin
        of the estimated propagator [5].

        References
        ----------
        .. [6] Hosseinbor et al. "Bessel fourier orientation reconstruction
        (bfor): An analytical diffusion propagator reconstruction for hybrid
        diffusion imaging and computation of q-space indices. NeuroImage 64,
        2013, 650–670.
        """
        ux, uy, uz = self.mu
        ind_mat = self.model.ind_mat

        if self.model.anisotropic_scaling:
            qiv = 0
            for i in range(ind_mat.shape[0]):
                nx, ny, nz = ind_mat[i]

                if not nx % 2 and not ny % 2 and not nz % 2:
                    numerator = 8 * np.pi ** 2 * (ux * uy * uz) ** 3 *\
                        np.sqrt(factorial(nx) * factorial(ny) * factorial(nz)) *\
                        gamma(0.5 - 0.5 * nx) * gamma(0.5 - 0.5 * ny) * \
                        gamma(0.5 - 0.5 * nz)

                    denominator = np.sqrt(2 ** (-1 + nx + ny + nz)) *\
                        ((1 + 2 * nx) * uy ** 2 * uz ** 2 + ux ** 2 *
                         ((1 + 2 * nz) * uy ** 2 + (1 + 2 * ny) * uz ** 2))
                    qiv += self._mapmri_coef[i] * (numerator / denominator)
        else:
            Bm = self.model.Bm
            j = ind_mat[:, 0]
            qiv_vec = ((8 * (-1) ** (1 - j) * np.sqrt(2) * np.pi ** (7 / 2.))
                       / ((4 * j - 1) * Bm))
            qiv_vec[-np.isfinite(qiv_vec)] = 0.
            qiv = ux ** 5 * qiv_vec * self._mapmri_coef
            qiv = qiv.sum()
        return qiv

    def ng(self):
        r""" Calculates the analytical non-Gaussiannity (NG) [1]_.

        References
        ----------
        .. [1] Ozarslan E. et. al, "Mean apparent propagator (MAP) MRI: A novel
        diffusion imaging method for mapping tissue microstructure",
        NeuroImage, 2013.

        .. [2] Avram et al. "Clinical feasibility of using mean apparent
        propagator (MAP) MRI to characterize brain tissue microstructure".
        NeuroImage 2015, in press.
        """
        if self.model.bval_threshold > 2000.:
            msg = 'model bval_threshold must be lower than 2000 for the '
            msg += 'non_Gaussianity to be physically meaningful [2].'
            warn(msg)
        if not self.model.anisotropic_scaling:
            msg = 'Parallel non-Gaussianity is not defined using '
            msg += 'isotropic scaling.'
            ValueError(msg)

        coef = self._mapmri_coef
        return np.sqrt(1 - coef[0] ** 2 / np.sum(coef ** 2))

    def ng_parallel(self):
        r""" Calculates the analytical parallel non-Gaussiannity (NG) [1]_.

        References
        ----------
        .. [1] Ozarslan E. et. al, "Mean apparent propagator (MAP) MRI: A novel
        diffusion imaging method for mapping tissue microstructure",
        NeuroImage, 2013.

        .. [2] Avram et al. "Clinical feasibility of using mean apparent
        propagator (MAP) MRI to characterize brain tissue microstructure".
        NeuroImage 2015, in press.
        """
        if self.model.bval_threshold > 2000.:
            msg = 'Model bval_threshold must be lower than 2000 for the '
            msg += 'non_Gaussianity to be physically meaningful [2].'
            warn(msg)
        if not self.model.anisotropic_scaling:
            msg = 'Parallel non-Gaussianity is not defined using '
            msg += 'isotropic scaling.'
            ValueError(msg)

        ind_mat = self.model.ind_mat
        coef = self._mapmri_coef
        a_par = np.zeros_like(coef)
        a0 = np.zeros_like(coef)

        for i in range(coef.shape[0]):
            n1, n2, n3 = ind_mat[i]
            if (n2 % 2 + n3 % 2) == 0:
                a_par[i] = coef[i] * (-1) ** ((n2 + n3) / 2) *\
                    np.sqrt(factorial(n2) * factorial(n3)) /\
                    (factorial2(n2) * factorial2(n3))
                if n1 == 0:
                    a0[i] = a_par[i]
        return np.sqrt(1 - np.sum(a0 ** 2) / np.sum(a_par ** 2))

    def ng_perpendicular(self):
        r""" Calculates the analytical perpendicular non-Gaussiannity (NG) [1]_.

        References
        ----------
        .. [1] Ozarslan E. et. al, "Mean apparent propagator (MAP) MRI: A novel
        diffusion imaging method for mapping tissue microstructure",
        NeuroImage, 2013.

        .. [2] Avram et al. "Clinical feasibility of using mean apparent
        propagator (MAP) MRI to characterize brain tissue microstructure".
        NeuroImage 2015, in press.
        """
        if self.model.bval_threshold > 2000.:
            msg = 'model bval_threshold must be lower than 2000 for the '
            msg += 'non_Gaussianity to be physically meaningful [2].'
            warn(msg)
        if not self.model.anisotropic_scaling:
            msg = 'Parallel non-Gaussianity is not defined using '
            msg += 'isotropic scaling.'
            ValueError(msg)

        ind_mat = self.model.ind_mat
        coef = self._mapmri_coef
        a_perp = np.zeros_like(coef)
        a00 = np.zeros_like(coef)

        for i in range(coef.shape[0]):
            n1, n2, n3 = ind_mat[i]
            if n1 % 2 == 0:
                if n2 % 2 == 0 and n3 % 2 == 0:
                    a_perp[i] = coef[i] * (-1) ** (n1 / 2) *\
                        np.sqrt(factorial(n1)) / factorial2(n1)
                    if n2 == 0 and n3 == 0:
                        a00[i] = a_perp[i]
        return np.sqrt(1 - np.sum(a00 ** 2) / np.sum(a_perp ** 2))

    def fitted_signal(self, gtab=None):
        """
        Recovers the fitted signal for the given gradient table. If no gradient
        table is given it recovers the signal for the gtab of the data.
        """
        if gtab is None:
            E = self.predict(self.model.gtab)
        else:
            E = self.predict(gtab)
        return E

    def predict(self, qvals_or_gtab, S0=1.):
        r'''Recovers the reconstructed signal for any qvalue array or
        gradient table.
        '''
        if isinstance(qvals_or_gtab, np.ndarray):
            q = qvals_or_gtab
            qvals = np.linalg.norm(q, axis=1)
        else:
            gtab = qvals_or_gtab
            qvals = np.sqrt(gtab.bvals / self.model.tau) / (2 * np.pi)
            q = qvals[:, None] * gtab.bvecs

        if self.model.anisotropic_scaling:
            q_rot = np.dot(q, self.R)
            M = mapmri_phi_matrix(self.radial_order, self.mu, q_rot.T)
        else:
            M = mapmri_isotropic_phi_matrix(self.radial_order, self.mu[0], q)

        E = S0 * np.dot(M, self._mapmri_coef)
        return E

    def pdf(self, r_points):
        """ Diffusion propagator on a given set of real points.
            if the array r_points is non writeable, then intermediate
            results are cached for faster recalculation
        """
        if self.model.anisotropic_scaling:
            r_point_rotated = np.dot(r_points, self.R)
            K = mapmri_psi_matrix(self.radial_order, self.mu, r_point_rotated)
            EAP = np.dot(K, self._mapmri_coef)
        else:
            if not r_points.flags.writeable:
                K_independent = self.model.cache_get(
                    'mapmri_matrix_pdf_independent', key=hash(r_points.data))
                if K_independent is None:
                    K_independent = mapmri_isotropic_K_mu_independent(
                        self.radial_order, r_points)
                    self.model.cache_set('mapmri_matrix_pdf_independent',
                                         hash(r_points.data), K_independent)
                K_dependent = mapmri_isotropic_K_mu_dependent(
                    self.radial_order, self.mu[0], r_points)
                K = K_dependent * K_independent
            else:
                K = mapmri_isotropic_psi_matrix(
                    self.radial_order, self.mu[0], r_points)
            EAP = np.dot(K, self._mapmri_coef)

        return EAP


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
        B[i] = (
            K * np.sqrt(factorial(n1) * factorial(n2) * factorial(n3)) /
            (factorial2(n1) * factorial2(n2) * factorial2(n3))
            )

    return B


def b_mat_isotropic(ind_mat_isotropic):
    r""" Calculates the isotropic B coefficients from [1]_ Appendix, Fig 8.

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

    b_mat = np.zeros((ind_mat_isotropic.shape[0]))
    for i in range(ind_mat_isotropic.shape[0]):
        if ind_mat_isotropic[i, 1] == 0:
            b_mat[i] = genlaguerre(ind_mat_isotropic[i, 0] - 1, 0.5)(0)

    return b_mat


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
                        (vertices[:, 1] / muy) ** 2 +
                        (vertices[:, 2] / muz) ** 2)
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


def mapmri_isotropic_phi_matrix(radial_order, mu, q):
    r""" Three dimensional isotropic MAPMRI signal basis function from [1]_
    Eq. 57.

    Parameters
    ----------
    n : array, shape (3,)
        order of the basis function for x, y, z
    q : array, shape (N,3)
        points in the q-space in which evaluate the basis
    mu : float,
        isotropic scale factor of the basis

    References
    ----------
    .. [1] Ozarslan E. et. al, "Mean apparent propagator (MAP) MRI: A novel
    diffusion imaging method for mapping tissue microstructure",
    NeuroImage, 2013.
    """
    qval, theta, phi = cart2sphere(q[:, 0], q[:, 1],
                                   q[:, 2])
    theta[np.isnan(theta)] = 0

    ind_mat = mapmri_isotropic_index_matrix(radial_order)

    n_elem = ind_mat.shape[0]
    n_qgrad = q.shape[0]
    M = np.zeros((n_qgrad, n_elem))

    counter = 0
    for n in range(0, radial_order + 1, 2):
        for j in range(1, 2 + n / 2):
            l = n + 2 - 2 * j
            const = (-1) ** (l / 2) * np.sqrt(4.0 * np.pi) *\
                (2 * np.pi ** 2 * mu ** 2 * qval ** 2) ** (l / 2) *\
                np.exp(-2 * np.pi ** 2 * mu ** 2 * qval ** 2) *\
                genlaguerre(j - 1, l + 0.5)(4 * np.pi ** 2 * mu ** 2 *
                                            qval ** 2)
            for m in range(-l, l+1):
                M[:, counter] = const * real_sph_harm(m, l, theta, phi)
                counter += 1
    return M


def mapmri_isotropic_M_mu_independent(radial_order, q):
    r"""Computed the mu independent part of the signal design matrix.
    """
    ind_mat = mapmri_isotropic_index_matrix(radial_order)

    qval, theta, phi = cart2sphere(q[:, 0], q[:, 1],
                                   q[:, 2])
    theta[np.isnan(theta)] = 0

    n_elem = ind_mat.shape[0]
    n_rgrad = theta.shape[0]
    Q_mu_independent = np.zeros((n_rgrad, n_elem))

    counter = 0
    for n in range(0, radial_order + 1, 2):
        for j in range(1, 2 + n / 2):
            l = n + 2 - 2 * j
            const = np.sqrt(4 * np.pi) * (-1) ** (-l / 2) * \
                (2 * np.pi ** 2 * qval ** 2) ** (l / 2)
            for m in range(-1 * (n + 2 - 2 * j), (n + 3 - 2 * j)):
                Q_mu_independent[:, counter] = const * \
                    real_sph_harm(m, l, theta, phi)
                counter += 1
    return Q_mu_independent


def mapmri_isotropic_M_mu_dependent(radial_order, mu, qval):
    '''Computed the mu dependent part of the signal design matrix.
    '''
    ind_mat = mapmri_isotropic_index_matrix(radial_order)

    n_elem = ind_mat.shape[0]
    n_qgrad = qval.shape[0]
    Q_u0_dependent = np.zeros((n_qgrad, n_elem))

    counter = 0

    for n in range(0, radial_order + 1, 2):
        for j in range(1, 2 + n / 2):
            l = n + 2 - 2 * j
            const = mu ** l * np.exp(-2 * np.pi ** 2 * mu ** 2 * qval ** 2) *\
                genlaguerre(j - 1, l + 0.5)(4 * np.pi ** 2 * mu ** 2 *
                                            qval ** 2)
            for m in range(-l, l + 1):
                Q_u0_dependent[:, counter] = const
                counter += 1

    return Q_u0_dependent


def mapmri_isotropic_psi_matrix(radial_order, mu, rgrad):
    r""" Three dimensional isotropic MAPMRI propagator basis function from [1]_
    Eq. 61.

    Parameters
    ----------
    n : array, shape (3,)
        order of the basis function for x, y, z
    rgrad : array, shape (N,3)
        points in the r-space in which evaluate the basis
    mu : array, shape (3,)
        isotropic scale factor of the basis

    References
    ----------
    .. [1] Ozarslan E. et. al, "Mean apparent propagator (MAP) MRI: A novel
    diffusion imaging method for mapping tissue microstructure",
    NeuroImage, 2013.
    """

    r, theta, phi = cart2sphere(rgrad[:, 0], rgrad[:, 1],
                                rgrad[:, 2])
    theta[np.isnan(theta)] = 0

    ind_mat = mapmri_isotropic_index_matrix(radial_order)

    n_elem = ind_mat.shape[0]
    n_rgrad = rgrad.shape[0]
    K = np.zeros((n_rgrad, n_elem))

    counter = 0
    for n in range(0, radial_order + 1, 2):
        for j in range(1, 2 + n / 2):
            l = n + 2 - 2 * j
            const = (-1) ** (j - 1) * \
                    (np.sqrt(2) * np.pi * mu ** 3) ** (-1) *\
                    (r ** 2 / (2 * mu ** 2)) ** (l / 2) *\
                np.exp(- r ** 2 / (2 * mu ** 2)) *\
                genlaguerre(j - 1, l + 0.5)(r ** 2 / mu ** 2)
            for m in range(-l, l + 1):
                K[:, counter] = const * real_sph_harm(m, l, theta, phi)
                counter += 1

    return K


def mapmri_isotropic_K_mu_independent(radial_order, rgrad):
    '''Computes mu independent part of K. Same trick as with M.
    '''
    r, theta, phi = cart2sphere(rgrad[:, 0], rgrad[:, 1],
                                rgrad[:, 2])
    theta[np.isnan(theta)] = 0

    ind_mat = mapmri_isotropic_index_matrix(radial_order)

    n_elem = ind_mat.shape[0]
    n_rgrad = rgrad.shape[0]
    K = np.zeros((n_rgrad, n_elem))

    counter = 0
    for n in range(0, radial_order + 1, 2):
        for j in range(1, 2 + n / 2):
            l = n + 2 - 2 * j
            const = (-1) ** (j - 1) *\
                (np.sqrt(2) * np.pi) ** (-1) *\
                (r ** 2 / 2) ** (l / 2)
            for m in range(-l, l+1):
                K[:, counter] = const * real_sph_harm(m, l, theta, phi)
                counter += 1
    return K


def mapmri_isotropic_K_mu_dependent(radial_order, mu, rgrad):
    '''Computes mu dependent part of M. Same trick as with M.
    '''
    r, theta, phi = cart2sphere(rgrad[:, 0], rgrad[:, 1],
                                rgrad[:, 2])
    theta[np.isnan(theta)] = 0

    ind_mat = mapmri_isotropic_index_matrix(radial_order)

    n_elem = ind_mat.shape[0]
    n_rgrad = rgrad.shape[0]
    K = np.zeros((n_rgrad, n_elem))

    counter = 0
    for n in range(0, radial_order + 1, 2):
        for j in range(1, 2 + n / 2):
            l = n + 2 - 2 * j
            const = (mu ** 3) ** (-1) * mu ** (-l) *\
                np.exp(-r ** 2 / (2 * mu ** 2)) *\
                genlaguerre(j - 1, l + 0.5)(r ** 2 / mu ** 2)
            for m in range(-l, l + 1):
                K[:, counter] = const
                counter += 1
    return K


def binomialfloat(n, k):
    """Custom Binomial function
    """
    return factorial(n) / (factorial(n - k) * factorial(k))


def mapmri_isotropic_odf_matrix(radial_order, mu, s, vertices):
    r"""Compute the isotropic MAPMRI ODF matrix [1]_ Eq. 32 but for the
    isotropic propagator in [1]_ eq. 60. Analytical derivation in [2]_.

    Parameters
    ----------
    radial_order : unsigned int,
        an even integer that represent the order of the basis
    mu : float,
        isotropic scale factor of the isotropic MAP-MRI basis
    s : unsigned int
        radial moment of the ODF
    vertices : array, shape (N,3)
        points of the sphere shell in the r-space in which evaluate the ODF

    Returns
    -------
    odf_mat : Matrix, shape (N_vertices, N_mapmri_coef)
        ODF design matrix to discrete sphere function

    References
    ----------
    .. [1] Ozarslan E. et. al, "Mean apparent propagator (MAP) MRI: A novel
    diffusion imaging method for mapping tissue microstructure",
    NeuroImage, 2013.

    .. [2]_ Fick et al. "MAPL: Tissue Microstructure Estimation Using
    Laplacian-Regularized MAP-MRI and its Application to HCP Data",
    NeuroImage, Under Review.
    """
    r, theta, phi = cart2sphere(vertices[:, 0], vertices[:, 1],
                                vertices[:, 2])

    theta[np.isnan(theta)] = 0
    ind_mat = mapmri_isotropic_index_matrix(radial_order)
    n_vert = vertices.shape[0]
    n_elem = ind_mat.shape[0]
    odf_mat = np.zeros((n_vert, n_elem))

    counter = 0
    for n in range(0, radial_order + 1, 2):
        for j in range(1, 2 + n / 2):
            l = n + 2 - 2 * j
            kappa = ((-1) ** (j - 1) * 2 ** (-(l + 3) / 2.0) * mu ** s) / np.pi
            matsum = 0
            for k in range(0, j):
                matsum += ((-1) ** k * binomialfloat(j + l - 0.5, j - k - 1) *
                           gamma((l + s + 3) / 2.0 + k)) /\
                    (factorial(k) * 0.5 ** ((l + s + 3) / 2.0 + k))
            for m in range(-l, l + 1):
                odf_mat[:, counter] = kappa * matsum *\
                    real_sph_harm(m, l, theta, phi)
                counter += 1

    return odf_mat


def mapmri_isotropic_odf_sh_matrix(radial_order, mu, s):
    r"""Compute the isotropic MAPMRI ODF matrix [1]_ Eq. 32 for the isotropic
    propagator in [1]_ eq. 60. Here we do not compute the sphere function but
    the spherical harmonics by only integrating the radial part of the
    propagator. We use the same derivation of the ODF in the isotropic
    implementation as in [2]_.

    Parameters
    ----------
    radial_order : unsigned int,
        an even integer that represent the order of the basis
    mu : float,
        isotropic scale factor of the isotropic MAP-MRI basis
    s : unsigned int
        radial moment of the ODF

    Returns
    -------
    odf_sh_mat : Matrix, shape (N_sh_coef, N_mapmri_coef)
        ODF design matrix to spherical harmonics

    References
    ----------
    .. [1] Ozarslan E. et. al, "Mean apparent propagator (MAP) MRI: A novel
    diffusion imaging method for mapping tissue microstructure",
    NeuroImage, 2013.

    .. [2]_ Fick et al. "MAPL: Tissue Microstructure Estimation Using
    Laplacian-Regularized MAP-MRI and its Application to HCP Data",
    NeuroImage, Under Review.
    """
    sh_mat = sph_harm_ind_list(radial_order)
    ind_mat = mapmri_isotropic_index_matrix(radial_order)
    n_elem_shore = ind_mat.shape[0]
    n_elem_sh = sh_mat[0].shape[0]
    odf_sh_mat = np.zeros((n_elem_sh, n_elem_shore))

    counter = 0
    for n in range(0, radial_order + 1, 2):
        for j in range(1, 2 + n / 2):
            l = n + 2 - 2 * j
            kappa = ((-1) ** (j - 1) * 2 ** (-(l + 3) / 2.0) * mu ** s) / np.pi
            matsum = 0
            for k in range(0, j):
                matsum += ((-1) ** k * binomialfloat(j + l - 0.5, j - k - 1) *
                           gamma((l + s + 3) / 2.0 + k)) /\
                    (factorial(k) * 0.5 ** ((l + s + 3) / 2.0 + k))
            for m in range(-l, l + 1):
                index_overlap = np.all([l == sh_mat[1], m == sh_mat[0]], 0)
                odf_sh_mat[:, counter] = kappa * matsum * index_overlap
                counter += 1

    return odf_sh_mat


def mapmri_isotropic_laplacian_reg_matrix(radial_order, mu):
    r''' Computes the Laplacian regularization matrix for MAP-MRI's isotropic
    implementation [1]_.

    Parameters
    ----------
    radial_order : unsigned int,
        an even integer that represent the order of the basis
    mu : float,
        isotropic scale factor of the isotropic MAP-MRI basis

    Returns
    -------
    LR : Matrix, shape (N_coef, N_coef)
        Laplacian regularization matrix

    References
    ----------
    .. [1]_ Fick et al. "MAPL: Tissue Microstructure Estimation Using
    Laplacian-Regularized MAP-MRI and its Application to HCP Data",
    NeuroImage, Under Review.
    '''
    ind_mat = mapmri_isotropic_index_matrix(radial_order)

    n_elem = ind_mat.shape[0]

    LR = np.zeros((n_elem, n_elem))

    for i in range(n_elem):
        for k in range(i, n_elem):
            if ind_mat[i, 1] == ind_mat[k, 1] and \
               ind_mat[i, 2] == ind_mat[k, 2]:
                ji = ind_mat[i, 0]
                jk = ind_mat[k, 0]
                l = ind_mat[i, 1]
                if ji == (jk + 2):
                    LR[i, k] = LR[k, i] = 2 ** (2 - l) * np.pi ** 2 * mu *\
                        gamma(5 / 2.0 + jk + l) / gamma(jk)
                elif ji == (jk + 1):
                    LR[i, k] = LR[k, i] = 2 ** (2 - l) * np.pi ** 2 * mu *\
                        (-3 + 4 * ji + 2 * l) * gamma(3 / 2.0 + jk + l) /\
                        gamma(jk)
                elif ji == jk:
                    LR[i, k] = 2 ** (-l) * np.pi ** 2 * mu *\
                        (3 + 24 * ji ** 2 + 4 * (-2 + l) *
                         l + 12 * ji * (-1 + 2 * l)) *\
                        gamma(1 / 2.0 + ji + l) / gamma(ji)
                elif ji == (jk - 1):
                    LR[i, k] = LR[k, i] = 2 ** (2 - l) * np.pi ** 2 * mu *\
                        (-3 + 4 * jk + 2 * l) * gamma(3 / 2.0 + ji + l) /\
                        gamma(ji)
                elif ji == (jk - 2):
                    LR[i, k] = LR[k, i] = 2 ** (2 - l) * np.pi ** 2 * mu *\
                        gamma(5 / 2.0 + ji + l) / gamma(ji)

    return LR


def mapmri_isotropic_index_matrix(radial_order):
    r""" Calculates the indices for the isotropic MAPMRI [1]_ basis.

    Parameters
    ----------
    radial_order : unsigned int
        radial order of isotropic MAPMRI basis

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
        for j in range(1, 2 + n / 2):
            for m in range(-1 * (n + 2 - 2 * j), (n + 3 - 2 * j)):
                index_matrix.append([j, n + 2 - 2 * j, m])

    return np.array(index_matrix)


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
    tab : array, shape (N,3)
        real space points in which calculates the pdf
    """

    radius = gridsize // 2
    vecs = []
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            for k in range(0, radius + 1):
                vecs.append([i, j, k])

    vecs = np.array(vecs, dtype=np.float32)

    # there are points in the corners farther than sphere radius
    points_inside_sphere = np.linalg.norm(vecs, axis=1) <= radius
    vecs_inside_sphere = vecs[points_inside_sphere]

    tab = vecs_inside_sphere / radius
    tab = tab * radius_max

    return tab


def delta(n, m):
    if n == m:
        return 1
    return 0


def map_laplace_s(n, m):
    """ S(n, m) static matrix for Laplacian regularization [1]_.
    Parameters
    ----------
    n, m : unsigned int
        basis order of the MAP-MRI basis in different directions

    Returns
    -------
    R, L, S : Matrices, shape (N_coef,N_coef)
        Regularization submatrices

    References
    ----------
    .. [1]_ Fick et al. "MAPL: Tissue Microstructure Estimation Using
    Laplacian-Regularized MAP-MRI and its Application to HCP Data",
    NeuroImage, Under Review.
    """
    return (-1) ** n * delta(n, m) / (2 * np.sqrt(np.pi))


def map_laplace_l(n, m):
    """ L(m, n) static matrix for Laplacian regularization [1]_.
    Parameters
    ----------
    n, m : unsigned int
        basis order of the MAP-MRI basis in different directions

    Returns
    -------
    R, L, S : Matrices, shape (N_coef,N_coef)
        Regularization submatrices

    References
    ----------
    .. [1]_ Fick et al. "MAPL: Tissue Microstructure Estimation Using
    Laplacian-Regularized MAP-MRI and its Application to HCP Data",
    NeuroImage, Under Review.
    """
    a = np.sqrt((m - 1) * m) * delta(m - 2, n)
    b = np.sqrt((n - 1) * n) * delta(n - 2, m)
    c = (2 * n + 1) * delta(m, n)
    return np.pi ** (3 / 2.) * (-1) ** (n + 1) * (a + b + c)


def map_laplace_r(n, m):
    """ R(m,n) static matrix for Laplacian regularization [1]_.
    Parameters
    ----------
    n, m : unsigned int
        basis order of the MAP-MRI basis in different directions

    Returns
    -------
    R, L, S : Matrices, shape (N_coef,N_coef)
        Regularization submatrices

    References
    ----------
    .. [1]_ Fick et al. "MAPL: Tissue Microstructure Estimation Using
    Laplacian-Regularized MAP-MRI and its Application to HCP Data",
    NeuroImage, Under Review.
    """

    k = 2 * np.pi ** (7 / 2.) * (-1) ** (n)

    a0 = 3 * (2 * n ** 2 + 2 * n + 1) * delta(n, m)

    sqmn = np.sqrt(gamma(m + 1) / gamma(n + 1))

    sqnm = 1 / sqmn

    an2 = 2 * (2 * n + 3) * sqmn * delta(m, n + 2)

    an4 = sqmn * delta(m, n + 4)

    am2 = 2 * (2 * m + 3) * sqnm * delta(m + 2, n)

    am4 = sqnm * delta(m + 4, n)

    return k * (a0 + an2 + an4 + am2 + am4)


def mapmri_RLS_reg_matrices(radial_order):
    """ Generates the static portions of the Laplacian regularization matrix
    according to [1]_.

    Parameters
    ----------
    radial_order : unsigned int,
        an even integer that represent the order of the basis

    Returns
    -------
    R, L, S : Matrices, shape (N_coef,N_coef)
        Regularization submatrices

    References
    ----------
    .. [1]_ Fick et al. "MAPL: Tissue Microstructure Estimation Using
    Laplacian-Regularized MAP-MRI and its Application to HCP Data",
    NeuroImage, Under Review.
    """
    R = np.zeros((radial_order + 1, radial_order + 1))
    for i in xrange(radial_order + 1):
        for j in xrange(radial_order + 1):
            R[i, j] = map_laplace_r(i, j)

    L = np.zeros((radial_order + 1, radial_order + 1))
    for i in xrange(radial_order + 1):
        for j in xrange(radial_order + 1):
            L[i, j] = map_laplace_l(i, j)

    S = np.zeros((radial_order + 1, radial_order + 1))
    for i in xrange(radial_order + 1):
        for j in xrange(radial_order + 1):
            S[i, j] = map_laplace_s(i, j)
    return R, L, S


def mapmri_laplacian_reg_matrix(ind_mat, mu, R_mat, L_mat, S_mat):
    """ Puts the Laplacian regularization matrix together [1]_.
    The static parts in R, L and S are multiplied and divided by the
    voxel-specific scale factors.

    Parameters
    ----------
    ind_mat : matrix (N_coef, 3),
        Basis order matrix
    mu : array, shape (3,)
        scale factors of the basis for x, y, z
    R, L, S : matrices, shape (N_coef,N_coef)
        Regularization submatrices

    Returns
    -------
    LR : matrix (N_coef, N_coef),
        Voxel-specific Laplacian regularization matrix

    References
    ----------
    .. [1]_ Fick et al. "MAPL: Tissue Microstructure Estimation Using
    Laplacian-Regularized MAP-MRI and its Application to HCP Data",
    NeuroImage, Under Review.
    """
    ux, uy, uz = mu

    x, y, z = ind_mat.T

    n_elem = ind_mat.shape[0]

    LR = np.zeros((n_elem, n_elem))

    for i in range(n_elem):
        for j in range(i, n_elem):
            if (
               (x[i] - x[j]) % 2 == 0 and
               (y[i] - y[j]) % 2 == 0 and
               (z[i] - z[j]) % 2 == 0
               ):
                LR[i, j] = LR[j, i] = \
                    (ux ** 3 / (uy * uz)) *\
                    R_mat[x[i], x[j]] * S_mat[y[i], y[j]] * S_mat[z[i], z[j]] +\
                    (uy ** 3 / (ux * uz)) *\
                    R_mat[y[i], y[j]] * S_mat[z[i], z[j]] * S_mat[x[i], x[j]] +\
                    (uz ** 3 / (ux * uy)) *\
                    R_mat[z[i], z[j]] * S_mat[x[i], x[j]] * S_mat[y[i], y[j]] +\
                    2 * ((ux * uy) / uz) *\
                    L_mat[x[i], x[j]] * L_mat[y[i], y[j]] * S_mat[z[i], z[j]] +\
                    2 * ((ux * uz) / uy) *\
                    L_mat[x[i], x[j]] * L_mat[z[i], z[j]] * S_mat[y[i], y[j]] +\
                    2 * ((uz * uy) / ux) *\
                    L_mat[z[i], z[j]] * L_mat[y[i], y[j]] * S_mat[x[i], x[j]]

    return LR


def generalized_crossvalidation(data, M, LR, weights_array=None):
    """Generalized Cross Validation Function [1]_.
    Here weights_array is a numpy array with all values that should be
    considered in the GCV. It will run through the weights until the cost
    function starts to increase, then stop and take the last value as the
    optimum weight.

    Parameters
    ----------
    data : array (N),
        Basis order matrix
    M : matrix, shape (N, Ncoef)
        mapmri observation matrix
    LR : matrix, shape (N_coef,N_coef)
        regularization matrix
    weights_array : array (N_of_weights)
        array of optional regularization weights

    Returns
    -------
    lopt : float,
        optimal regularization weight

    References
    ----------
    .. [1]_ Craven et al. "Smoothing Noisy Data with Spline Functions."
        NUMER MATH 31.4 (1978): 377-403.
    """

    if weights_array is None:
        lrange = np.linspace(0, 1, 21)[1:]  # reasonably fast standard range
    else:
        lrange = weights_array

    samples = lrange.shape[0]
    MMt = np.dot(M.T, M)
    K = len(data)
    gcvold = gcvnew = 10e10
    i = -1
    while gcvold >= gcvnew and i < samples - 2:
        gcvold = gcvnew
        i = i + 1
        S = np.dot(np.dot(M, np.linalg.pinv(MMt + lrange[i] * LR)), M.T)
        trS = np.matrix.trace(S)
        normyytilde = np.linalg.norm(data - np.dot(S, data), 2)
        gcvnew = normyytilde / (K - trS)

    lopt = lrange[i - 1]
    return lopt
