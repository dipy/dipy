import numpy as np

from dipy.reconst.multi_voxel import multi_voxel_fit
from dipy.reconst.base import ReconstModel, ReconstFit
from dipy.reconst.cache import Cache
from scipy.special import hermite, gamma, genlaguerre
try:  # preferred scipy >= 0.14, required scipy >= 1.0
    from scipy.special import factorial as sfactorial
    from scipy.special import factorial2
except ImportError:
    from scipy.misc import factorial as sfactorial
    from scipy.misc import factorial2
from math import factorial as mfactorial
from dipy.core.geometry import cart2sphere
from dipy.reconst.shm import real_sh_descoteaux_from_index, sph_harm_ind_list
import dipy.reconst.dti as dti
from warnings import warn
from dipy.core.gradients import gradient_table
from dipy.utils.optpkg import optional_package
from dipy.core.optimize import Optimizer, PositiveDefiniteLeastSquares
from dipy.data import load_sdp_constraints

cvxpy, have_cvxpy, _ = optional_package("cvxpy", min_version="1.4.1")


class MapmriModel(ReconstModel, Cache):

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
    .. [1] Ozarslan E. et al., "Mean apparent propagator (MAP) MRI: A novel
           diffusion imaging method for mapping tissue microstructure",
           NeuroImage, 2013.

    .. [2] Ozarslan E. et al., "Simple harmonic oscillator based reconstruction
           and estimation for one-dimensional q-space magnetic resonance
           1D-SHORE)", eapoc Intl Soc Mag Reson Med, vol. 16, p. 35., 2008.

    .. [3] Merlet S. et al., "Continuous diffusion signal, EAP and ODF
           estimation via Compressive Sensing in diffusion MRI", Medical
           Image Analysis, 2013.

    .. [4] Fick, Rutger HJ, et al. "MAPL: Tissue microstructure estimation
           using Laplacian-regularized MAP-MRI and its application to HCP
           data." NeuroImage (2016).

    .. [5] Cheng, J., 2014. Estimation and Processing of Ensemble Average
           Propagator and Its Features in Diffusion MRI. Ph.D. Thesis.

    .. [6] Hosseinbor et al. "Bessel fourier orientation reconstruction
           (bfor): An analytical diffusion propagator reconstruction for hybrid
           diffusion imaging and computation of q-space indices". NeuroImage
           64, 2013, 650-670.

    .. [7] Craven et al. "Smoothing Noisy Data with Spline Functions."
           NUMER MATH 31.4 (1978): 377-403.

    .. [8] Avram et al. "Clinical feasibility of using mean apparent
           propagator (MAP) MRI to characterize brain tissue microstructure".
           NeuroImage 2015, in press.

    .. [9] Dela Haije et al. "Enforcing necessary non-negativity constraints
           for common diffusion MRI models using sum of squares programming".
           NeuroImage 209, 2020, 116405.
    """

    def __init__(self,
                 gtab,
                 radial_order=6,
                 laplacian_regularization=True,
                 laplacian_weighting=0.2,
                 positivity_constraint=False,
                 global_constraints=False,
                 pos_grid=15,
                 pos_radius='adaptive',
                 anisotropic_scaling=True,
                 eigenvalue_threshold=1e-04,
                 bval_threshold=np.inf,
                 dti_scale_estimation=True,
                 static_diffusivity=0.7e-3,
                 cvxpy_solver=None):
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
        constraint proposed in [1]_ or [4]_ and/or the laplacian regularization
        proposed in [5]_.

        For the estimation of q-space indices we recommend using the 'regular'
        anisotropic implementation of MAPMRI. However, it has been shown that
        the ODF estimation in this implementation has a bias which
        'squeezes together' the ODF peaks when there is a crossing at an angle
        smaller than 90 degrees [5]_. When you want to estimate ODFs for
        tractography we therefore recommend using the isotropic implementation
        (which is equivalent to [3]_).

        The switch between isotropic and anisotropic can be easily made through
        the anisotropic_scaling option.


        Parameters
        ----------
        gtab : GradientTable,
            gradient directions and bvalues container class.
            the gradient table has to include b0-images.
        radial_order : unsigned int,
            an even integer that represent the order of the basis
        laplacian_regularization: bool,
            Regularize using the Laplacian of the MAP-MRI basis.
        laplacian_weighting: string or scalar,
            The string 'GCV' makes it use generalized cross-validation to find
            the regularization weight [4]. A scalar sets the regularization
            weight to that value and an array will make it selected the
            optimal weight from the values in the array.
        positivity_constraint : bool,
            Constrain the propagator to be positive.
        global_constraints : bool, optional
            If set to False, positivity is enforced on a grid determined by
            pos_grid and pos_radius. If set to True, positivity is enforced
            everywhere using the constraints of [6]_. Global constraints are
            currently supported for anisotropic_scaling=True and for
            radial_order <= 10. Default: False.
        pos_grid : integer,
            The number of points in the grid that is used in the local
            positivity constraint.
        pos_radius : float or string,
            If set to a float, the maximum distance the local positivity
            constraint constrains to posivity is that value. If set to
            'adaptive', the maximum distance is dependent on the estimated
            tissue diffusivity.
        anisotropic_scaling : bool,
            If True, uses the standard anisotropic MAP-MRI basis. If False,
            uses the isotropic MAP-MRI basis (equal to 3D-SHORE).
        eigenvalue_threshold : float,
            Sets the minimum of the tensor eigenvalues in order to avoid
            stability problem.
        bval_threshold : float,
            Sets the b-value threshold to be used in the scale factor
            estimation. In order for the estimated non-Gaussianity to have
            meaning this value should set to a lower value (b<2000 s/mm^2)
            such that the scale factors are estimated on signal points that
            reasonably represent the spins at Gaussian diffusion.
        dti_scale_estimation : bool,
            Whether or not DTI fitting is used to estimate the isotropic scale
            factor for isotropic MAP-MRI.
            When set to False the algorithm presets the isotropic tissue
            diffusivity to static_diffusivity. This vastly increases fitting
            speed but at the cost of slightly reduced fitting quality. Can
            still be used in combination with regularization and constraints.
        static_diffusivity : float,
            the tissue diffusivity that is used when dti_scale_estimation is
            set to False. The default is that of typical white matter
            D=0.7e-3 _[5].
        cvxpy_solver : str, optional
            cvxpy solver name. Optionally optimize the positivity constraint
            with a particular cvxpy solver. See https://www.cvxpy.org/ for
            details.
            Default: None (cvxpy chooses its own solver)

        References
        ----------
        .. [1] Ozarslan E. et al., "Mean apparent propagator (MAP) MRI: A novel
               diffusion imaging method for mapping tissue microstructure",
               NeuroImage, 2013.

        .. [2] Ozarslan E. et al., "Simple harmonic oscillator based
               reconstruction and estimation for one-dimensional q-space
               magnetic resonance 1D-SHORE)", Proc Intl Soc Mag Reson Med,
               vol. 16, p. 35., 2008.

        .. [3] Ozarslan E. et al., "Simple harmonic oscillator based
               reconstruction and estimation for three-dimensional q-space
               mri", ISMRM 2009.

        .. [4] Dela Haije et al. "Enforcing necessary non-negativity
               constraints for common diffusion MRI models using sum of squares
               programming". NeuroImage 209, 2020, 116405.

        .. [5] Fick, Rutger HJ, et al. "MAPL: Tissue microstructure estimation
               using Laplacian-regularized MAP-MRI and its application to HCP
               data." NeuroImage (2016).

        .. [6] Merlet S. et al., "Continuous diffusion signal, EAP and ODF
               estimation via Compressive Sensing in diffusion MRI", Medical
               Image Analysis, 2013.

        Examples
        --------
        In this example, where the data, gradient table and sphere tessellation
        used for reconstruction are provided, we model the diffusion signal
        with respect to the SHORE basis and compute the real and analytical
        ODF.

        >>> from dipy.data import dsi_voxels, default_sphere
        >>> from dipy.core.gradients import gradient_table
        >>> _, gtab_ = dsi_voxels()
        >>> gtab = gradient_table(gtab_.bvals, gtab_.bvecs,
        ...                       b0_threshold=gtab_.bvals.min())
        >>> from dipy.sims.voxel import sticks_and_ball
        >>> data, golden_directions = sticks_and_ball(gtab, d=0.0015, S0=1,
        ...                                           angles=[(0, 0),
        ...                                                   (90, 0)],
        ...                                           fractions=[50, 50],
        ...                                           snr=None)
        >>> from dipy.reconst.mapmri import MapmriModel
        >>> radial_order = 4
        >>> map_model = MapmriModel(gtab, radial_order=radial_order)
        >>> mapfit = map_model.fit(data)
        >>> odf = mapfit.odf(default_sphere)
        """

        if np.sum(gtab.b0s_mask) == 0:
            raise ValueError("gtab does not have any b0s, check in the "
                             "gradient_table if b0_threshold needs to be "
                             "increased.")
        self.gtab = gtab

        if radial_order < 0 or radial_order % 2:
            raise ValueError("radial_order must be a positive, even number.")
        self.radial_order = radial_order

        self.bval_threshold = bval_threshold
        self.dti_scale_estimation = dti_scale_estimation

        if laplacian_regularization:
            msg = ("Laplacian Regularization weighting must be 'GCV',"
                   " a positive float or an array of positive floats.")
            if isinstance(laplacian_weighting, str):
                if not laplacian_weighting == 'GCV':
                    raise ValueError(msg)
            elif isinstance(laplacian_weighting, (float, np.ndarray)):
                if np.sum(laplacian_weighting < 0) > 0:
                    raise ValueError(msg)
            self.laplacian_weighting = laplacian_weighting
        self.laplacian_regularization = laplacian_regularization

        if positivity_constraint:
            if not have_cvxpy:
                raise ImportError('CVXPY package needed to enforce '
                                  'constraints.')
            if cvxpy_solver is not None:
                if cvxpy_solver not in cvxpy.installed_solvers():
                    installed_solvers = ', '.join(cvxpy.installed_solvers())
                    raise ValueError(f"Input `cvxpy_solver` was set to"
                                     f" {cvxpy_solver}. One of"
                                     f" {installed_solvers} was expected.")
            self.cvxpy_solver = cvxpy_solver
            if global_constraints:
                if not anisotropic_scaling:
                    raise ValueError('Global constraints only available for'
                                     ' anistropic_scaling=True.')
                if radial_order > 10:
                    self.sdp_constraints = load_sdp_constraints('hermite', 10)
                    warn('Global constraints are currently supported for'
                         ' radial_order <= 10.')
                else:
                    self.sdp_constraints = load_sdp_constraints('hermite',
                                                                radial_order)
                m = (2 + radial_order)*(4 + radial_order)*(3 + 2*radial_order)
                m = m//24
                self.sdp = PositiveDefiniteLeastSquares(m,
                                                        A=self.sdp_constraints)
            else:
                msg = "pos_radius must be 'adaptive' or a positive float."
                if isinstance(pos_radius, str):
                    if pos_radius != 'adaptive':
                        raise ValueError(msg)
                elif isinstance(pos_radius, (float, int)):
                    if pos_radius <= 0:
                        raise ValueError(msg)
                    self.constraint_grid = create_rspace(pos_grid, pos_radius)
                    if not anisotropic_scaling:
                        self.pos_K_independent = \
                            mapmri_isotropic_K_mu_independent(
                                radial_order, self.constraint_grid)
                else:
                    raise ValueError(msg)
                self.pos_grid = pos_grid
                self.pos_radius = pos_radius
            self.global_constraints = global_constraints
        self.positivity_constraint = positivity_constraint
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

        if self.anisotropic_scaling:
            self.ind_mat = mapmri_index_matrix(self.radial_order)
            self.Bm = b_mat(self.ind_mat)
            self.S_mat, self.T_mat, self.U_mat = mapmri_STU_reg_matrices(
                radial_order)
        else:
            self.ind_mat = mapmri_isotropic_index_matrix(self.radial_order)
            self.Bm = b_mat_isotropic(self.ind_mat)
            self.laplacian_matrix = mapmri_isotropic_laplacian_reg_matrix(
                radial_order, 1.)

            qvals = np.sqrt(self.gtab.bvals / self.tau) / (2 * np.pi)
            q = gtab.bvecs * qvals[:, None]
            if self.dti_scale_estimation:
                self.M_mu_independent = mapmri_isotropic_M_mu_independent(
                    self.radial_order, q)
            else:
                D = static_diffusivity
                mumean = np.sqrt(2 * D * self.tau)
                self.mu = np.array([mumean, mumean, mumean])
                self.M = mapmri_isotropic_phi_matrix(radial_order, mumean, q)
                if (self.laplacian_regularization and
                   isinstance(laplacian_weighting, float) and
                   not positivity_constraint):
                    MMt = (np.dot(self.M.T, self.M) +
                           laplacian_weighting * mumean *
                           self.laplacian_matrix)
                    self.MMt_inv_Mt = np.dot(np.linalg.pinv(MMt), self.M.T)

    @multi_voxel_fit
    def fit(self, data):
        errorcode = 0
        tenfit = self.tenmodel.fit(data[self.cutoff])
        evals = tenfit.evals
        R = tenfit.evecs
        evals = np.clip(evals, self.eigenvalue_threshold, evals.max())
        qvals = np.sqrt(self.gtab.bvals / self.tau) / (2 * np.pi)
        mu_max = max(np.sqrt(evals * 2 * self.tau))  # used for constraint
        if self.anisotropic_scaling:
            mu = np.sqrt(evals * 2 * self.tau)
            qvecs = np.dot(self.gtab.bvecs, R)
            q = qvecs * qvals[:, None]
            M = mapmri_phi_matrix(self.radial_order, mu, q)
        else:
            try:
                # self.MMt_inv_Mt
                lopt = self.laplacian_weighting
                coef = np.dot(self.MMt_inv_Mt, data)
                coef = coef / sum(coef * self.Bm)
                return MapmriFit(self, coef, self.mu, R, lopt, errorcode)
            except AttributeError:
                try:
                    M = self.M
                    mu = self.mu
                except AttributeError:
                    u0 = isotropic_scale_factor(evals * 2 * self.tau)
                    mu = np.array([u0, u0, u0])
                    M_mu_dependent = mapmri_isotropic_M_mu_dependent(
                        self.radial_order, mu[0], qvals)
                    M = M_mu_dependent * self.M_mu_independent

        if self.laplacian_regularization:
            if self.anisotropic_scaling:
                laplacian_matrix = mapmri_laplacian_reg_matrix(
                    self.ind_mat, mu, self.S_mat, self.T_mat, self.U_mat)
            else:
                laplacian_matrix = self.laplacian_matrix * mu[0]

            if (isinstance(self.laplacian_weighting, str) and
                    self.laplacian_weighting.upper() == 'GCV'):
                try:
                    lopt = generalized_crossvalidation(data, M,
                                                       laplacian_matrix)
                except np.linalg.linalg.LinAlgError:
                    # 1/0.
                    lopt = 0.05
                    errorcode = 1
            elif np.isscalar(self.laplacian_weighting):
                lopt = self.laplacian_weighting
            else:
                lopt = generalized_crossvalidation_array(
                                    data,
                                    M,
                                    laplacian_matrix,
                                    self.laplacian_weighting)

        else:
            lopt = 0.
            laplacian_matrix = np.ones((self.ind_mat.shape[0],
                                        self.ind_mat.shape[0]))

        if self.positivity_constraint:
            data_norm = np.asarray(data / data[self.gtab.b0s_mask].mean())
            if self.global_constraints:
                coef = self.sdp.solve(M, data_norm, solver=self.cvxpy_solver)
            else:
                c = cvxpy.Variable(M.shape[1])
                design_matrix = cvxpy.Constant(M) @ c
                # workaround for the bug on cvxpy 1.0.15 when lopt = 0
                # See https://github.com/cvxgrp/cvxpy/issues/672
                if not lopt:
                    objective = cvxpy.Minimize(
                        cvxpy.sum_squares(design_matrix - data_norm))
                else:
                    objective = cvxpy.Minimize(
                        cvxpy.sum_squares(design_matrix - data_norm) +
                        lopt * cvxpy.quad_form(c, laplacian_matrix)
                    )
                if self.pos_radius == 'adaptive':
                    # custom constraint grid based on scale factor [Avram2015]
                    constraint_grid = create_rspace(self.pos_grid,
                                                    np.sqrt(5) * mu_max)
                else:
                    constraint_grid = self.constraint_grid
                if self.anisotropic_scaling:
                    K = mapmri_psi_matrix(self.radial_order, mu,
                                          constraint_grid)
                else:
                    if self.pos_radius == 'adaptive':
                        # grid changes per voxel. Recompute entire K matrix.
                        K = mapmri_isotropic_psi_matrix(self.radial_order,
                                                        mu[0], constraint_grid)
                    else:
                        # grid is static. Only compute mu-dependent part of K.
                        K_dependent = mapmri_isotropic_K_mu_dependent(
                            self.radial_order, mu[0], constraint_grid)
                        K = K_dependent * self.pos_K_independent

                M0 = M[self.gtab.b0s_mask, :]
                constraints = [(M0[0] @ c) == 1, (K @ c) >= -0.1]
                prob = cvxpy.Problem(objective, constraints)
                try:
                    prob.solve(solver=self.cvxpy_solver)
                    coef = np.asarray(c.value).squeeze()
                except Exception:
                    errorcode = 2
                    warn('Optimization did not find a solution')
                    try:
                        coef = np.dot(np.linalg.pinv(M), data)  # least squares
                    except np.linalg.linalg.LinAlgError:
                        errorcode = 3
                        coef = np.zeros(M.shape[1])
                        return MapmriFit(self, coef, mu, R, lopt, errorcode)
        else:
            try:
                pseudoInv = np.dot(
                    np.linalg.inv(np.dot(M.T, M) + lopt * laplacian_matrix),
                    M.T)
                coef = np.dot(pseudoInv, data)
            except np.linalg.linalg.LinAlgError:
                errorcode = 1
                coef = np.zeros(M.shape[1])
                return MapmriFit(self, coef, mu, R, lopt, errorcode)

        coef = coef / sum(coef * self.Bm)

        return MapmriFit(self, coef, mu, R, lopt, errorcode)


class MapmriFit(ReconstFit):

    def __init__(self, model, mapmri_coef, mu, R, lopt, errorcode=0):
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
        lopt : float,
            regularization weight used for laplacian regularization
        errorcode : int
            provides information on whether errors occurred in the fitting
            of each voxel. 0 means no problem, 1 means a LinAlgError
            occurred when trying to invert the design matrix. 2 means the
            positivity constraint was unable to solve the problem. 3 means
            that after positivity constraint failed, also matrix inversion
            failed.
        """
        self.model = model
        self._mapmri_coef = mapmri_coef
        self.gtab = model.gtab
        self.radial_order = model.radial_order
        self.mu = mu
        self.R = R
        self.lopt = lopt
        self.errorcode = errorcode

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
        from the signal [1]_ Eq. (32).

        Parameters
        ----------
        sphere : Sphere
            A Sphere instance with vertices, edges and faces attributes.
        s : unsigned int
            radial moment of the ODF

        References
        ----------
        .. [1] Ozarslan E. et al., "Mean apparent propagator (MAP) MRI: A novel
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
        and radial moment [1]_ eq. (32). The radial moment s acts as a
        sharpening method. The analytical equation for the spherical ODF basis
        is given in [2]_ eq. (C8).

        References
        ----------
        .. [1] Ozarslan E. et al., "Mean apparent propagator (MAP) MRI: A novel
        diffusion imaging method for mapping tissue microstructure",
        NeuroImage, 2013.

        .. [2] Fick, Rutger HJ, et al. "MAPL: Tissue microstructure estimation
        using Laplacian-regularized MAP-MRI and its application to HCP data."
        NeuroImage (2016).
        """
        if self.model.anisotropic_scaling:
            raise ValueError('odf in spherical harmonics not yet implemented '
                             'for anisotropic implementation')
        I = self.model.cache_get('ODF_sh_matrix', key=(self.radial_order, s))

        if I is None:
            I = mapmri_isotropic_odf_sh_matrix(self.radial_order, 1, s)
            self.model.cache_set('ODF_sh_matrix', (self.radial_order, s), I)

        odf = self.mu[0] ** s * np.dot(I, self._mapmri_coef)

        return odf

    def rtpp(self):
        r""" Calculates the analytical return to the plane probability (RTPP)
        [1]_ eq. (42). The analytical formula for the isotropic MAP-MRI
        basis was derived in [2]_ eq. (C11).

        References
        ----------
        .. [1] Ozarslan E. et al., "Mean apparent propagator (MAP) MRI: A novel
        diffusion imaging method for mapping tissue microstructure",
        NeuroImage, 2013.

        .. [2] Fick, Rutger HJ, et al. "MAPL: Tissue microstructure estimation
        using Laplacian-regularized MAP-MRI and its application to HCP data."
        NeuroImage (2016).
        """
        Bm = self.model.Bm
        ind_mat = self.model.ind_mat
        if self.model.anisotropic_scaling:
            sel = Bm > 0.  # select only relevant coefficients
            const = 1 / (np.sqrt(2 * np.pi) * self.mu[0])
            ind_sum = (-1.0) ** (ind_mat[sel, 0] / 2.0)
            rtpp_vec = const * Bm[sel] * ind_sum * self._mapmri_coef[sel]
            rtpp = rtpp_vec.sum()
            return rtpp

        else:
            rtpp_vec = np.zeros((ind_mat.shape[0]))
            count = 0
            for n in range(0, self.model.radial_order + 1, 2):
                    for j in range(1, 2 + n // 2):
                        l = n + 2 - 2 * j
                        const = (-1/2.0) ** (l/2) / np.sqrt(np.pi)
                        matsum = 0
                        for k in range(0, j):
                            matsum += (-1) ** k * \
                                binomialfloat(j + l - 0.5, j - k - 1) *\
                                gamma(l / 2 + k + 1 / 2.0) /\
                                (sfactorial(k) * 0.5 ** (l / 2 + 1 / 2.0 + k))
                        for m in range(-l, l + 1):
                            rtpp_vec[count] = const * matsum
                            count += 1

            direction = np.array(self.R[:, 0], ndmin=2)
            r, theta, phi = cart2sphere(direction[:, 0], direction[:, 1],
                                        direction[:, 2])

            rtpp = self._mapmri_coef * (1 / self.mu[0]) *\
                rtpp_vec * real_sh_descoteaux_from_index(
                    ind_mat[:, 2], ind_mat[:, 1],
                                         theta, phi)

            return rtpp.sum()

    def rtap(self):
        r""" Calculates the analytical return to the axis probability (RTAP)
        [1]_ eq. (40, 44a). The analytical formula for the isotropic MAP-MRI
        basis was derived in [2]_ eq. (C11).

        References
        ----------
        .. [1] Ozarslan E. et al., "Mean apparent propagator (MAP) MRI: A novel
        diffusion imaging method for mapping tissue microstructure",
        NeuroImage, 2013.

        .. [2] Fick, Rutger HJ, et al. "MAPL: Tissue microstructure estimation
        using Laplacian-regularized MAP-MRI and its application to HCP data."
        NeuroImage (2016).
        """
        Bm = self.model.Bm
        ind_mat = self.model.ind_mat
        if self.model.anisotropic_scaling:
            sel = Bm > 0.  # select only relevant coefficients
            const = 1 / (2 * np.pi * np.prod(self.mu[1:]))
            ind_sum = (-1.0) ** (np.sum(ind_mat[sel, 1:], axis=1) / 2.0)
            rtap_vec = const * Bm[sel] * ind_sum * self._mapmri_coef[sel]
            rtap = np.sum(rtap_vec)
        else:
            rtap_vec = np.zeros((ind_mat.shape[0]))
            count = 0

            for n in range(0, self.model.radial_order + 1, 2):
                for j in range(1, 2 + n // 2):
                    l = n + 2 - 2 * j
                    kappa = ((-1) ** (j - 1) * 2 ** (-(l + 3) / 2.0)) / np.pi
                    matsum = 0
                    for k in range(0, j):
                        matsum += ((-1) ** k *
                                   binomialfloat(j + l - 0.5, j - k - 1) *
                                   gamma((l + 1) / 2.0 + k)) /\
                            (sfactorial(k) * 0.5 ** ((l + 1) / 2.0 + k))
                    for m in range(-l, l + 1):
                        rtap_vec[count] = kappa * matsum
                        count += 1
            rtap_vec *= 2

            direction = np.array(self.R[:, 0], ndmin=2)
            r, theta, phi = cart2sphere(direction[:, 0],
                                        direction[:, 1], direction[:, 2])
            rtap_vec = self._mapmri_coef * (1 / self.mu[0] ** 2) *\
                rtap_vec * real_sh_descoteaux_from_index(
                    ind_mat[:, 2], ind_mat[:, 1], theta, phi)
            rtap = rtap_vec.sum()
        return rtap

    def rtop(self):
        r""" Calculates the analytical return to the origin probability (RTOP)
        [1]_ eq. (36, 43). The analytical formula for the isotropic MAP-MRI
        basis was derived in [2]_ eq. (C11).

        References
        ----------
        .. [1] Ozarslan E. et al., "Mean apparent propagator (MAP) MRI: A novel
        diffusion imaging method for mapping tissue microstructure",
        NeuroImage, 2013.

        .. [2] Fick, Rutger HJ, et al. "MAPL: Tissue microstructure estimation
        using Laplacian-regularized MAP-MRI and its application to HCP data."
        NeuroImage (2016).
        """
        Bm = self.model.Bm

        if self.model.anisotropic_scaling:
            const = 1 / (np.sqrt(8 * np.pi ** 3) * np.prod(self.mu))
            ind_sum = (-1.0) ** (np.sum(self.model.ind_mat, axis=1) / 2)
            rtop_vec = const * ind_sum * Bm * self._mapmri_coef
            rtop = rtop_vec.sum()
        else:
            const = 1 / (2 * np.sqrt(2.0) * np.pi ** (3 / 2.0))
            rtop_vec = const * (-1.0) ** (self.model.ind_mat[:, 0] - 1) * Bm
            rtop = (1 / self.mu[0] ** 3) * rtop_vec * self._mapmri_coef
            rtop = rtop.sum()
        return rtop

    def msd(self):
        r""" Calculates the analytical Mean Squared Displacement (MSD).
        It is defined as the Laplacian of the origin of the estimated signal
        [1]_. The analytical formula for the MAP-MRI basis was derived in [2]_
        eq. (C13, D1).

        References
        ----------
        .. [1] Cheng, J., 2014. Estimation and Processing of Ensemble Average
        Propagator and Its Features in Diffusion MRI. Ph.D. Thesis.

        .. [2] Fick, Rutger HJ, et al. "MAPL: Tissue microstructure estimation
        using Laplacian-regularized MAP-MRI and its application to HCP data."
        NeuroImage (2016).
        """

        mu = self.mu
        ind_mat = self.model.ind_mat
        Bm = self.model.Bm
        sel = self.model.Bm > 0.  # select only relevant coefficients
        mapmri_coef = self._mapmri_coef[sel]
        if self.model.anisotropic_scaling:
            ind_sum = np.sum(ind_mat[sel], axis=1)
            nx, ny, nz = ind_mat[sel].T

            numerator = (-1) ** (0.5 * (-ind_sum)) * np.pi ** (3 / 2.0) *\
                ((1 + 2 * nx) * mu[0] ** 2 + (1 + 2 * ny) *
                 mu[1] ** 2 + (1 + 2 * nz) * mu[2] ** 2)

            denominator = np.sqrt(2. ** (-ind_sum) * sfactorial(nx) *
                                  sfactorial(ny) * sfactorial(nz)) *\
                gamma(0.5 - 0.5 * nx) * gamma(0.5 - 0.5 * ny) *\
                gamma(0.5 - 0.5 * nz)

            msd_vec = self._mapmri_coef[sel] * (numerator / denominator)
            msd = msd_vec.sum()
        else:
            msd_vec = (4 * ind_mat[sel, 0] - 1) * Bm[sel]
            msd = self.mu[0] ** 2 * msd_vec * mapmri_coef
            msd = msd.sum()
        return msd

    def qiv(self):
        r""" Calculates the analytical Q-space Inverse Variance (QIV).
        It is defined as the inverse of the Laplacian of the origin of the
        estimated propagator [1]_ eq. (22). The analytical formula for the
        MAP-MRI basis was derived in [2]_ eq. (C14, D2).

        References
        ----------
        .. [1] Hosseinbor et al. "Bessel fourier orientation reconstruction
        (bfor): An analytical diffusion propagator reconstruction for hybrid
        diffusion imaging and computation of q-space indices. NeuroImage 64,
        2013, 650-670.

        .. [2] Fick, Rutger HJ, et al. "MAPL: Tissue microstructure estimation
        using Laplacian-regularized MAP-MRI and its application to HCP data."
        NeuroImage (2016).
        """
        ux, uy, uz = self.mu
        ind_mat = self.model.ind_mat
        if self.model.anisotropic_scaling:
            sel = self.model.Bm > 0  # select only relevant coefficients
            nx, ny, nz = ind_mat[sel].T

            numerator = 8 * np.pi ** 2 * (ux * uy * uz) ** 3 *\
                np.sqrt(sfactorial(nx) * sfactorial(ny) * sfactorial(nz)) *\
                gamma(0.5 - 0.5 * nx) * gamma(0.5 - 0.5 * ny) * \
                gamma(0.5 - 0.5 * nz)

            denominator = np.sqrt(2. ** (-1 + nx + ny + nz)) *\
                ((1 + 2 * nx) * uy ** 2 * uz ** 2 + ux ** 2 *
                 ((1 + 2 * nz) * uy ** 2 + (1 + 2 * ny) * uz ** 2))

            qiv_vec = self._mapmri_coef[sel] * (numerator / denominator)
            qiv = qiv_vec.sum()
        else:
            sel = self.model.Bm > 0.  # select only relevant coefficients
            j = ind_mat[sel, 0]
            qiv_vec = ((8 * (-1.0) ** (1 - j) *
                        np.sqrt(2) * np.pi ** (7 / 2.)) / ((4.0 * j - 1) *
                                                           self.model.Bm[sel]))
            qiv = ux ** 5 * qiv_vec * self._mapmri_coef[sel]
            qiv = qiv.sum()
        return qiv

    def ng(self):
        r""" Calculates the analytical non-Gaussiannity (NG) [1]_.
        For the NG to be meaningful the mapmri scale factors must be
        estimated only on data representing Gaussian diffusion of spins, i.e.,
        bvals smaller than about 2000 s/mm^2 [2]_.

        References
        ----------
        .. [1] Ozarslan E. et al., "Mean apparent propagator (MAP) MRI: A novel
        diffusion imaging method for mapping tissue microstructure",
        NeuroImage, 2013.

        .. [2] Avram et al. "Clinical feasibility of using mean apparent
        propagator (MAP) MRI to characterize brain tissue microstructure".
        NeuroImage 2015, in press.
        """
        if self.model.bval_threshold > 2000.:
            warn('model bval_threshold must be lower than 2000 for the '
                 'non_Gaussianity to be physically meaningful [2].')
        if not self.model.anisotropic_scaling:
            raise ValueError('Parallel non-Gaussianity is not defined using '
                             'isotropic scaling.')

        coef = self._mapmri_coef
        return np.sqrt(1 - coef[0] ** 2 / np.sum(coef ** 2))

    def ng_parallel(self):
        r""" Calculates the analytical parallel non-Gaussiannity (NG) [1]_.
        For the NG to be meaningful the mapmri scale factors must be
        estimated only on data representing Gaussian diffusion of spins, i.e.,
        bvals smaller than about 2000 s/mm^2 [2]_.

        References
        ----------
        .. [1] Ozarslan E. et al., "Mean apparent propagator (MAP) MRI: A novel
        diffusion imaging method for mapping tissue microstructure",
        NeuroImage, 2013.

        .. [2] Avram et al. "Clinical feasibility of using mean apparent
        propagator (MAP) MRI to characterize brain tissue microstructure".
        NeuroImage 2015, in press.
        """
        if self.model.bval_threshold > 2000.:
            warn('Model bval_threshold must be lower than 2000 for the '
                 'non_Gaussianity to be physically meaningful [2].')
        if not self.model.anisotropic_scaling:
            raise ValueError('Parallel non-Gaussianity is not defined using '
                             'isotropic scaling.')

        ind_mat = self.model.ind_mat
        coef = self._mapmri_coef
        a_par = np.zeros_like(coef)
        a0 = np.zeros_like(coef)

        for i in range(coef.shape[0]):
            n1, n2, n3 = ind_mat[i]
            if (n2 % 2 + n3 % 2) == 0:
                a_par[i] = coef[i] * (-1) ** ((n2 + n3) / 2) *\
                    np.sqrt(sfactorial(n2) * sfactorial(n3)) /\
                    (factorial2(n2) * factorial2(n3))
                if n1 == 0:
                    a0[i] = a_par[i]
        return np.sqrt(1 - np.sum(a0 ** 2) / np.sum(a_par ** 2))

    def ng_perpendicular(self):
        r""" Calculates the analytical perpendicular non-Gaussiannity (NG)
        [1]_. For the NG to be meaningful the mapmri scale factors must be
        estimated only on data representing Gaussian diffusion of spins, i.e.,
        bvals smaller than about 2000 s/mm^2 [2]_.

        References
        ----------
        .. [1] Ozarslan E. et al., "Mean apparent propagator (MAP) MRI: A novel
        diffusion imaging method for mapping tissue microstructure",
        NeuroImage, 2013.

        .. [2] Avram et al. "Clinical feasibility of using mean apparent
        propagator (MAP) MRI to characterize brain tissue microstructure".
        NeuroImage 2015, in press.
        """
        if self.model.bval_threshold > 2000.:
            warn('model bval_threshold must be lower than 2000 for the '
                 'non_Gaussianity to be physically meaningful [2].')
        if not self.model.anisotropic_scaling:
            raise ValueError('Parallel non-Gaussianity is not defined using '
                             'isotropic scaling.')

        ind_mat = self.model.ind_mat
        coef = self._mapmri_coef
        a_perp = np.zeros_like(coef)
        a00 = np.zeros_like(coef)

        for i in range(coef.shape[0]):
            n1, n2, n3 = ind_mat[i]
            if n1 % 2 == 0:
                if n2 % 2 == 0 and n3 % 2 == 0:
                    a_perp[i] = coef[i] * (-1) ** (n1 / 2) *\
                        np.sqrt(sfactorial(n1)) / factorial2(n1)
                    if n2 == 0 and n3 == 0:
                        a00[i] = a_perp[i]
        return np.sqrt(1 - np.sum(a00 ** 2) / np.sum(a_perp ** 2))

    def norm_of_laplacian_signal(self):
        """ Calculates the norm of the laplacian of the fitted signal [1]_.
        This information could be useful to assess if the extrapolation of the
        fitted signal contains spurious oscillations. A high laplacian may
        indicate that these are present, and any q-space indices that
        use integrals of the signal may be corrupted (e.g. RTOP, RTAP, RTPP,
        QIV).

        References
        ----------
        .. [1] Fick, Rutger HJ, et al. "MAPL: Tissue microstructure estimation
        using Laplacian-regularized MAP-MRI and its application to HCP data."
        NeuroImage (2016).
        """
        if self.model.anisotropic_scaling:
            laplacian_matrix = mapmri_laplacian_reg_matrix(
                    self.model.ind_mat, self.mu,
                    self.model.S_mat, self.model.T_mat, self.model.U_mat)
        else:
            laplacian_matrix = self.mu[0] * self.model.laplacian_matrix

        norm_of_laplacian = np.linalg.multi_dot([self._mapmri_coef,
                                                laplacian_matrix,
                                                self._mapmri_coef])
        return norm_of_laplacian

    def fitted_signal(self, gtab=None):
        """
        Recovers the fitted signal for the given gradient table. If no gradient
        table is given it recovers the signal for the gtab of the model object.
        """
        if gtab is None:
            E = self.predict(self.model.gtab, S0=1.)
        else:
            E = self.predict(gtab, S0=1.)
        return E

    def predict(self, qvals_or_gtab, S0=100.):
        r"""Recovers the reconstructed signal for any qvalue array or
        gradient table.
        """
        if isinstance(qvals_or_gtab, np.ndarray):
            q = qvals_or_gtab
            # qvals = np.linalg.norm(q, axis=1)
        else:
            gtab = qvals_or_gtab
            qvals = np.sqrt(gtab.bvals / self.model.tau) / (2 * np.pi)
            q = qvals[:, None] * gtab.bvecs

        if self.model.anisotropic_scaling:
            q_rot = np.dot(q, self.R)
            M = mapmri_phi_matrix(self.radial_order, self.mu, q_rot)
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


def isotropic_scale_factor(mu_squared):
    r"""Estimated isotropic scaling factor _[1] Eq. (49).

    Parameters
    ----------
    mu_squared : array, shape (N,3)
        squared scale factors of mapmri basis in x, y, z

    Returns
    -------
    u0 : float
        closest isotropic scale factor for the isotropic basis

    References
    ----------
    .. [1] Ozarslan E. et al., "Mean apparent propagator (MAP) MRI: A novel
    diffusion imaging method for mapping tissue microstructure",
    NeuroImage, 2013.
    """
    X, Y, Z = mu_squared
    coef_array = np.array([-3, -(X + Y + Z), (X * Y + X * Z + Y * Z),
                           3 * X * Y * Z])
    # take the real, positive root of the problem.
    u0 = np.sqrt(np.real(np.roots(coef_array).max()))
    return u0


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
    .. [1] Ozarslan E. et al., "Mean apparent propagator (MAP) MRI: A novel
    diffusion imaging method for mapping tissue microstructure",
    NeuroImage, 2013.
    """
    index_matrix = []
    for n in range(0, radial_order + 1, 2):
        for i in range(0, n + 1):
            for j in range(0, n - i + 1):
                index_matrix.append([n - i - j, j, i])

    return np.array(index_matrix)


def b_mat(index_matrix):
    r""" Calculates the B coefficients from [1]_ Eq. (27).

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
    .. [1] Ozarslan E. et al., "Mean apparent propagator (MAP) MRI: A novel
    diffusion imaging method for mapping tissue microstructure",
    NeuroImage, 2013.
    """

    B = np.zeros(index_matrix.shape[0])
    for i in range(index_matrix.shape[0]):
        n1, n2, n3 = index_matrix[i]
        K = int(not(n1 % 2) and not(n2 % 2) and not(n3 % 2))
        B[i] = (
            K * np.sqrt(sfactorial(n1) * sfactorial(n2) * sfactorial(n3)) /
            (factorial2(n1) * factorial2(n2) * factorial2(n3))
            )

    return B


def b_mat_isotropic(index_matrix):
    r""" Calculates the isotropic B coefficients from [1]_ Fig 8.

    Parameters
    ----------
    index_matrix : array, shape (N,3)
        ordering of the isotropic basis in j, l, m

    Returns
    -------
    B : array, shape (N,)
        B coefficients for the isotropic basis

    References
    ----------
    .. [1] Ozarslan E. et al., "Mean apparent propagator (MAP) MRI: A novel
    diffusion imaging method for mapping tissue microstructure",
    NeuroImage, 2013.
    """

    B = np.zeros((index_matrix.shape[0]))
    for i in range(index_matrix.shape[0]):
        if index_matrix[i, 1] == 0:
            B[i] = genlaguerre(index_matrix[i, 0] - 1, 0.5)(0)

    return B


def mapmri_phi_1d(n, q, mu):
    r""" One dimensional MAPMRI basis function from [1]_ Eq. (4).

    Parameters
    ----------
    n : unsigned int
        order of the basis
    q : array, shape (N,)
        points in the q-space in which evaluate the basis
    mu : float
        scale factor of the basis

    References
    ----------
    .. [1] Ozarslan E. et al., "Mean apparent propagator (MAP) MRI: A novel
    diffusion imaging method for mapping tissue microstructure",
    NeuroImage, 2013.
    """

    qn = 2 * np.pi * mu * q
    H = hermite(n)(qn)
    i = complex(0, 1)
    f = mfactorial(n)

    k = i ** (-n) / np.sqrt(2 ** n * f)
    phi = k * np.exp(- qn ** 2 / 2) * H

    return phi


def mapmri_phi_matrix(radial_order, mu, q_gradients):
    r"""Compute the MAPMRI phi matrix for the signal [1]_ eq. (23).

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
    .. [1] Ozarslan E. et al., "Mean apparent propagator (MAP) MRI: A novel
    diffusion imaging method for mapping tissue microstructure",
    NeuroImage, 2013.
    """

    ind_mat = mapmri_index_matrix(radial_order)
    n_elem = ind_mat.shape[0]
    n_qgrad = q_gradients.shape[0]

    qx, qy, qz = q_gradients.T
    mux, muy, muz = mu

    Mx_storage = np.array(np.zeros((n_qgrad, radial_order + 1)),
                          dtype=complex)
    My_storage = np.array(np.zeros((n_qgrad, radial_order + 1)),
                          dtype=complex)
    Mz_storage = np.array(np.zeros((n_qgrad, radial_order + 1)),
                          dtype=complex)
    M = np.zeros((n_qgrad, n_elem))

    for n in range(radial_order + 1):
        Mx_storage[:, n] = mapmri_phi_1d(n, qx, mux)
        My_storage[:, n] = mapmri_phi_1d(n, qy, muy)
        Mz_storage[:, n] = mapmri_phi_1d(n, qz, muz)

    counter = 0
    for nx, ny, nz in ind_mat:
        M[:, counter] = (
            np.real(Mx_storage[:, nx] * My_storage[:, ny] * Mz_storage[:, nz])
            )
        counter += 1

    return M


def mapmri_psi_1d(n, x, mu):
    r""" One dimensional MAPMRI propagator basis function from [1]_ Eq. (10).

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
    .. [1] Ozarslan E. et al., "Mean apparent propagator (MAP) MRI: A novel
    diffusion imaging method for mapping tissue microstructure",
    NeuroImage, 2013.
    """

    H = hermite(n)(x / mu)
    f = mfactorial(n)
    k = 1 / (np.sqrt(2 ** (n + 1) * np.pi * f) * mu)
    psi = k * np.exp(- x ** 2 / (2 * mu ** 2)) * H

    return psi


def mapmri_psi_matrix(radial_order, mu, rgrad):
    r"""Compute the MAPMRI psi matrix for the propagator [1]_ eq. (22).

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
    .. [1] Ozarslan E. et al., "Mean apparent propagator (MAP) MRI: A novel
    diffusion imaging method for mapping tissue microstructure",
    NeuroImage, 2013.
    """

    ind_mat = mapmri_index_matrix(radial_order)
    n_elem = ind_mat.shape[0]
    n_qgrad = rgrad.shape[0]
    rx, ry, rz = rgrad.T
    mux, muy, muz = mu

    Kx_storage = np.zeros((n_qgrad, radial_order + 1))
    Ky_storage = np.zeros((n_qgrad, radial_order + 1))
    Kz_storage = np.zeros((n_qgrad, radial_order + 1))
    K = np.zeros((n_qgrad, n_elem))

    for n in range(radial_order + 1):
        Kx_storage[:, n] = mapmri_psi_1d(n, rx, mux)
        Ky_storage[:, n] = mapmri_psi_1d(n, ry, muy)
        Kz_storage[:, n] = mapmri_psi_1d(n, rz, muz)

    counter = 0
    for nx, ny, nz in ind_mat:
        K[:, counter] = (
            Kx_storage[:, nx] * Ky_storage[:, ny] * Kz_storage[:, nz]
            )
        counter += 1

    return K


def mapmri_odf_matrix(radial_order, mu, s, vertices):
    r"""Compute the MAPMRI ODF matrix [1]_  Eq. (33).

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
    .. [1] Ozarslan E. et al., "Mean apparent propagator (MAP) MRI: A novel
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
        f = np.sqrt(sfactorial(n1) * sfactorial(n2) * sfactorial(n3))
        odf_mat[:, j] = const * f * \
            _odf_cfunc(n1, n2, n3, alpha, beta, gamma, s)

    return odf_mat


def _odf_cfunc(n1, n2, n3, a, b, g, s):
    r"""Compute the MAPMRI ODF function from [1]_  Eq. (34).

    References
    ----------
    .. [1] Ozarslan E. et al., "Mean apparent propagator (MAP) MRI: A novel
    diffusion imaging method for mapping tissue microstructure",
    NeuroImage, 2013.
    """

    f = mfactorial
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
    Eq. (61).

    Parameters
    ----------
    radial_order : unsigned int,
        radial order of the mapmri basis.
    mu : float,
        positive isotropic scale factor of the basis
    q : array, shape (N,3)
        points in the q-space in which evaluate the basis

    References
    ----------
    .. [1] Ozarslan E. et al., "Mean apparent propagator (MAP) MRI: A novel
    diffusion imaging method for mapping tissue microstructure",
    NeuroImage, 2013.
    """
    qval, theta, phi = cart2sphere(q[:, 0], q[:, 1], q[:, 2])
    theta[np.isnan(theta)] = 0

    ind_mat = mapmri_isotropic_index_matrix(radial_order)

    n_elem = ind_mat.shape[0]
    n_qgrad = q.shape[0]
    M = np.zeros((n_qgrad, n_elem))

    counter = 0
    for n in range(0, radial_order + 1, 2):
        for j in range(1, 2 + n // 2):
            l_value = n + 2 - 2 * j
            const = mapmri_isotropic_radial_signal_basis(j, l_value, mu, qval)
            for m_value in range(-l_value, l_value+1):
                M[:, counter] = const * real_sh_descoteaux_from_index(
                    m_value, l_value, theta, phi)
                counter += 1
    return M


def mapmri_isotropic_radial_signal_basis(j, l_value, mu, qval):
    r"""Radial part of the isotropic 1D-SHORE signal basis [1]_ eq. (61).

    Parameters
    ----------
    j : unsigned int,
        a positive integer related to the radial order
    l_value : unsigned int,
        the spherical harmonic order (l)
    mu : float,
        isotropic scale factor of the basis
    qval : float,
        points in the q-space in which evaluate the basis

    References
    ----------
    .. [1] Ozarslan E. et al., "Mean apparent propagator (MAP) MRI: A novel
    diffusion imaging method for mapping tissue microstructure",
    NeuroImage, 2013.
    """
    pi2_mu2_q2 = 2 * np.pi ** 2 * mu ** 2 * qval ** 2
    const = (
        (-1) ** (l_value / 2) * np.sqrt(4.0 * np.pi) *
        pi2_mu2_q2 ** (l_value / 2) * np.exp(-pi2_mu2_q2) *
        genlaguerre(j - 1, l_value + 0.5)(2 * pi2_mu2_q2)
        )
    return const


def mapmri_isotropic_M_mu_independent(radial_order, q):
    r"""Computed the mu independent part of the signal design matrix.
    """
    ind_mat = mapmri_isotropic_index_matrix(radial_order)

    qval, theta, phi = cart2sphere(q[:, 0], q[:, 1], q[:, 2])
    theta[np.isnan(theta)] = 0

    n_elem = ind_mat.shape[0]
    n_rgrad = theta.shape[0]
    Q_mu_independent = np.zeros((n_rgrad, n_elem))

    counter = 0
    for n in range(0, radial_order + 1, 2):
        for j in range(1, 2 + n // 2):
            l_value = n + 2 - 2 * j
            const = np.sqrt(4 * np.pi) * (-1) ** (-l_value / 2) * \
                (2 * np.pi ** 2 * qval ** 2) ** (l_value / 2)
            for m_value in range(-1 * (n + 2 - 2 * j), (n + 3 - 2 * j)):
                Q_mu_independent[:, counter] = const * \
                    real_sh_descoteaux_from_index(m_value, l_value, theta, phi)
                counter += 1
    return Q_mu_independent


def mapmri_isotropic_M_mu_dependent(radial_order, mu, qval):
    """Computed the mu dependent part of the signal design matrix.
    """
    ind_mat = mapmri_isotropic_index_matrix(radial_order)

    n_elem = ind_mat.shape[0]
    n_qgrad = qval.shape[0]
    Q_u0_dependent = np.zeros((n_qgrad, n_elem))
    pi2q2mu2 = 2 * np.pi ** 2 * mu ** 2 * qval ** 2

    counter = 0
    for n in range(0, radial_order + 1, 2):
        for j in range(1, 2 + n // 2):
            l_value = n + 2 - 2 * j
            const = mu ** l_value * np.exp(-pi2q2mu2) *\
                genlaguerre(j - 1, l_value + 0.5)(2 * pi2q2mu2)
            for m_value in range(-l_value, l_value + 1):
                Q_u0_dependent[:, counter] = const
                counter += 1

    return Q_u0_dependent


def mapmri_isotropic_psi_matrix(radial_order, mu, rgrad):
    r""" Three dimensional isotropic MAPMRI propagator basis function from [1]_
    Eq. (61).

    Parameters
    ----------
    radial_order : unsigned int,
        radial order of the mapmri basis.
    mu : float,
        positive isotropic scale factor of the basis
    rgrad : array, shape (N,3)
        points in the r-space in which evaluate the basis

    References
    ----------
    .. [1] Ozarslan E. et al., "Mean apparent propagator (MAP) MRI: A novel
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
        for j in range(1, 2 + n // 2):
            l_value = n + 2 - 2 * j
            const = mapmri_isotropic_radial_pdf_basis(j, l_value, mu, r)
            for m_value in range(-l_value, l_value + 1):
                K[:, counter] = const * real_sh_descoteaux_from_index(
                    m_value, l_value, theta, phi)
                counter += 1
    return K


def mapmri_isotropic_radial_pdf_basis(j, l_value, mu, r):
    r"""Radial part of the isotropic 1D-SHORE propagator basis [1]_ eq. (61).

    Parameters
    ----------
    j : unsigned int,
        a positive integer related to the radial order
    l_value : unsigned int,
        the spherical harmonic order (l)
    mu : float,
        isotropic scale factor of the basis
    r : float,
        points in the r-space in which evaluate the basis

    References
    ----------
    .. [1] Ozarslan E. et al., "Mean apparent propagator (MAP) MRI: A novel
    diffusion imaging method for mapping tissue microstructure",
    NeuroImage, 2013.
    """
    r2u2 = r ** 2 / (2 * mu ** 2)
    const = (
        (-1) ** (j - 1) / (np.sqrt(2) * np.pi * mu ** 3) *
        r2u2 ** (l_value / 2) * np.exp(-r2u2) * 
        genlaguerre(j - 1, l_value + 0.5)(2 * r2u2)
        )
    return const


def mapmri_isotropic_K_mu_independent(radial_order, rgrad):
    """Computes mu independent part of K. Same trick as with M.
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
        for j in range(1, 2 + n // 2):
            l = n + 2 - 2 * j
            const = (-1) ** (j - 1) *\
                (np.sqrt(2) * np.pi) ** (-1) *\
                (r ** 2 / 2) ** (l / 2)
            for m in range(-l, l+1):
                K[:, counter] = const * real_sh_descoteaux_from_index(
                    m, l, theta, phi)
                counter += 1
    return K


def mapmri_isotropic_K_mu_dependent(radial_order, mu, rgrad):
    """Computes mu dependent part of M. Same trick as with M.
    """
    r, theta, phi = cart2sphere(rgrad[:, 0], rgrad[:, 1],
                                rgrad[:, 2])
    theta[np.isnan(theta)] = 0
    ind_mat = mapmri_isotropic_index_matrix(radial_order)
    n_elem = ind_mat.shape[0]
    n_rgrad = rgrad.shape[0]
    K = np.zeros((n_rgrad, n_elem))
    r2mu2 = r ** 2 / (2 * mu ** 2)

    counter = 0
    for n in range(0, radial_order + 1, 2):
        for j in range(1, 2 + n // 2):
            l = n + 2 - 2 * j
            const = (mu ** 3) ** (-1) * mu ** (-l) *\
                np.exp(-r2mu2) * genlaguerre(j - 1, l + 0.5)(2 * r2mu2)
            for m in range(-l, l + 1):
                K[:, counter] = const
                counter += 1
    return K


def binomialfloat(n, k):
    """Custom Binomial function
    """
    return sfactorial(n) / (sfactorial(n - k) * sfactorial(k))


def mapmri_isotropic_odf_matrix(radial_order, mu, s, vertices):
    r"""Compute the isotropic MAPMRI ODF matrix [1]_ Eq. 32 but for the
    isotropic propagator in [1]_ eq. (60). Analytical derivation in
    [2]_ eq. (C8).

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
    .. [1] Ozarslan E. et al., "Mean apparent propagator (MAP) MRI: A novel
    diffusion imaging method for mapping tissue microstructure",
    NeuroImage, 2013.

    .. [2] Fick, Rutger HJ, et al. "MAPL: Tissue microstructure estimation
    using Laplacian-regularized MAP-MRI and its application to HCP data."
    NeuroImage (2016).

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
        for j in range(1, 2 + n // 2):
            l = n + 2 - 2 * j
            kappa = ((-1) ** (j - 1) * 2 ** (-(l + 3) / 2.0) * mu ** s) / np.pi
            matsum = 0
            for k in range(0, j):
                matsum += ((-1) ** k * binomialfloat(j + l - 0.5, j - k - 1) *
                           gamma((l + s + 3) / 2.0 + k)) /\
                    (mfactorial(k) * 0.5 ** ((l + s + 3) / 2.0 + k))
            for m in range(-l, l + 1):
                odf_mat[:, counter] = kappa * matsum *\
                    real_sh_descoteaux_from_index(m, l, theta, phi)
                counter += 1

    return odf_mat


def mapmri_isotropic_odf_sh_matrix(radial_order, mu, s):
    r"""Compute the isotropic MAPMRI ODF matrix [1]_ Eq. 32 for the isotropic
    propagator in [1]_ eq. (60). Here we do not compute the sphere function but
    the spherical harmonics by only integrating the radial part of the
    propagator. We use the same derivation of the ODF in the isotropic
    implementation as in [2]_ eq. (C8).

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
    .. [1] Ozarslan E. et al., "Mean apparent propagator (MAP) MRI: A novel
    diffusion imaging method for mapping tissue microstructure",
    NeuroImage, 2013.

    .. [2] Fick, Rutger HJ, et al. "MAPL: Tissue microstructure estimation
    using Laplacian-regularized MAP-MRI and its application to HCP data."
    NeuroImage (2016).

    """
    sh_mat = sph_harm_ind_list(radial_order)
    ind_mat = mapmri_isotropic_index_matrix(radial_order)
    n_elem_shore = ind_mat.shape[0]
    n_elem_sh = sh_mat[0].shape[0]
    odf_sh_mat = np.zeros((n_elem_sh, n_elem_shore))

    counter = 0
    for n in range(0, radial_order + 1, 2):
        for j in range(1, 2 + n // 2):
            l = n + 2 - 2 * j
            kappa = ((-1) ** (j - 1) * 2 ** (-(l + 3) / 2.0) * mu ** s) / np.pi
            matsum = 0
            for k in range(0, j):
                matsum += ((-1) ** k * binomialfloat(j + l - 0.5, j - k - 1) *
                           gamma((l + s + 3) / 2.0 + k)) /\
                    (mfactorial(k) * 0.5 ** ((l + s + 3) / 2.0 + k))
            for m in range(-l, l + 1):
                index_overlap = np.all([l == sh_mat[1], m == sh_mat[0]], 0)
                odf_sh_mat[:, counter] = kappa * matsum * index_overlap
                counter += 1

    return odf_sh_mat


def mapmri_isotropic_laplacian_reg_matrix(radial_order, mu):
    r""" Computes the Laplacian regularization matrix for MAP-MRI's isotropic
    implementation [1]_ eq. (C7).

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
    .. [1] Fick, Rutger HJ, et al. "MAPL: Tissue microstructure estimation
    using Laplacian-regularized MAP-MRI and its application to HCP data."
    NeuroImage (2016).

    """
    ind_mat = mapmri_isotropic_index_matrix(radial_order)
    return mapmri_isotropic_laplacian_reg_matrix_from_index_matrix(
        ind_mat, mu
    )


def mapmri_isotropic_laplacian_reg_matrix_from_index_matrix(ind_mat, mu):
    r""" Computes the Laplacian regularization matrix for MAP-MRI's isotropic
    implementation [1]_ eq. (C7).

    Parameters
    ----------
    ind_mat : matrix (N_coef, 3),
            Basis order matrix
    mu : float,
        isotropic scale factor of the isotropic MAP-MRI basis

    Returns
    -------
    LR : Matrix, shape (N_coef, N_coef)
        Laplacian regularization matrix

    References
    ----------
    .. [1] Fick, Rutger HJ, et al. "MAPL: Tissue microstructure estimation
    using Laplacian-regularized MAP-MRI and its application to HCP data."
    NeuroImage (2016).

    """
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
                    LR[i, k] = LR[k, i] = 2.0 ** (2 - l) * np.pi ** 2 * mu *\
                        gamma(5 / 2.0 + jk + l) / gamma(jk)
                elif ji == (jk + 1):
                    LR[i, k] = LR[k, i] = 2.0 ** (2 - l) * np.pi ** 2 * mu *\
                        (-3 + 4 * ji + 2 * l) * gamma(3 / 2.0 + jk + l) /\
                        gamma(jk)
                elif ji == jk:
                    LR[i, k] = 2.0 ** (-l) * np.pi ** 2 * mu *\
                        (3 + 24 * ji ** 2 + 4 * (-2 + l) *
                         l + 12 * ji * (-1 + 2 * l)) *\
                        gamma(1 / 2.0 + ji + l) / gamma(ji)
                elif ji == (jk - 1):
                    LR[i, k] = LR[k, i] = 2.0 ** (2 - l) * np.pi ** 2 * mu *\
                        (-3 + 4 * jk + 2 * l) * gamma(3 / 2.0 + ji + l) /\
                        gamma(ji)
                elif ji == (jk - 2):
                    LR[i, k] = LR[k, i] = 2.0 ** (2 - l) * np.pi ** 2 * mu *\
                        gamma(5 / 2.0 + ji + l) / gamma(ji)

    return LR


def mapmri_isotropic_index_matrix(radial_order):
    r""" Calculates the indices for the isotropic MAPMRI basis [1]_ Fig 8.

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
    .. [1] Ozarslan E. et al., "Mean apparent propagator (MAP) MRI: A novel
    diffusion imaging method for mapping tissue microstructure",
    NeuroImage, 2013.

    """
    index_matrix = []
    for n in range(0, radial_order + 1, 2):
        for j in range(1, 2 + n // 2):
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
    points_inside_sphere = np.sqrt(np.einsum('ij,ij->i', vecs, vecs)) <= radius
    vecs_inside_sphere = vecs[points_inside_sphere]

    tab = vecs_inside_sphere / radius
    tab = tab * radius_max

    return tab


def delta(n, m):
    if n == m:
        return 1
    return 0


def map_laplace_u(n, m):
    r"""S(n, m) static matrix for Laplacian regularization [1]_ eq. (13).

    Parameters
    ----------
    n, m : unsigned int
        basis order of the MAP-MRI basis in different directions

    Returns
    -------
    U : float,
        Analytical integral of :math:`\phi_n(q) * \phi_m(q)`

    References
    ----------
    .. [1] Fick, Rutger HJ, et al. "MAPL: Tissue microstructure estimation
    using Laplacian-regularized MAP-MRI and its application to HCP data."
    NeuroImage (2016).

    """
    return (-1) ** n * delta(n, m) / (2 * np.sqrt(np.pi))


def map_laplace_t(n, m):
    r"""L(m, n) static matrix for Laplacian regularization [1]_ eq. (12).

    Parameters
    ----------
    n, m : unsigned int
        basis order of the MAP-MRI basis in different directions

    Returns
    -------
    T : float
        Analytical integral of :math:`\phi_n(q) * \phi_m''(q)`

    References
    ----------
    .. [1] Fick, Rutger HJ, et al. "MAPL: Tissue microstructure estimation
    using Laplacian-regularized MAP-MRI and its application to HCP data."
    NeuroImage (2016).

    """
    a = np.sqrt((m - 1) * m) * delta(m - 2, n)
    b = np.sqrt((n - 1) * n) * delta(n - 2, m)
    c = (2 * n + 1) * delta(m, n)
    return np.pi ** (3 / 2.) * (-1) ** (n + 1) * (a + b + c)


def map_laplace_s(n, m):
    r"""R(m,n) static matrix for Laplacian regularization [1]_ eq. (11).

    Parameters
    ----------
    n, m : unsigned int
        basis order of the MAP-MRI basis in different directions

    Returns
    -------
    S : float
        Analytical integral of :math:`\phi_n''(q) * \phi_m''(q)`

    References
    ----------
    .. [1] Fick, Rutger HJ, et al. "MAPL: Tissue microstructure estimation
    using Laplacian-regularized MAP-MRI and its application to HCP data."
    NeuroImage (2016).

    """

    k = 2 * np.pi ** (7 / 2.) * (-1) ** n
    a0 = 3 * (2 * n ** 2 + 2 * n + 1) * delta(n, m)
    sqmn = np.sqrt(gamma(m + 1) / gamma(n + 1))
    sqnm = 1 / sqmn
    an2 = 2 * (2 * n + 3) * sqmn * delta(m, n + 2)
    an4 = sqmn * delta(m, n + 4)
    am2 = 2 * (2 * m + 3) * sqnm * delta(m + 2, n)
    am4 = sqnm * delta(m + 4, n)

    return k * (a0 + an2 + an4 + am2 + am4)


def mapmri_STU_reg_matrices(radial_order):
    """Generate the static portions of the Laplacian regularization matrix
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
    .. [1] Fick, Rutger HJ, et al. "MAPL: Tissue microstructure estimation
    using Laplacian-regularized MAP-MRI and its application to HCP data."
    NeuroImage (2016).

    """
    S = np.zeros((radial_order + 1, radial_order + 1))
    for i in range(radial_order + 1):
        for j in range(radial_order + 1):
            S[i, j] = map_laplace_s(i, j)

    T = np.zeros((radial_order + 1, radial_order + 1))
    for i in range(radial_order + 1):
        for j in range(radial_order + 1):
            T[i, j] = map_laplace_t(i, j)

    U = np.zeros((radial_order + 1, radial_order + 1))
    for i in range(radial_order + 1):
        for j in range(radial_order + 1):
            U[i, j] = map_laplace_u(i, j)
    return S, T, U


def mapmri_laplacian_reg_matrix(ind_mat, mu, S_mat, T_mat, U_mat):
    """Put the Laplacian regularization matrix together [1]_ eq. (10).

    The static parts in S, T and U are multiplied and divided by the
    voxel-specific scale factors.

    Parameters
    ----------
    ind_mat : matrix (N_coef, 3),
        Basis order matrix
    mu : array, shape (3,)
        scale factors of the basis for x, y, z
    S, T, U : matrices, shape (N_coef,N_coef)
        Regularization submatrices

    Returns
    -------
    LR : matrix (N_coef, N_coef),
        Voxel-specific Laplacian regularization matrix

    References
    ----------
    .. [1] Fick, Rutger HJ, et al. "MAPL: Tissue microstructure estimation
    using Laplacian-regularized MAP-MRI and its application to HCP data."
    NeuroImage (2016).

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
                  S_mat[x[i], x[j]] * U_mat[y[i], y[j]] * U_mat[z[i], z[j]] +\
                  (uy ** 3 / (ux * uz)) *\
                  S_mat[y[i], y[j]] * U_mat[z[i], z[j]] * U_mat[x[i], x[j]] +\
                  (uz ** 3 / (ux * uy)) *\
                  S_mat[z[i], z[j]] * U_mat[x[i], x[j]] * U_mat[y[i], y[j]] +\
                  2 * ((ux * uy) / uz) *\
                  T_mat[x[i], x[j]] * T_mat[y[i], y[j]] * U_mat[z[i], z[j]] +\
                  2 * ((ux * uz) / uy) *\
                  T_mat[x[i], x[j]] * T_mat[z[i], z[j]] * U_mat[y[i], y[j]] +\
                  2 * ((uz * uy) / ux) *\
                  T_mat[z[i], z[j]] * T_mat[y[i], y[j]] * U_mat[x[i], x[j]]

    return LR


def generalized_crossvalidation_array(data, M, LR, weights_array=None):
    """Generalized Cross Validation Function [1]_ eq. (15).

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
    LR : matrix, shape (N_coef, N_coef)
        regularization matrix
    weights_array : array (N_of_weights)
        array of optional regularization weights

    """
    if weights_array is None:
        lrange = np.linspace(0.05, 1, 20)  # reasonably fast standard range
    else:
        lrange = weights_array

    samples = lrange.shape[0]
    MMt = np.dot(M.T, M)
    K = len(data)
    gcvold = gcvnew = 10e10  # set initialization gcv threshold very high
    i = -1
    while gcvold >= gcvnew and i < samples - 2:
        gcvold = gcvnew
        i = i + 1
        S = np.linalg.multi_dot([M, np.linalg.pinv(MMt + lrange[i] * LR), M.T])
        trS = np.trace(S)
        normyytilde = np.linalg.norm(data - np.dot(S, data), 2)
        gcvnew = normyytilde / (K - trS)
    lopt = lrange[i - 1]
    return lopt


def generalized_crossvalidation(data, M, LR, gcv_startpoint=5e-2):
    """Generalized Cross Validation Function [1]_ eq. (15).

    Finds optimal regularization weight based on generalized cross-validation.

    Parameters
    ----------
    data : array (N),
        data array
    M : matrix, shape (N, Ncoef)
        mapmri observation matrix
    LR : matrix, shape (N_coef, N_coef)
        regularization matrix
    gcv_startpoint : float
        startpoint for the gcv optimization

    Returns
    -------
    optimal_lambda : float,
        optimal regularization weight

    References
    ----------
    .. [1] Craven et al. "Smoothing Noisy Data with Spline Functions."
        NUMER MATH 31.4 (1978): 377-403.

    """
    MMt = np.dot(M.T, M)
    K = len(data)
    bounds = ((1e-5, 10),)
    solver = Optimizer(fun=gcv_cost_function,
                       x0=(gcv_startpoint,),
                       args=((data, M, MMt, K, LR),),
                       bounds=bounds)

    optimal_lambda = solver.xopt
    return optimal_lambda


def gcv_cost_function(weight, args):
    """The GCV cost function that is iterated [4]."""
    data, M, MMt, K, LR = args
    S = np.linalg.multi_dot([M, np.linalg.pinv(MMt + weight * LR), M.T])
    trS = np.trace(S)
    normyytilde = np.linalg.norm(data - np.dot(S, data), 2)
    gcv_value = normyytilde / (K - trS)
    return gcv_value
