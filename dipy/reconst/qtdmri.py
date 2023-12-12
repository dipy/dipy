import numpy as np

from dipy.reconst.cache import Cache
from dipy.core.geometry import cart2sphere
from dipy.reconst.multi_voxel import multi_voxel_fit
from scipy.special import genlaguerre, gamma
from dipy.core.gradients import gradient_table_from_gradient_strength_bvecs
from scipy import special
from warnings import warn
from dipy.reconst import mapmri
try:  # preferred scipy >= 0.14, required scipy >= 1.0
    from scipy.special import factorial, factorial2
except ImportError:
    from scipy.misc import factorial, factorial2
from scipy.optimize import fmin_l_bfgs_b
from dipy.reconst.shm import real_sh_descoteaux_from_index
import dipy.reconst.dti as dti
from dipy.utils.optpkg import optional_package
import random

cvxpy, have_cvxpy, _ = optional_package("cvxpy", min_version="1.4.1")
plt, have_plt, _ = optional_package("matplotlib.pyplot")


class QtdmriModel(Cache):
    r"""The q$\tau$-dMRI model [1] to analytically and continuously represent
        the q$\tau$ diffusion signal attenuation over diffusion sensitization
        q and diffusion time $\tau$. The model can be seen as an extension of
        the MAP-MRI basis [2] towards different diffusion times.

        The main idea is to model the diffusion signal over time and space as
        a linear combination of continuous functions,

        ..math::
            :nowrap:
                \begin{equation}
                    \hat{E}(\textbf{q},\tau;\textbf{c}) =
                    \sum_i^{N_{\textbf{q}}}\sum_k^{N_\tau} \textbf{c}_{ik}
                    \,\Phi_i(\textbf{q})\,T_k(\tau),
                \end{equation}

        where $\Phi$ and $T$ are the spatial and temporal basis functions,
        $N_{\textbf{q}}$ and $N_\tau$ are the maximum spatial and temporal
        order, and $i,k$ are basis order iterators.

        The estimation of the coefficients $c_i$ can be regularized using
        either analytic Laplacian regularization, sparsity regularization using
        the l1-norm, or both to do a type of elastic net regularization.

        From the coefficients, there exists an analytical formula to estimate
        the ODF, RTOP, RTAP, RTPP, QIV and MSD, for any diffusion time.

        Parameters
        ----------
        gtab : GradientTable,
            gradient directions and bvalues container class. The bvalues
            should be in the normal s/mm^2. big_delta and small_delta need to
            given in seconds.
        radial_order : unsigned int,
            an even integer representing the spatial/radial order of the basis.
        time_order : unsigned int,
            an integer larger or equal than zero representing the time order
            of the basis.
        laplacian_regularization : bool,
            Regularize using the Laplacian of the qt-dMRI basis.
        laplacian_weighting: string or scalar,
            The string 'GCV' makes it use generalized cross-validation to find
            the regularization weight [3]. A scalar sets the regularization
            weight to that value.
        l1_regularization : bool,
            Regularize by imposing sparsity in the coefficients using the
            l1-norm.
        l1_weighting : 'CV' or scalar,
            The string 'CV' makes it use five-fold cross-validation to find
            the regularization weight. A scalar sets the regularization weight
            to that value.
        cartesian : bool
            Whether to use the Cartesian or spherical implementation of the
            qt-dMRI basis, which we first explored in [4].
        anisotropic_scaling : bool
            Whether to use anisotropic scaling or isotropic scaling. This
            option can be used to test if the Cartesian implementation is
            equivalent with the spherical one when using the same scaling.
        normalization : bool
            Whether to normalize the basis functions such that their inner
            product is equal to one. Normalization is only necessary when
            imposing sparsity in the spherical basis if cartesian=False.
        constrain_q0 : bool
            whether to constrain the q0 point to unity along the tau-space.
            This is necessary to ensure that $E(0,\tau)=1$.
        bval_threshold : float
            the threshold b-value to be used, such that only data points below
            that threshold are used when estimating the scale factors.
        eigenvalue_threshold : float,
            Sets the minimum of the tensor eigenvalues in order to avoid
            stability problem.
        cvxpy_solver : str, optional
            cvxpy solver name. Optionally optimize the positivity constraint
            with a particular cvxpy solver. See See https://www.cvxpy.org/ for
            details. Default: ECOS.

        References
        ----------
        .. [1] Fick, Rutger HJ, et al. "Non-Parametric GraphNet-Regularized
           Representation of dMRI in Space and Time", Medical Image Analysis,
           2017.

        .. [2] Ozarslan E. et al., "Mean apparent propagator (MAP) MRI: A novel
           diffusion imaging method for mapping tissue microstructure",
           NeuroImage, 2013.

        .. [3] Craven et al. "Smoothing Noisy Data with Spline Functions."
           NUMER MATH 31.4 (1978): 377-403.

        .. [4] Fick, Rutger HJ, et al. "A unifying framework for spatial and
           temporal diffusion in diffusion mri." International Conference on
           Information Processing in Medical Imaging. Springer, Cham, 2015.
        """

    def __init__(self,
                 gtab,
                 radial_order=6,
                 time_order=2,
                 laplacian_regularization=False,
                 laplacian_weighting=0.2,
                 l1_regularization=False,
                 l1_weighting=0.1,
                 cartesian=True,
                 anisotropic_scaling=True,
                 normalization=False,
                 constrain_q0=True,
                 bval_threshold=1e10,
                 eigenvalue_threshold=1e-04,
                 cvxpy_solver="ECOS"
                 ):

        if radial_order % 2 or radial_order < 0:
            msg = "radial_order must be zero or an even positive integer."
            msg += " radial_order %s was given." % radial_order
            raise ValueError(msg)

        if time_order < 0:
            msg = "time_order must be larger or equal than zero integer."
            msg += " time_order %s was given." % time_order
            raise ValueError(msg)

        if not isinstance(laplacian_regularization, bool):
            msg = "laplacian_regularization must be True or False."
            msg += " Input value was %s." % laplacian_regularization
            raise ValueError(msg)

        if laplacian_regularization:
            msg = "laplacian_regularization weighting must be 'GCV' "
            msg += "or a float larger or equal than zero."
            msg += " Input value was %s." % laplacian_weighting
            if isinstance(laplacian_weighting, str):
                if laplacian_weighting != 'GCV':
                    raise ValueError(msg)
            elif isinstance(laplacian_weighting, float):
                if laplacian_weighting < 0:
                    raise ValueError(msg)
            else:
                raise ValueError(msg)

        if not isinstance(l1_regularization, bool):
            msg = "l1_regularization must be True or False."
            msg += " Input value was %s." % l1_regularization
            raise ValueError(msg)

        if l1_regularization:
            msg = "l1_weighting weighting must be 'CV' "
            msg += "or a float larger or equal than zero."
            msg += " Input value was %s." % l1_weighting
            if isinstance(l1_weighting, str):
                if l1_weighting != 'CV':
                    raise ValueError(msg)
            elif isinstance(l1_weighting, float):
                if l1_weighting < 0:
                    raise ValueError(msg)
            else:
                raise ValueError(msg)

        if not isinstance(cartesian, bool):
            msg = "cartesian must be True or False."
            msg += " Input value was %s." % cartesian
            raise ValueError(msg)

        if not isinstance(anisotropic_scaling, bool):
            msg = "anisotropic_scaling must be True or False."
            msg += " Input value was %s." % anisotropic_scaling
            raise ValueError(msg)

        if not isinstance(constrain_q0, bool):
            msg = "constrain_q0 must be True or False."
            msg += " Input value was %s." % constrain_q0
            raise ValueError(msg)

        if (not isinstance(bval_threshold, float) or
                bval_threshold < 0):
            msg = "bval_threshold must be a positive float."
            msg += " Input value was %s." % bval_threshold
            raise ValueError(msg)

        if (not isinstance(eigenvalue_threshold, float) or
                eigenvalue_threshold < 0):
            msg = "eigenvalue_threshold must be a positive float."
            msg += " Input value was %s." % eigenvalue_threshold
            raise ValueError(msg)

        if laplacian_regularization or l1_regularization:
            if not have_cvxpy:
                msg = "cvxpy must be installed for Laplacian or l1 "
                msg += "regularization."
                raise ImportError(msg)
            if cvxpy_solver is not None:
                if cvxpy_solver not in cvxpy.installed_solvers():
                    msg = "Input `cvxpy_solver` was set to %s." % cvxpy_solver
                    msg += " One of %s" % ', '.join(cvxpy.installed_solvers())
                    msg += " was expected."
                    raise ValueError(msg)

        if l1_regularization and not cartesian and not normalization:
            msg = "The non-Cartesian implementation must be normalized for the"
            msg += " l1-norm sparsity regularization to work. Set "
            msg += "normalization=True to proceed."
            raise ValueError(msg)

        self.gtab = gtab
        self.radial_order = radial_order
        self.time_order = time_order
        self.laplacian_regularization = laplacian_regularization
        self.laplacian_weighting = laplacian_weighting
        self.l1_regularization = l1_regularization
        self.l1_weighting = l1_weighting
        self.cartesian = cartesian
        self.anisotropic_scaling = anisotropic_scaling
        self.normalization = normalization
        self.constrain_q0 = constrain_q0
        self.bval_threshold = bval_threshold
        self.eigenvalue_threshold = eigenvalue_threshold
        self.cvxpy_solver = cvxpy_solver

        if self.cartesian:
            self.ind_mat = qtdmri_index_matrix(radial_order, time_order)
        else:
            self.ind_mat = qtdmri_isotropic_index_matrix(radial_order,
                                                         time_order)

        # precompute parts of laplacian regularization matrices
        self.part4_reg_mat_tau = part4_reg_matrix_tau(self.ind_mat, 1.)
        self.part23_reg_mat_tau = part23_reg_matrix_tau(self.ind_mat, 1.)
        self.part1_reg_mat_tau = part1_reg_matrix_tau(self.ind_mat, 1.)
        if self.cartesian:
            self.S_mat, self.T_mat, self.U_mat = (
                mapmri.mapmri_STU_reg_matrices(radial_order)
            )
        else:
            self.part1_uq_iso_precomp = (
                mapmri.mapmri_isotropic_laplacian_reg_matrix_from_index_matrix(
                    self.ind_mat[:, :3], 1.
                )
            )

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
                us, ut, R = qtdmri_anisotropic_scaling(data_norm[bval_mask],
                                                       qvals[bval_mask],
                                                       bvecs[bval_mask],
                                                       tau[bval_mask])
                tau_scaling = ut / us.mean()
                tau_scaled = tau * tau_scaling
                ut /= tau_scaling
                us = np.clip(us, self.eigenvalue_threshold, np.inf)
                q = np.dot(bvecs, R) * qvals[:, None]
                M = qtdmri_signal_matrix_(
                    self.radial_order, self.time_order, us, ut, q, tau_scaled,
                    self.normalization
                )
            else:
                us, ut = qtdmri_isotropic_scaling(data_norm, qvals, tau)
                tau_scaling = ut / us
                tau_scaled = tau * tau_scaling
                ut /= tau_scaling
                R = np.eye(3)
                us = np.tile(us, 3)
                q = bvecs * qvals[:, None]
                M = qtdmri_signal_matrix_(
                    self.radial_order, self.time_order, us, ut, q, tau_scaled,
                    self.normalization
                )
        else:
            us, ut = qtdmri_isotropic_scaling(data_norm, qvals, tau)
            tau_scaling = ut / us
            tau_scaled = tau * tau_scaling
            ut /= tau_scaling
            R = np.eye(3)
            us = np.tile(us, 3)
            q = bvecs * qvals[:, None]
            M = qtdmri_isotropic_signal_matrix_(
                self.radial_order, self.time_order, us[0], ut, q, tau_scaled,
                normalization=self.normalization
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
        if self.laplacian_regularization and not self.l1_regularization:
            if self.cartesian:
                laplacian_matrix = qtdmri_laplacian_reg_matrix(
                    self.ind_mat, us, ut, self.S_mat, self.T_mat, self.U_mat,
                    self.part1_reg_mat_tau,
                    self.part23_reg_mat_tau,
                    self.part4_reg_mat_tau,
                    normalization=self.normalization
                )
            else:
                laplacian_matrix = qtdmri_isotropic_laplacian_reg_matrix(
                    self.ind_mat, us, ut, self.part1_uq_iso_precomp,
                    self.part1_reg_mat_tau, self.part23_reg_mat_tau,
                    self.part4_reg_mat_tau,
                    normalization=self.normalization
                )
            if self.laplacian_weighting == 'GCV':
                try:
                    lopt = generalized_crossvalidation(data_norm, M,
                                                       laplacian_matrix)
                except BaseException:
                    msg = "Laplacian GCV failed. lopt defaulted to 2e-4."
                    warn(msg)
                    lopt = 2e-4
            elif np.isscalar(self.laplacian_weighting):
                lopt = self.laplacian_weighting
            c = cvxpy.Variable(M.shape[1])
            design_matrix = cvxpy.Constant(M) @ c
            objective = cvxpy.Minimize(
                cvxpy.sum_squares(design_matrix - data_norm) +
                lopt * cvxpy.quad_form(c, laplacian_matrix)
            )
            if self.constrain_q0:
                # just constraint first and last, otherwise the solver fails
                constraints = [M0[0] @ c == 1, M0[-1] @ c == 1]
            else:
                constraints = []
            prob = cvxpy.Problem(objective, constraints)
            try:
                prob.solve(solver=self.cvxpy_solver, verbose=False)
                cvxpy_solution_optimal = prob.status == 'optimal'
                qtdmri_coef = np.asarray(c.value).squeeze()
            except BaseException:
                qtdmri_coef = np.zeros(M.shape[1])
                cvxpy_solution_optimal = False
        elif self.l1_regularization and not self.laplacian_regularization:
            if self.l1_weighting == 'CV':
                alpha = l1_crossvalidation(b0s_mask, data_norm, M)
            elif np.isscalar(self.l1_weighting):
                alpha = self.l1_weighting
            c = cvxpy.Variable(M.shape[1])
            design_matrix = cvxpy.Constant(M) @ c
            objective = cvxpy.Minimize(
                cvxpy.sum_squares(design_matrix - data_norm) +
                alpha * cvxpy.norm1(c)
            )
            if self.constrain_q0:
                # just constraint first and last, otherwise the solver fails
                constraints = [M0[0] @ c == 1, M0[-1] @ c == 1]
            else:
                constraints = []
            prob = cvxpy.Problem(objective, constraints)
            try:
                prob.solve(solver=self.cvxpy_solver, verbose=False)
                cvxpy_solution_optimal = prob.status == 'optimal'
                qtdmri_coef = np.asarray(c.value).squeeze()
            except BaseException:
                qtdmri_coef = np.zeros(M.shape[1])
                cvxpy_solution_optimal = False
        elif self.l1_regularization and self.laplacian_regularization:
            if self.cartesian:
                laplacian_matrix = qtdmri_laplacian_reg_matrix(
                    self.ind_mat, us, ut, self.S_mat, self.T_mat, self.U_mat,
                    self.part1_reg_mat_tau,
                    self.part23_reg_mat_tau,
                    self.part4_reg_mat_tau,
                    normalization=self.normalization
                )
            else:
                laplacian_matrix = qtdmri_isotropic_laplacian_reg_matrix(
                    self.ind_mat, us, ut, self.part1_uq_iso_precomp,
                    self.part1_reg_mat_tau, self.part23_reg_mat_tau,
                    self.part4_reg_mat_tau,
                    normalization=self.normalization
                )
            if self.laplacian_weighting == 'GCV':
                lopt = generalized_crossvalidation(data_norm, M,
                                                   laplacian_matrix)
            elif np.isscalar(self.laplacian_weighting):
                lopt = self.laplacian_weighting
            if self.l1_weighting == 'CV':
                alpha = elastic_crossvalidation(b0s_mask, data_norm, M,
                                                laplacian_matrix, lopt)
            elif np.isscalar(self.l1_weighting):
                alpha = self.l1_weighting
            c = cvxpy.Variable(M.shape[1])
            design_matrix = cvxpy.Constant(M) @ c
            objective = cvxpy.Minimize(
                cvxpy.sum_squares(design_matrix - data_norm) +
                alpha * cvxpy.norm1(c) +
                lopt * cvxpy.quad_form(c, laplacian_matrix)
            )
            if self.constrain_q0:
                # just constraint first and last, otherwise the solver fails
                constraints = [M0[0] @ c == 1, M0[-1] @ c == 1]
            else:
                constraints = []
            prob = cvxpy.Problem(objective, constraints)
            try:
                prob.solve(solver=self.cvxpy_solver, verbose=False)
                cvxpy_solution_optimal = prob.status == 'optimal'
                qtdmri_coef = np.asarray(c.value).squeeze()
            except BaseException:
                qtdmri_coef = np.zeros(M.shape[1])
                cvxpy_solution_optimal = False
        elif not self.l1_regularization and not self.laplacian_regularization:
            # just use least squares with the observation matrix
            pseudoInv = np.linalg.pinv(M)
            qtdmri_coef = np.dot(pseudoInv, data_norm)
            # if cvxpy is used to constraint q0 without regularization the
            # solver often fails, so only first tau-position is manually
            # normalized.
            qtdmri_coef /= np.dot(M0[0], qtdmri_coef)
            cvxpy_solution_optimal = None

        if cvxpy_solution_optimal is False:
            msg = "cvxpy optimization resulted in non-optimal solution. Check "
            msg += "cvxpy_solution_optimal attribute in fitted object to see "
            msg += "which voxels are affected."
            warn(msg)
        return QtdmriFit(
            self, qtdmri_coef, us, ut, tau_scaling, R, lopt, alpha,
            cvxpy_solution_optimal)


class QtdmriFit:

    def __init__(self, model, qtdmri_coef, us, ut, tau_scaling, R, lopt,
                 alpha, cvxpy_solution_optimal):
        """ Calculates diffusion properties for a single voxel

        Parameters
        ----------
        model : object,
            AnalyticalModel
        qtdmri_coef : 1d ndarray,
            qtdmri coefficients
        us : array, 3 x 1
            spatial scaling factors
        ut : float
            temporal scaling factor
        tau_scaling : float,
            the temporal scaling that used to scale tau to the size of us
        R : 3x3 numpy array,
            tensor eigenvectors
        lopt : float,
            laplacian regularization weight
        alpha : float,
            the l1 regularization weight
        cvxpy_solution_optimal: bool,
            indicates whether the cvxpy coefficient estimation reach an optimal
            solution
        """

        self.model = model
        self._qtdmri_coef = qtdmri_coef
        self.us = us
        self.ut = ut
        self.tau_scaling = tau_scaling
        self.R = R
        self.lopt = lopt
        self.alpha = alpha
        self.cvxpy_solution_optimal = cvxpy_solution_optimal

    def qtdmri_to_mapmri_coef(self, tau):
        """This function converts the qtdmri coefficients to mapmri
        coefficients for a given tau [1]_. The conversion is performed by a
        matrix multiplication that evaluates the time-depenent part of the
        basis and multiplies it with the coefficients, after which coefficients
        with the same spatial orders are summed up, resulting in mapmri
        coefficients.

        Parameters
        ----------
        tau : float
            diffusion time (big_delta - small_delta / 3.) in seconds

        References
        ----------
        .. [1] Fick, Rutger HJ, et al. "Non-Parametric GraphNet-Regularized
            Representation of dMRI in Space and Time", Medical Image Analysis,
            2017.
        """
        if self.model.cartesian:
            II = self.model.cache_get('qtdmri_to_mapmri_matrix',
                                      key=tau)
            if II is None:
                II = qtdmri_to_mapmri_matrix(self.model.radial_order,
                                             self.model.time_order, self.ut,
                                             self.tau_scaling * tau)
                self.model.cache_set('qtdmri_to_mapmri_matrix',
                                     tau, II)
        else:
            II = self.model.cache_get('qtdmri_isotropic_to_mapmri_matrix',
                                      key=tau)
            if II is None:
                II = qtdmri_isotropic_to_mapmri_matrix(self.model.radial_order,
                                                       self.model.time_order,
                                                       self.ut,
                                                       self.tau_scaling * tau)
                self.model.cache_set('qtdmri_isotropic_to_mapmri_matrix',
                                     tau, II)
        mapmri_coef = np.dot(II, self._qtdmri_coef)
        return mapmri_coef

    def sparsity_abs(self, threshold=0.99):
        """As a measure of sparsity, calculates the number of largest
        coefficients needed to absolute sum up to 99% of the total absolute sum
        of all coefficients"""
        if not 0. < threshold < 1.:
            msg = "sparsity threshold must be between zero and one"
            raise ValueError(msg)
        total_weight = np.sum(abs(self._qtdmri_coef))
        absolute_normalized_coef_array = (
            np.sort(abs(self._qtdmri_coef))[::-1] / total_weight)
        current_weight = 0.
        counter = 0
        while current_weight < threshold:
            current_weight += absolute_normalized_coef_array[counter]
            counter += 1
        return counter

    def sparsity_density(self, threshold=0.99):
        """As a measure of sparsity, calculates the number of largest
        coefficients needed to squared sum up to 99% of the total squared sum
        of all coefficients"""
        if not 0. < threshold < 1.:
            msg = "sparsity threshold must be between zero and one"
            raise ValueError(msg)
        total_weight = np.sum(self._qtdmri_coef ** 2)
        squared_normalized_coef_array = (
            np.sort(self._qtdmri_coef ** 2)[::-1] / total_weight)
        current_weight = 0.
        counter = 0
        while current_weight < threshold:
            current_weight += squared_normalized_coef_array[counter]
            counter += 1
        return counter

    def odf(self, sphere, tau, s=2):
        r""" Calculates the analytical Orientation Distribution Function (ODF)
        for a given diffusion time tau from the signal, [1]_ Eq. (32). The
        qtdmri coefficients are first converted to mapmri coefficients
        following [2].

        Parameters
        ----------
        sphere : dipy sphere object
            sphere object with vertice orientations to compute the ODF on.
        tau : float
            diffusion time (big_delta - small_delta / 3.) in seconds
        s : unsigned int
            radial moment of the ODF

        References
        ----------
        .. [1] Ozarslan E. et. al, "Mean apparent propagator (MAP) MRI: A novel
            diffusion imaging method for mapping tissue microstructure",
            NeuroImage, 2013.
        .. [2] Fick, Rutger HJ, et al. "Non-Parametric GraphNet-Regularized
            Representation of dMRI in Space and Time", Medical Image Analysis,
            2017.
        """
        mapmri_coef = self.qtdmri_to_mapmri_coef(tau)
        if self.model.cartesian:
            v_ = sphere.vertices
            v = np.dot(v_, self.R)
            I_s = mapmri.mapmri_odf_matrix(self.model.radial_order, self.us,
                                           s, v)
            odf = np.dot(I_s, mapmri_coef)
        else:
            II = self.model.cache_get('ODF_matrix', key=(sphere, s))
            if II is None:
                II = mapmri.mapmri_isotropic_odf_matrix(
                    self.model.radial_order, 1, s, sphere.vertices)
                self.model.cache_set('ODF_matrix', (sphere, s), II)

            odf = self.us[0] ** s * np.dot(II, mapmri_coef)
        return odf

    def odf_sh(self, tau, s=2):
        r""" Calculates the real analytical odf for a given discrete sphere.
        Computes the design matrix of the ODF for the given sphere vertices
        and radial moment [1]_ eq. (32). The radial moment s acts as a
        sharpening method. The analytical equation for the spherical ODF basis
        is given in [2]_ eq. (C8). The qtdmri coefficients are first converted
        to mapmri coefficients following [3].

        Parameters
        ----------
        tau : float
            diffusion time (big_delta - small_delta / 3.) in seconds
        s : unsigned int
            radial moment of the ODF

        References
        ----------
        .. [1] Ozarslan E. et. al, "Mean apparent propagator (MAP) MRI: A novel
            diffusion imaging method for mapping tissue microstructure",
            NeuroImage, 2013.
        .. [2]_ Fick, Rutger HJ, et al. "MAPL: Tissue microstructure estimation
            using Laplacian-regularized MAP-MRI and its application to HCP
            data." NeuroImage (2016).
        .. [3] Fick, Rutger HJ, et al. "Non-Parametric GraphNet-Regularized
            Representation of dMRI in Space and Time", Medical Image Analysis,
            2017.
        """
        mapmri_coef = self.qtdmri_to_mapmri_coef(tau)
        if self.model.cartesian:
            msg = 'odf in spherical harmonics not yet implemented for '
            msg += 'cartesian implementation'
            raise ValueError(msg)
        II = self.model.cache_get('ODF_sh_matrix',
                                  key=(self.model.radial_order, s))

        if II is None:
            II = mapmri.mapmri_isotropic_odf_sh_matrix(self.model.radial_order,
                                                       1, s)
            self.model.cache_set('ODF_sh_matrix', (self.model.radial_order, s),
                                 II)

        odf = self.us[0] ** s * np.dot(II, mapmri_coef)
        return odf

    def rtpp(self, tau):
        r""" Calculates the analytical return to the plane probability (RTPP)
        for a given diffusion time tau, [1]_ eq. (42). The analytical formula
        for the isotropic MAP-MRI basis was derived in [2]_ eq. (C11). The
        qtdmri coefficients are first converted to mapmri coefficients
        following [3].

        Parameters
        ----------
        tau : float
            diffusion time (big_delta - small_delta / 3.) in seconds

        References
        ----------
        .. [1] Ozarslan E. et. al, "Mean apparent propagator (MAP) MRI: A novel
            diffusion imaging method for mapping tissue microstructure",
            NeuroImage, 2013.
        .. [2]_ Fick, Rutger HJ, et al. "MAPL: Tissue microstructure estimation
            using Laplacian-regularized MAP-MRI and its application to HCP
            data." NeuroImage (2016).
        .. [3] Fick, Rutger HJ, et al. "Non-Parametric GraphNet-Regularized
            Representation of dMRI in Space and Time", Medical Image Analysis,
            2017.
        """
        mapmri_coef = self.qtdmri_to_mapmri_coef(tau)

        if self.model.cartesian:
            ind_mat = mapmri.mapmri_index_matrix(self.model.radial_order)
            Bm = mapmri.b_mat(ind_mat)
            sel = Bm > 0.  # select only relevant coefficients
            const = 1 / (np.sqrt(2 * np.pi) * self.us[0])
            ind_sum = (-1.0) ** (ind_mat[sel, 0] / 2.0)
            rtpp_vec = const * Bm[sel] * ind_sum * mapmri_coef[sel]
            rtpp = rtpp_vec.sum()
            return rtpp
        else:
            ind_mat = mapmri.mapmri_isotropic_index_matrix(
                self.model.radial_order
            )
            rtpp_vec = np.zeros(int(ind_mat.shape[0]))
            count = 0
            for n in range(0, self.model.radial_order + 1, 2):
                    for j in range(1, 2 + n // 2):
                        ll = n + 2 - 2 * j
                        const = (-1 / 2.0) ** (ll / 2) / np.sqrt(np.pi)
                        matsum = 0
                        for k in range(0, j):
                            matsum += (
                                (-1) ** k *
                                mapmri.binomialfloat(j + ll - 0.5, j - k - 1) *
                                gamma(ll / 2 + k + 1 / 2.0) /
                                (factorial(k) * 0.5 ** (ll / 2 + 1 / 2.0 + k)))
                        for m in range(-ll, ll + 1):
                            rtpp_vec[count] = const * matsum
                            count += 1
            direction = np.array(self.R[:, 0], ndmin=2)
            r, theta, phi = cart2sphere(direction[:, 0], direction[:, 1],
                                        direction[:, 2])

            rtpp = mapmri_coef * (1 / self.us[0]) *\
                rtpp_vec * real_sh_descoteaux_from_index(
                    ind_mat[:, 2], ind_mat[:, 1], theta, phi)
            return rtpp.sum()

    def rtap(self, tau):
        r""" Calculates the analytical return to the axis probability (RTAP)
        for a given diffusion time tau, [1]_ eq. (40, 44a). The analytical
        formula for the isotropic MAP-MRI basis was derived in [2]_ eq. (C11).
        The qtdmri coefficients are first converted to mapmri coefficients
        following [3].

        Parameters
        ----------
        tau : float
            diffusion time (big_delta - small_delta / 3.) in seconds

        References
        ----------
        .. [1] Ozarslan E. et. al, "Mean apparent propagator (MAP) MRI: A novel
            diffusion imaging method for mapping tissue microstructure",
            NeuroImage, 2013.
        .. [2]_ Fick, Rutger HJ, et al. "MAPL: Tissue microstructure estimation
            using Laplacian-regularized MAP-MRI and its application to HCP
            data." NeuroImage (2016).
        .. [3] Fick, Rutger HJ, et al. "Non-Parametric GraphNet-Regularized
            Representation of dMRI in Space and Time", Medical Image Analysis,
            2017.
        """
        mapmri_coef = self.qtdmri_to_mapmri_coef(tau)

        if self.model.cartesian:
            ind_mat = mapmri.mapmri_index_matrix(self.model.radial_order)
            Bm = mapmri.b_mat(ind_mat)
            sel = Bm > 0.  # select only relevant coefficients
            const = 1 / (2 * np.pi * np.prod(self.us[1:]))
            ind_sum = (-1.0) ** (np.sum(ind_mat[sel, 1:], axis=1) / 2.0)
            rtap_vec = const * Bm[sel] * ind_sum * mapmri_coef[sel]
            rtap = np.sum(rtap_vec)
        else:
            ind_mat = mapmri.mapmri_isotropic_index_matrix(
                self.model.radial_order
            )
            rtap_vec = np.zeros(int(ind_mat.shape[0]))
            count = 0

            for n in range(0, self.model.radial_order + 1, 2):
                for j in range(1, 2 + n // 2):
                    ll = n + 2 - 2 * j
                    kappa = ((-1) ** (j - 1) * 2. ** (-(ll + 3) / 2.0)) / np.pi
                    matsum = 0
                    for k in range(0, j):
                        matsum += ((-1) ** k *
                                   mapmri.binomialfloat(j + ll - 0.5,
                                                        j - k - 1) *
                                   gamma((ll + 1) / 2.0 + k)) /\
                            (factorial(k) * 0.5 ** ((ll + 1) / 2.0 + k))
                    for m in range(-ll, ll + 1):
                        rtap_vec[count] = kappa * matsum
                        count += 1
            rtap_vec *= 2

            direction = np.array(self.R[:, 0], ndmin=2)
            r, theta, phi = cart2sphere(direction[:, 0],
                                        direction[:, 1], direction[:, 2])
            rtap_vec = mapmri_coef * (1 / self.us[0] ** 2) *\
                rtap_vec * real_sh_descoteaux_from_index(
                    ind_mat[:, 2], ind_mat[:, 1], theta, phi)
            rtap = rtap_vec.sum()
        return rtap

    def rtop(self, tau):
        r""" Calculates the analytical return to the origin probability (RTOP)
        for a given diffusion time tau [1]_ eq. (36, 43). The analytical
        formula for the isotropic MAP-MRI basis was derived in [2]_ eq. (C11).
        The qtdmri coefficients are first converted to mapmri coefficients
        following [3].

        Parameters
        ----------
        tau : float
            diffusion time (big_delta - small_delta / 3.) in seconds

        References
        ----------
        .. [1] Ozarslan E. et. al, "Mean apparent propagator (MAP) MRI: A novel
            diffusion imaging method for mapping tissue microstructure",
            NeuroImage, 2013.
        .. [2]_ Fick, Rutger HJ, et al. "MAPL: Tissue microstructure estimation
            using Laplacian-regularized MAP-MRI and its application to HCP
            data." NeuroImage (2016).
        .. [3] Fick, Rutger HJ, et al. "Non-Parametric GraphNet-Regularized
            Representation of dMRI in Space and Time", Medical Image Analysis,
            2017.
        """
        mapmri_coef = self.qtdmri_to_mapmri_coef(tau)

        if self.model.cartesian:
            ind_mat = mapmri.mapmri_index_matrix(self.model.radial_order)
            Bm = mapmri.b_mat(ind_mat)
            const = 1 / (np.sqrt(8 * np.pi ** 3) * np.prod(self.us))
            ind_sum = (-1.0) ** (np.sum(ind_mat, axis=1) / 2)
            rtop_vec = const * ind_sum * Bm * mapmri_coef
            rtop = rtop_vec.sum()
        else:
            ind_mat = mapmri.mapmri_isotropic_index_matrix(
                self.model.radial_order
            )
            Bm = mapmri.b_mat_isotropic(ind_mat)
            const = 1 / (2 * np.sqrt(2.0) * np.pi ** (3 / 2.0))
            rtop_vec = const * (-1.0) ** (ind_mat[:, 0] - 1) * Bm
            rtop = (1 / self.us[0] ** 3) * rtop_vec * mapmri_coef
            rtop = rtop.sum()
        return rtop

    def msd(self, tau):
        r""" Calculates the analytical Mean Squared Displacement (MSD) for a
        given diffusion time tau. It is defined as the Laplacian of the origin
        of the estimated signal [1]_. The analytical formula for the MAP-MRI
        basis was derived in [2]_ eq. (C13, D1). The qtdmri coefficients are
        first converted to mapmri coefficients following [3].

        Parameters
        ----------
        tau : float
            diffusion time (big_delta - small_delta / 3.) in seconds

        References
        ----------
        .. [1] Cheng, J., 2014. Estimation and Processing of Ensemble Average
            Propagator and Its Features in Diffusion MRI. Ph.D. Thesis.
        .. [2]_ Fick, Rutger HJ, et al. "MAPL: Tissue microstructure estimation
            using Laplacian-regularized MAP-MRI and its application to HCP
            data." NeuroImage (2016).
        .. [3] Fick, Rutger HJ, et al. "Non-Parametric GraphNet-Regularized
            Representation of dMRI in Space and Time", Medical Image Analysis,
            2017.
        """
        mapmri_coef = self.qtdmri_to_mapmri_coef(tau)
        mu = self.us
        if self.model.cartesian:
            ind_mat = mapmri.mapmri_index_matrix(self.model.radial_order)
            Bm = mapmri.b_mat(ind_mat)
            sel = Bm > 0.  # select only relevant coefficients
            ind_sum = np.sum(ind_mat[sel], axis=1)
            nx, ny, nz = ind_mat[sel].T

            numerator = (-1) ** (0.5 * (-ind_sum)) * np.pi ** (3 / 2.0) *\
                ((1 + 2 * nx) * mu[0] ** 2 + (1 + 2 * ny) *
                 mu[1] ** 2 + (1 + 2 * nz) * mu[2] ** 2)

            denominator = np.sqrt(2. ** (-ind_sum) * factorial(nx) *
                                  factorial(ny) * factorial(nz)) *\
                gamma(0.5 - 0.5 * nx) * gamma(0.5 - 0.5 * ny) *\
                gamma(0.5 - 0.5 * nz)

            msd_vec = mapmri_coef[sel] * (numerator / denominator)
            msd = msd_vec.sum()
        else:
            ind_mat = mapmri.mapmri_isotropic_index_matrix(
                self.model.radial_order
            )
            Bm = mapmri.b_mat_isotropic(ind_mat)
            sel = Bm > 0.  # select only relevant coefficients
            msd_vec = (4 * ind_mat[sel, 0] - 1) * Bm[sel]
            msd = self.us[0] ** 2 * msd_vec * mapmri_coef[sel]
            msd = msd.sum()
        return msd

    def qiv(self, tau):
        r""" Calculates the analytical Q-space Inverse Variance (QIV) for given
        diffusion time tau.
        It is defined as the inverse of the Laplacian of the origin of the
        estimated propagator [1]_ eq. (22). The analytical formula for the
        MAP-MRI basis was derived in [2]_ eq. (C14, D2). The qtdmri
        coefficients are first converted to mapmri coefficients following [3].

        Parameters
        ----------
        tau : float
            diffusion time (big_delta - small_delta / 3.) in seconds

        References
        ----------
        .. [1] Hosseinbor et al. "Bessel fourier orientation reconstruction
            (bfor): An analytical diffusion propagator reconstruction for
            hybrid diffusion imaging and computation of q-space indices.
            NeuroImage 64, 2013, 650–670.
        .. [2]_ Fick, Rutger HJ, et al. "MAPL: Tissue microstructure estimation
            using Laplacian-regularized MAP-MRI and its application to HCP
            data." NeuroImage (2016).
        .. [3] Fick, Rutger HJ, et al. "Non-Parametric GraphNet-Regularized
            Representation of dMRI in Space and Time", Medical Image Analysis,
            2017.
        """
        mapmri_coef = self.qtdmri_to_mapmri_coef(tau)
        ux, uy, uz = self.us
        if self.model.cartesian:
            ind_mat = mapmri.mapmri_index_matrix(self.model.radial_order)
            Bm = mapmri.b_mat(ind_mat)
            sel = Bm > 0  # select only relevant coefficients
            nx, ny, nz = ind_mat[sel].T

            numerator = 8 * np.pi ** 2 * (ux * uy * uz) ** 3 *\
                np.sqrt(factorial(nx) * factorial(ny) * factorial(nz)) *\
                gamma(0.5 - 0.5 * nx) * gamma(0.5 - 0.5 * ny) * \
                gamma(0.5 - 0.5 * nz)

            denominator = np.sqrt(2. ** (-1 + nx + ny + nz)) *\
                ((1 + 2 * nx) * uy ** 2 * uz ** 2 + ux ** 2 *
                 ((1 + 2 * nz) * uy ** 2 + (1 + 2 * ny) * uz ** 2))

            qiv_vec = mapmri_coef[sel] * (numerator / denominator)
            qiv = qiv_vec.sum()
        else:
            ind_mat = mapmri.mapmri_isotropic_index_matrix(
                self.model.radial_order
            )
            Bm = mapmri.b_mat_isotropic(ind_mat)
            sel = Bm > 0.  # select only relevant coefficients
            j = ind_mat[sel, 0]
            qiv_vec = ((8 * (-1.0) ** (1 - j) *
                        np.sqrt(2) * np.pi ** (7 / 2.)) / ((4.0 * j - 1) *
                                                           Bm[sel]))
            qiv = ux ** 5 * qiv_vec * mapmri_coef[sel]
            qiv = qiv.sum()
        return qiv

    def fitted_signal(self, gtab=None):
        """
        Recovers the fitted signal for the given gradient table. If no gradient
        table is given it recovers the signal for the gtab of the model object.
        """
        if gtab is None:
            E = self.predict(self.model.gtab)
        else:
            E = self.predict(gtab)
        return E

    def predict(self, qvals_or_gtab, S0=1.):
        r"""Recovers the reconstructed signal for any qvalue array or
        gradient table.
        """
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
                M = qtdmri_signal_matrix_(self.model.radial_order,
                                          self.model.time_order,
                                          self.us, self.ut, q_rot, tau,
                                          self.model.normalization)
            else:
                M = qtdmri_signal_matrix_(self.model.radial_order,
                                          self.model.time_order,
                                          self.us, self.ut, q, tau,
                                          self.model.normalization)
        else:
            M = qtdmri_isotropic_signal_matrix_(
                self.model.radial_order, self.model.time_order,
                self.us[0], self.ut, q, tau,
                normalization=self.model.normalization)
        E = S0 * np.dot(M, self._qtdmri_coef)
        return E

    def norm_of_laplacian_signal(self):
        """ Calculates the norm of the laplacian of the fitted signal [1]_.
        This information could be useful to assess if the extrapolation of the
        fitted signal contains spurious oscillations. A high laplacian norm may
        indicate that these are present, and any q-space indices that
        use integrals of the signal may be corrupted (e.g. RTOP, RTAP, RTPP,
        QIV). In contrast to [1], the Laplacian now describes oscillations in
        the 4-dimensional qt-signal [2].

        References
        ----------
        .. [1]_ Fick, Rutger HJ, et al. "MAPL: Tissue microstructure estimation
            using Laplacian-regularized MAP-MRI and its application to HCP
            data." NeuroImage (2016).
        .. [2] Fick, Rutger HJ, et al. "Non-Parametric GraphNet-Regularized
            Representation of dMRI in Space and Time", Medical Image Analysis,
            2017.
        """
        if self.model.cartesian:
            lap_matrix = qtdmri_laplacian_reg_matrix(
                self.model.ind_mat, self.us, self.ut,
                self.model.S_mat, self.model.T_mat, self.model.U_mat,
                self.model.part1_reg_mat_tau,
                self.model.part23_reg_mat_tau,
                self.model.part4_reg_mat_tau,
                normalization=self.model.normalization
            )
        else:
            lap_matrix = qtdmri_isotropic_laplacian_reg_matrix(
                self.model.ind_mat, self.us, self.ut,
                self.model.part1_uq_iso_precomp,
                self.model.part1_reg_mat_tau,
                self.model.part23_reg_mat_tau,
                self.model.part4_reg_mat_tau,
                normalization=self.model.normalization
            )
        norm_laplacian = np.dot(self._qtdmri_coef,
                                np.dot(self._qtdmri_coef, lap_matrix))
        return norm_laplacian

    def pdf(self, rt_points):
        """ Diffusion propagator on a given set of real points.
            if the array r_points is non writeable, then intermediate
            results are cached for faster recalculation
        """
        tau_scaling = self.tau_scaling
        rt_points_ = rt_points * np.r_[1, 1, 1, tau_scaling]
        if self.model.cartesian:
            K = qtdmri_eap_matrix_(self.model.radial_order,
                                   self.model.time_order,
                                   self.us, self.ut, rt_points_,
                                   self.model.normalization)
        else:
            K = qtdmri_isotropic_eap_matrix_(
                self.model.radial_order, self.model.time_order,
                self.us[0], self.ut, rt_points_,
                normalization=self.model.normalization
            )
        eap = np.dot(K, self._qtdmri_coef)
        return eap


def _qtdmri_to_mapmri_matrix(radial_order, time_order, ut, tau, isotropic):
    """Generate the matrix that maps the spherical qtdmri coefficients to
    MAP-MRI coefficients. The conversion is done by only evaluating the time
    basis for a diffusion time tau and summing up coefficients with the same
    spatial basis orders [1].

    Parameters
    ----------
    radial_order : unsigned int,
        an even integer representing the spatial/radial order of the basis.
    time_order : unsigned int,
        an integer larger or equal than zero representing the time order
        of the basis.
    ut : float
        temporal scaling factor
    tau : float
        diffusion time (big_delta - small_delta / 3.) in seconds
    isotropic : bool
        `True` if the case is isotropic.


    References
    ----------
    .. [1] Fick, Rutger HJ, et al. "Non-Parametric GraphNet-Regularized
        Representation of dMRI in Space and Time", Medical Image Analysis,
        2017.
    """
    if isotropic:
        mapmri_ind_mat = mapmri.mapmri_isotropic_index_matrix(radial_order)
        qtdmri_ind_mat = qtdmri_isotropic_index_matrix(radial_order, time_order)
    else:
        mapmri_ind_mat = mapmri.mapmri_index_matrix(radial_order)
        qtdmri_ind_mat = qtdmri_index_matrix(radial_order, time_order)

    n_elem_mapmri = int(mapmri_ind_mat.shape[0])
    n_elem_qtdmri = int(qtdmri_ind_mat.shape[0])

    temporal_storage = np.zeros(time_order + 1)
    for o in range(time_order + 1):
        temporal_storage[o] = temporal_basis(o, ut, tau)

    counter = 0
    mapmri_mat = np.zeros((n_elem_mapmri, n_elem_qtdmri))
    for j, ll, m, o in qtdmri_ind_mat:
        index_overlap = np.all([j == mapmri_ind_mat[:, 0],
                                ll == mapmri_ind_mat[:, 1],
                                m == mapmri_ind_mat[:, 2]], 0)
        mapmri_mat[:, counter] = temporal_storage[o] * index_overlap
        counter += 1
    return mapmri_mat


def qtdmri_to_mapmri_matrix(radial_order, time_order, ut, tau):
    """Generate the matrix that maps the qtdmri coefficients to MAP-MRI
    coefficients for the anisotropic case. The conversion is done by only
    evaluating the time basis for a diffusion time tau and summing up
    coefficients with the same spatial basis orders [1].

    Parameters
    ----------
    radial_order : unsigned int,
        an even integer representing the spatial/radial order of the basis.
    time_order : unsigned int,
        an integer larger or equal than zero representing the time order
        of the basis.
    ut : float
        temporal scaling factor
    tau : float
        diffusion time (big_delta - small_delta / 3.) in seconds

    References
    ----------
    .. [1] Fick, Rutger HJ, et al. "Non-Parametric GraphNet-Regularized
        Representation of dMRI in Space and Time", Medical Image Analysis,
        2017.
    """

    return _qtdmri_to_mapmri_matrix(radial_order, time_order, ut, tau, False)


def qtdmri_isotropic_to_mapmri_matrix(radial_order, time_order, ut, tau):
    """Generate the matrix that maps the spherical qtdmri coefficients to
    MAP-MRI coefficients for the isotropic case. The conversion is done by only
    evaluating the time basis for a diffusion time tau and summing up
    coefficients with the same spatial basis orders [1].

    Parameters
    ----------
    radial_order : unsigned int,
        an even integer representing the spatial/radial order of the basis.
    time_order : unsigned int,
        an integer larger or equal than zero representing the time order
        of the basis.
    ut : float
        temporal scaling factor
    tau : float
        diffusion time (big_delta - small_delta / 3.) in seconds

    References
    ----------
    .. [1] Fick, Rutger HJ, et al. "Non-Parametric GraphNet-Regularized
        Representation of dMRI in Space and Time", Medical Image Analysis,
        2017.
    """

    return _qtdmri_to_mapmri_matrix(radial_order, time_order, ut, tau, True)


def qtdmri_temporal_normalization(ut):
    """Normalization factor for the temporal basis"""
    return np.sqrt(ut)


def qtdmri_mapmri_normalization(mu):
    """Normalization factor for Cartesian MAP-MRI basis. The scaling is the
        same for every basis function depending only on the spatial scaling
        mu.
    """
    sqrtC = np.sqrt(8 * np.prod(mu)) * np.pi ** (3. / 4.)
    return sqrtC


def qtdmri_mapmri_isotropic_normalization(j, l, u0):
    """Normalization factor for Spherical MAP-MRI basis. The normalization
       for a basis function with orders [j,l,m] depends only on orders j,l and
       the isotropic scale factor.
    """
    sqrtC = ((2 * np.pi) ** (3. / 2.) *
             np.sqrt(2 ** l * u0 ** 3 * gamma(j) / gamma(j + l + 1. / 2.)))
    return sqrtC


def qtdmri_signal_matrix_(radial_order, time_order, us, ut, q, tau,
                          normalization=False):
    """Function to generate the qtdmri signal basis."""
    M = qtdmri_signal_matrix(radial_order, time_order, us, ut, q, tau)
    if normalization:
        sqrtC = qtdmri_mapmri_normalization(us)
        sqrtut = qtdmri_temporal_normalization(ut)
        sqrtCut = sqrtC * sqrtut
        M *= sqrtCut
    return M


def qtdmri_signal_matrix(radial_order, time_order, us, ut, q, tau):
    r"""Constructs the design matrix as a product of 3 separated radial,
    angular and temporal design matrices. It precomputes the relevant basis
    orders for each one and finally puts them together according to the index
    matrix
    """
    ind_mat = qtdmri_index_matrix(radial_order, time_order)

    n_dat = int(q.shape[0])
    n_elem = int(ind_mat.shape[0])
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


def qtdmri_eap_matrix(radial_order, time_order, us, ut, grid):
    r"""Constructs the design matrix as a product of 3 separated radial,
    angular and temporal design matrices. It precomputes the relevant basis
    orders for each one and finally puts them together according to the index
    matrix
    """
    ind_mat = qtdmri_index_matrix(radial_order, time_order)
    rx, ry, rz, tau = grid.T

    n_dat = int(rx.shape[0])
    n_elem = int(ind_mat.shape[0])
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


def qtdmri_isotropic_signal_matrix_(radial_order, time_order, us, ut, q, tau,
                                    normalization=False):
    M = qtdmri_isotropic_signal_matrix(
        radial_order, time_order, us, ut, q, tau
    )
    if normalization:
        ind_mat = qtdmri_isotropic_index_matrix(radial_order, time_order)
        j, ll = ind_mat[:, :2].T
        sqrtut = qtdmri_temporal_normalization(ut)
        sqrtC = qtdmri_mapmri_isotropic_normalization(j, ll, us)
        sqrtCut = sqrtC * sqrtut
        M = M * sqrtCut[None, :]
    return M


def qtdmri_isotropic_signal_matrix(radial_order, time_order, us, ut, q, tau):
    ind_mat = qtdmri_isotropic_index_matrix(radial_order, time_order)
    qvals, theta, phi = cart2sphere(q[:, 0], q[:, 1], q[:, 2])

    n_dat = int(qvals.shape[0])
    n_elem = int(ind_mat.shape[0])

    num_j = int(np.max(ind_mat[:, 0]))
    num_o = int(time_order + 1)
    num_l = int(radial_order // 2 + 1)
    num_m = int(radial_order * 2 + 1)

    # Radial Basis
    radial_storage = np.zeros([num_j, num_l, n_dat])
    for j in range(1, num_j + 1):
        for ll in range(0, radial_order + 1, 2):
            radial_storage[j - 1, ll // 2, :] = radial_basis_opt(
                j, ll, us, qvals)

    # Angular Basis
    angular_storage = np.zeros([num_l, num_m, n_dat])
    for ll in range(0, radial_order + 1, 2):
        for m in range(-ll, ll + 1):
            angular_storage[ll // 2, m + ll, :] = (
                angular_basis_opt(ll, m, qvals, theta, phi)
            )

    # Temporal Basis
    temporal_storage = np.zeros([num_o + 1, n_dat])
    for o in range(0, num_o + 1):
        temporal_storage[o, :] = temporal_basis(o, ut, tau)

    # Construct full design matrix
    M = np.zeros((n_dat, n_elem))
    counter = 0
    for j, ll, m, o in ind_mat:
        M[:, counter] = (radial_storage[j - 1, ll // 2, :] *
                         angular_storage[ll // 2, m + ll, :] *
                         temporal_storage[o, :])
        counter += 1
    return M


def qtdmri_eap_matrix_(radial_order, time_order, us, ut, grid,
                       normalization=False):
    sqrtCut = 1.
    if normalization:
        sqrtC = qtdmri_mapmri_normalization(us)
        sqrtut = qtdmri_temporal_normalization(ut)
        sqrtCut = sqrtC * sqrtut
    K_tau = (
        qtdmri_eap_matrix(radial_order, time_order, us, ut, grid) * sqrtCut
    )
    return K_tau


def qtdmri_isotropic_eap_matrix_(radial_order, time_order, us, ut, grid,
                                 normalization=False):
    K = qtdmri_isotropic_eap_matrix(
        radial_order, time_order, us, ut, grid
    )
    if normalization:
        ind_mat = qtdmri_isotropic_index_matrix(radial_order, time_order)
        j, ll = ind_mat[:, :2].T
        sqrtut = qtdmri_temporal_normalization(ut)
        sqrtC = qtdmri_mapmri_isotropic_normalization(j, ll, us)
        sqrtCut = sqrtC * sqrtut
        K = K * sqrtCut[None, :]
    return K


def qtdmri_isotropic_eap_matrix(radial_order, time_order, us, ut, grid):
    r"""Constructs the design matrix as a product of 3 separated radial,
    angular and temporal design matrices. It precomputes the relevant basis
    orders for each one and finally puts them together according to the index
    matrix
    """

    rx, ry, rz, tau = grid.T
    R, theta, phi = cart2sphere(rx, ry, rz)
    theta[np.isnan(theta)] = 0

    ind_mat = qtdmri_isotropic_index_matrix(radial_order, time_order)
    n_dat = int(R.shape[0])
    n_elem = int(ind_mat.shape[0])

    num_j = int(np.max(ind_mat[:, 0]))
    num_o = int(time_order + 1)
    num_l = int(radial_order / 2 + 1)
    num_m = int(radial_order * 2 + 1)

    # Radial Basis
    radial_storage = np.zeros([num_j, num_l, n_dat])
    for j in range(1, num_j + 1):
        for ll in range(0, radial_order + 1, 2):
            radial_storage[j - 1, ll // 2, :] = radial_basis_EAP_opt(
                j, ll, us, R)

    # Angular Basis
    angular_storage = np.zeros([num_j, num_l, num_m, n_dat])
    for j in range(1, num_j + 1):
        for ll in range(0, radial_order + 1, 2):
            for m in range(-ll, ll + 1):
                angular_storage[j - 1, ll // 2, m + ll, :] = (
                    angular_basis_EAP_opt(j, ll, m, R, theta, phi)
                )

    # Temporal Basis
    temporal_storage = np.zeros([num_o + 1, n_dat])
    for o in range(0, num_o + 1):
        temporal_storage[o, :] = temporal_basis(o, ut, tau)

    # Construct full design matrix
    M = np.zeros((n_dat, n_elem))
    counter = 0
    for j, ll, m, o in ind_mat:
        M[:, counter] = (radial_storage[j - 1, ll // 2, :] *
                         angular_storage[j - 1, ll // 2, m + ll, :] *
                         temporal_storage[o, :])
        counter += 1
    return M


def radial_basis_opt(j, l, us, q):
    """ Spatial basis dependent on spatial scaling factor us
    """
    const = (
        us ** l * np.exp(-2 * np.pi ** 2 * us ** 2 * q ** 2) *
        genlaguerre(j - 1, l + 0.5)(4 * np.pi ** 2 * us ** 2 * q ** 2)
    )
    return const


def angular_basis_opt(l, m, q, theta, phi):
    """ Angular basis independent of spatial scaling factor us. Though it
    includes q, it is independent of the data and can be precomputed.
    """
    const = (
        (-1) ** (l / 2) * np.sqrt(4.0 * np.pi) *
        (2 * np.pi ** 2 * q ** 2) ** (l / 2) *
        real_sh_descoteaux_from_index(m, l, theta, phi)
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
        (r ** 2 / 2) ** (l / 2) * real_sh_descoteaux_from_index(
            m, l, theta, phi)
    )
    return angular_part


def temporal_basis(o, ut, tau):
    """ Temporal basis dependent on temporal scaling factor ut
    """
    const = np.exp(-ut * tau / 2.0) * special.laguerre(o)(ut * tau)
    return const


def qtdmri_index_matrix(radial_order, time_order):
    """Computes the SHORE basis order indices according to [1].
    """
    index_matrix = []
    for n in range(0, radial_order + 1, 2):
        for i in range(0, n + 1):
            for j in range(0, n - i + 1):
                for o in range(0, time_order + 1):
                    index_matrix.append([n - i - j, j, i, o])

    return np.array(index_matrix)


def qtdmri_isotropic_index_matrix(radial_order, time_order):
    """Computes the SHORE basis order indices according to [1].
    """
    index_matrix = []
    for n in range(0, radial_order + 1, 2):
        for j in range(1, 2 + n // 2):
            ll = n + 2 - 2 * j
            for m in range(-ll, ll + 1):
                for o in range(0, time_order + 1):
                    index_matrix.append([j, ll, m, o])
    return np.array(index_matrix)


def qtdmri_laplacian_reg_matrix(ind_mat, us, ut,
                                S_mat=None, T_mat=None, U_mat=None,
                                part1_ut_precomp=None,
                                part23_ut_precomp=None,
                                part4_ut_precomp=None,
                                normalization=False):
    """Computes the cartesian qt-dMRI Laplacian regularization matrix. If
    given, uses precomputed matrices for temporal and spatial regularization
    matrices to speed up computation. Follows the the formulation of Appendix B
    in [1].

    References
    ----------
    .. [1] Fick, Rutger HJ, et al. "Non-Parametric GraphNet-Regularized
        Representation of dMRI in Space and Time", Medical Image Analysis,
        2017.
    """
    if S_mat is None or T_mat is None or U_mat is None:
        radial_order = ind_mat[:, :3].max()
        S_mat, T_mat, U_mat = mapmri.mapmri_STU_reg_matrices(radial_order)

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

    if normalization:
        temporal_normalization = qtdmri_temporal_normalization(ut) ** 2
        spatial_normalization = qtdmri_mapmri_normalization(us) ** 2
        regularization_matrix *= temporal_normalization * spatial_normalization
    return regularization_matrix


def qtdmri_isotropic_laplacian_reg_matrix(ind_mat, us, ut,
                                          part1_uq_iso_precomp=None,
                                          part1_ut_precomp=None,
                                          part23_ut_precomp=None,
                                          part4_ut_precomp=None,
                                          normalization=False):
    """Computes the spherical qt-dMRI Laplacian regularization matrix. If
    given, uses precomputed matrices for temporal and spatial regularization
    matrices to speed up computation. Follows the the formulation of Appendix C
    in [1].

    References
    ----------
    .. [1] Fick, Rutger HJ, et al. "Non-Parametric GraphNet-Regularized
        Representation of dMRI in Space and Time", Medical Image Analysis,
        2017.
    """
    if part1_uq_iso_precomp is None:
        part1_us = (
            mapmri.mapmri_isotropic_laplacian_reg_matrix_from_index_matrix(
                ind_mat[:, :3], us[0]
            )
        )
    else:
        part1_us = part1_uq_iso_precomp * us[0]

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

    part23_us = part23_iso_reg_matrix_q(ind_mat, us[0])
    part4_us = part4_iso_reg_matrix_q(ind_mat, us[0])

    regularization_matrix = (
        part1_us * part1_ut + part23_us * part23_ut + part4_us * part4_ut
    )

    if normalization:
        temporal_normalization = qtdmri_temporal_normalization(ut) ** 2
        j, ll = ind_mat[:, :2].T
        pre_spatial_norm = qtdmri_mapmri_isotropic_normalization(j, ll, us[0])
        spatial_normalization = np.outer(pre_spatial_norm, pre_spatial_norm)
        regularization_matrix *= temporal_normalization * spatial_normalization
    return regularization_matrix


def part23_reg_matrix_q(ind_mat, U_mat, T_mat, us):
    """Partial cartesian spatial Laplacian regularization matrix following
    second line of Eq. (B2) in [1].

    References
    ----------
    .. [1] Fick, Rutger HJ, et al. "Non-Parametric GraphNet-Regularized
        Representation of dMRI in Space and Time", Medical Image Analysis,
        2017.
    """
    ux, uy, uz = us
    x, y, z, _ = ind_mat.T
    n_elem = int(ind_mat.shape[0])
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
    """Partial spherical spatial Laplacian regularization matrix following the
    equation below Eq. (C4) in [1].

    References
    ----------
    .. [1] Fick, Rutger HJ, et al. "Non-Parametric GraphNet-Regularized
        Representation of dMRI in Space and Time", Medical Image Analysis,
        2017.
    """
    n_elem = int(ind_mat.shape[0])

    LR = np.zeros((n_elem, n_elem))

    for i in range(n_elem):
        for k in range(i, n_elem):
            if ind_mat[i, 1] == ind_mat[k, 1] and \
               ind_mat[i, 2] == ind_mat[k, 2]:
                ji = ind_mat[i, 0]
                jk = ind_mat[k, 0]
                ll = ind_mat[i, 1]
                if ji == (jk + 1):
                    LR[i, k] = LR[k, i] = (
                        2. ** (-ll) * -gamma(3 / 2.0 + jk + ll) / gamma(jk)
                    )
                elif ji == jk:
                    LR[i, k] = LR[k, i] = 2. ** (-(ll + 1)) *\
                        (1 - 4 * ji - 2 * ll) *\
                        gamma(1 / 2.0 + ji + ll) / gamma(ji)
                elif ji == (jk - 1):
                    LR[i, k] = LR[k, i] = 2. ** (-ll) *\
                        -gamma(3 / 2.0 + ji + ll) / gamma(ji)
    return LR / us


def part4_reg_matrix_q(ind_mat, U_mat, us):
    """Partial cartesian spatial Laplacian regularization matrix following
    equation Eq. (B2) in [1].

    References
    ----------
    .. [1] Fick, Rutger HJ, et al. "Non-Parametric GraphNet-Regularized
        Representation of dMRI in Space and Time", Medical Image Analysis,
        2017.
    """
    ux, uy, uz = us
    x, y, z, _ = ind_mat.T
    n_elem = int(ind_mat.shape[0])
    LR = np.zeros((n_elem, n_elem))
    for i in range(n_elem):
        for k in range(i, n_elem):
            if x[i] == x[k] and y[i] == y[k] and z[i] == z[k]:
                LR[i, k] = LR[k, i] = (
                    (1. / (ux * uy * uz)) * U_mat[x[i], x[k]] *
                    U_mat[y[i], y[k]] * U_mat[z[i], z[k]]
                )
    return LR


def part4_iso_reg_matrix_q(ind_mat, us):
    """Partial spherical spatial Laplacian regularization matrix following the
    equation below Eq. (C4) in [1].

    References
    ----------
    .. [1] Fick, Rutger HJ, et al. "Non-Parametric GraphNet-Regularized
        Representation of dMRI in Space and Time", Medical Image Analysis,
        2017.
    """
    n_elem = int(ind_mat.shape[0])
    LR = np.zeros((n_elem, n_elem))
    for i in range(n_elem):
        for k in range(i, n_elem):
            if ind_mat[i, 0] == ind_mat[k, 0] and \
               ind_mat[i, 1] == ind_mat[k, 1] and \
               ind_mat[i, 2] == ind_mat[k, 2]:
                ji = ind_mat[i, 0]
                ll = ind_mat[i, 1]
                LR[i, k] = LR[k, i] = (
                    2. ** (-(ll + 2)) * gamma(1 / 2.0 + ji + ll) /
                    (np.pi ** 2 * gamma(ji))
                )

    return LR / us ** 3


def part1_reg_matrix_tau(ind_mat, ut):
    """Partial temporal Laplacian regularization matrix following
    Appendix B in [1].

    References
    ----------
    .. [1] Fick, Rutger HJ, et al. "Non-Parametric GraphNet-Regularized
        Representation of dMRI in Space and Time", Medical Image Analysis,
        2017.
    """
    n_elem = int(ind_mat.shape[0])
    LD = np.zeros((n_elem, n_elem))
    for i in range(n_elem):
        for k in range(i, n_elem):
            oi = ind_mat[i, 3]
            ok = ind_mat[k, 3]
            if oi == ok:
                LD[i, k] = LD[k, i] = 1. / ut
    return LD


def part23_reg_matrix_tau(ind_mat, ut):
    """Partial temporal Laplacian regularization matrix following
    Appendix B in [1].

    References
    ----------
    .. [1] Fick, Rutger HJ, et al. "Non-Parametric GraphNet-Regularized
        Representation of dMRI in Space and Time", Medical Image Analysis,
        2017.
    """
    n_elem = int(ind_mat.shape[0])
    LD = np.zeros((n_elem, n_elem))
    for i in range(n_elem):
        for k in range(i, n_elem):
            oi = ind_mat[i, 3]
            ok = ind_mat[k, 3]
            if oi == ok:
                LD[i, k] = LD[k, i] = 1 / 2.
            else:
                LD[i, k] = LD[k, i] = np.abs(oi - ok)
    return ut * LD


def part4_reg_matrix_tau(ind_mat, ut):
    """Partial temporal Laplacian regularization matrix following
    Appendix B in [1].

    References
    ----------
    .. [1] Fick, Rutger HJ, et al. "Non-Parametric GraphNet-Regularized
        Representation of dMRI in Space and Time", Medical Image Analysis,
        2017.
    """
    n_elem = int(ind_mat.shape[0])
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


def H(value):
    """Step function of H(x)=1 if x>=0 and zero otherwise. Used for the
    temporal laplacian matrix."""
    if value >= 0:
        return 1
    return 0


def generalized_crossvalidation(data, M, LR, startpoint=5e-4):
    r"""Generalized Cross Validation Function [1].

    References
    ----------
    .. [1] Craven et al. "Smoothing Noisy Data with Spline Functions."
        NUMER MATH 31.4 (1978): 377-403.
    """
    startpoint = 1e-4
    MMt = np.dot(M.T, M)
    K = len(data)
    input_stuff = (data, M, MMt, K, LR)

    bounds = ((1e-5, 1),)
    res = fmin_l_bfgs_b(GCV_cost_function,
                        startpoint, args=(input_stuff,), approx_grad=True,
                        bounds=bounds, disp=False, pgtol=1e-10, factr=10.)
    return res[0][0]


def GCV_cost_function(weight, arguments):
    r"""Generalized Cross Validation Function that is iterated [1].

    References
    ----------
    .. [1] Craven et al. "Smoothing Noisy Data with Spline Functions."
        NUMER MATH 31.4 (1978): 377-403.
    """
    data, M, MMt, K, LR = arguments
    S = np.dot(np.dot(M, np.linalg.pinv(MMt + weight * LR)), M.T)
    trS = np.trace(S)
    normyytilde = np.linalg.norm(data - np.dot(S, data), 2)
    gcv_value = normyytilde / (K - trS)
    return gcv_value


def qtdmri_isotropic_scaling(data, q, tau):
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

    us = np.sqrt(np.dot(logE_q, inv_B_q)).item()
    ut = np.dot(logE_tau, inv_B_tau).item()
    return us, ut


def qtdmri_anisotropic_scaling(data, q, bvecs, tau):
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

    ut = np.dot(logE_tau, inv_B_tau).item()

    return us, ut, R


def design_matrix_spatial(bvecs, qvals):
    """  Constructs design matrix for DTI weighted least squares or
    least squares fitting. (Basser et al., 1994a)

    Parameters
    ----------
    bvecs : array (N x 3)
        unit b-vectors of the acquisition.
    qvals : array (N,)
        corresponding q-values in 1/mm

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


def qtdmri_number_of_coefficients(radial_order, time_order):
    """Computes the total number of coefficients of the qtdmri basis given a
    radial and temporal order. Equation given below Eq (9) in [1].

    References
    ----------
    .. [1] Fick, Rutger HJ, et al. "Non-Parametric GraphNet-Regularized
        Representation of dMRI in Space and Time", Medical Image Analysis,
        2017.
    """
    F = np.floor(radial_order / 2.)
    Msym = (F + 1) * (F + 2) * (4 * F + 3) / 6
    M_total = Msym * (time_order + 1)
    return M_total


def l1_crossvalidation(b0s_mask, E, M, weight_array=np.linspace(0, .4, 21)):
    """cross-validation function to find the optimal weight of alpha for
    sparsity regularization"""
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
            design_matrix = cvxpy.Constant(M[test]) @ c
            recovered_signal = cvxpy.Constant(M[sub]) @ c
            data = cvxpy.Constant(E[test])
            objective = cvxpy.Minimize(
                cvxpy.sum_squares(design_matrix - data) +
                alpha * cvxpy.norm1(c)
            )
            constraints = []
            prob = cvxpy.Problem(objective, constraints)
            prob.solve(solver="ECOS", verbose=False)
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
    """cross-validation function to find the optimal weight of alpha for
    sparsity regularization when also Laplacian regularization is used."""
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
        c = cvxpy.Variable(M.shape[1])
        design_matrix = cvxpy.Constant(M[test]) @ c
        recovered_signal = cvxpy.Constant(M[sub]) @ c

        data = cvxpy.Constant(E[test])
        constraints = []
        while cv_old >= cv_new and counter < weight_array.shape[0]:
            alpha = weight_array[counter]
            objective = cvxpy.Minimize(
                cvxpy.sum_squares(design_matrix - data) +
                alpha * cvxpy.norm1(c) +
                lopt * cvxpy.quad_form(c, L)
            )
            prob = cvxpy.Problem(objective, constraints)
            prob.solve(solver="ECOS", verbose=False)
            errorlist[i, counter] = np.mean(
                (E[sub] - np.asarray(recovered_signal.value).squeeze()) ** 2)
            cv_old = errorlist[i, counter - 1]
            cv_new = errorlist[i, counter]
            counter += 1
        optimal_alpha_sub[i] = weight_array[counter - 1]
    optimal_alpha = optimal_alpha_sub.mean()
    return optimal_alpha


def visualise_gradient_table_G_Delta_rainbow(
        gtab,
        big_delta_start=None, big_delta_end=None, G_start=None, G_end=None,
        bval_isolines=np.r_[0, 250, 1000, 2500, 5000, 7500, 10000, 14000],
        alpha_shading=0.6):
    """This function visualizes a q-tau acquisition scheme as a function of
    gradient strength and pulse separation (big_delta). It represents every
    measurements at its G and big_delta position regardless of b-vector, with a
    background of b-value isolines for reference. It assumes there is only one
    unique pulse length (small_delta) in the acquisition scheme.

    Parameters
    ----------
    gtab : GradientTable object
        constructed gradient table with big_delta and small_delta given as
        inputs.
    big_delta_start : float,
        optional minimum big_delta that is plotted in seconds
    big_delta_end : float,
        optional maximum big_delta that is plotted in seconds
    G_start : float,
        optional minimum gradient strength that is plotted in T/m
    G_end : float,
        optional maximum gradient strength that is plotted in T/m
    bval_isolines : array,
        optional array of bvalue isolines that are plotted in the background
    alpha_shading : float between [0-1]
        optional shading of the bvalue colors in the background
    """
    Delta = gtab.big_delta  # in seconds
    delta = gtab.small_delta  # in seconds
    G = gtab.gradient_strength * 1e3  # in SI units T/m

    if len(np.unique(delta)) > 1:
        msg = "This acquisition has multiple small_delta values. "
        msg += "This visualization assumes there is only one small_delta."
        raise ValueError(msg)

    if big_delta_start is None:
        big_delta_start = 0.005
    if big_delta_end is None:
        big_delta_end = Delta.max() + 0.004
    if G_start is None:
        G_start = 0.
    if G_end is None:
        G_end = G.max() + .05

    Delta_ = np.linspace(big_delta_start, big_delta_end, 50)
    G_ = np.linspace(G_start, G_end, 50)
    Delta_grid, G_grid = np.meshgrid(Delta_, G_)
    dummy_bvecs = np.tile([0, 0, 1], (len(G_grid.ravel()), 1))
    gtab_grid = gradient_table_from_gradient_strength_bvecs(
        G_grid.ravel() / 1e3, dummy_bvecs, Delta_grid.ravel(), delta[0]
    )
    bvals_ = gtab_grid.bvals.reshape(G_grid.shape)

    plt.contourf(Delta_, G_, bvals_,
                 levels=bval_isolines,
                 cmap='rainbow', alpha=alpha_shading)
    cb = plt.colorbar(spacing="proportional")
    cb.ax.tick_params(labelsize=16)
    plt.scatter(Delta, G, c='k', s=25)

    plt.xlim(big_delta_start, big_delta_end)
    plt.ylim(G_start, G_end)
    cb.set_label('b-value ($s$/$mm^2$)', fontsize=18)
    plt.xlabel(r'Pulse Separation $\Delta$ [sec]', fontsize=18)
    plt.ylabel('Gradient Strength [T/m]', fontsize=18)
    return None
