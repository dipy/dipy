from __future__ import division
import warnings
from math import factorial
import numpy as np
from scipy.special import genlaguerre, gamma, hyp2f1
from dipy.reconst.cache import Cache
from dipy.reconst.multi_voxel import multi_voxel_fit
from .shm import real_sym_sh_brainsuite
from dipy.core.geometry import cart2sphere
from dipy.utils.optpkg import optional_package

cvxpy, have_cvxpy, _ = optional_package("cvxpy")
sklearn, have_sklearn, _ = optional_package("sklearn")
if have_sklearn:
    from sklearn.linear_model import Lasso, LassoCV
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.metrics import r2_score

class BrainSuiteShoreModel(Cache):
    r"""Simple Harmonic Oscillator based Reconstruction and Estimation
    (SHORE) [1]_ of the diffusion signal.

    The main idea is to model the diffusion signal as a linear combination of
    continuous functions $\phi_i$,

    ..math::
        :nowrap:
            \begin{equation}
                S(\mathbf{q})= \sum_{i=0}^I  c_{i} \phi_{i}(\mathbf{q}).
            \end{equation}

    where $\mathbf{q}$ is the wave vector which corresponds to different
    gradient directions. Numerous continuous functions $\phi_i$ can be used to
    model $S$. This specifically comes from [1].

    References
    ----------

    .. [1] Merlet S. et al., "Continuous diffusion signal, EAP and ODF
           estimation via Compressive Sensing in diffusion MRI", Medical
           Image Analysis, 2013.

    Notes
    -----
    The implementation of SHORE depends on CVXPY (http://www.cvxpy.org/).

    """

    def __init__(
            self,
            gtab,
            regularization="L1",
            radial_order=6,
            zeta=700,
            tau=1. / (4 * np.pi**2),
            # For L2 method
            lambdaN=1e-8,
            lambdaL=1e-8,
            # For L1 method
            regularization_weighting="CV",
            l1_positive_constraint=False,
            l1_cv=3,
            l1_maxiter=1000,
            l1_verbose=False,
            l1_alpha=1.0,
            pos_grid=11,
            pos_radius=20e-03):
        r""" Analytical and continuous modeling of the diffusion signal with
        respect to the SHORE basis [1,2]_.
        This implementation is a modification of SHORE presented in [1]_.
        The modification was made to obtain the same ordering of the basis
        presented in [2,3]_.

        The main idea is to model the diffusion signal as a linear
        combination of continuous functions $\phi_i$,

        ..math::
            :nowrap:
                \begin{equation}
                    S(\mathbf{q})= \sum_{i=0}^I  c_{i} \phi_{i}(\mathbf{q}).
                \end{equation}

        where $\mathbf{q}$ is the wave vector which corresponds to different
        gradient directions.

        From the $c_i$ coefficients, there exists an analytical formula to
        estimate the ODF.


        Parameters
        ----------
        gtab : GradientTable,
            gradient directions and bvalues container class
        radial_order : unsigned int,
            an even integer that represent the order of the basis
        zeta : unsigned int,
            scale factor
        lambdaN : float,
            radial regularisation constant
        lambdaL : float,
            angular regularisation constant
        tau : float,
            diffusion time. By default the value that makes q equal to the
            square root of the b-value.
        pos_grid : int,
            Grid that define the points of the EAP in which we want to enforce
            positivity.
        pos_radius : float,
            Radius of the grid of the EAP in which enforce positivity in
            millimeters. By default 20e-03 mm.


        References
        ----------
        .. [1] Merlet S. et al., "Continuous diffusion signal, EAP and
        ODF estimation via Compressive Sensing in diffusion MRI", Medical
        Image Analysis, 2013.

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
        data, golden_directions = SticksAndBall(
            gtab, d=0.0015, S0=1., angles=[(0, 0), (90, 0)],
            fractions=[50, 50], snr=None)
        from dipy.reconst.canal import ShoreModel
        radial_order = 4
        zeta = 700
        asm = ShoreModel(gtab, radial_order=radial_order, zeta=zeta,
                         lambdaN=1e-8, lambdaL=1e-8)
        asmfit = asm.fit(data)
        odf= asmfit.odf(sphere)
        """

        self.bvals = gtab.bvals
        self.bvecs = gtab.bvecs
        self.gtab = gtab

        self.radial_order = radial_order
        self.ind_mat = shore_index_matrix(self.radial_order)
        self.n_coefs = self.ind_mat.shape[0]
        self.zeta = zeta
        self.regularization = regularization

        # Effective diffusion time
        if (gtab.big_delta is None) or (gtab.small_delta is None):
            self.tau = tau
        else:
            self.tau = gtab.big_delta - gtab.small_delta / 3.0

        # L2 parameters
        self.lambdaL = lambdaL
        self.lambdaN = lambdaN
        self.Nshore = self._n_shore()
        self.Lshore = self._l_shore()

        # L1 parameters
        self.l1_positive_constraint = l1_positive_constraint
        self.regularization_weighting = regularization_weighting
        self.l1_cv = l1_cv
        self.l1_maxiter = l1_maxiter
        self.l1_verbose = l1_verbose
        self.l1_alpha = l1_alpha

        # For computing EAP
        self.pos_grid = pos_grid
        self.pos_radius = pos_radius

    def _n_shore(self):
        n = self.ind_mat[:, 0]
        return np.diag((n * (n + 1))**2)

    def _l_shore(self):
        ell = self.ind_mat[:, 1]
        return np.diag((ell * (ell + 1))**2)

    @multi_voxel_fit
    def fit(self, data):

        # Generate the SHORE basis
        M = self.cache_get('shore_matrix', key=self.gtab)
        if M is None:
            M = brainsuite_shore_basis(
                self.radial_order, self.zeta, self.gtab, self.tau)
            self.cache_set('shore_matrix', self.gtab, M)
        MpseudoInv = self.cache_get('shore_matrix_reg_pinv', key=self.gtab)
        if MpseudoInv is None:
            MpseudoInv = np.linalg.solve(
                np.dot(M.T, M) + self.lambdaN * self.Nshore
                + self.lambdaL * self.Lshore,
                M.T)
            self.cache_set('shore_matrix_reg_pinv', self.gtab, MpseudoInv)

        # Compute the signal coefficients in SHORE basis
        l2_fallback = False
        if self.regularization == "L1":
            regularization = 1
            if self.regularization_weighting == "CV":
                lasso = LassoCV(
                    fit_intercept=False,
                    cv=self.l1_cv,
                    positive=self.l1_positive_constraint,
                    max_iter=self.l1_maxiter,
                    verbose=self.l1_verbose)
            else:
                lasso = Lasso(
                    fit_intercept=False,
                    alpha=self.l1_alpha,
                    positive=self.l1_positive_constraint,
                    max_iter=self.l1_maxiter)

            # Try the L1 fit
            with warnings.catch_warnings():
                warnings.filterwarnings('error', category=ConvergenceWarning)

                try:
                    lasso_fit = lasso.fit(M, data)
                    coef = lasso_fit.coef_
                    alpha = lasso_fit.alpha_
                    fitted = lasso_fit.predict(M)

                except Warning as this_warning:
                    if isinstance(this_warning, ConvergenceWarning):
                        l2_fallback = True
                    else:
                        raise this_warning

        if self.regularization == "L2" or l2_fallback:
            regularization = 2
            coef = np.dot(MpseudoInv, data)
            fitted = np.dot(M, coef)
            alpha = 0

        r2 = r2_score(data, fitted)
        cnr = np.nan_to_num(np.var(fitted) / np.var(fitted - data))
        return BrainSuiteShoreFit(self, coef, regularization, alpha, r2, cnr)


class BrainSuiteShoreFit():
    def __init__(self, model, shore_coef, regularization=0,
                 alpha=0., r2=0., cnr=0.):
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
        self._alpha = alpha
        self._r2 = r2
        self._cnr = cnr
        self._regularization = regularization
        self.gtab = model.gtab
        self.radial_order = model.radial_order
        self.zeta = model.zeta

    def pdf_grid(self, gridsize, radius_max):
        r""" Applies the analytical FFT on $S$ to generate the diffusion
        propagator. This is calculated on a discrete 3D grid in order to
        obtain an EAP similar to that which is obtained with DSI.

        Parameters
        ----------
        gridsize : unsigned int
            dimension of the propagator grid
        radius_max : float
            maximal radius in which to compute the propagator

        Returns
        -------
        eap : ndarray
            the ensemble average propagator in the 3D grid

        """
        # Create the grid in which to compute the pdf
        rgrid_rtab = self.model.cache_get(
            'pdf_grid', key=(gridsize, radius_max))
        if rgrid_rtab is None:
            rgrid_rtab = create_rspace(gridsize, radius_max)
            self.model.cache_set('pdf_grid', (gridsize, radius_max), rgrid_rtab)
        rgrid, rtab = rgrid_rtab

        psi = self.model.cache_get(
            'shore_matrix_pdf', key=(gridsize, radius_max))
        if psi is None:
            psi = brainsuite_shore_matrix_pdf(
                self.radial_order, self.zeta, rtab)
            self.model.cache_set(
                'shore_matrix_pdf', (gridsize, radius_max), psi)

        propagator = np.dot(psi, self._shore_coef)
        eap = np.empty((gridsize, gridsize, gridsize), dtype=float)
        eap[tuple(rgrid.astype(int).T)] = propagator
        eap *= (2 * radius_max / (gridsize - 1))**3

        return eap

    def pdf(self, r_points):
        """ Diffusion propagator on a given set of real points.
            if the array r_points is non writeable, then intermediate
            results are cached for faster recalculation
        """
        if not r_points.flags.writeable:
            psi = self.model.cache_get(
                'shore_matrix_pdf', key=hash(r_points.data))
        else:
            psi = None
        if psi is None:
            psi = brainsuite_shore_matrix_pdf(
                self.radial_order, self.zeta, r_points)
            if not r_points.flags.writeable:
                self.model.cache_set(
                    'shore_matrix_pdf', hash(r_points.data), psi)

        eap = np.dot(psi, self._shore_coef)

        return np.clip(eap, 0, eap.max())

    def odf_sh(self):
        """ Calculates the real analytical ODF in
        terms of Spherical Harmonics."""
        # Number of Spherical Harmonics involved in the estimation
        J = (self.radial_order + 1) * (self.radial_order + 2) // 2

        # Compute the Spherical Harmonics Coefficients
        c_sh = np.zeros(J)
        counter = 0
        for n in range(self.radial_order + 1):
            for l in range(0, n + 1, 2):
                for m in range(-l, l + 1):

                    j = int(l + m + (2 * np.array(range(0, l, 2)) + 1).sum())

                    Cnl = (
                        ((-1)**(n - l / 2))
                        / (2.0 * (4.0 * np.pi**2 * self.zeta) ** (3.0 / 2.0))
                        * ((2.0 * (4.0 * np.pi**2 * self.zeta) ** (3.0 / 2.0)
                            * factorial(n - l)) / (gamma(n + 3.0 / 2.0)))
                        ** (1.0 / 2.0))
                    Gnl = (
                        gamma(l / 2 + 3.0 / 2.0) * gamma(3.0 / 2.0 + n)) \
                        / (gamma(l + 3.0 / 2.0) * factorial(n - l)) \
                        * (1.0 / 2.0) ** (-l / 2 - 3.0 / 2.0)
                    Fnl = hyp2f1(
                        -n + l, l / 2 + 3.0 / 2.0, l + 3.0 / 2.0, 2.0)

                    c_sh[j] += self._shore_coef[counter] * Cnl * Gnl * Fnl
                    counter += 1

        return c_sh

    def odf(self, sphere):
        r""" Calculates the ODF for a given discrete sphere.
        """
        upsilon = self.model.cache_get('shore_matrix_odf', key=sphere)
        if upsilon is None:
            upsilon = shore_matrix_odf(
                self.radial_order, self.zeta, sphere.vertices)
            self.model.cache_set('shore_matrix_odf', sphere, upsilon)

        odf = np.dot(upsilon, self._shore_coef)
        return odf

    def rtop_signal(self):
        r""" Calculates the analytical return to origin probability (RTOP)
        from the signal [1]_.

        References
        ----------
        .. [1] Ozarslan E. et al., "Mean apparent propagator (MAP) MRI: A novel
        diffusion imaging method for mapping tissue microstructure",
        NeuroImage, 2013.
        """
        rtop = 0
        c = self._shore_coef

        for n in range(int(self.radial_order / 2) + 1):
            rtop += c[n] * (-1) ** n * \
                ((16 * np.pi * self.zeta ** 1.5 * gamma(n + 1.5)) / (
                 factorial(n))) ** 0.5

        return np.clip(rtop, 0, rtop.max())

    def rtop_pdf(self):
        r""" Calculates the analytical return to origin probability (RTOP)
        from the pdf [1]_.

        References
        ----------
        .. [1] Ozarslan E. et al., "Mean apparent propagator (MAP) MRI: A novel
        diffusion imaging method for mapping tissue microstructure",
        NeuroImage, 2013.
        """
        rtop = 0
        c = self._shore_coef
        for n in range(int(self.radial_order / 2) + 1):
            rtop += c[n] * (-1) ** n \
                * ((4 * np.pi ** 2 * self.zeta ** 1.5 * factorial(n))
                   / (gamma(n + 1.5))) ** 0.5 \
                * genlaguerre(n, 0.5)(0)

        return np.clip(rtop, 0, rtop.max())

    def msd(self):
        r""" Calculates the analytical mean squared displacement (MSD) [1]_

        ..math::
            :nowrap:
                \begin{equation}
                    MSD:{DSI}=\int_{-\infty}^{\infty}\int_{-\infty}^{\infty}
                    \int_{-\infty}^{\infty} P(\hat{\mathbf{r}}) \cdot
                    \hat{\mathbf{r}}^{2} \ dr_x \ dr_y \ dr_z
                \end{equation}

        where $\hat{\mathbf{r}}$ is a point in the 3D propagator space (see Wu
        et al. [1]_).

        References
        ----------
        .. [1] Wu Y. et al., "Hybrid diffusion imaging", NeuroImage, vol 36,
        p. 617-629, 2007.
        """
        msd = 0
        c = self._shore_coef

        for n in range(int(self.radial_order / 2) + 1):
            msd += c[n] * (-1) ** n \
                * (9 * (gamma(n + 1.5))
                   / (8 * np.pi ** 6 * self.zeta ** 3.5
                   * factorial(n))) ** 0.5 \
                * hyp2f1(-n, 2.5, 1.5, 2)

        return np.clip(msd, 0, msd.max())

    def fitted_signal(self):
        """ The fitted signal."""
        phi = self.model.cache_get('shore_matrix', key=self.model.gtab)
        return np.dot(phi, self._shore_coef)

    def predict(self, gtab, S0=100.):
        """Recovers the reconstructed signal for any qvalue array or
        gradient table."""
        qvals = np.sqrt(gtab.bvals / self.model.tau) / (2 * np.pi)
        M = brainsuite_shore_basis(
            self.radial_order, self.zeta, gtab, self.model.tau)
        E = S0 * np.dot(M, self._shore_coef)
        return E

    @property
    def shore_coeff(self):
        """The SHORE coefficients."""
        return self._shore_coef

    @property
    def alpha(self):
        """The alpha used for the L1 fit."""
        return self._alpha

    @property
    def cnr(self):
        """Contrast to Noise ratio."""
        return self._cnr

    @property
    def regularization(self):
        """Regularization used for fitting coefficients."""
        return self._regularization

    @property
    def r2(self):
        """Model r^2."""
        return self._r2


def _kappa(zeta, n, l):
    return np.sqrt((2 * factorial(n - l))
                   / (zeta**1.5 * gamma(n + 1.5)))


def brainsuite_shore_basis(radial_order, zeta, gtab, tau=1 / (4 * np.pi**2)):
    """Calculate the brainsuite shore basis functions."""

    # If deltas are defined, use them
    try:
        qvals = gtab.qvals
    except TypeError:
        qvals = np.sqrt(gtab.bvals / (4 * np.pi**2 * tau))

    qvals[gtab.b0s_mask] = 0
    bvecs = gtab.bvecs
    qgradients = qvals[:, None] * bvecs

    r, theta, phi = cart2sphere(
        qgradients[:, 0], qgradients[:, 1], qgradients[:, 2])
    theta[np.isnan(theta)] = 0

    # Angular part of the basis - Spherical harmonics
    S, Z, L = real_sym_sh_brainsuite(radial_order, theta, phi)
    Snew = []
    R = []
    for n in range(radial_order + 1):
        for l in range(0, n + 1, 2):
            Snew.append(S[:, L == l])
            for m in range(-l, l + 1):
                # Radial part
                R.append(
                    genlaguerre(n - l, l + 0.5)(r**2 / zeta)
                    * np.exp(-r**2 / (2.0 * zeta))
                    * _kappa(zeta, n, l)
                    * (r**2 / zeta) ** (l / 2))
    R = np.column_stack(R)
    Snew = np.column_stack(Snew)
    Sh = R * Snew
    return Sh


def brainsuite_shore_matrix_pdf(radial_order, zeta, rtab):
    r"""Compute the SHORE propagator matrix [1]_"

    Parameters
    ----------
    radial_order : unsigned int,
        an even integer that represent the order of the basis
    zeta : unsigned int,
        scale factor
    rtab : array, shape (N,3)
        real space points in which calculates the pdf

    References
    ----------
    .. [1] Merlet S. et al., "Continuous diffusion signal, EAP and
    ODF estimation via Compressive Sensing in diffusion MRI", Medical
    Image Analysis, 2013.
    """

    r, theta, phi = cart2sphere(rtab[:, 0], rtab[:, 1], rtab[:, 2])
    theta[np.isnan(theta)] = 0
    psi = []

    # Angular part of the basis - Spherical harmonics
    S, Z, L = real_sym_sh_brainsuite(radial_order, theta, phi)
    Snew = []

    for n in range(radial_order + 1):
        for l in range(0, n + 1, 2):
            Snew.append(S[:, L == l])
            for m in range(-l, l + 1):
                psi.append(
                    genlaguerre(n - l, l + 0.5)(4 * np.pi ** 2
                                                * zeta * r ** 2)
                    * np.exp(-2 * np.pi ** 2 * zeta * r ** 2)
                    * _kappa_pdf(zeta, n, l)
                    * (4 * np.pi ** 2 * zeta * r ** 2) ** (l / 2)
                    * (-1) ** (n - l / 2))

    return np.column_stack(psi) * np.column_stack(Snew)


def _kappa_pdf(zeta, n, l):
    return np.sqrt(
        (16 * np.pi**3 * zeta**1.5 * factorial(n - l))
        / gamma(n + 1.5))


def shore_matrix_odf(radial_order, zeta, sphere_vertices):
    r"""Compute the SHORE ODF matrix [1]_"

    Parameters
    ----------
    radial_order : unsigned int,
        an even integer that represent the order of the basis
    zeta : unsigned int,
        scale factor
    sphere_vertices : array, shape (N,3)
        vertices of the odf sphere

    References
    ----------
    .. [1] Merlet S. et al., "Continuous diffusion signal, EAP and
    ODF estimation via Compressive Sensing in diffusion MRI", Medical
    Image Analysis, 2013.
    """

    r, theta, phi = cart2sphere(sphere_vertices[:, 0],
                                sphere_vertices[:, 1],
                                sphere_vertices[:, 2])
    theta[np.isnan(theta)] = 0
    upsilon = []

    # Angular part of the basis - Spherical harmonics
    S, Z, L = real_sym_sh_brainsuite(radial_order, theta, phi)
    Snew = []

    for n in range(radial_order + 1):
        for l in range(0, n + 1, 2):
            Snew.append(S[:, L == l])
            for m in range(-l, l + 1):
                upsilon.append( (-1) ** (n - l / 2.0)
                    * _kappa_odf(zeta, n, l) \
                    * hyp2f1(l - n, l / 2.0 + 1.5, l + 1.5, 2.0))

    return np.column_stack(upsilon) * np.column_stack(Snew)


def _kappa_odf(zeta, n, l):
    return np.sqrt(
        (gamma(l / 2.0 + 1.5)**2 * gamma(n + 1.5)
         * 2 ** (l + 3))
        / (16 * np.pi**3 * (zeta) ** 1.5
           * factorial(n - l) * gamma(l + 1.5) ** 2))


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


def shore_index_matrix(radial_order):
    indices = []
    for n in range(radial_order + 1):
        for l in range(0, n + 1, 2):
            for m in range(-l, l + 1):
                indices.append((n, l, m))
    return np.array(indices).astype(np.int)
