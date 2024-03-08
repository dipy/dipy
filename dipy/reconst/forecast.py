from warnings import warn

import numpy as np

from dipy.reconst.cache import Cache
from dipy.reconst.multi_voxel import multi_voxel_fit
from dipy.reconst.csdeconv import csdeconv
from dipy.reconst.shm import real_sh_descoteaux_from_index
from scipy.special import gamma, hyp1f1
from dipy.core.geometry import cart2sphere
from dipy.data import default_sphere
from dipy.reconst.odf import OdfModel, OdfFit
from scipy.optimize import leastsq
from dipy.utils.optpkg import optional_package
from dipy.utils.deprecator import deprecated_params

cvxpy, have_cvxpy, _ = optional_package("cvxpy", min_version="1.4.1")


class ForecastModel(OdfModel, Cache):
    r"""Fiber ORientation Estimated using Continuous Axially Symmetric Tensors
    (FORECAST) [1,2,3]_. FORECAST is a Spherical Deconvolution reconstruction
    model for multi-shell diffusion data which enables the calculation of a
    voxel adaptive response function using the Spherical Mean Technique (SMT)
    [2,3]_.

    With FORECAST it is possible to calculate crossing invariant parallel
    diffusivity, perpendicular diffusivity, mean diffusivity, and fractional
    anisotropy [2]_

    References
    ----------
    .. [1] Anderson A. W., "Measurement of Fiber Orientation Distributions
           Using High Angular Resolution Diffusion Imaging", Magnetic
           Resonance in Medicine, 2005.

    .. [2] Kaden E. et al., "Quantitative Mapping of the Per-Axon Diffusion
           Coefficients in Brain White Matter", Magnetic Resonance in
           Medicine, 2016.

    .. [3] Zucchelli E. et al., "A generalized SMT-based framework for
           Diffusion MRI microstructural model estimation", MICCAI Workshop
           on Computational DIFFUSION MRI (CDMRI), 2017.

    Notes
    -----
    The implementation of FORECAST may require CVXPY (https://www.cvxpy.org/).
    """
    @deprecated_params('sh_order', 'sh_order_max', since='1.9', until='2.0')
    def __init__(self,
                 gtab,
                 sh_order_max=8,
                 lambda_lb=1e-3,
                 dec_alg='CSD',
                 sphere=None,
                 lambda_csd=1.0):
        r""" Analytical and continuous modeling of the diffusion signal with
        respect to the FORECAST basis [1,2,3]_.
        This implementation is a modification of the original FORECAST
        model presented in [1]_ adapted for multi-shell data as in [2,3]_ .

        The main idea is to model the diffusion signal as the combination of a
        single fiber response function $F(\mathbf{b})$ times the fODF
        $\rho(\mathbf{v})$

        ..math::
            :nowrap:
                \begin{equation}
                    E(\mathbf{b}) = \int_{\mathbf{v} \in \mathcal{S}^2} \rho(\mathbf{v}) F({\mathbf{b}} | \mathbf{v}) d \mathbf{v}
                \end{equation}

        where $\mathbf{b}$ is the b-vector (b-value times gradient direction)
        and $\mathbf{v}$ is an unit vector representing a fiber direction.

        In FORECAST $\rho$ is modeled using real symmetric Spherical Harmonics
        (SH) and $F(\mathbf(b))$ is an axially symmetric tensor.


        Parameters
        ----------
        gtab : GradientTable,
            gradient directions and bvalues container class.
        sh_order_max : unsigned int,
            an even integer that represent the maximal SH order (l) of the 
            basis (max 12)
        lambda_lb: float,
            Laplace-Beltrami regularization weight.
        dec_alg : str,
            Spherical deconvolution algorithm. The possible values are Weighted Least Squares ('WLS'),
            Positivity Constraints using CVXPY ('POS') and the Constraint
            Spherical Deconvolution algorithm ('CSD'). Default is 'CSD'.
        sphere : array, shape (N,3),
            sphere points where to enforce positivity when 'POS' or 'CSD'
            dec_alg are selected.
        lambda_csd : float,
            CSD regularization weight.

        References
        ----------
        .. [1] Anderson A. W., "Measurement of Fiber Orientation Distributions
               Using High Angular Resolution Diffusion Imaging", Magnetic
               Resonance in Medicine, 2005.

        .. [2] Kaden E. et al., "Quantitative Mapping of the Per-Axon Diffusion
               Coefficients in Brain White Matter", Magnetic Resonance in
               Medicine, 2016.

        .. [3] Zucchelli M. et al., "A generalized SMT-based framework for
               Diffusion MRI microstructural model estimation", MICCAI Workshop
               on Computational DIFFUSION MRI (CDMRI), 2017.

        Examples
        --------
        In this example, where the data, gradient table and sphere tessellation
        used for reconstruction are provided, we model the diffusion signal
        with respect to the FORECAST and compute the fODF, parallel and
        perpendicular diffusivity.

        >>> import warnings
        >>> from dipy.data import default_sphere, get_3shell_gtab
        >>> gtab = get_3shell_gtab()
        >>> from dipy.sims.voxel import multi_tensor
        >>> mevals = np.array(([0.0017, 0.0003, 0.0003],
        ...                    [0.0017, 0.0003, 0.0003]))
        >>> angl = [(0, 0), (60, 0)]
        >>> data, sticks = multi_tensor(gtab,
        ...                             mevals,
        ...                             S0=100.0,
        ...                             angles=angl,
        ...                             fractions=[50, 50],
        ...                             snr=None)
        >>> from dipy.reconst.forecast import ForecastModel
        >>> from dipy.reconst.shm import descoteaux07_legacy_msg
        >>> with warnings.catch_warnings():
        ...     warnings.filterwarnings(
        ...         "ignore", message=descoteaux07_legacy_msg,
        ...         category=PendingDeprecationWarning)
        ...     fm = ForecastModel(gtab, sh_order_max=6)
        >>> f_fit = fm.fit(data)
        >>> d_par = f_fit.dpar
        >>> d_perp = f_fit.dperp
        >>> with warnings.catch_warnings():
        ...     warnings.filterwarnings(
        ...         "ignore", message=descoteaux07_legacy_msg,
        ...         category=PendingDeprecationWarning)
        ...     fodf = f_fit.odf(default_sphere)
        """
        OdfModel.__init__(self, gtab)

        # round the bvals in order to avoid numerical errors
        self.bvals = np.round(gtab.bvals / 100) * 100
        self.bvecs = gtab.bvecs

        if 0 <= sh_order_max <= 12 and not bool(sh_order_max % 2):
            self.sh_order_max = sh_order_max
        else:
            msg = "sh_order_max must be a non-zero even positive number "
            msg += "between 2 and 12"
            raise ValueError(msg)

        if sphere is None:
            sphere = default_sphere
            self.vertices = sphere.vertices[
                0:int(sphere.vertices.shape[0]/2), :]

        else:
            self.vertices = sphere

        self.b0s_mask = self.bvals == 0
        self.one_0_bvals = np.r_[0, self.bvals[~self.b0s_mask]]
        self.one_0_bvecs = np.r_[np.array([0, 0, 0]).reshape(
            1, 3), self.bvecs[~self.b0s_mask, :]]

        self.rho = rho_matrix(self.sh_order_max, self.one_0_bvecs)

        # signal regularization matrix
        self.srm = rho_matrix(4, self.one_0_bvecs)
        self.lb_matrix_signal = lb_forecast(4)

        self.b_unique = np.sort(np.unique(self.bvals[self.bvals > 0]))
        self.wls = True
        self.csd = False
        self.pos = False

        if dec_alg.upper() == 'POS':
            if not have_cvxpy:
                cvxpy.import_error()

            self.wls = False
            self.pos = True

        if dec_alg.upper() == 'CSD':
            self.csd = True

        self.lb_matrix = lb_forecast(self.sh_order_max)
        self.lambda_lb = lambda_lb
        self.lambda_csd = lambda_csd
        self.fod = rho_matrix(sh_order_max, self.vertices)

    @multi_voxel_fit
    def fit(self, data):

        data_b0 = data[self.b0s_mask].mean()
        data_single_b0 = np.r_[data_b0, data[~self.b0s_mask]] / data_b0

        # calculates the mean signal at each b_values
        means = find_signal_means(self.b_unique,
                                  data_single_b0,
                                  self.one_0_bvals,
                                  self.srm,
                                  self.lb_matrix_signal)

        # average diffusivity initialization
        x = np.array([np.pi / 4, np.pi / 4])

        x, status = leastsq(forecast_error_func, x,
                            args=(self.b_unique, means))

        # transform to bound the diffusivities from 0 to 3e-03
        d_par = np.cos(x[0])**2 * 3e-03
        d_perp = np.cos(x[1])**2 * 3e-03

        if d_perp >= d_par:
            d_par, d_perp = d_perp, d_par

        # round to avoid memory explosion
        diff_key = str(int(np.round(d_par * 1e05))) + \
            str(int(np.round(d_perp * 1e05)))

        M_diff = self.cache_get('forecast_matrix', key=diff_key)
        if M_diff is None:
            M_diff = forecast_matrix(
                self.sh_order_max, d_par, d_perp, self.one_0_bvals)
            self.cache_set('forecast_matrix', key=diff_key, value=M_diff)

        M = M_diff * self.rho
        M0 = M[:, 0]
        c0 = np.sqrt(1.0/(4*np.pi))

        # coefficients vector initialization
        n_c = int((self.sh_order_max + 1)*(self.sh_order_max + 2)/2)
        coef = np.zeros(n_c)
        coef[0] = c0
        if int(np.round(d_par*1e05)) > int(np.round(d_perp*1e05)):
            if self.wls:
                data_r = data_single_b0 - M0*c0

                Mr = M[:, 1:]
                Lr = self.lb_matrix[1:, 1:]

                pseudo_inv = np.dot(np.linalg.inv(
                    np.dot(Mr.T, Mr) + self.lambda_lb*Lr), Mr.T)
                coef = np.dot(pseudo_inv, data_r)
                coef = np.r_[c0, coef]

            if self.csd:
                coef, _ = csdeconv(data_single_b0, M, self.fod, tau=0.1,
                                   convergence=50)
                coef = coef / coef[0] * c0

            if self.pos:
                c = cvxpy.Variable(M.shape[1])
                design_matrix = cvxpy.Constant(M) @ c
                objective = cvxpy.Minimize(
                    cvxpy.sum_squares(design_matrix - data_single_b0) +
                    self.lambda_lb * cvxpy.quad_form(c, self.lb_matrix))

                constraints = [c[0] == c0, self.fod @ c >= 0]
                prob = cvxpy.Problem(objective, constraints)
                try:
                    prob.solve(solver=cvxpy.OSQP, eps_abs=1e-05, eps_rel=1e-05)
                    coef = np.asarray(c.value).squeeze()
                except Exception:
                    warn('Optimization did not find a solution')
                    coef = np.zeros(M.shape[1])
                    coef[0] = c0

        return ForecastFit(self, data, coef, d_par, d_perp)


class ForecastFit(OdfFit):

    def __init__(self, model, data, sh_coef, d_par, d_perp):
        """ Calculates diffusion properties for a single voxel

        Parameters
        ----------
        model : object,
            AnalyticalModel
        data : 1d ndarray,
            fitted data
        sh_coef : 1d ndarray,
            forecast sh coefficients
        d_par : float,
            parallel diffusivity
        d_perp : float,
            perpendicular diffusivity
        """
        OdfFit.__init__(self, model, data)
        self.model = model
        self._sh_coef = sh_coef
        self.gtab = model.gtab
        self.sh_order_max = model.sh_order_max
        self.d_par = d_par
        self.d_perp = d_perp

        self.rho = None

    def odf(self, sphere, clip_negative=True):
        r""" Calculates the fODF for a given discrete sphere.

        Parameters
        ----------
        sphere : Sphere,
            the odf sphere
        clip_negative : boolean, optional
            if True clip the negative odf values to 0, default True
        """
        if self.rho is None:
            self.rho = rho_matrix(self.sh_order_max, sphere.vertices)

        odf = np.dot(self.rho, self._sh_coef)

        if clip_negative:
            odf = np.clip(odf, 0, odf.max())

        return odf

    def fractional_anisotropy(self):
        r""" Calculates the fractional anisotropy.
        """
        fa = np.sqrt(0.5 * (2*(self.d_par - self.d_perp)**2) /
                     (self.d_par**2 + 2*self.d_perp**2))
        return fa

    def mean_diffusivity(self):
        r""" Calculates the mean diffusivity.
        """
        md = (self.d_par + 2*self.d_perp)/3.0
        return md

    def predict(self, gtab=None, S0=1.0):
        r""" Calculates the fODF for a given discrete sphere.

        Parameters
        ----------
        gtab : GradientTable, optional
            gradient directions and bvalues container class.
        S0 : float, optional
            the signal at b-value=0

        """
        if gtab is None:
            gtab = self.gtab

        M_diff = forecast_matrix(self.sh_order_max,
                                 self.d_par,
                                 self.d_perp,
                                 gtab.bvals)

        rho = rho_matrix(self.sh_order_max, gtab.bvecs)
        M = M_diff * rho
        S = S0 * np.dot(M, self._sh_coef)

        return S

    @property
    def sh_coeff(self):
        """The FORECAST SH coefficients
        """
        return self._sh_coef

    @property
    def dpar(self):
        """The parallel diffusivity
        """
        return self.d_par

    @property
    def dperp(self):
        """The perpendicular diffusivity
        """
        return self.d_perp


def find_signal_means(b_unique, data_norm, bvals, rho, lb_matrix, w=1e-03):
    r"""Calculate the mean signal for each shell.

    Parameters
    ----------
    b_unique : 1d ndarray,
        unique b-values in a vector excluding zero
    data_norm : 1d ndarray,
        normalized diffusion signal
    bvals : 1d ndarray,
        the b-values
    rho : 2d ndarray,
        SH basis matrix for fitting the signal on each shell
    lb_matrix : 2d ndarray,
        Laplace-Beltrami regularization matrix
    w : float,
        weight for the Laplace-Beltrami regularization

    Returns
    -------
    means : 1d ndarray
        the average of the signal for each b-values

    """
    lb = len(b_unique)
    means = np.zeros(lb)
    for u in range(lb):
        ind = bvals == b_unique[u]
        shell = data_norm[ind]
        if np.sum(ind) > 20:
            M = rho[ind, :]
            coef = np.linalg.multi_dot([np.linalg.inv(
                np.dot(M.T, M) + w*lb_matrix), M.T, shell])

            means[u] = coef[0] / np.sqrt(4*np.pi)
        else:
            means[u] = shell.mean()

    return means


def forecast_error_func(x, b_unique, E):
    r""" Calculates the difference between the mean signal calculated using
    the parameter vector x and the average signal E using FORECAST and SMT
    """
    d_par = np.cos(x[0])**2 * 3e-03
    d_perp = np.cos(x[1])**2 * 3e-03

    if d_perp >= d_par:
        d_par, d_perp = d_perp, d_par

    E_reconst = 0.5 * np.exp(-b_unique * d_perp) * psi_l(0, (b_unique * (d_par - d_perp)))

    v = E-E_reconst
    return v


def psi_l(l, b):
    n = l//2
    v = (-b)**n
    v *= gamma(n + 1./2) / gamma(2*n + 3./2)
    v *= hyp1f1(n + 1./2, 2*n + 3./2, -b)
    return v

@deprecated_params('sh_order', 'sh_order_max', since='1.9', until='2.0')
def forecast_matrix(sh_order_max, d_par, d_perp, bvals):
    r"""Compute the FORECAST radial matrix
    """
    n_c = int((sh_order_max + 1) * (sh_order_max + 2) / 2)
    M = np.zeros((bvals.shape[0], n_c))
    counter = 0
    for l in range(0, sh_order_max + 1, 2):
        for m in range(-l, l + 1):
            M[:, counter] = 2 * np.pi * \
                np.exp(-bvals * d_perp) * psi_l(l, bvals * (d_par - d_perp))
            counter += 1
    return M

@deprecated_params('sh_order', 'sh_order_max', since='1.9', until='2.0')
def rho_matrix(sh_order_max, vecs):
    r"""Compute the SH matrix $\rho$
    """

    r, theta, phi = cart2sphere(vecs[:, 0], vecs[:, 1], vecs[:, 2])
    theta[np.isnan(theta)] = 0

    n_c = int((sh_order_max + 1) * (sh_order_max + 2) / 2)
    rho = np.zeros((vecs.shape[0], n_c))
    counter = 0
    for l_values in range(0, sh_order_max + 1, 2):
        for m_values in range(-l_values, l_values + 1):
            rho[:, counter] = real_sh_descoteaux_from_index(
                m_values, l_values, theta, phi)
            counter += 1
    return rho

@deprecated_params('sh_order', 'sh_order_max', since='1.9', until='2.0')
def lb_forecast(sh_order_max):
    r"""Returns the Laplace-Beltrami regularization matrix for FORECAST
    """
    n_c = int((sh_order_max + 1)*(sh_order_max + 2)/2)
    diag_lb = np.zeros(n_c)
    counter = 0
    for j in range(0, sh_order_max + 1, 2):
        stop = 2 * j + 1 + counter
        diag_lb[counter:stop] = (j * (j + 1)) ** 2
        counter = stop

    return np.diag(diag_lb)
