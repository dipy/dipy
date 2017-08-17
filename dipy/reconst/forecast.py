from __future__ import division

from warnings import warn
import numpy as np
from dipy.reconst.cache import Cache
from dipy.reconst.multi_voxel import multi_voxel_fit
from dipy.reconst.shm import real_sph_harm
from dipy.core.gradients import gradient_table
from scipy.special import erf
from math import factorial
from dipy.core.geometry import cart2sphere
from dipy.data import get_sphere
from dipy.reconst.odf import OdfModel, OdfFit
from scipy.optimize import leastsq
from dipy.utils.optpkg import optional_package
cvxpy, have_cvxpy, _ = optional_package("cvxpy")


class ForecastModel(OdfModel, Cache):
    r"""Fiber ORientation Estimated using Continuous Axially Symmetric Tensors 
    (FORECAST) [1,2,3]_. FORECAST is a Spherical Deconvolution reconstruction model
    for multi-shell diffusion data which enables the calculation of a voxel
    adaptive response function using the Spherical Mean Tecnique (SMT) [2,3]_. 

    With FORECAST it is possible to calculate crossing invariant parallel
    diffusivity, perpendicular diffusivity, mean diffusivity, and fractional
    anisotropy [2]_

    References
    ----------
    .. [1] Anderson A. W., "Measurement of Fiber Orientation Distributions
           Using High Angular Resolution Diffusion Imaging", Magnetic
           Resonance in Medicine, 2005.

    .. [2] Kaden E. et. al, "Quantitative Mapping of the Per-Axon Diffusion 
           Coefficients in Brain White Matter", Magnetic Resonance in 
           Medicine, 2016.

    .. [3] Zucchelli E. et. al, "A generalized SMT-based framework for
           Diffusion MRI microstructural model estimation", MICCAI Workshop
           on Computational DIFFUSION MRI (CDMRI), 2017.
    Notes
    -----
    The implementation of FORECAST may require CVXPY (http://www.cvxpy.org/).
    """

    def __init__(self,
                 gtab,
                 sh_order=8,
                 lambda_LB=1e-3,
                 optimizer='wls',
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
        sh_order : unsigned int,
            an even integer that represent the SH order of the basis (max 12)
        lambda_LB: float,
            Laplace-Beltrami regularization weight.
        optimizer : str,
            optimizer. The possible values are Weighted Least Squares ('wls'),
            Positivity Constraints using CVXPY ('pos') and the Constraint 
            Spherical Deconvolution algorithm ('csd'). Default is 'wls'.
        sphere : array, shape (N,3),
            sphere points where to enforce positivity when 'pos' or 'csd'
            optimizer are selected.
        lambda_csd : float,
            csd regularization weight.

        References
        ----------
        .. [1] Anderson A. W., "Measurement of Fiber Orientation Distributions
               Using High Angular Resolution Diffusion Imaging", Magnetic
               Resonance in Medicine, 2005.

        .. [2] Kaden E. et. al, "Quantitative Mapping of the Per-Axon Diffusion 
               Coefficients in Brain White Matter", Magnetic Resonance in 
               Medicine, 2016.

        .. [3] Zucchelli M. et. al, "A generalized SMT-based framework for
               Diffusion MRI microstructural model estimation", MICCAI Workshop
               on Computational DIFFUSION MRI (CDMRI), 2017.

        Examples
        --------
        In this example, where the data, gradient table and sphere tessellation
        used for reconstruction are provided, we model the diffusion signal
        with respect to the FORECAST and compute the fODF, parallel and 
        perpendicular diffusivity.

        from dipy.data import get_sphere, get_3shell_gtab
        gtab = get_3shell_gtab()

        from dipy.sims.voxel import MultiTensor
        mevals = np.array(([0.0017, 0.0003, 0.0003],
                            [0.0017, 0.0003, 0.0003]))
        angl = [(0, 0), (60, 0)]
        data, sticks = MultiTensor(
            gtab, mevals, S0=100.0, angles=angl,
            fractions=[50, 50], snr=None)

        from dipy.reconst.forecast import ForecastModel
        fm= ForecastModel(gtab, sh_order=6)
        f_fit = fm.fit(data)
        d_par = f_fit.dpar
        d_perp = f_fit.dperp

        sphere = get_sphere('symmetric724')
        fodf = f_fit.odf(sphere)
        """

        # round the bvals in order to avoid numerical errors
        self.bvals = np.round(gtab.bvals/100) * 100
        self.bvecs = gtab.bvecs
        self.gtab = gtab
        if sh_order >= 0 and not(bool(sh_order % 2)) and sh_order<=12:
            self.sh_order = sh_order
        else:
            msg = "sh_order must be a non-zero even positive number "
            msg += "between 2 and 12"
            raise ValueError(msg)

        if sphere is None:
            sphere = get_sphere('repulsion100')
            self.vertices = sphere.vertices[
                0:int(sphere.vertices.shape[0]/2), :]

        else:
            self.vertices = sphere

        self.b0s_mask = gtab.b0s_mask
        self.one_0_bvals = np.r_[0, self.bvals[~self.b0s_mask]]
        self.one_0_bvecs = np.r_[np.array([0, 0, 0]).reshape(
            1, 3), self.bvecs[~self.b0s_mask, :]]

        self.rho = rho_matrix(self.sh_order, self.one_0_bvecs)

        self.b_unique = np.sort(np.unique(self.bvals[self.bvals > 0]))
        self.wls = True
        self.csd = False
        self.pos = False

        if optimizer == 'pos':
            if have_cvxpy:
                self.wls = False
                self.pos = True
            else:
                msg = 'cvxpy is needed to inforce positivity constraints.'
                raise ValueError(msg)

        if optimizer == 'csd':
            self.csd = True

        self.L = lb_forecast(self.sh_order)
        self.lambda_lb = lambda_LB
        self.lambda_csd = lambda_csd
        self.fod = rho_matrix(sh_order, self.vertices)

    @multi_voxel_fit
    def fit(self, data):

        data_b0 = data[self.b0s_mask].mean()
        data_single_b0 = np.r_[data_b0, data[~self.b0s_mask]] / data_b0
        data_single_b0 = np.clip(data_single_b0, 0, 1.0)

        # calculates the mean signal at each b_values
        means = find_signal_means(self.b_unique, data_single_b0, self.bvals)

        n_c = int((self.sh_order + 1)*(self.sh_order + 2)/2)

        lv = self.vertices.shape[0]

        # average diffusivity initialization
        x = np.array([np.pi/4, np.pi/4])

        x, status = leastsq(forecast_error_func, x,
                            args=(self.b_unique, means))

        # squared to avoid negative diffusivities
        c0 = np.cos(x[0])**2
        c1 = np.cos(x[1])**2

        if c0 >= c1:
            d_par = c0 * 3e-03
            d_perp = c1 * 3e-03
        else:
            d_par = c1 * 3e-03
            d_perp = c0 * 3e-03

        # round to avoid memory explosion
        diff_key = str(int(np.round(d_par*1e05))) + \
            str(int(np.round(d_perp*1e05)))

        M_diff = self.cache_get('forecast_matrix', key=diff_key)
        if M_diff is None:
            M_diff = forecast_matrix(
                self.sh_order,  d_par, d_perp, self.one_0_bvals)
            self.cache_set('forecast_matrix', key=diff_key, value=M_diff)

        M = M_diff * self.rho
        M0 = M[:, 0]
        c0 = np.sqrt(1.0/(4*np.pi))

        if self.wls:
            data_r = data_single_b0 - M0*c0

            Mr = M[:, 1:]
            Lr = self.L[1:, 1:]

            pseudoInv = np.dot(np.linalg.inv(
                np.dot(Mr.T, Mr) + self.lambda_lb*Lr), Mr.T)
            coef = np.dot(pseudoInv, data_r)
            coef = np.r_[c0, coef]

        if self.csd:
            values = np.dot(self.fod, coef)
            low_peak_mean = np.mean(values)
            low_peaks = values < (0.1*low_peak_mean)

            lpl = np.ones((300, 1))

            L = self.fod[low_peaks, :]
            counter = 0
            while not np.array_equal(low_peaks, lpl) and L.shape[0] > 0:
                lpl = low_peaks
                pseudoInv = np.linalg.inv(
                    np.dot(M.T, M) + self.lambda_csd*np.dot(L.T, L))
                data_corr = np.dot(M.T, data_single_b0[:, None])

                coef = np.dot(pseudoInv, data_corr)[:, 0]

                coef = coef/coef[0] * c0

                values = np.dot(self.fod, coef)

                low_peak_mean = np.mean(values)
                low_peaks = values < (0.1*low_peak_mean)

                L = self.fod[low_peaks, :]
                counter = counter + 1

                if counter > 50:
                    warn('CSD convergence not reached')
                    break

        if self.pos:
            c = cvxpy.Variable(M.shape[1])
            design_matrix = cvxpy.Constant(M)
            objective = cvxpy.Minimize(
                cvxpy.sum_squares(design_matrix * c - data_single_b0) +
                self.lambda_lb * cvxpy.quad_form(c, self.L))

            constraints = [c[0] == c0, self.fod * c >= 0]
            prob = cvxpy.Problem(objective, constraints)
            try:
                prob.solve()
                coef = np.asarray(c.value).squeeze()
            except:
                warn('Optimization did not find a solution')
                coef = np.zeros(M.shape[1])
                coef[0] = c0

        return ForecastFit(self, coef, d_par, d_perp)


class ForecastFit(OdfFit):

    def __init__(self, model, sh_coef, d_par, d_perp):
        """ Calculates diffusion properties for a single voxel

        Parameters
        ----------
        model : object,
            AnalyticalModel
        sh_coef : 1d ndarray,
            forecast sh coefficients
        d_par : float,
            parallel diffusivity
        d_perp : float,
            perpendicular diffusivity
        """

        self.model = model
        self._sh_coef = sh_coef
        self.gtab = model.gtab
        self.sh_order = model.sh_order

        self.d_par = d_par
        self.d_perp = d_perp

        self.M = None
        self.rho = None

        self.Y_ = None

    def odf(self, sphere):
        r""" Calculates the fODF for a given discrete sphere.

        Parameters
        ----------
        sphere : Sphere,
            the odf sphere
        """
        if self.rho == None:
            self.rho = rho_matrix(self.sh_order, sphere.vertices)

        odf = np.dot(self.rho, self._sh_coef)

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


def find_signal_means(b_unique, data_norm, bvals):
    r"""Calculates the mean signal for each shell

    Parameters
    ----------
    b_unique : 1d ndarray,
        unique b-values in a vector excluding zero
    data_norm : 1d ndarray,
        normalized diffusion signal
    bvals : 1d ndarray,
        the b-values.

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

        means[u] = np.mean(shell)

    return means


def forecast_error_func(x, b_unique, E):
    r""" Calculates the difference between the mean signal calculated using 
    the parameter vector x and the average signal E using FORECAST and SMT
    """
    c0 = np.cos(x[0])**2
    c1 = np.cos(x[1])**2

    if c0 >= c1:
        d_par = c0 * 3e-03
        d_perp = c1 * 3e-03
    else:
        d_par = c1 * 3e-03
        d_perp = c0 * 3e-03

    E_ = 0.5 * np.exp(-b_unique*d_perp) * Psi_l(0, (b_unique * (d_par-d_perp)))

    v = E-E_
    return v


def I_l(l, b):
    if np.isscalar(b):
        if b <= 1e-08:
            v = 2.0/(l+1)
        else:
            if l == 0:
                v = (np.sqrt(np.pi)*erf(np.sqrt(b))) / (b**(0.5))

            if l == 2:
                v = (np.sqrt(np.pi)*erf(np.sqrt(b)) - 2*np.exp(-b) *
                     np.sqrt(b)) / (2*b**(1.5))

            if l == 4:
                v = (3 * np.sqrt(np.pi)*erf(np.sqrt(b)) - 2*np.exp(-b) *
                     np.sqrt(b)*(2*b+3)) / (4*b**(2.5))

            if l == 6:
                v = (15 * np.sqrt(np.pi)*erf(np.sqrt(b)) - 2*np.exp(-b) *
                     np.sqrt(b)*(4*b**2 + 10*b+15)) / (8*b**(3.5))

            if l == 8:
                v = (105 * np.sqrt(np.pi)*erf(np.sqrt(b)) - 2*np.exp(-b) *
                     np.sqrt(b)*(8*b**3 + 28*b**2 + 70*b+105)) / (16*b**(4.5))

            if l == 10:
                v = (945 * np.sqrt(np.pi)*erf(np.sqrt(b)) - 2*np.exp(-b) *
                     np.sqrt(b)*(16*b**4 + 72*b**3 + 252*b**2 + 630*b+945)) /\
                    (32*b**(5.5))

            if l == 12:
                v = (10395 * np.sqrt(np.pi)*erf(np.sqrt(b)) - 2*np.exp(-b) *
                     np.sqrt(b)*(32*b**5 + 176*b**4 + 792*b**3 + 2772*b**2 +
                     6930*b+10395)) / (64*b**(6.5))

    else:
        v = np.zeros(len(b))
        ind = b <= 1e-08
        v[ind] = 2.0/(l+1)
        b_not = b[~ind]
        if l == 0:
            v[~ind] = (np.sqrt(np.pi)*erf(np.sqrt(b_not))) / (b_not**(0.5))

        if l == 2:
            v[~ind] = (np.sqrt(np.pi)*erf(np.sqrt(b_not)) - 2*np.exp(-b_not) *
                       np.sqrt(b_not)) / (2*b_not**(1.5))

        if l == 4:
            v[~ind] = (3 * np.sqrt(np.pi)*erf(np.sqrt(b_not)) - 2 * \
                       np.exp(-b_not)*np.sqrt(b_not)*(2*b_not+3)) / \
                      (4*b_not**(2.5))

        if l == 6:
            v[~ind] = (15 * np.sqrt(np.pi)*erf(np.sqrt(b_not)) - 2 * \
                       np.exp(-b_not) * np.sqrt(b_not)*(4*b_not**2 + \
                       10*b_not+15)) / (8*b_not**(3.5))

        if l == 8:
            v[~ind] = (105 * np.sqrt(np.pi)*erf(np.sqrt(b_not)) - \
                       2*np.exp(-b_not) * np.sqrt(b_not)*(8*b_not**3 + 28 * \
                       b_not**2 + 70*b_not+105)) / (16*b_not**(4.5))

        if l == 10:
            v[~ind] = (945 * np.sqrt(np.pi)*erf(np.sqrt(b_not)) - 2 * \
                       np.exp(-b_not)*np.sqrt(b_not) * (16*b_not**4 + 72 * \
                       b_not**3 + 252*b_not**2 + 630*b_not+945)) / \
                      (32*b_not**(5.5))

        if l == 12:
            v[~ind] = (10395 * np.sqrt(np.pi)*erf(np.sqrt(b_not)) - 2 *\
                       np.exp(-b_not)*np.sqrt(b_not)*(32*b_not**5 + \
                       176*b_not**4 + 792*b_not**3 + 2772*b_not**2 + 6930*
                       b_not+10395)) / (64*b_not**(6.5))

    return v


def Psi_l(l, b):
    if l == 0:
        v = I_l(0, b)

    if l == 2:
        v = 0.5*(3*I_l(2, b) - I_l(0, b))

    if l == 4:
        v = (1.0/8)*(35*I_l(4, b) - 30*I_l(2, b) + 3*I_l(0, b))

    if l == 6:
        v = (1.0/16)*(231*I_l(6, b) - 315 * \
             I_l(4, b) + 105*I_l(2, b) - 5*I_l(0, b))

    if l == 8:
        v = (1.0/128)*(6435*I_l(8, b) - 12012*I_l(6, b) + \
            6930 * I_l(4, b) - 1260*I_l(2, b) + 35*I_l(0, b))

    if l == 10:
        v = (46189/256.0)*I_l(10, b) - (109395/256.0)*I_l(8, b) + \
            (90090/256.0) * I_l(6, b)-(30030/256.0)*I_l(4, b) + \
            (3465/256.0)*I_l(2, b) - (63/256.0)*I_l(0, b)

    if l == 12:
        v = (676039/1024.0)*I_l(12, b) - (1939938/1024.0)*I_l(10, b) + \
            (2078505/1024.0)*I_l(8, b) - (1021020 / 1024.0)*I_l(6, b) + \
            (225225/1024.0)*I_l(4, b) - (18018/1024.0)*I_l(2, b) + \
            (231/1024.0)*I_l(0, b)

    return v


def forecast_matrix(sh_order,  d_par, d_perp, bvals):
    r"""Compute the FORECAST radial matrix 
    """
    n_c = int((sh_order + 1)*(sh_order + 2)/2)
    M = np.zeros((bvals.shape[0], n_c))
    counter = 0
    for l in range(0, sh_order + 1, 2):
        for m in range(-l, l + 1):
            M[:, counter] = 2*np.pi * \
                np.exp(-bvals*d_perp) * Psi_l(l, bvals*(d_par-d_perp))
            counter += 1
    return M


def rho_matrix(sh_order, vecs):
    r"""Compute the SH matrix $\rho$
    """

    r, theta, phi = cart2sphere(vecs[:, 0], vecs[:, 1], vecs[:, 2])
    theta[np.isnan(theta)] = 0

    n_c = int((sh_order + 1)*(sh_order + 2)/2)
    rho = np.zeros((vecs.shape[0], n_c))
    counter = 0
    for l in range(0, sh_order + 1, 2):
        for m in range(-l, l + 1):
            rho[:, counter] = real_sph_harm(m, l, theta, phi)
            counter += 1
    return rho


def lb_forecast(sh_order):
    r"""Returns the Laplace Beltrami regularization matrix for FORECAST
    """
    n_c = int((sh_order + 1)*(sh_order + 2)/2)
    diagL = np.zeros(n_c)
    counter = 0
    for l in range(0, sh_order + 1, 2):
        for m in range(-l, l + 1):
            diagL[counter] = (l * (l + 1)) ** 2
            counter += 1

    return np.diag(diagL)
