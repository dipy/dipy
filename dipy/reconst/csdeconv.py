from __future__ import division, print_function, absolute_import
import warnings

import numpy as np
from scipy.integrate import quad
from scipy.special import lpn, gamma
import scipy.linalg as la
import scipy.linalg.lapack as ll

from dipy.data import small_sphere, get_sphere, default_sphere

from dipy.core.geometry import cart2sphere
from dipy.core.ndindex import ndindex
from dipy.sims.voxel import single_tensor
from dipy.utils.six.moves import range

from dipy.reconst.multi_voxel import multi_voxel_fit
from dipy.reconst.dti import TensorModel, fractional_anisotropy
from dipy.reconst.shm import (sph_harm_ind_list, real_sph_harm,
                              sph_harm_lookup, lazy_index, SphHarmFit,
                              real_sym_sh_basis, sh_to_rh, forward_sdeconv_mat,
                              SphHarmModel)

from dipy.direction.peaks import peaks_from_model
from dipy.core.geometry import vec2vec_rotmat


class AxSymShResponse(object):
    """A simple wrapper for response functions represented using only axially
    symmetric, even spherical harmonic functions (ie, m == 0 and n even).

    Parameters:
    -----------
    S0 : float
        Signal with no diffusion weighting.
    dwi_response : array
        Response function signal as coefficients to axially symmetric, even
        spherical harmonic.

    """

    def __init__(self, S0, dwi_response, bvalue=None):
        self.S0 = S0
        self.dwi_response = dwi_response
        self.bvalue = bvalue
        self.m = np.zeros(len(dwi_response))
        self.sh_order = 2 * (len(dwi_response) - 1)
        self.n = np.arange(0, self.sh_order + 1, 2)

    def basis(self, sphere):
        """A basis that maps the response coefficients onto a sphere."""
        theta = sphere.theta[:, None]
        phi = sphere.phi[:, None]
        return real_sph_harm(self.m, self.n, theta, phi)

    def on_sphere(self, sphere):
        """Evaluates the response function on sphere."""
        B = self.basis(sphere)
        return np.dot(self.dwi_response, B.T)


class ConstrainedSphericalDeconvModel(SphHarmModel):

    def __init__(self, gtab, response, reg_sphere=None, sh_order=8, lambda_=1,
                 tau=0.1):
        r""" Constrained Spherical Deconvolution (CSD) [1]_.

        Spherical deconvolution computes a fiber orientation distribution
        (FOD), also called fiber ODF (fODF) [2]_, as opposed to a diffusion ODF
        as the QballModel or the CsaOdfModel. This results in a sharper angular
        profile with better angular resolution that is the best object to be
        used for later deterministic and probabilistic tractography [3]_.

        A sharp fODF is obtained because a single fiber *response* function is
        injected as *a priori* knowledge. The response function is often
        data-driven and is thus provided as input to the
        ConstrainedSphericalDeconvModel. It will be used as deconvolution
        kernel, as described in [1]_.

        Parameters
        ----------
        gtab : GradientTable
        response : tuple or AxSymShResponse object
            A tuple with two elements. The first is the eigen-values as an (3,)
            ndarray and the second is the signal value for the response
            function without diffusion weighting.  This is to be able to
            generate a single fiber synthetic signal. The response function
            will be used as deconvolution kernel ([1]_)
        reg_sphere : Sphere (optional)
            sphere used to build the regularization B matrix.
            Default: 'symmetric362'.
        sh_order : int (optional)
            maximal spherical harmonics order. Default: 8
        lambda_ : float (optional)
            weight given to the constrained-positivity regularization part of
            the deconvolution equation (see [1]_). Default: 1
        tau : float (optional)
            threshold controlling the amplitude below which the corresponding
            fODF is assumed to be zero.  Ideally, tau should be set to
            zero. However, to improve the stability of the algorithm, tau is
            set to tau*100 % of the mean fODF amplitude (here, 10% by default)
            (see [1]_). Default: 0.1

        References
        ----------
        .. [1] Tournier, J.D., et al. NeuroImage 2007. Robust determination of
               the fibre orientation distribution in diffusion MRI:
               Non-negativity constrained super-resolved spherical
               deconvolution
        .. [2] Descoteaux, M., et al. IEEE TMI 2009. Deterministic and
               Probabilistic Tractography Based on Complex Fibre Orientation
               Distributions
        .. [3] C\^ot\'e, M-A., et al. Medical Image Analysis 2013. Tractometer:
               Towards validation of tractography pipelines
        .. [4] Tournier, J.D, et al. Imaging Systems and Technology
               2012. MRtrix: Diffusion Tractography in Crossing Fiber Regions
        """
        # Initialize the parent class:
        SphHarmModel.__init__(self, gtab)
        m, n = sph_harm_ind_list(sh_order)
        self.m, self.n = m, n
        self._where_b0s = lazy_index(gtab.b0s_mask)
        self._where_dwi = lazy_index(~gtab.b0s_mask)

        no_params = ((sh_order + 1) * (sh_order + 2)) / 2

        if no_params > np.sum(~gtab.b0s_mask):
            msg = "Number of parameters required for the fit are more "
            msg += "than the actual data points"
            warnings.warn(msg, UserWarning)

        x, y, z = gtab.gradients[self._where_dwi].T
        r, theta, phi = cart2sphere(x, y, z)
        # for the gradient sphere
        self.B_dwi = real_sph_harm(m, n, theta[:, None], phi[:, None])

        # for the sphere used in the regularization positivity constraint
        if reg_sphere is None:
            self.sphere = small_sphere
        else:
            self.sphere = reg_sphere

        r, theta, phi = cart2sphere(
            self.sphere.x,
            self.sphere.y,
            self.sphere.z
        )
        self.B_reg = real_sph_harm(m, n, theta[:, None], phi[:, None])

        if response is None:
            response = (np.array([0.0015, 0.0003, 0.0003]), 1)

        self.response = response
        if isinstance(response, AxSymShResponse):
            r_sh = response.dwi_response
            self.response_scaling = response.S0
            n_response = response.n
            m_response = response.m
        else:
            self.S_r = estimate_response(gtab, self.response[0],
                                         self.response[1])
            r_sh = np.linalg.lstsq(self.B_dwi, self.S_r[self._where_dwi],
                                   rcond=-1)[0]
            n_response = n
            m_response = m
            self.response_scaling = response[1]
        r_rh = sh_to_rh(r_sh, m_response, n_response)
        self.R = forward_sdeconv_mat(r_rh, n)

        # scale lambda_ to account for differences in the number of
        # SH coefficients and number of mapped directions
        # This is exactly what is done in [4]_
        lambda_ = (lambda_ * self.R.shape[0] * r_rh[0] /
                   (np.sqrt(self.B_reg.shape[0]) * np.sqrt(362.)))
        self.B_reg *= lambda_
        self.sh_order = sh_order
        self.tau = tau
        self._X = X = self.R.diagonal() * self.B_dwi
        self._P = np.dot(X.T, X)

    @multi_voxel_fit
    def fit(self, data):
        dwi_data = data[self._where_dwi]
        shm_coeff, _ = csdeconv(dwi_data, self._X, self.B_reg, self.tau,
                                P=self._P)
        return SphHarmFit(self, shm_coeff, None)

    def predict(self, sh_coeff, gtab=None, S0=1.):
        """Compute a signal prediction given spherical harmonic coefficients
        for the provided GradientTable class instance.

        Parameters
        ----------
        sh_coeff : ndarray
            The spherical harmonic representation of the FOD from which to make
            the signal prediction.
        gtab : GradientTable
            The gradients for which the signal will be predicted. Use the
            model's gradient table by default.
        S0 : ndarray or float
            The non diffusion-weighted signal value.

        Returns
        -------
        pred_sig : ndarray
            The predicted signal.

        """
        if gtab is None or gtab is self.gtab:
            SH_basis = self.B_dwi
            gtab = self.gtab
        else:
            x, y, z = gtab.gradients[~gtab.b0s_mask].T
            r, theta, phi = cart2sphere(x, y, z)
            SH_basis, m, n = real_sym_sh_basis(self.sh_order, theta, phi)

        # Because R is diagonal, the matrix multiply is written as a multiply
        predict_matrix = SH_basis * self.R.diagonal()
        S0 = np.asarray(S0)[..., None]
        scaling = S0 / self.response_scaling
        # This is the key operation: convolve and multiply by S0:
        pre_pred_sig = scaling * np.dot(sh_coeff, predict_matrix.T)

        # Now put everything in its right place:
        pred_sig = np.zeros(pre_pred_sig.shape[:-1] + (gtab.bvals.shape[0],))
        pred_sig[..., ~gtab.b0s_mask] = pre_pred_sig
        pred_sig[..., gtab.b0s_mask] = S0

        return pred_sig


class ConstrainedSDTModel(SphHarmModel):

    def __init__(self, gtab, ratio, reg_sphere=None, sh_order=8, lambda_=1.,
                 tau=0.1):
        r""" Spherical Deconvolution Transform (SDT) [1]_.

        The SDT computes a fiber orientation distribution (FOD) as opposed to a
        diffusion ODF as the QballModel or the CsaOdfModel. This results in a
        sharper angular profile with better angular resolution. The Constrained
        SDTModel is similar to the Constrained CSDModel but mathematically it
        deconvolves the q-ball ODF as oppposed to the HARDI signal (see [1]_
        for a comparison and a through discussion).

        A sharp fODF is obtained because a single fiber *response* function is
        injected as *a priori* knowledge. In the SDTModel, this response is a
        single fiber q-ball ODF as opposed to a single fiber signal function
        for the CSDModel. The response function will be used as deconvolution
        kernel.

        Parameters
        ----------
        gtab : GradientTable
        ratio : float
            ratio of the smallest vs the largest eigenvalue of the single
            prolate tensor response function
        reg_sphere : Sphere
            sphere used to build the regularization B matrix
        sh_order : int
            maximal spherical harmonics order
        lambda_ : float
            weight given to the constrained-positivity regularization part of
            the deconvolution equation
        tau : float
            threshold (tau *mean(fODF)) controlling the amplitude below
            which the corresponding fODF is assumed to be zero.

        References
        ----------
        .. [1] Descoteaux, M., et al. IEEE TMI 2009. Deterministic and
               Probabilistic Tractography Based on Complex Fibre Orientation
               Distributions.

        """
        SphHarmModel.__init__(self, gtab)
        m, n = sph_harm_ind_list(sh_order)
        self.m, self.n = m, n
        self._where_b0s = lazy_index(gtab.b0s_mask)
        self._where_dwi = lazy_index(~gtab.b0s_mask)

        no_params = ((sh_order + 1) * (sh_order + 2)) / 2

        if no_params > np.sum(~gtab.b0s_mask):
            msg = "Number of parameters required for the fit are more "
            msg += "than the actual data points"
            warnings.warn(msg, UserWarning)

        x, y, z = gtab.gradients[self._where_dwi].T
        r, theta, phi = cart2sphere(x, y, z)
        # for the gradient sphere
        self.B_dwi = real_sph_harm(m, n, theta[:, None], phi[:, None])

        # for the odf sphere
        if reg_sphere is None:
            self.sphere = get_sphere('symmetric362')
        else:
            self.sphere = reg_sphere

        r, theta, phi = cart2sphere(
            self.sphere.x,
            self.sphere.y,
            self.sphere.z
        )
        self.B_reg = real_sph_harm(m, n, theta[:, None], phi[:, None])

        self.R, self.P = forward_sdt_deconv_mat(ratio, n)

        # scale lambda_ to account for differences in the number of
        # SH coefficients and number of mapped directions
        self.lambda_ = (lambda_ * self.R.shape[0] * self.R[0, 0] /
                        self.B_reg.shape[0])
        self.tau = tau
        self.sh_order = sh_order

    @multi_voxel_fit
    def fit(self, data):
        s_sh = np.linalg.lstsq(self.B_dwi, data[self._where_dwi],
                               rcond=-1)[0]
        # initial ODF estimation
        odf_sh = np.dot(self.P, s_sh)
        qball_odf = np.dot(self.B_reg, odf_sh)
        Z = np.linalg.norm(qball_odf)
        # normalize ODF
        odf_sh /= Z

        shm_coeff, num_it = odf_deconv(odf_sh, self.R, self.B_reg,
                                       self.lambda_, self.tau)
        # print 'SDT CSD converged after %d iterations' % num_it
        return SphHarmFit(self, shm_coeff, None)


def estimate_response(gtab, evals, S0):
    """ Estimate single fiber response function

    Parameters
    ----------
    gtab : GradientTable
    evals : ndarray
    S0 : float
        non diffusion weighted

    Returns
    -------
    S : estimated signal

    """
    evecs = np.array([[0, 0, 1],
                      [0, 1, 0],
                      [1, 0, 0]])

    return single_tensor(gtab, S0, evals, evecs, snr=None)


def forward_sdt_deconv_mat(ratio, n, r2_term=False):
    """ Build forward sharpening deconvolution transform (SDT) matrix

    Parameters
    ----------
    ratio : float
        ratio = $\frac{\lambda_2}{\lambda_1}$ of the single fiber response
        function
    n : ndarray (N,)
        The degree of spherical harmonic function associated with each row of
        the deconvolution matrix. Only even degrees are allowed.
    r2_term : bool
        True if ODF comes from an ODF computed from a model using the $r^2$
        term in the integral. For example, DSI, GQI, SHORE, CSA, Tensor,
        Multi-tensor ODFs. This results in using the proper analytical response
        function solution solving from the single-fiber ODF with the r^2 term.
        This derivation is not published anywhere but is very similar to [1]_.

    Returns
    -------
    R : ndarray (N, N)
        SDT deconvolution matrix
    P : ndarray (N, N)
        Funk-Radon Transform (FRT) matrix

    References
    ----------
    .. [1] Descoteaux, M. PhD Thesis. INRIA Sophia-Antipolis. 2008.

    """
    if np.any(n % 2):
        raise ValueError("n has odd degrees, expecting only even degrees")
    n_degrees = n.max() // 2 + 1
    sdt = np.zeros(n_degrees)  # SDT matrix
    frt = np.zeros(n_degrees)  # FRT (Funk-Radon transform) q-ball matrix

    for l in np.arange(0, n_degrees * 2, 2):
        if r2_term:
            sharp = quad(lambda z: lpn(l, z)[0][-1] * gamma(1.5) *
                         np.sqrt(ratio / (4 * np.pi ** 3)) /
                         np.power((1 - (1 - ratio) * z ** 2), 1.5), -1., 1.)
        else:
            sharp = quad(lambda z: lpn(l, z)[0][-1] *
                         np.sqrt(1 / (1 - (1 - ratio) * z * z)), -1., 1.)

        sdt[l // 2] = sharp[0]
        frt[l // 2] = 2 * np.pi * lpn(l, 0)[0][-1]

    idx = n // 2
    b = sdt[idx]
    bb = frt[idx]
    return np.diag(b), np.diag(bb)


potrf, potrs = ll.get_lapack_funcs(('potrf', 'potrs'))


def _solve_cholesky(Q, z):
    L, info = potrf(Q, lower=False, overwrite_a=False, clean=False)
    if info > 0:
        msg = "%d-th leading minor not positive definite" % info
        raise la.LinAlgError(msg)
    if info < 0:
        msg = 'illegal value in %d-th argument of internal potrf' % -info
        raise ValueError(msg)
    f, info = potrs(L, z, lower=False, overwrite_b=False)
    if info != 0:
        msg = 'illegal value in %d-th argument of internal potrs' % -info
        raise ValueError(msg)
    return f


def csdeconv(dwsignal, X, B_reg, tau=0.1, convergence=50, P=None):
    r""" Constrained-regularized spherical deconvolution (CSD) [1]_

    Deconvolves the axially symmetric single fiber response function `r_rh` in
    rotational harmonics coefficients from the diffusion weighted signal in
    `dwsignal`.

    Parameters
    ----------
    dwsignal : array
        Diffusion weighted signals to be deconvolved.
    X : array
        Prediction matrix which estimates diffusion weighted signals from FOD
        coefficients.
    B_reg : array (N, B)
        SH basis matrix which maps FOD coefficients to FOD values on the
        surface of the sphere. B_reg should be scaled to account for lambda.
    tau : float
        Threshold controlling the amplitude below which the corresponding fODF
        is assumed to be zero.  Ideally, tau should be set to zero. However, to
        improve the stability of the algorithm, tau is set to tau*100 % of the
        max fODF amplitude (here, 10% by default). This is similar to peak
        detection where peaks below 0.1 amplitude are usually considered noise
        peaks. Because SDT is based on a q-ball ODF deconvolution, and not
        signal deconvolution, using the max instead of mean (as in CSD), is
        more stable.
    convergence : int
        Maximum number of iterations to allow the deconvolution to converge.
    P : ndarray
        This is an optimization to avoid computing ``dot(X.T, X)`` many times.
        If the same ``X`` is used many times, ``P`` can be precomputed and
        passed to this function.

    Returns
    -------
    fodf_sh : ndarray (``(sh_order + 1)*(sh_order + 2)/2``,)
         Spherical harmonics coefficients of the constrained-regularized fiber
         ODF.
    num_it : int
         Number of iterations in the constrained-regularization used for
         convergence.

    Notes
    -----
    This section describes how the fitting of the SH coefficients is done.
    Problem is to minimise per iteration:

    $F(f_n) = ||Xf_n - S||^2 + \lambda^2 ||H_{n-1} f_n||^2$

    Where $X$ maps current FOD SH coefficients $f_n$ to DW signals $s$ and
    $H_{n-1}$ maps FOD SH coefficients $f_n$ to amplitudes along set of
    negative directions identified in previous iteration, i.e. the matrix
    formed by the rows of $B_{reg}$ for which $Hf_{n-1}<0$ where $B_{reg}$
    maps $f_n$ to FOD amplitude on a sphere.

    Solve by differentiating and setting to zero:

    $\Rightarrow \frac{\delta F}{\delta f_n} = 2X^T(Xf_n - S) + 2 \lambda^2
    H_{n-1}^TH_{n-1}f_n=0$

    Or:

    $(X^TX + \lambda^2 H_{n-1}^TH_{n-1})f_n = X^Ts$

    Define $Q = X^TX + \lambda^2 H_{n-1}^TH_{n-1}$ , which by construction is a
    square positive definite symmetric matrix of size $n_{SH} by n_{SH}$. If
    needed, positive definiteness can be enforced with a small minimum norm
    regulariser (helps a lot with poorly conditioned direction sets and/or
    superresolution):

    $Q = X^TX + (\lambda H_{n-1}^T) (\lambda H_{n-1}) + \mu I$

    Solve $Qf_n = X^Ts$ using Cholesky decomposition:

    $Q = LL^T$

    where $L$ is lower triangular. Then problem can be solved by
    back-substitution:

    $L_y = X^Ts$

    $L^Tf_n = y$

    To speeds things up further, form $P = X^TX + \mu I$, and update to form
    $Q$ by rankn update with $H_{n-1}$. The dipy implementation looks like:

        form initially $P = X^T X + \mu I$ and $\lambda B_{reg}$

        for each voxel: form $z = X^Ts$

            estimate $f_0$ by solving $Pf_0=z$. We use a simplified $l_{max}=4$
            solution here, but it might not make a big difference.

            Then iterate until no change in rows of $H$ used in $H_n$

                form $H_{n}$ given $f_{n-1}$

                form $Q = P + (\lambda H_{n-1}^T) (\lambda H_{n-1}$) (this can
                be done by rankn update, but we currently do not use rankn
                update).

                solve $Qf_n = z$ using Cholesky decomposition

    We'd like to thanks Donald Tournier for his help with describing and
    implementing this algorithm.

    References
    ----------
    .. [1] Tournier, J.D., et al. NeuroImage 2007. Robust determination of the
           fibre orientation distribution in diffusion MRI: Non-negativity
           constrained super-resolved spherical deconvolution.

    """
    mu = 1e-5
    if P is None:
        P = np.dot(X.T, X)
    z = np.dot(X.T, dwsignal)

    try:
        fodf_sh = _solve_cholesky(P, z)
    except la.LinAlgError:
        P = P + mu * np.eye(P.shape[0])
        fodf_sh = _solve_cholesky(P, z)
    # For the first iteration we use a smooth FOD that only uses SH orders up
    # to 4 (the first 15 coefficients).
    fodf = np.dot(B_reg[:, :15], fodf_sh[:15])
    # The mean of an fodf can be computed by taking $Y_{0,0} * coeff_{0,0}$
    threshold = B_reg[0, 0] * fodf_sh[0] * tau
    where_fodf_small = (fodf < threshold).nonzero()[0]

    # If the low-order fodf does not have any values less than threshold, the
    # full-order fodf is used.
    if len(where_fodf_small) == 0:
        fodf = np.dot(B_reg, fodf_sh)
        where_fodf_small = (fodf < threshold).nonzero()[0]
        # If the fodf still has no values less than threshold, return the fodf.
        if len(where_fodf_small) == 0:
            return fodf_sh, 0

    for num_it in range(1, convergence + 1):
        # This is the super-resolved trick.  Wherever there is a negative
        # amplitude value on the fODF, it concatenates a value to the S vector
        # so that the estimation can focus on trying to eliminate it. In a
        # sense, this "adds" a measurement, which can help to better estimate
        # the fodf_sh, even if you have more SH coefficients to estimate than
        # actual S measurements.
        H = B_reg.take(where_fodf_small, axis=0)

        # We use the Cholesky decomposition to solve for the SH coefficients.
        Q = P + np.dot(H.T, H)
        fodf_sh = _solve_cholesky(Q, z)

        # Sample the FOD using the regularization sphere and compute k.
        fodf = np.dot(B_reg, fodf_sh)
        where_fodf_small_last = where_fodf_small
        where_fodf_small = (fodf < threshold).nonzero()[0]

        if (len(where_fodf_small) == len(where_fodf_small_last) and
                (where_fodf_small == where_fodf_small_last).all()):
            break
    else:
        msg = 'maximum number of iterations exceeded - failed to converge'
        warnings.warn(msg)

    return fodf_sh, num_it


def odf_deconv(odf_sh, R, B_reg, lambda_=1., tau=0.1, r2_term=False):
    r""" ODF constrained-regularized spherical deconvolution using
    the Sharpening Deconvolution Transform (SDT) [1]_, [2]_.

    Parameters
    ----------
    odf_sh : ndarray (``(sh_order + 1)*(sh_order + 2)/2``,)
         ndarray of SH coefficients for the ODF spherical function to be
         deconvolved
    R : ndarray (``(sh_order + 1)(sh_order + 2)/2``, ``(sh_order + 1)(sh_order + 2)/2``)
         SDT matrix in SH basis
    B_reg : ndarray (``(sh_order + 1)(sh_order + 2)/2``, ``(sh_order + 1)(sh_order + 2)/2``)
         SH basis matrix used for deconvolution
    lambda_ : float
         lambda parameter in minimization equation (default 1.0)
    tau : float
         threshold (tau *max(fODF)) controlling the amplitude below
         which the corresponding fODF is assumed to be zero.
    r2_term : bool
         True if ODF is computed from model that uses the $r^2$ term in the
         integral.  Recall that Tuch's ODF (used in Q-ball Imaging [1]_) and
         the true normalized ODF definition differ from a $r^2$ term in the ODF
         integral. The original Sharpening Deconvolution Transform (SDT)
         technique [2]_ is expecting Tuch's ODF without the $r^2$ (see [3]_ for
         the mathematical details).  Now, this function supports ODF that have
         been computed using the $r^2$ term because the proper analytical
         response function has be derived.  For example, models such as DSI,
         GQI, SHORE, CSA, Tensor, Multi-tensor ODFs, should now be deconvolved
         with the r2_term=True.

    Returns
    -------
    fodf_sh : ndarray (``(sh_order + 1)(sh_order + 2)/2``,)
         Spherical harmonics coefficients of the constrained-regularized fiber
         ODF
    num_it : int
         Number of iterations in the constrained-regularization used for
         convergence

    References
    ----------
    .. [1] Tuch, D. MRM 2004. Q-Ball Imaging.
    .. [2] Descoteaux, M., et al. IEEE TMI 2009. Deterministic and
           Probabilistic Tractography Based on Complex Fibre Orientation
           Distributions
    .. [3] Descoteaux, M, PhD thesis, INRIA Sophia-Antipolis, 2008.
    """
    # In ConstrainedSDTModel.fit, odf_sh is divided by its norm (Z) and
    # sometimes the norm is 0 which creates NaNs.
    if np.any(np.isnan(odf_sh)):
        return np.zeros_like(odf_sh), 0

    # Generate initial fODF estimate, which is the ODF truncated at SH order 4
    fodf_sh = np.linalg.lstsq(R, odf_sh, rcond=-1)[0]
    fodf_sh[15:] = 0

    fodf = np.dot(B_reg, fodf_sh)

    # if sharpening a q-ball odf (it is NOT properly normalized), we need to
    # force normalization otherwise, for DSI, CSA, SHORE, Tensor odfs, they are
    # normalized by construction
    if ~r2_term:
        Z = np.linalg.norm(fodf)
        fodf_sh /= Z

    threshold = tau * np.max(np.dot(B_reg, fodf_sh))

    k = []
    convergence = 50
    for num_it in range(1, convergence + 1):
        A = np.dot(B_reg, fodf_sh)
        k2 = np.nonzero(A < threshold)[0]

        if (k2.shape[0] + R.shape[0]) < B_reg.shape[1]:
            warnings.warn(
                'too few negative directions identified - failed to converge')
            return fodf_sh, num_it

        if num_it > 1 and k.shape[0] == k2.shape[0]:
            if (k == k2).all():
                return fodf_sh, num_it

        k = k2
        M = np.concatenate((R, lambda_ * B_reg[k, :]))
        ODF = np.concatenate((odf_sh, np.zeros(k.shape)))
        try:
            fodf_sh = np.linalg.lstsq(M, ODF, rcond=-1)[0]
        except np.linalg.LinAlgError:
            # SVD did not converge in Linear Least Squares in current
            # voxel. Proceeding with initial SH estimate for this voxel.
            pass

    warnings.warn('maximum number of iterations exceeded - failed to converge')
    return fodf_sh, num_it


def odf_sh_to_sharp(odfs_sh, sphere, basis=None, ratio=3 / 15., sh_order=8,
                    lambda_=1., tau=0.1, r2_term=False):
    r""" Sharpen odfs using the sharpening deconvolution transform [2]_

    This function can be used to sharpen any smooth ODF spherical function. In
    theory, this should only be used to sharpen QballModel ODFs, but in
    practice, one can play with the deconvolution ratio and sharpen almost any
    ODF-like spherical function. The constrained-regularization is stable and
    will not only sharpen the ODF peaks but also regularize the noisy peaks.

    Parameters
    ----------
    odfs_sh : ndarray (``(sh_order + 1)*(sh_order + 2)/2``, )
        array of odfs expressed as spherical harmonics coefficients
    sphere : Sphere
        sphere used to build the regularization matrix
    basis : {None, 'mrtrix', 'fibernav'}
        different spherical harmonic basis. None is the fibernav basis as well.
    ratio : float,
        ratio of the smallest vs the largest eigenvalue of the single prolate
        tensor response function (:math:`\frac{\lambda_2}{\lambda_1}`)
    sh_order : int
        maximal SH order of the SH representation
    lambda_ : float
        lambda parameter (see odfdeconv) (default 1.0)
    tau : float
        tau parameter in the L matrix construction (see odfdeconv)
        (default 0.1)
    r2_term : bool
         True if ODF is computed from model that uses the $r^2$ term in the
         integral.  Recall that Tuch's ODF (used in Q-ball Imaging [1]_) and
         the true normalized ODF definition differ from a $r^2$ term in the ODF
         integral. The original Sharpening Deconvolution Transform (SDT)
         technique [2]_ is expecting Tuch's ODF without the $r^2$ (see [3]_ for
         the mathematical details).  Now, this function supports ODF that have
         been computed using the $r^2$ term because the proper analytical
         response function has be derived.  For example, models such as DSI,
         GQI, SHORE, CSA, Tensor, Multi-tensor ODFs, should now be deconvolved
         with the r2_term=True.

    Returns
    -------
    fodf_sh : ndarray
        sharpened odf expressed as spherical harmonics coefficients

    References
    ----------
    .. [1] Tuch, D. MRM 2004. Q-Ball Imaging.
    .. [2] Descoteaux, M., et al. IEEE TMI 2009. Deterministic and
           Probabilistic Tractography Based on Complex Fibre Orientation
           Distributions
    .. [3] Descoteaux, M, et al. MRM 2007. Fast, Regularized and Analytical
           Q-Ball Imaging
    """
    r, theta, phi = cart2sphere(sphere.x, sphere.y, sphere.z)
    real_sym_sh = sph_harm_lookup[basis]

    B_reg, m, n = real_sym_sh(sh_order, theta, phi)
    R, P = forward_sdt_deconv_mat(ratio, n, r2_term=r2_term)

    # scale lambda to account for differences in the number of
    # SH coefficients and number of mapped directions
    lambda_ = lambda_ * R.shape[0] * R[0, 0] / B_reg.shape[0]

    fodf_sh = np.zeros(odfs_sh.shape)

    for index in ndindex(odfs_sh.shape[:-1]):
        fodf_sh[index], num_it = odf_deconv(odfs_sh[index], R, B_reg,
                                            lambda_=lambda_, tau=tau,
                                            r2_term=r2_term)

    return fodf_sh


def fa_superior(FA, fa_thr):
    """ Check that the FA is greater than the FA threshold

        Parameters
        ----------
        FA : array
            Fractional Anisotropy
        fa_thr : int
            FA threshold

        Returns
        -------
        True when the FA value is greater than the FA threshold, otherwise False.
    """
    return FA > fa_thr


def fa_inferior(FA, fa_thr):
    """ Check that the FA is lower than the FA threshold

        Parameters
        ----------
        FA : array
            Fractional Anisotropy
        fa_thr : int
            FA threshold

        Returns
        -------
        True when the FA value is lower than the FA threshold, otherwise False.
    """
    return FA < fa_thr


def auto_response(gtab, data, roi_center=None, roi_radius=10, fa_thr=0.7,
                  fa_callable=fa_superior, return_number_of_voxels=False):
    """ Automatic estimation of response function using FA.

    Parameters
    ----------
    gtab : GradientTable
    data : ndarray
        diffusion data
    roi_center : tuple, (3,)
        Center of ROI in data. If center is None, it is assumed that it is
        the center of the volume with shape `data.shape[:3]`.
    roi_radius : int
        radius of cubic ROI
    fa_thr : float
        FA threshold
    fa_callable : callable
        A callable that defines an operation that compares FA with the fa_thr. The operator
        should have two positional arguments (e.g., `fa_operator(FA, fa_thr)`) and it should
        return a bool array.
    return_number_of_voxels : bool
        If True, returns the number of voxels used for estimating the response
        function.

    Returns
    -------
    response : tuple, (2,)
        (`evals`, `S0`)
    ratio : float
        The ratio between smallest versus largest eigenvalue of the response.
    number of voxels : int (optional)
        The number of voxels used for estimating the response function.

    Notes
    -----
    In CSD there is an important pre-processing step: the estimation of the
    fiber response function. In order to do this we look for voxels with very
    anisotropic configurations. For example we can use an ROI (20x20x20) at
    the center of the volume and store the signal values for the voxels with
    FA values higher than 0.7. Of course, if we haven't precalculated FA we
    need to fit a Tensor model to the datasets. Which is what we do in this
    function.

    For the response we also need to find the average S0 in the ROI. This is
    possible using `gtab.b0s_mask()` we can find all the S0 volumes (which
    correspond to b-values equal 0) in the dataset.

    The `response` consists always of a prolate tensor created by averaging
    the highest and second highest eigenvalues in the ROI with FA higher than
    threshold. We also include the average S0s.

    We also return the `ratio` which is used for the SDT models. If requested,
    the number of voxels used for estimating the response function is also
    returned, which can be used to judge the fidelity of the response function.
    As a rule of thumb, at least 300 voxels should be used to estimate a good
    response function (see [1]_).

    References
    ----------
    .. [1] Tournier, J.D., et al. NeuroImage 2004. Direct estimation of the
    fiber orientation density function from diffusion-weighted MRI
    data using spherical deconvolution
    """

    ten = TensorModel(gtab)
    if roi_center is None:
        ci, cj, ck = np.array(data.shape[:3]) // 2
    else:
        ci, cj, ck = roi_center
    w = roi_radius
    roi = data[int(ci - w): int(ci + w),
               int(cj - w): int(cj + w),
               int(ck - w): int(ck + w)]
    tenfit = ten.fit(roi)
    FA = fractional_anisotropy(tenfit.evals)
    FA[np.isnan(FA)] = 0
    indices = np.where(fa_callable(FA, fa_thr))

    if indices[0].size == 0:
        msg = "No voxel with a FA higher than " + str(fa_thr) + " were found."
        msg += " Try a larger roi or a lower threshold."
        warnings.warn(msg, UserWarning)

    lambdas = tenfit.evals[indices][:, :2]
    S0s = roi[indices][:, np.nonzero(gtab.b0s_mask)[0]]

    response, ratio = _get_response(S0s, lambdas)

    if return_number_of_voxels:
        return response, ratio, indices[0].size

    return response, ratio


def response_from_mask(gtab, data, mask):
    """ Estimate the response function from a given mask.

    Parameters
    ----------
    gtab : GradientTable
    data : ndarray
        Diffusion data
    mask : ndarray
        Mask to use for the estimation of the response function. For example a
        mask of the white matter voxels with FA values higher than 0.7
        (see [1]_).

    Returns
    -------
    response : tuple, (2,)
        (`evals`, `S0`)
    ratio : float
        The ratio between smallest versus largest eigenvalue of the response.

    Notes
    -----
    See csdeconv.auto_response() or csdeconv.recursive_response() if you don't
    have a computed mask for the response function estimation.

    References
    ----------
    .. [1] Tournier, J.D., et al. NeuroImage 2004. Direct estimation of the
    fiber orientation density function from diffusion-weighted MRI
    data using spherical deconvolution
    """

    ten = TensorModel(gtab)
    indices = np.where(mask > 0)

    if indices[0].size == 0:
        msg = "No voxel in mask with value > 0 were found."
        warnings.warn(msg, UserWarning)
        return (np.nan, np.nan), np.nan

    tenfit = ten.fit(data[indices])
    lambdas = tenfit.evals[:, :2]
    S0s = data[indices][:, np.nonzero(gtab.b0s_mask)[0]]

    return _get_response(S0s, lambdas)


def _get_response(S0s, lambdas):
    S0 = np.mean(S0s)
    l01 = np.mean(lambdas, axis=0)
    evals = np.array([l01[0], l01[1], l01[1]])
    response = (evals, S0)
    ratio = evals[1] / evals[0]
    return response, ratio


def recursive_response(gtab, data, mask=None, sh_order=8, peak_thr=0.01,
                       init_fa=0.08, init_trace=0.0021, iter=8,
                       convergence=0.001, parallel=True, nbr_processes=None,
                       sphere=default_sphere):
    """ Recursive calibration of response function using peak threshold

    Parameters
    ----------
    gtab : GradientTable
    data : ndarray
        diffusion data
    mask : ndarray, optional
        mask for recursive calibration, for example a white matter mask. It has
        shape `data.shape[0:3]` and dtype=bool. Default: use the entire data
        array.
    sh_order : int, optional
        maximal spherical harmonics order. Default: 8
    peak_thr : float, optional
        peak threshold, how large the second peak can be relative to the first
        peak in order to call it a single fiber population [1]. Default: 0.01
    init_fa : float, optional
        FA of the initial 'fat' response function (tensor). Default: 0.08
    init_trace : float, optional
        trace of the initial 'fat' response function (tensor). Default: 0.0021
    iter : int, optional
        maximum number of iterations for calibration. Default: 8.
    convergence : float, optional
        convergence criterion, maximum relative change of SH
        coefficients. Default: 0.001.
    parallel : bool, optional
        Whether to use parallelization in peak-finding during the calibration
        procedure. Default: True
    nbr_processes: int
        If `parallel` is True, the number of subprocesses to use
        (default multiprocessing.cpu_count()).
    sphere : Sphere, optional.
        The sphere used for peak finding. Default: default_sphere.

    Returns
    -------
    response : ndarray
        response function in SH coefficients

    Notes
    -----
    In CSD there is an important pre-processing step: the estimation of the
    fiber response function. Using an FA threshold is not a very robust method.
    It is dependent on the dataset (non-informed used subjectivity), and still
    depends on the diffusion tensor (FA and first eigenvector),
    which has low accuracy at high b-value. This function recursively
    calibrates the response function, for more information see [1].

    References
    ----------
    .. [1] Tax, C.M.W., et al. NeuroImage 2014. Recursive calibration of
           the fiber response function for spherical deconvolution of
           diffusion MRI data.
    """
    S0 = 1.
    evals = fa_trace_to_lambdas(init_fa, init_trace)
    res_obj = (evals, S0)

    if mask is None:
        data = data.reshape(-1, data.shape[-1])
    else:
        data = data[mask]

    n = np.arange(0, sh_order + 1, 2)
    where_dwi = lazy_index(~gtab.b0s_mask)
    response_p = np.ones(len(n))

    for _ in range(iter):
        r_sh_all = np.zeros(len(n))
        csd_model = ConstrainedSphericalDeconvModel(gtab, res_obj,
                                                    sh_order=sh_order)

        csd_peaks = peaks_from_model(model=csd_model,
                                     data=data,
                                     sphere=sphere,
                                     relative_peak_threshold=peak_thr,
                                     min_separation_angle=25,
                                     parallel=parallel,
                                     nbr_processes=nbr_processes)

        dirs = csd_peaks.peak_dirs
        vals = csd_peaks.peak_values
        single_peak_mask = (vals[:, 1] / vals[:, 0]) < peak_thr
        data = data[single_peak_mask]
        dirs = dirs[single_peak_mask]

        for num_vox in range(data.shape[0]):
            rotmat = vec2vec_rotmat(dirs[num_vox, 0], np.array([0, 0, 1]))

            rot_gradients = np.dot(rotmat, gtab.gradients.T).T

            x, y, z = rot_gradients[where_dwi].T
            r, theta, phi = cart2sphere(x, y, z)
            # for the gradient sphere
            B_dwi = real_sph_harm(0, n, theta[:, None], phi[:, None])
            r_sh_all += np.linalg.lstsq(B_dwi, data[num_vox, where_dwi],
                                        rcond=-1)[0]

        response = r_sh_all / data.shape[0]
        res_obj = AxSymShResponse(data[:, gtab.b0s_mask].mean(), response)

        change = abs((response_p - response) / response_p)
        if all(change < convergence):
            break

        response_p = response

    return res_obj


def fa_trace_to_lambdas(fa=0.08, trace=0.0021):
    lambda1 = (trace / 3.) * (1 + 2 * fa / (3 - 2 * fa ** 2) ** (1 / 2.))
    lambda2 = (trace / 3.) * (1 - fa / (3 - 2 * fa ** 2) ** (1 / 2.))
    evals = np.array([lambda1, lambda2, lambda2])
    return evals
