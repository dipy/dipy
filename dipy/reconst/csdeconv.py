from __future__ import division, print_function, absolute_import
import warnings
import numpy as np
from scipy.integrate import quad
from dipy.reconst.odf import OdfModel
from dipy.reconst.cache import Cache
from dipy.reconst.multi_voxel import multi_voxel_fit
from dipy.reconst.shm import (sph_harm_ind_list, real_sph_harm,
                              sph_harm_lookup, lazy_index, SphHarmFit)
from dipy.data import get_sphere
from dipy.core.geometry import cart2sphere
from ..core.sphere import Sphere
from dipy.core.ndindex import ndindex
from dipy.sims.voxel import single_tensor, all_tensor_evecs
import dipy.core.gradients as grad

from scipy.special import lpn, gamma
from dipy.reconst.dti import TensorModel, fractional_anisotropy
from scipy.integrate import quad

class ConstrainedSphericalDeconvModel(OdfModel, Cache):

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
        data-driven and thus, comes as input to the
        ConstrainedSphericalDeconvModel. It will be used as deconvolution
        kernel, as described in [1]_.

        Parameters
        ----------
        gtab : GradientTable
        response : tuple
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
            weight given to the constrained-positivity regularization part of the
            deconvolution equation (see [1]_). Default: 1
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
        OdfModel.__init__(self, gtab)
        m, n = sph_harm_ind_list(sh_order)
        self.m, self.n = m, n
        self._where_b0s = lazy_index(gtab.b0s_mask)
        self._where_dwi = lazy_index(~gtab.b0s_mask)

        no_params = ((sh_order + 1) * (sh_order + 2)) / 2

        if no_params > np.sum(gtab.b0s_mask == False):
            msg = "Number of parameters required for the fit are more "
            msg += "than the actual data points"
            warnings.warn(msg, UserWarning)

        x, y, z = gtab.gradients[self._where_dwi].T
        r, theta, phi = cart2sphere(x, y, z)
        # for the gradient sphere
        self.B_dwi = real_sph_harm(m, n, theta[:, None], phi[:, None])

        # for the sphere used in the regularization positivity constraint
        if reg_sphere is None:
            self.sphere = get_sphere('symmetric362')
        else:
            self.sphere = reg_sphere

        r, theta, phi = cart2sphere(self.sphere.x, self.sphere.y, self.sphere.z)
        self.B_reg = real_sph_harm(m, n, theta[:, None], phi[:, None])

        if response is None:
            self.response = (np.array([0.0015, 0.0003, 0.0003]), 1)
        else:
            self.response = response
            
        self.S_r = estimate_response(gtab, self.response[0], self.response[1])

        r_sh = np.linalg.lstsq(self.B_dwi, self.S_r[self._where_dwi])[0]
        r_rh = sh_to_rh(r_sh, m, n)

        self.R = forward_sdeconv_mat(r_rh, n)

        # scale lambda_ to account for differences in the number of
        # SH coefficients and number of mapped directions
        # This is exactly what is done in [4]_
        self.lambda_ = lambda_ * self.R.shape[0] * r_rh[0] / self.B_reg.shape[0]
        self.sh_order = sh_order
        self.tau = tau

    @multi_voxel_fit
    def fit(self, data):
        s_sh = np.linalg.lstsq(self.B_dwi, data[self._where_dwi])[0]
        shm_coeff, num_it = csdeconv(s_sh, self.sh_order, self.R, self.B_reg,
                                     self.lambda_, self.tau)
        return ConstrainedSphericalDeconvFit(self, shm_coeff)


class ConstrainedSphericalDeconvFit(SphHarmFit):
    def __init__(self, model, shm_coeff):
        """
        Initialize a ConstrainedSphericalDeconvFit class instance

        Parameters
        ----------
        model : A ConstrainedSphericalDeconvModel class instance

        shm_coeff : ndarray
           Spherical harmonic coefficients estimated with `csdeconv`

        Returns
        -------
        ConstrainedSphericalDeconvFit class instance
        """
        SphHarmFit.__init__(self, model, shm_coeff, None)


    def predict(self, gtab, S0=1):
        """
        Predict the diffusion signal from the model coefficients.

        Parameters
        ----------
        gtab : a GradientTable class instance
            The directions and bvalues on which prediction is desired

        S0 : float array
           The mean non-diffusion-weighted signal in each voxel. Default: 1 in
           all voxels

        """
        # Initalize the prediction to be equal to S0 everywhere, we'll
        # allocate the prediction to all the other locations:
        prediction = np.ones(gtab.bvals.shape) * S0
        b0s_mask = gtab.b0s_mask
        # To invert the model, we estimate the ODF on the provided sphere and
        # then convolve with the response function, rotated to each one of the
        # gtab bvecs:
        sphere = Sphere(xyz=gtab.bvecs[~b0s_mask])
        est_odf = self.odf(sphere)
        est_odf = est_odf/np.sum(est_odf)
        prediction_matrix = np.zeros((est_odf.shape[0], est_odf.shape[0]))
        pred_gtab = grad.gradient_table(gtab.bvals[~b0s_mask],
                                        gtab.bvecs[~b0s_mask])
        for ii in xrange(est_odf.shape[0]):
            evals = self.model.response[0]
            evecs = all_tensor_evecs(sphere.vertices[ii])
            prediction_matrix[ii] = single_tensor(pred_gtab, S0,
                                                  evals, evecs, snr=None)

        prediction[~b0s_mask] = np.dot(est_odf, prediction_matrix.T)
        return prediction


class ConstrainedSDTModel(OdfModel, Cache):

    def __init__(self, gtab, ratio, reg_sphere=None, sh_order=8, lambda_=1.,
                 tau=0.1):
        r""" Spherical Deconvolution Transform (SDT) [1]_.

        The SDT computes a fiber orientation distribution (FOD) as opposed to a
        diffusion ODF as the QballModel or the CsaOdfModel. This results in a
        sharper angular profile with better angular resolution. The Contrained
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
            weight given to the constrained-positivity regularization part of the
            deconvolution equation
        tau : float
            threshold (tau *mean(fODF)) controlling the amplitude below
            which the corresponding fODF is assumed to be zero.

        References
        ----------
        .. [1] Descoteaux, M., et al. IEEE TMI 2009. Deterministic and
               Probabilistic Tractography Based on Complex Fibre Orientation
               Distributions.
        """

        m, n = sph_harm_ind_list(sh_order)
        self.m, self.n = m, n
        self._where_b0s = lazy_index(gtab.b0s_mask)
        self._where_dwi = lazy_index(~gtab.b0s_mask)

        no_params = ((sh_order + 1) * (sh_order + 2)) / 2

        if no_params > np.sum(gtab.b0s_mask == False):
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

        r, theta, phi = cart2sphere(self.sphere.x, self.sphere.y, self.sphere.z)
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
        s_sh = np.linalg.lstsq(self.B_dwi, data[self._where_dwi])[0]
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


def sh_to_rh(r_sh, m, n):
    """ Spherical harmonics (SH) to rotational harmonics (RH)

    Calculate the rotational harmonic decomposition up to
    harmonic sh_order for an axially and antipodally
    symmetric function. Note that all ``m != 0`` coefficients
    will be ignored as axial symmetry is assumed. Hence, there
    will be ``(sh_order/2 + 1)`` non-zero coefficients.

    Parameters
    ----------
    r_sh : ndarray (N,)
        ndarray of SH coefficients for the single fiber response function.
        These coefficients must correspond to the real spherical harmonic
        functions produced by `shm.real_sph_harm`.
    m : ndarray (N,)
        The order of the spherical harmonic function associated with each
        coefficient.
    n : ndarray (N,)
        The degree of the spherical harmonic function associated with each
        coefficient.

    Returns
    -------
    r_rh : ndarray (``(sh_order + 1)*(sh_order + 2)/2``,)
        Rotational harmonics coefficients representing the input `r_sh`

    See Also
    --------
    shm.real_sph_harm, shm.real_sym_sh_basis

    References
    ----------
    .. [1] Tournier, J.D., et al. NeuroImage 2007. Robust determination of the
        fibre orientation distribution in diffusion MRI: Non-negativity
        constrained super-resolved spherical deconvolution

    """
    mask = m == 0
    # The delta function at theta = phi = 0 is known to have zero coefficients
    # where m != 0, therefore we need only compute the coefficients at m=0.
    dirac_sh = gen_dirac(0, n[mask], 0, 0)
    r_rh = r_sh[mask] / dirac_sh
    return r_rh


def gen_dirac(m, n, theta, phi):
    """ Generate Dirac delta function orientated in (theta, phi) on the sphere

    The spherical harmonics (SH) representation of this Dirac is returned as
    coefficients to spherical harmonic functions produced by
    `shm.real_sph_harm`.

    Parameters
    ----------
    m : ndarray (N,)
        The order of the spherical harmonic function associated with each
        coefficient.
    n : ndarray (N,)
        The degree of the spherical harmonic function associated with each
        coefficient.
    theta : float [0, 2*pi]
        The azimuthal (longitudinal) coordinate.
    phi : float [0, pi]
        The polar (colatitudinal) coordinate.

    See Also
    --------
    shm.real_sph_harm, shm.real_sym_sh_basis

    Returns
    -------
    dirac : ndarray
        SH coefficients representing the Dirac function

    """
    return real_sph_harm(m, n, theta, phi)


def forward_sdeconv_mat(r_rh, n):
    """ Build forward spherical deconvolution matrix

    Parameters
    ----------
    r_rh : ndarray
        ndarray of rotational harmonics coefficients for the single fiber
        response function. Each element `rh[i]` is associated with spherical
        harmonics of degree `2*i`.
    n : ndarray
        The degree of spherical harmonic function associated with each row of
        the deconvolution matrix. Only even degrees are allowed

    Returns
    -------
    R : ndarray (N, N)
        Deconvolution matrix with shape (N, N)

    """

    if np.any(n % 2):
        raise ValueError("n has odd degrees, expecting only even degrees")
    return np.diag(r_rh[n // 2])


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
        True if ODF comes from an ODF computed from a model using the $r^2$ term
        in the integral. For example, DSI, GQI, SHORE, CSA, Tensor, Multi-tensor
        ODFs. This results in using the proper analytical response function
        solution solving from the single-fiber ODF with the r^2 term. This
        derivation is not published anywhere but is very similar to [1]_.

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
    sdt = np.zeros(n_degrees) # SDT matrix
    frt = np.zeros(n_degrees) # FRT (Funk-Radon transform) q-ball matrix

    for l in np.arange(0, n_degrees*2, 2):
        if r2_term :
            sharp = quad(lambda z: lpn(l, z)[0][-1] * gamma(1.5) *
                         np.sqrt( ratio / (4 * np.pi ** 3) ) /
                         np.power((1 - (1 - ratio) * z ** 2), 1.5), -1., 1.)
        else :
            sharp = quad(lambda z: lpn(l, z)[0][-1] *
                         np.sqrt(1 / (1 - (1 - ratio) * z * z)), -1., 1.)

        sdt[l / 2] = sharp[0]
        frt[l / 2] = 2 * np.pi * lpn(l, 0)[0][-1]

    idx = n // 2
    b = sdt[idx]
    bb = frt[idx]
    return np.diag(b), np.diag(bb)


def csdeconv(s_sh, sh_order, R, B_reg, lambda_=1., tau=0.1):
    r""" Constrained-regularized spherical deconvolution (CSD) [1]_

    Deconvolves the axially symmetric single fiber response
    function `r_rh` in rotational harmonics coefficients from the spherical
    function `s_sh` in SH coefficients.

    Parameters
    ----------
    s_sh : ndarray (``(sh_order + 1)*(sh_order + 2)/2``,)
         ndarray of SH coefficients for the spherical function to be deconvolved
    sh_order : int
         maximal SH order of the SH representation
    R : ndarray (``(sh_order + 1)*(sh_order + 2)/2``, ``(sh_order + 1)*(sh_order + 2)/2``)
        forward spherical harmonics matrix
    B_reg : ndarray (``(sh_order + 1)*(sh_order + 2)/2``, ``(sh_order + 1)*(sh_order + 2)/2``)
         SH basis matrix used for deconvolution
    lambda_ : float
         lambda parameter in minimization equation (default 1.0)
    tau : float
         threshold controlling the amplitude below which the corresponding fODF
         is assumed to be zero.  Ideally, tau should be set to zero. However,
         to improve the stability of the algorithm, tau is set to tau*100 % of
         the max fODF amplitude (here, 10% by default). This is similar to peak
         detection where peaks below 0.1 amplitude are usually considered noise
         peaks. Because SDT is based on a q-ball ODF deconvolution, and not
         signal deconvolution, using the max instead of mean (as in CSD), is
         more stable.

    Returns
    -------
    fodf_sh : ndarray (``(sh_order + 1)*(sh_order + 2)/2``,)
         Spherical harmonics coefficients of the constrained-regularized fiber
         ODF
    num_it : int
         Number of iterations in the constrained-regularization used for
         convergence

    References
    ----------

    .. [1] Tournier, J.D., et al. NeuroImage 2007. Robust determination of the
           fibre orientation distribution in diffusion MRI: Non-negativity
           constrained super-resolved spherical deconvolution.
    """

    # generate initial fODF estimate, truncated at SH order 4
    fodf_sh = np.linalg.lstsq(R, s_sh)[0]
    fodf_sh[15:] = 0

    fodf = np.dot(B_reg, fodf_sh)
    # set threshold on FOD amplitude used to identify 'negative' values
    threshold = tau * np.mean(np.dot(B_reg, fodf_sh))
    #print(np.min(fodf), np.max(fodf), np.mean(fodf), threshold, tau)

    k = []
    convergence = 50
    for num_it in range(1, convergence + 1):
        fodf = np.dot(B_reg, fodf_sh)

        k2 = np.nonzero(fodf < threshold)[0]

        if (k2.shape[0] + R.shape[0]) < B_reg.shape[1]:
            warnings.warn(
            'too few negative directions identified - failed to converge')
            return fodf_sh, num_it

        if num_it > 1 and k.shape[0] == k2.shape[0]:
            if (k == k2).all():
                return fodf_sh, num_it

        k = k2

        # This is the super-resolved trick.
        # Wherever there is a negative amplitude value on the fODF, it
        # concatenates a value to the S vector so that the estimation can
        # focus on trying to eliminate it. In a sense, this "adds" a
        # measurement, which can help to better estimate the fodf_sh, even if
        # you have more SH coeffcients to estimate than actual S measurements.
        M = np.concatenate((R, lambda_ * B_reg[k, :]))
        S = np.concatenate((s_sh, np.zeros(k.shape)))
        try:
            fodf_sh = np.linalg.lstsq(M, S)[0]
        except np.linalg.LinAlgError as lae:
            # SVD did not converge in Linear Least Squares in current
            # voxel. Proceeding with initial SH estimate for this voxel.
            pass

    warnings.warn('maximum number of iterations exceeded - failed to converge')
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
    # Generate initial fODF estimate, which is the ODF truncated at SH order 4
    fodf_sh = np.linalg.lstsq(R, odf_sh)[0]
    fodf_sh[15:] = 0

    fodf = np.dot(B_reg, fodf_sh)

    # if sharpening a q-ball odf (it is NOT properly normalized), we need to
    # force normalization otherwise, for DSI, CSA, SHORE, Tensor odfs, they are
    # normalized by construction
    if ~r2_term :
        Z = np.linalg.norm(fodf)
        fodf_sh /= Z

    fodf = np.dot(B_reg, fodf_sh)
    threshold = tau * np.max(np.dot(B_reg, fodf_sh))
    #print(np.min(fodf), np.max(fodf), np.mean(fodf), threshold, tau)

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
            fodf_sh = np.linalg.lstsq(M, ODF)[0]
        except np.linalg.LinAlgError as lae:
            # SVD did not converge in Linear Least Squares in current
            # voxel. Proceeding with initial SH estimate for this voxel.
            pass

    warnings.warn('maximum number of iterations exceeded - failed to converge')
    return fodf_sh, num_it


def odf_sh_to_sharp(odfs_sh, sphere, basis=None, ratio=3 / 15., sh_order=8,
                    lambda_=1., tau=0.1, r2_term=False):
    r""" Sharpen odfs using the spherical deconvolution transform [1]_

    This function can be used to sharpen any smooth ODF spherical function. In
    theory, this should only be used to sharpen QballModel ODFs, but in
    practice, one can play with the deconvolution ratio and sharpen almost any
    ODF-like spherical function. The constrained-regularization is stable and
    will not only sharp the ODF peaks but also regularize the noisy peaks.

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
        tau parameter in the L matrix construction (see odfdeconv) (default 0.1)
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
    R, P = forward_sdt_deconv_mat(ratio, n)

    # scale lambda to account for differences in the number of
    # SH coefficients and number of mapped directions
    lambda_ = lambda_ * R.shape[0] * R[0, 0] / B_reg.shape[0]

    fodf_sh = np.zeros(odfs_sh.shape)

    for index in ndindex(odfs_sh.shape[:-1]):
        fodf_sh[index], num_it = odf_deconv(odfs_sh[index], R, B_reg,
                                            lambda_=lambda_, tau=tau,
                                            r2_term=r2_term)

    return fodf_sh


def auto_response(gtab, data, roi_center=None, roi_radius=10, fa_thr=0.7):
    """ Automatic estimation of response function using FA

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

    Returns
    -------
    response : tuple, (2,)
        (`evals`, `S0`)
    ratio : float
        the ratio between smallest versus largest eigenvalue of the response

    Notes
    -----
    In CSD there is an important pre-processing step: the estimation of the
    fiber response function. In order to do this we look for voxels with very
    anisotropic configurations. For example we can use an ROI (20x20x20) at
    the center of the volume and store the signal values for the voxels with
    FA values higher than 0.7. Of course, if we haven't precalculated FA we
    need to fit a Tensor model to the datasets. Which is what we do  in this
    function.

    For the response we also need to find the average S0 in the ROI. This is
    possible using `gtab.b0s_mask()` we can find all the S0 volumes (which
    correspond to b-values equal 0) in the dataset.

    The `response` consists always of a prolate tensor created by averaging
    the highest and second highest eigenvalues in the ROI with FA higher than
    threshold. We also include the average S0s.

    Finally, we also return the `ratio` which is used for the SDT models.
    """

    ten = TensorModel(gtab)
    if roi_center is None:
        ci, cj, ck = np.array(data.shape[:3]) / 2
    else:
        ci, cj, ck = roi_center
    w = roi_radius
    roi = data[ci - w: ci + w, cj - w: cj + w, ck - w: ck + w]
    tenfit = ten.fit(roi)
    FA = fractional_anisotropy(tenfit.evals)
    FA[np.isnan(FA)] = 0
    indices = np.where(FA > fa_thr)
    lambdas = tenfit.evals[indices][:, :2]
    S0s = roi[indices][:, np.nonzero(gtab.b0s_mask)[0]]
    S0 = np.mean(S0s)
    l01 = np.mean(lambdas, axis=0)
    evals = np.array([l01[0], l01[1], l01[1]])
    response = (evals, S0)
    ratio = evals[1]/evals[0]
    return response, ratio
