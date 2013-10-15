from __future__ import division, print_function, absolute_import
import warnings
import numpy as np
from dipy.reconst.odf import OdfModel
from dipy.reconst.cache import Cache
from dipy.reconst.multi_voxel import multi_voxel_fit
from dipy.reconst.shm import (sph_harm_ind_list, real_sph_harm,
                              sph_harm_lookup, lazy_index, SphHarmFit)
from dipy.data import get_sphere
from dipy.core.geometry import cart2sphere
from dipy.core.ndindex import ndindex
from dipy.sims.voxel import single_tensor
from scipy.special import lpn
from dipy.reconst.dti import TensorModel, fractional_anisotropy


class ConstrainedSphericalDeconvModel(OdfModel, Cache):

    def __init__(self, gtab, response, reg_sphere=None, sh_order=8, lambda_=1, tau=0.1):
        r""" Constrained Spherical Deconvolution (CSD) [1]_.

        Spherical deconvolution computes a fiber orientation distribution (FOD), also
        called fiber ODF (fODF) [2]_, as opposed to a diffusion ODF as the QballModel
        or the CsaOdfModel. This results in a sharper angular profile with better
        angular resolution that is the best object to be used for later deterministic
        and probabilistic tractography [3]_.

        A sharp fODF is obtained because a single fiber *response* function is injected
        as *a priori* knowledge. The response function is often data-driven and thus,
        comes as input to the ConstrainedSphericalDeconvModel. It will be used as deconvolution
        kernel, as described in [1]_.
    
        Parameters
        ----------
        gtab : GradientTable
        response : tuple or callable
            If tuple, then it should have two elements. The first is the eigen-values as an (3,) ndarray
            and the second is the signal value for the response function without diffusion weighting.
            This is to be able to generate a single fiber synthetic signal. If callable then the function
            should return an ndarray with the all the signal values for the response function. The response
            function will be used as deconvolution kernel ([1]_)
        reg_sphere : Sphere
            sphere used to build the regularization B matrix
        sh_order : int
            maximal spherical harmonics order
        lambda_ : float
            weight given to the constrained-positivity regularization part of the
            deconvolution equation (see [1]_)
        tau : float
            threshold controlling the amplitude below which the corresponding fODF is assumed to be zero.
            Ideally, tau should be set to zero. However, to improve the stability of the algorithm, tau
            is set to tau*100 % of the mean fODF amplitude (here, 10% by default) (see [1]_)

        References
        ----------
        .. [1] Tournier, J.D., et al. NeuroImage 2007. Robust determination of the fibre orientation
               distribution in diffusion MRI: Non-negativity constrained super-resolved spherical
               deconvolution
        .. [2] Descoteaux, M., et al. IEEE TMI 2009. Deterministic and Probabilistic Tractography Based
               on Complex Fibre Orientation Distributions
        .. [3] C\^ot\'e, M-A., et al. Medical Image Analysis 2013. Tractometer: Towards validation
               of tractography pipelines
        .. [4] Tournier, J.D, et al. Imaging Systems and Technology 2012. MRtrix: Diffusion
               Tractography in Crossing Fiber Regions
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

        # for the sphere used in the regularization positivity constraint
        if reg_sphere is None:
            self.sphere = get_sphere('symmetric362')
        else:
            self.sphere = reg_sphere

        r, theta, phi = cart2sphere(self.sphere.x, self.sphere.y, self.sphere.z)
        self.B_reg = real_sph_harm(m, n, theta[:, None], phi[:, None])

        if callable(response):
            S_r = response
        else:
            if response is None:
                S_r = estimate_response(gtab, np.array([0.0015, 0.0003, 0.0003]), 1)
            else:
                S_r = estimate_response(gtab, response[0], response[1])

        r_sh = np.linalg.lstsq(self.B_dwi, S_r[self._where_dwi])[0]
        r_rh = sh_to_rh(r_sh, sh_order)

        self.R = forward_sdeconv_mat(r_rh, sh_order)

        # scale lambda_ to account for differences in the number of
        # SH coefficients and number of mapped directions
        # This is exactly what is done in [4]_ 
        self.lambda_ = lambda_ * self.R.shape[0] * r_rh[0] / self.B_reg.shape[0]
        self.sh_order = sh_order
        self.tau = tau

    @multi_voxel_fit
    def fit(self, data):
        s_sh = np.linalg.lstsq(self.B_dwi, data[self._where_dwi])[0]
        shm_coeff, num_it = csdeconv(s_sh, self.sh_order, self.R, self.B_reg, self.lambda_, self.tau)
        return SphHarmFit(self, shm_coeff, None)


class ConstrainedSDTModel(OdfModel, Cache):

    def __init__(self, gtab, ratio, reg_sphere=None, sh_order=8, lambda_=1., tau=0.1):
        r""" Spherical Deconvolution Transform (SDT) [1]_.
        
        The SDT computes a fiber orientation distribution (FOD) as opposed to a diffusion
        ODF as the QballModel or the CsaOdfModel. This results in a sharper angular
        profile with better angular resolution. The Contrained SDTModel is similar
        to the Constrained CSDModel but mathematically it deconvolves the q-ball ODF
        as oppposed to the HARDI signal (see [1]_ for a comparison and a through discussion).
        
        A sharp fODF is obtained because a single fiber *response* function is injected
        as *a priori* knowledge. In the SDTModel, this response is a single fiber q-ball
        ODF as opposed to a single fiber signal function for the CSDModel. The response function
        will be used as deconvolution kernel.

        Parameters
        ----------
        gtab : GradientTable
        ratio : float
            ratio of the smallest vs the largest eigenvalue of the single prolate tensor response function
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
        .. [1] Descoteaux, M., et al. IEEE TMI 2009. Deterministic and Probabilistic Tractography Based
               on Complex Fibre Orientation Distributions.
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

        self.R, self.P = forward_sdt_deconv_mat(ratio, sh_order)

        # scale lambda_ to account for differences in the number of
        # SH coefficients and number of mapped directions
        self.lambda_ = lambda_ * self.R.shape[0] * self.R[0, 0] / self.B_reg.shape[0]
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
        shm_coeff, num_it = odf_deconv(odf_sh, self.sh_order, self.R, self.B_reg, self.lambda_, self.tau)
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


def sh_to_rh(r_sh, sh_order):
    """ Spherical harmonics (SH) to rotational harmonics (RH)

    Calculate the rotational harmonic decomposition up to
    harmonic sh_order for an axially and antipodally
    symmetric function. Note that all ``m != 0`` coefficients
    will be ignored as axial symmetry is assumed. Hence, there
    will be ``(sh_order/2 + 1)`` non-zero coefficients.

    Parameters
    ----------
    r_sh : ndarray (``sh_order/2 + 1``,)
        ndarray of SH coefficients for the single fiber response function
    sh_order : int
        maximal SH order of the SH representation

    Returns
    -------
    r_rh : ndarray (``(sh_order + 1)*(sh_order + 2)/2``,)
        Rotational harmonics coefficients representing the input `r_sh`
 
    References
    ----------
    .. [1] Tournier, J.D., et al. NeuroImage 2007. Robust determination of the fibre orientation
           distribution in diffusion MRI: Non-negativity constrained super-resolved spherical
           deconvolution
    """

    dirac_sh = gen_dirac(0, 0, sh_order)
    k, = np.nonzero(dirac_sh)
    r_rh = r_sh[k] / dirac_sh[k]

    return r_rh


def gen_dirac(pol, azi, sh_order):
    """ Generate Dirac delta function orientated in (theta, phi) = (azi, pol)
    on the sphere. The spherical harmonics (SH) representation of this Dirac is
    returned. 

    Parameters
    ----------
    pol : float [0, pi]
        The polar (colatitudinal) coordinate (phi)
    az : float [0, 2*pi]
        The azimuthal (longitudinal) coordinate (theta)
    sh_order : int
        maximal SH order of the SH representation

    Returns
    -------
    dirac : ndarray (``(sh_order + 1)(sh_order + 2)/2``,)
        SH coefficients representing the Dirac function
    """
    m, n = sph_harm_ind_list(sh_order)
    dirac = np.zeros(m.shape)
    i = 0
    for l in np.arange(0, sh_order + 1, 2):
        for m in np.arange(-l, l + 1):
            if m == 0:
                dirac[i] = real_sph_harm(0, l, azi, pol)

            i = i + 1

    return dirac


def forward_sdeconv_mat(r_rh, sh_order):
    """ Build forward spherical deconvolution matrix

    Parameters
    ----------
    r_rh : ndarray (``(sh_order + 1)*(sh_order + 2)/2``,)
        ndarray of rotational harmonics coefficients for the single
        fiber response function
    sh_order : int
        maximal SH order

    Returns
    -------
    R : ndarray (``(sh_order + 1)*(sh_order + 2)/2``, ``(sh_order + 1)*(sh_order + 2)/2``)

    """

    m, n = sph_harm_ind_list(sh_order)

    b = np.zeros(m.shape)
    i = 0
    for l in np.arange(0, sh_order + 1, 2):
        for m in np.arange(-l, l + 1):
            b[i] = r_rh[l / 2]
            i = i + 1
    return np.diag(b)


def forward_sdt_deconv_mat(ratio, sh_order):
    """ Build forward sharpening deconvolution transform (SDT) matrix

    Parameters
    ----------
    ratio : float
        ratio = $\frac{\lambda_2}{\lambda_1}$ of the single fiber response function
    sh_order : int
        spherical harmonic order

    Returns
    -------
    R : ndarray (``(sh_order + 1)*(sh_order + 2)/2``, ``(sh_order + 1)*(sh_order + 2)/2``)
        SDT deconvolution matrix
    P : ndarray (``(sh_order + 1)*(sh_order + 2)/2``, ``(sh_order + 1)*(sh_order + 2)/2``)
        Funk-Radon Transform (FRT) matrix
    """
    m, n = sph_harm_ind_list(sh_order)

    sdt = np.zeros(m.shape) # SDT matrix
    frt = np.zeros(m.shape) # FRT (Funk-Radon transform) q-ball matrix
    b = np.zeros(m.shape)
    bb = np.zeros(m.shape)

    for l in np.arange(0, sh_order + 1, 2):
        from scipy.integrate import quad
        sharp = quad(lambda z: lpn(l, z)[0][-1] * np.sqrt(1 / (1 - (1 - ratio) * z * z)), -1., 1.)

        sdt[l / 2] = sharp[0]
        frt[l / 2] = 2 * np.pi * lpn(l, 0)[0][-1]

    i = 0
    for l in np.arange(0, sh_order + 1, 2):
        for m in np.arange(-l, l + 1):
            b[i] = sdt[l / 2]
            bb[i] = frt[l / 2]
            i = i + 1

    return np.diag(b), np.diag(bb)


def csdeconv(s_sh, sh_order, R, B_reg, lambda_=1., tau=0.1):
    r""" Constrained-regularized spherical deconvolution (CSD) [1]_

    Deconvolves the axially symmetric single fiber response
    function `r_rh` in rotational harmonics coefficients from the spherical function
    `s_sh` in SH coefficients.

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
         threshold controlling the amplitude below which the corresponding fODF is assumed to be zero.
         Ideally, tau should be set to zero. However, to improve the stability of the algorithm, tau
         is set to tau*100 % of the max fODF amplitude (here, 10% by default). This is similar to peak
         detection where peaks below 0.1 amplitude are usually considered noise peaks. Because SDT
         is based on a q-ball ODF deconvolution, and not signal deconvolution, using the max instead
         of mean (as in CSD), is more stable.

    Returns
    -------
    fodf_sh : ndarray (``(sh_order + 1)*(sh_order + 2)/2``,)
         Spherical harmonics coefficients of the constrained-regarized fiber ODF
    num_it : int
         Number of iterations in the constrained-regarization used for convergence

    References
    ----------
    .. [1] Tournier, J.D., et al. NeuroImage 2007. Robust determination of the fibre orientation
           distribution in diffusion MRI: Non-negativity constrained super-resolved spherical
           deconvolution
    """

    # generate initial fODF estimate, truncated at SH order 4
    fodf_sh = np.linalg.lstsq(R, s_sh)[0] #fodf_sh, = np.linalg.lstsq(R, s_sh) # R\s_sh
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
            warnings.warn('too few negative directions identified - failed to converge')
            return fodf_sh, num_it

        if num_it > 1 and k.shape[0] == k2.shape[0]:
            if (k == k2).all():
                return fodf_sh, num_it

        k = k2

        # This is the super-resolved trick. 
        # Wherever there is a negative amplitude value on the fODF, it concatinates a value
        # to the S vector so that the estimation can focus on trying to eliminate it.
        # In a sense, this "adds" a measurement, which can help to better estimate the fodf_sh,
        # even if you have more SH coeffcients to estimate than actual S measurements. 
        M = np.concatenate((R, lambda_ * B_reg[k, :]))
        S = np.concatenate((s_sh, np.zeros(k.shape)))
        fodf_sh = np.linalg.lstsq(M, S)[0]

    warnings.warn('maximum number of iterations exceeded - failed to converge')
    return fodf_sh, num_it


def odf_deconv(odf_sh, sh_order, R, B_reg, lambda_=1., tau=0.1):
    r""" ODF constrained-regularized sherical deconvolution using
    the Sharpening Deconvolution Transform (SDT) [1]_, [2]_.

    Parameters
    ----------
    odf_sh : ndarray (``(sh_order + 1)*(sh_order + 2)/2``,)
         ndarray of SH coefficients for the ODF spherical function to be deconvolved
    sh_order : int
         maximal SH order of the SH representation
    R : ndarray (``(sh_order + 1)(sh_order + 2)/2``, ``(sh_order + 1)(sh_order + 2)/2``)
         SDT matrix in SH basis
    B_reg : ndarray (``(sh_order + 1)(sh_order + 2)/2``, ``(sh_order + 1)(sh_order + 2)/2``)
         SH basis matrix used for deconvolution
    lambda_ : float
         lambda parameter in minimization equation (default 1.0)
    tau : float
         threshold (tau *max(fODF)) controlling the amplitude below
         which the corresponding fODF is assumed to be zero.

    Returns
    -------
    fodf_sh : ndarray (``(sh_order + 1)(sh_order + 2)/2``,)
         Spherical harmonics coefficients of the constrained-regularized fiber ODF
    num_it : int
         Number of iterations in the constrained-regularization used for convergence

    References
    ----------
    .. [1] Descoteaux, M., et al. IEEE TMI 2009. Deterministic and Probabilistic Tractography Based
           on Complex Fibre Orientation Distributions
    .. [2] Descoteaux, M, PhD thesis, INRIA Sophia-Antipolis, 2008.
    """
    m, n = sph_harm_ind_list(sh_order)

    # Generate initial fODF estimate, which is the ODF truncated at SH order 4
    fodf_sh = np.linalg.lstsq(R, odf_sh)[0]
    fodf_sh[15:] = 0

    fodf = np.dot(B_reg, fodf_sh)

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
            warnings.warn('too few negative directions identified - failed to converge')
            return fodf_sh, num_it

        if num_it > 1 and k.shape[0] == k2.shape[0]:
            if (k == k2).all():
                return fodf_sh, num_it

        k = k2
        M = np.concatenate((R, lambda_ * B_reg[k, :]))
        ODF = np.concatenate((odf_sh, np.zeros(k.shape)))
        fodf_sh = np.linalg.lstsq(M, ODF)[0]  

    warnings.warn('maximum number of iterations exceeded - failed to converge')
    return fodf_sh, num_it


def odf_sh_to_sharp(odfs_sh, sphere, basis=None, ratio=3 / 15., sh_order=8, lambda_=1., tau=0.1):
    r""" Sharpen odfs using the spherical deconvolution transform [1]_

    This function can be used to sharpen any smooth ODF spherical function. In theory, this should
    only be used to sharpen QballModel ODFs, but in practice, one can play with the deconvolution
    ratio and sharpen almost any ODF-like spherical function. The constrained-regularization is stable
    and will not only sharp the ODF peaks but also regularize the noisy peaks.

    Parameters
    ---------- 
    odfs_sh : ndarray (``(sh_order + 1)*(sh_order + 2)/2``, )
        array of odfs expressed as spherical harmonics coefficients
    sphere : Sphere
        sphere used to build the regularization matrix    
    basis : {None, 'mrtrix', 'fibernav'}
        different spherical harmonic basis. None is the fibernav basis as well.
    ratio : float, 
        ratio of the smallest vs the largest eigenvalue of the single prolate tensor response function
        (:math:`\frac{\lambda_2}{\lambda_1}`)
    sh_order : int
        maximal SH order of the SH representation
    lambda_ : float
        lambda parameter (see odfdeconv) (default 1.0)
    tau : float
        tau parameter in the L matrix construction (see odfdeconv) (default 0.1)

    Returns
    -------
    fodf_sh : ndarray
        sharpened odf expressed as spherical harmonics coefficients

    References
    ----------
    .. [1] Descoteaux, M., et al. IEEE TMI 2009. Deterministic and Probabilistic Tractography Based
           on Complex Fibre Orientation Distributions
    """
    m, n = sph_harm_ind_list(sh_order)
    r, theta, phi = cart2sphere(sphere.x, sphere.y, sphere.z)

    real_sym_sh = sph_harm_lookup[basis]

    B_reg, m, n = real_sym_sh(sh_order, theta[:, None], phi[:, None])
    
    R, P = forward_sdt_deconv_mat(ratio, sh_order)

    # scale lambda to account for differences in the number of
    # SH coefficients and number of mapped directions
    lambda_ = lambda_ * R.shape[0] * R[0, 0] / B_reg.shape[0]

    fodf_sh = np.zeros(odfs_sh.shape)

    for index in ndindex(odfs_sh.shape[:-1]):

        fodf_sh[index], num_it = odf_deconv(odfs_sh[index], sh_order, R, B_reg, lambda_=lambda_, tau=tau)

    return fodf_sh


def auto_response(gtab, data, center=None, w=10, fa_thr=0.7):
    """ Automatic estimation of response function using FA 

    Parameters
    ----------
    gtab : GradientTable
    data : ndarray
        diffusion data
    center : tuple, (3,)
        Center of ROI in data. If center is None, it is assumed that is
        the center of the volume with shape `data.shape[:3]`.
    w : int
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
    fiber response function. In order to do this we look for voxel with very 
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
    if center is None:
        ci, cj, ck = np.array(data.shape[:3]) / 2
    else:
        ci, cj, ck = center
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
