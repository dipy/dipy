from __future__ import division, print_function, absolute_import
import numpy as np
from dipy.reconst.odf import OdfModel, OdfFit
from dipy.reconst.cache import Cache
from dipy.reconst.multi_voxel import multi_voxel_model
from dipy.reconst.shm import (sph_harm_ind_list,
                              real_sph_harm,
                              real_sph_harm_mrtrix,
                              lazy_index)
from dipy.data import get_sphere
from dipy.core.geometry import cart2sphere
from dipy.core.ndindex import ndindex
from dipy.sims.voxel import single_tensor
from scipy.special import lpn


@multi_voxel_model
class ConstrainedSphericalDeconvModel(OdfModel, Cache):

    def __init__(self, gtab, response, regul_sphere=None, sh_order=8, Lambda=1, tau=0.1):
        r""" Constrained Spherical Deconvolution [1]_.

        Parameters
        ----------
        gtab : GradientTable
        response : tuple
            tuple with two elements the first are the eigen-values as an (3,) ndarray
            and the second is the S0.
        regul_sphere : Sphere
            sphere used to build the regularized B matrix
        sh_order : int
            spherical harmonics order

        References
        ----------
        .. [1] Tournier, J.D., et. al. NeuroImage 2007.
        """

        m, n = sph_harm_ind_list(sh_order)
        self.m, self.n = m, n
        self._where_b0s = lazy_index(gtab.b0s_mask)
        self._where_dwi = lazy_index(~gtab.b0s_mask)
        x, y, z = gtab.gradients[self._where_dwi].T
        r, pol, azi = cart2sphere(x, y, z)
        # for the gradient sphere
        self.B_dwi = real_sph_harm(m, n, pol[:, None], azi[:, None])

        # for the odf sphere
        if regul_sphere is None:
            self.sphere = get_sphere('symmetric362')
        else:
            self.sphere = regul_sphere

        r, pol, azi = cart2sphere(self.sphere.x, self.sphere.y, self.sphere.z)
        self.B_regul = real_sph_harm(m, n, pol[:, None], azi[:, None])

        if response is None:
            S_r = estimate_response(gtab, np.array([0.0015, 0.0003, 0.0003]), 1)
        else:
            S_r = estimate_response(gtab, response[0], response[1])

        r_sh = np.linalg.lstsq(self.B_dwi, S_r[self._where_dwi])[0]

        r_rh = sh_to_rh(r_sh, sh_order)
        self.R = forward_sdeconv_mat(r_rh, sh_order)

        # scale lambda to account for differences in the number of
        # SH coefficients and number of mapped directions

        self.Lambda = Lambda * self.R.shape[0] * r_rh[0] / self.B_regul.shape[0]
        self.tau = 0.1
        self.sh_order = sh_order

    def fit(self, data):
        s_sh = np.linalg.lstsq(self.B_dwi, data[self._where_dwi])[0]
        shm_coeff, num_it = csdeconv(s_sh, self.sh_order, self.R, self.B_regul, self.Lambda, self.tau)
        return ConstrainedSphericalDeconvFit(self, shm_coeff)


class ConstrainedSphericalDeconvFit(OdfFit):

    def __init__(self, model, fodf_sh):
        self.shm_coeff = fodf_sh
        self.model = model

    def odf(self, sphere):

        sampling_matrix = self.model.cache_get("sampling_matrix", sphere)
        if sampling_matrix is None:
            phi = sphere.phi.reshape((-1, 1))
            theta = sphere.theta.reshape((-1, 1))
            sampling_matrix = real_sph_harm(self.model.m, self.model.n, theta, phi)
            self.model.cache_set("sampling_matrix", sphere, sampling_matrix)

        return np.dot(self.shm_coeff, sampling_matrix.T)


@multi_voxel_model
class ConstrainedSDTModel(OdfModel, Cache):

    def __init__(self, gtab, ratio, regul_sphere=None, sh_order=8, Lambda=1., tau=1.):
        r""" Spherical Deconvolution Transform [1]_.

        Parameters
        ----------
        gtab : GradientTable
        ratio : float
            ratio = \frac{\lambda_2}{\lambda_1} of the single tensor response function
        regul_sphere : Sphere
            sphere used to build the regularized B matrix
        sh_order : int
            spherical harmonics order

        References
        ----------
        .. [1] Descoteaux, M., et. al. TMI 2009.
        """

        m, n = sph_harm_ind_list(sh_order)
        self.m, self.n = m, n
        self._where_b0s = lazy_index(gtab.b0s_mask)
        self._where_dwi = lazy_index(~gtab.b0s_mask)
        x, y, z = gtab.gradients[self._where_dwi].T
        r, pol, azi = cart2sphere(x, y, z)
        # for the gradient sphere
        self.B_dwi = real_sph_harm(m, n, pol[:, None], azi[:, None])

        # for the odf sphere
        if regul_sphere is None:
            self.sphere = get_sphere('symmetric362')
        else:
            self.sphere = regul_sphere

        r, pol, azi = cart2sphere(self.sphere.x, self.sphere.y, self.sphere.z)
        self.B_regul = real_sph_harm(m, n, pol[:, None], azi[:, None])

        self.R, self.P = forward_sdt_deconv_mat(ratio, sh_order)

        # scale lambda to account for differences in the number of
        # SH coefficients and number of mapped directions
        self.Lambda = Lambda * self.R.shape[0] * self.R[0, 0] / self.B_regul.shape[0]
        self.tau = tau
        self.sh_order = sh_order

    def fit(self, data):
        s_sh = np.linalg.lstsq(self.B_dwi, data[self._where_dwi])[0]
        # initial ODF estimation
        odf_sh = np.dot(self.P, s_sh)
        qball_odf = np.dot(self.B_regul, odf_sh)
        Z = np.linalg.norm(qball_odf)
        # normalize ODF
        odf_sh /= Z
        shm_coeff, num_it = odf_deconv(odf_sh, self.sh_order, self.R, self.B_regul, self.Lambda, self.tau)
        # print 'SDT CSD converged after %d iterations' % num_it

        return ConstrainedSDTFit(self, shm_coeff)


class ConstrainedSDTFit(OdfFit):

    def __init__(self, model, fodf_sh):
        self.shm_coeff = fodf_sh
        self.model = model

    def odf(self, sphere):

        sampling_matrix = self.model.cache_get("sampling_matrix", sphere)
        if sampling_matrix is None:
            phi = sphere.phi.reshape((-1, 1))
            theta = sphere.theta.reshape((-1, 1))
            sampling_matrix = real_sph_harm(self.model.m, self.model.n, theta, phi)
            self.model.cache_set("sampling_matrix", sphere, sampling_matrix)

        return np.dot(self.shm_coeff, sampling_matrix.T)


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
    # evals = np.array([0.0015, 0.0003, 0.0003])
    evecs = np.array([[0, 0, 1],
                      [0, 1, 0],
                      [1, 0, 0]])

    return single_tensor(gtab, S0, evals, evecs, snr=None)


def sh_to_rh(r_sh, sh_order):
    """ Spherical harmonics (SH) to rotational harmonics (RH)

    Calculate the rotational harmonic decomposition up to
    harmonic sh_order for an axially and antipodally
    symmetric function. Note that all m != 0 coefficients
    will be ignored as axial symmetry is assumed.

    Parameters
    ----------
    r_sh : ndarray
        ndarray of SH coefficients for the single fiber response function
    sh_order : int
        maximal SH order of the SH representation

    Returns
    -------
    r_rh : ndarray
        Rotational harmonics coefficients representing the input `r_sh`
    """

    dirac_sh = gen_dirac(0, 0, sh_order)
    k = np.nonzero(dirac_sh)[0]
    r_rh = r_sh[k] / dirac_sh[k]
    return r_rh


def gen_dirac(pol, azi, sh_order):
    m, n = sph_harm_ind_list(sh_order)
    rsh = real_sph_harm
    # other bases support TO DO

    dirac = np.zeros((m.shape))
    i = 0
    for l in np.arange(0, sh_order + 1, 2):
        for m in np.arange(-l, l + 1):
            if m == 0:
                dirac[i] = rsh(0, l, azi, pol)

            i = i + 1

    return dirac


def forward_sdeconv_mat(r_rh, sh_order):
    """ Build forward spherical deconvolution matrix

    Parameters
    ----------
    r_rh : ndarray
        ndarray of rotational harmonics coefficients for the single
        fiber response function
    sh_order : int
        spherical harmonic order

    Returns
    -------
    R : ndarray

    """

    m, n = sph_harm_ind_list(sh_order)

    b = np.zeros((m.shape))
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
        ratio = \frac{\lambda_2}{\lambda_1} of the single fiber response function
    sh_order : int
        spherical harmonic order

    Returns
    -------
    R : ndarray
        SDT deconvolution matrix
    P : ndarray
        Funk-Radon Transform (FRT) matrix
    """
    m, n = sph_harm_ind_list(sh_order)
    b = np.zeros((m.shape))

    num = 1000
    delta = 1.0 / num
    # n = (sh_order + 1.0) + (sh_order + 2.0) / 2.0

    sdt = np.zeros((m.shape))
    frt = np.zeros((m.shape))
    b = np.zeros((m.shape))
    bb = np.zeros((m.shape))

    l = 0
    for l in np.arange(0, sh_order + 1, 2):
        sharp = 0.0
        integral = 0.0

        # Trapezoidal integration
        # 1/2 [ f(x0) + 2f(x1) + ... + 2f(x{n-1}) + f(xn) ] delta
        for z in np.linspace(-1, 1, num):
            if z == -1 or z == 1:
                sharp += lpn(l, z)[0][-1] * np.sqrt(1 / (1 - (1 - ratio) * z * z))
                integral += np.sqrt(1 / (1 - (1 - ratio) * z * z))
            else:
                sharp += 2 * lpn(l, z)[0][-1] * np.sqrt(1 / (1 - (1 - ratio) * z * z))
                integral += 2 * np.sqrt(1 / (1 - (1 - ratio) * z * z))

        integral /= 2
        integral *= delta
        sharp /= 2
        sharp *= delta
        sharp /= integral
        sdt[l / 2] = sharp
        frt[l / 2] = 2 * np.pi * lpn(l, 0)[0][-1]

    # print sdt
    # std = [ 1.          0.0987961   0.0214013   0.00570876  0.00169231]
    # for sh_order = 8 and num = 1000

    # print frt
    # frt =  [6.28318531 -3.14159265  2.35619449 -1.96349541  1.71805848]
    i = 0
    for l in np.arange(0, sh_order + 1, 2):
        for m in np.arange(-l, l + 1):
            b[i] = sdt[l / 2]
            bb[i] = frt[l / 2]
            i = i + 1

    return np.diag(b), np.diag(bb)


def csdeconv(s_sh, sh_order, R, B_regul, Lambda=1., tau=0.1):
    """ Constrained-regularized spherical deconvolution (CSD)

    Deconvolves the axially symmetric single fiber response
    function `r_rh` in rotational harmonics coefficients from the spherical function
    `s_sh` in SH coefficients.

    Parameters
    ----------
    s_sh : ndarray
         ndarray of SH coefficients for the spherical function to be deconvolved
    sh_order : int
         maximal SH order of the SH representation
    R : ndarray
        forward spherical harmonics matrix
    B_regul : ndarray
         SH basis matrix used for deconvolution
    Lambda : float
         lambda parameter in minimization equation (default 1.0)
    tau : float
         tau parameter in the L matrix construction (default 0.1)

    Returns
    -------
    fodf_sh : ndarray
         Spherical harmonics coefficients of the constrained-regularized fiber ODF
    num_it : int
         Number of iterations in the constrained-regularization used for convergence

    References
    ----------
    Tournier, J.D., et. al. NeuroImage 2007.
    """

    # generate initial fODF estimate, truncated at SH order 4
    fodf_sh = np.linalg.lstsq(R, s_sh)[0]  # R\s_sh
    fodf_sh[15:] = 0

    # set threshold on FOD amplitude used to identify 'negative' values
    threshold = tau * np.mean(np.dot(B_regul, fodf_sh))

    k = []
    convergence = 50
    for num_it in range(1, convergence + 1):
        A = np.dot(B_regul, fodf_sh)
        k2 = np.nonzero(A < threshold)[0]

        if (k2.shape[0] + R.shape[0]) < B_regul.shape[1]:
            print('too few negative directions identified - failed to converge')
            return fodf_sh, num_it

        if num_it > 1 and k.shape[0] == k2.shape[0]:
            if (k == k2).all():
                return fodf_sh, num_it

        k = k2
        M = np.concatenate((R, Lambda * B_regul[k, :]))
        S = np.concatenate((s_sh, np.zeros(k.shape)))
        fodf_sh = np.linalg.lstsq(M, S)[0]
        #fodf_sh = np.linalg.pinv(M).dot(S)

    print('maximum number of iterations exceeded - failed to converge')
    return fodf_sh, num_it


def odf_deconv(odf_sh, sh_order, R, B_regul, Lambda=1., tau=1.):
    """ ODF constrained-regularized sherical deconvolution using
    the Sharpening Deconvolution Transform (SDT)

    Parameters
    ----------
    odf_sh : ndarray
         ndarray of SH coefficients for the ODF spherical function to be deconvolved
    sh_order : int
         maximal SH order of the SH representation
    R : ndarray
         SDT matrix in SH basis
    B_regul : ndarray
         SH basis matrix used for deconvolution
    Lambda : float
         lambda parameter in minimization equation (default 1.0)
    tau : float
         tau parameter in the L matrix construction (default 1.0)
         You should not play with this parameter. It is quite sensitive and actually
         initiated directly from the fodf mean value.

    Returns
    -------
    fodf_sh : ndarray
         Spherical harmonics coefficients of the constrained-regularized fiber ODF
    num_it : int
         Number of iterations in the constrained-regularization used for convergence

    References
    ----------
    Descoteaux, M, et al. TMI 2009.
    Descoteaux, M, PhD thesis 2008.
    """
    m, n = sph_harm_ind_list(sh_order)

    # Generate initial fODF estimate, which is the ODF truncated at SH order 4
    fodf_sh = np.linalg.lstsq(R, odf_sh)[0]
    fodf_sh[15:] = 0

    fodf = np.dot(B_regul, fodf_sh)
    #     psphere = get_sphere('symmetric362')
    #     from dipy.viz import fvtk
    #     r = fvtk.ren()
    #     fvtk.add(r, fvtk.sphere_funcs(fodf, psphere))
    #     fvtk.show(r)

    Z = np.linalg.norm(fodf)
    # should be around 1.5
    #    print Z
    fodf_sh /= Z

    # This should be cleaned up... Because right now the tau parameter is useless
    # tau should be more or less around 0.025 from my experience
    # a good heuristic choice is just the mean of the fodf on the sphere.
    threshold = tau * np.mean(np.dot(B_regul, fodf_sh))

    # print Lambda,threshold
    # Typical values that work well: 0.124309392265 0.0339565336195

    k = []
    convergence = 50
    for num_it in range(1, convergence + 1):
        A = np.dot(B_regul, fodf_sh)
        k2 = np.nonzero(A < threshold)[0]

        if (k2.shape[0] + R.shape[0]) < B_regul.shape[1]:
            print('to few negative directions identified - failed to converge')
            return fodf_sh, num_it

        if num_it > 1 and k.shape[0] == k2.shape[0]:
            if (k == k2).all():
                return fodf_sh, num_it

        k = k2
        M = np.concatenate((R, Lambda * B_regul[k, :]))
        ODF = np.concatenate((odf_sh, np.zeros(k.shape)))
        fodf_sh = np.linalg.lstsq(M, ODF)[0]  # M\ODF

    print('maximum number of iterations exceeded - failed to converge')
    return fodf_sh, num_it


def odf_sh_to_sharp(odfs_sh, sphere, basis='mrtrix', ratio=3 / 15., sh_order=8, Lambda=1., tau=1.):
    """ Sharpen odfs
    """
    m, n = sph_harm_ind_list(sh_order)
    r, theta, phi = cart2sphere(sphere.x, sphere.y, sphere.z)

    if basis == 'mrtrix':
        B_regul, m, n = real_sph_harm_mrtrix(sh_order, theta[:, None], phi[:, None])
    else:
        B_regul = real_sph_harm(m, n, theta[:, None], phi[:, None])

    R, P = forward_sdt_deconv_mat(ratio, sh_order)

    # scale lambda to account for differences in the number of
    # SH coefficients and number of mapped directions
    Lambda = Lambda * R.shape[0] * R[0, 0] / B_regul.shape[0]

    fodf_sh = np.zeros(odfs_sh.shape)

    for index in ndindex(odfs_sh.shape[:-1]):

        fodf_sh[index], num_it = odf_deconv(odfs_sh[index], sh_order, R, B_regul, Lambda=Lambda, tau=tau)

    return fodf_sh
