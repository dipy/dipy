import numpy as np
from dipy.reconst.odf import OdfModel, OdfFit
from dipy.reconst.cache import Cache
from dipy.reconst.multi_voxel import multi_voxel_model
from dipy.reconst.shm import (sph_harm_ind_list,
                              real_sph_harm,
                              lazy_index,
                              sh_to_sf)
from dipy.data import get_sphere
from dipy.core.geometry import cart2sphere
from dipy.sims.voxel import single_tensor


@multi_voxel_model
class ConstrainedSphericalDeconvModel(OdfModel, Cache):
    def __init__(self, gtab, response_function=None, sh_order=8, Lambda=1, tau=0.1):
        r""" Constrained Spherical Deconvolution

        Parameters
        ----------

        gtab : GradientTable
        response_function : ndarray
                default is None
        sh_order : int
                spherical harmonics order

        Notes
        ------
        The method used here can be described in the following way.
        0 Estimate single fiber repsonse function
            From a masked FA get all voxels with FA > 0.7. Estimate eigen-vector,
            for each one and align it with z-axis.
        1 Build reconstruction matrices
            - B_dwi (original signal)
            - B_regul (regularized)
            - R (single fiber)
        2 Call csdeconv for every voxel

        References
        ----------
        Tournier, J.D., et. al. NeuroImage 2007.
        """

        m, n = sph_harm_ind_list(sh_order)
        self._where_b0s = lazy_index(gtab.b0s_mask)
        self._where_dwi = lazy_index(~gtab.b0s_mask)
        x, y, z = gtab.gradients[self._where_dwi].T
        r, pol, azi = cart2sphere(x, y, z)
        # for the gradient sphere
        self.B_dwi = real_sph_harm(m, n, azi[:, None], pol[:, None])

        # for the odf sphere
        self.sphere = get_sphere('symmetric362')
        r, pol, azi = cart2sphere(self.sphere.x, self.sphere.y, self.sphere.z)
        self.B_regul = real_sph_harm(m, n, azi[:, None], pol[:, None])

        S_r = estimate_response(gtab, 1)
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
        # return sh_to_sf(self.shm_coeff, sphere, self.model.sh_order)
        return np.dot(self.shm_coeff, self.model.B_regul.T)


def estimate_response(gtab, S0=100):
    """ Estimate response function

    """
    s_mevals = np.array([0.0015, 0.0003, 0.0003])
    s_mevecs = np.array([[0, 0, 1],
                         [0, 1, 0],
                         [1, 0, 0]])

    S_r = single_tensor(gtab, S0, s_mevals, s_mevecs, snr=None)
    return S_r


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
    fodf_sh = np.linalg.lstsq(R, s_sh)[0]  # a\b
    fodf_sh[15:] = 0

    # set threshold on FOD amplitude used to identify 'negative' values
    threshold = tau * np.mean(np.dot(B_regul, fodf_sh))

    k = []
    convergence = 50
    for num_it in xrange(1, convergence + 1):
        A = np.dot(B_regul, fodf_sh)
        k2 = np.nonzero(A < threshold)[0]

        if (k2.shape[0] + R.shape[0]) < B_regul.shape[1]:
            print 'too few negative directions identified - failed to converge'
            return fodf_sh, num_it

        if num_it > 1 and k.shape[0] == k2.shape[0]:
            if (k == k2).all():
                return fodf_sh, num_it

        k = k2
        M = np.concatenate((R, Lambda * B_regul[k, :]))
        S = np.concatenate((s_sh, np.zeros(k.shape)))
        fodf_sh = np.linalg.lstsq(M, S)[0] 

    print 'maximum number of iterations exceeded - failed to converge'
    return fodf_sh, num_it
