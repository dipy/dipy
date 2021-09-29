'''Robust and Unbiased Model-BAsed Spherical Deconvolution (RUMBA-SD)'''
import logging
import warnings

import numpy as np

from dipy.sims.voxel import single_tensor, all_tensor_evecs
from dipy.core.geometry import vec2vec_rotmat
from dipy.core.gradients import gradient_table, unique_bvals_tolerance, \
    get_bval_indices
from dipy.core.sphere import Sphere
from dipy.reconst.shm import lazy_index, normalize_data
from dipy.reconst.multi_voxel import multi_voxel_fit
from dipy.reconst.odf import OdfModel, OdfFit
from dipy.reconst.cache import Cache
from dipy.reconst.csdeconv import AxSymShResponse
from dipy.segment.mask import bounding_box, crop

# Machine precision for numerical stability in division
_EPS = np.finfo(float).eps
logger = logging.getLogger(__name__)


class RumbaSD(OdfModel, Cache):

    def __init__(self, gtab, wm_response=np.array([1.7e-3, 0.2e-3]),
                 gm_response=0.8e-3, csf_response=3.0e-3, n_iter=600,
                 recon_type='smf', n_coils=1, R=1):
        '''
        Robust and Unbiased Model-BAsed Spherical Deconvolution (RUMBA-SD) [1]_

        Modification of the Richardson-Lucy algorithm accounting for Rician
        and Noncentral Chi noise distributions, which more accurately
        represent MRI noise. Computes a maximum likelihood estimation of the
        fiber orientation density function (fODF) at each voxel. Includes
        optional isotropic compartments to account for partial volume
        effects.

        Kernel for deconvolution constructed using a priori knowledge of white
        matter resposne function, as well as the mean diffusivity of grey
        matter and/or CSF. RUMBA-SD is robust against impulse response
        imprecision, and thus the default diffusivity values are often
        adequate [2]_.


        Parameters
        ----------
        gtab : GradientTable
        wm_response : 1d ndarray or 2d ndarray or AxSymShResponse, optional
            Tensor eigenvalues as a (2,) ndarray, multishell eigenvalues as
            a (len(unique_bvals_tolerance(gtab.bvals))-1, 2) ndarray in
            order of smallest to largest b-value, or an AxSymShResponse.
            Default: np.array([1.7e-3, 0.2e-3])
        gm_response : float, optional
            Mean diffusivity for GM compartment. If `None`, then grey
            matter volume fraction is not computed. Default: 0.8e-3
        csf_response : float, optional
            Mean diffusivity for CSF compartment. If `None`, then CSF
            volume fraction is not computed. Default: 3.0e-3
        n_iter : int, optional
            Number of iterations for fODF estimation. Must be a positive int.
            Default: 600
        recon_type : {'smf', 'sos'}, optional
            MRI reconstruction method: spatial matched filter (SMF) or
            sum-of-squares (SoS). SMF reconstruction generates Rician noise
            while SoS reconstruction generates Noncentral Chi noise.
            Default: 'smf'
        n_coils : int, optional
            Number of coils in MRI scanner -- only relevant in SoS
            reconstruction. Must be a positive int. Default: 1
        R : int, optional
            Acceleration factor of the acquisition. For SIEMENS,
            R = iPAT factor. For GE, R = ASSET factor. For PHILIPS,
            R = SENSE factor. Typical values are 1 or 2. Must be a positive
            int. Default: 1

        References
        ----------
        .. [1] Canales-Rodríguez, E. J., Daducci, A., Sotiropoulos, S. N.,
               Caruyer, E., Aja-Fernández, S., Radua, J., Mendizabal, J. M. Y.,
               Iturria-Medina, Y., Melie-García, L., Alemán-Gómez, Y.,
               Thiran, J.-P., Sarró, S., Pomarol-Clotet, E., & Salvador, R.
               (2015). Spherical Deconvolution of Multichannel Diffusion MRI
               Data with Non-Gaussian Noise Models and Spatial Regularization.
               PLOS ONE, 10(10), e0138910.
               https://doi.org/10.1371/journal.pone.0138910

        .. [2] Dell’Acqua, F., Rizzo, G., Scifo, P., Clarke, R., Scotti, G., &
               Fazio, F. (2007). A Model-Based Deconvolution Approach to Solve
               Fiber Crossing in Diffusion-Weighted MR Imaging. IEEE
               Transactions on Bio-Medical Engineering, 54, 462–472.
               https://doi.org/10.1109/TBME.2006.888830


        '''

        if not np.any(gtab.b0s_mask):
            raise ValueError("Gradient table has no b0 measurements")

        # Masks to extract b0/non-b0 measurements
        self.where_b0s = lazy_index(gtab.b0s_mask)
        self.where_dwi = lazy_index(~gtab.b0s_mask)

        # Correct gradient table to contain b0 data at the beginning
        bvals_cor = np.concatenate(([0], gtab.bvals[self.where_dwi]))
        bvecs_cor = np.concatenate(([[0, 0, 0]], gtab.bvecs[self.where_dwi]))
        gtab_cor = gradient_table(bvals_cor, bvecs_cor)

        # Initialize self.gtab
        OdfModel.__init__(self, gtab_cor)

        # Store responses
        self.wm_response = wm_response
        self.gm_response = gm_response
        self.csf_response = csf_response

        # Initializing remaining parameters
        if R < 1 or n_iter < 1 or n_coils < 1:
            raise ValueError(f"R, n_iter, and n_coils must be >= 1, but R={R},"
                             + f"n_iter={n_iter}, and n_coils={n_coils} ")

        self.R = R
        self.n_iter = n_iter
        self.recon_type = recon_type
        self.n_coils = n_coils

    @multi_voxel_fit
    def fit(self, data):
        # Normalize data to mean b0 image
        data = normalize_data(data, self.where_b0s, min_signal=_EPS)
        # Rearrange data to match corrected gradient table
        data = np.concatenate(([1], data[self.where_dwi]))
        data[data > 1] = 1  # Clip values between 0 and 1

        return RumbaFit(self, data)


class RumbaFit(OdfFit):

    def __init__(self, model, data):
        '''
        Computes fODF and isotropic volume fractions for a single voxel.

        fODF and isotropic fractions are normalized to collectively sum to 1.

        Parameters
        ----------
        model : RumbaSD
            RUMBA-SD model.
        data : 1d ndarray
            Signal values for a single voxel.

        '''
        OdfFit.__init__(self, model, data)
        self.kernel = None

        self._f_gm = None
        self._f_csf = None
        self._f_iso = None
        self._f_wm = None
        self._odf = None
        self._combined = None
        self._sphere = None

    def odf(self, sphere):
        '''
        Computes fODF at discrete vertices on sphere.

        Parameters
        ----------
        sphere : Sphere
            Sphere on which to construct fODF.

        Returns
        -------
        odf : 1d ndarray
            fODF computed at each vertex on `sphere`.

        '''

        # Check if previously fit on same sphere
        if self._odf is None or self._sphere != sphere:
            self._fit(sphere)

        return self._odf

    def f_gm(self, sphere):
        '''
        Computes GM volume fraction of voxel.

        Parameters
        ----------
        sphere : Sphere
            Sphere on which to construct fODF.

        Returns
        -------
        f_gm : float
            GM volume fraction.
        '''

        # Check if previously fit on same sphere
        if self._f_gm is None or self._sphere != sphere:
            self._fit(sphere)

        return self._f_gm

    def f_csf(self, sphere):
        '''
        Computes CSF volume fraction of voxel.

        Parameters
        ----------
        sphere : Sphere
            Sphere on which to construct fODF.

        Returns
        -------
        f_csf : float
            CSF volume fraction.
        '''

        # Check if previously fit on same sphere
        if self._f_csf is None or self._sphere != sphere:
            self._fit(sphere)

        return self._f_csf
      
    def f_wm(self, sphere):
        '''
        Computes white matter fraction of voxel.

        Equivalent to sum of fODF.

        Parameters
        ----------
        sphere : Sphere
            Sphere on which to construct fODF.

        Returns
        -------
        f_wm : float
            White matter volume fraction.
        '''

        # Check if previously fit on same sphere
        if self._f_wm is None or self._sphere != sphere:
            self._fit(sphere)

        return self._f_wm

    def f_iso(self, sphere):
        '''
        Computes isotropic volume fraction of voxel.

        Equivalent to sum of GM and CSF volume fractions.

        Parameters
        ----------
        sphere : Sphere
            Sphere on which to construct fODF.

        Returns
        -------
        f_iso : float
            Isotropic volume fraction.
        '''

        # Check if previously fit on same sphere
        if self._f_iso is None or self._sphere != sphere:
            self._fit(sphere)

        return self._f_iso

    def combined_odf_iso(self, sphere):
        '''
        Combine fODF and isotropic volume fractionss.

        Distributes isotropic compartments evenly along each fODF direction.
        Sums to 1.

        Parameters
        ----------
        sphere : Sphere
            Sphere on which to construct fODF.

        Returns
        -------
        combined : 1d ndarray
            fODF combined with isotropic volume fractions.
        '''

        # Check if previously fit on same sphere
        if self._combined is None or self._sphere != sphere:
            self._fit(sphere)

        return self._combined

    def _fit(self, sphere):
        '''
        Fit fODF and isotropic volume fractions on a given sphere
        '''

        # Check if kernel previously constructed
        self.kernel = self.model.cache_get('kernel', key=sphere)

        if self.kernel is None:
            self.kernel = generate_kernel(self.model.gtab, sphere,
                                          self.model.wm_response,
                                          self.model.gm_response,
                                          self.model.csf_response)
            self.model.cache_set('kernel', sphere, self.kernel)

        # Fitting
        fodf, f_gm, f_csf, f_wm, f_iso, combined = \
            rumba_deconv(self.data,
                         self.kernel,
                         self.model.n_iter,
                         self.model.recon_type,
                         self.model.n_coils)
        self._f_wm = f_wm
        self._f_gm = f_gm
        self._f_csf = f_csf
        self._f_iso = f_iso
        self._odf = fodf
        self._combined = combined
        self._sphere = sphere
        return


def rumba_deconv(data, kernel, n_iter=600, recon_type='smf', n_coils=1):
    '''
    Fit fODF for a single voxel using RUMBA-SD [1]_.

    Deconvolves the kernel from the diffusion-weighted signal by computing a
    maximum likelihood estimation of the fODF. Minimizes the negative
    log-likelihood of the data under Rician or Noncentral Chi noise
    distributions by adapting the iterative technique developed in
    Richardson-Lucy deconvolution.

    Parameters
    ----------
    data : 1d ndarray (N,)
        Signal values for a single voxel.
    kernel : 2d ndarray (N, M)
        Deconvolution kernel mapping volume fractions of the M compartments to
        N-length signal. Last two columns should be for GM and CSF.
    n_iter : int, optional
        Number of iterations for fODF estimation. Must be a positive int.
        Default: 600
    recon_type : {'smf', 'sos'}, optional
        MRI reconstruction method: spatial matched filter (SMF) or
        sum-of-squares (SoS). SMF reconstruction generates Rician noise while
        SoS reconstruction generates Noncentral Chi noise. Default: 'smf'
    n_coils : int, optional
        Number of coils in MRI scanner -- only relevant in SoS reconstruction.
        Must be a positive int. Default: 1

    Returns
    -------
    fodf : 1d ndarray (M-1,)
        fODF for white matter compartments.
    f_gm : float
        GM volume fraction.
    f_csf : float
        CSF volume fraction.
    f_wm : float
        White matter volume fraction.
    f_iso : float
        Isotropic volume fraction (GM + CSF).
    combined : 1d ndarray (M-1, )
        fODF combined with isotropic compartments.

    Notes
    -----
    The diffusion MRI signal measured at a given voxel is a sum of
    contributions from each intra-voxel compartment, including parallel white
    matter (WM) fiber populations in a given orientation as well as effects
    from GM and CSF. The equation governing these  contributions is:

    $ S_i = S_0\left(\sum_{j=1}^{M}f_j\exp(-b_i\bold{v}_i^T\bold{D}_j
      \bold{v}_i) + f_{GM}\exp(-b_iD_{GM})+f_{CSF}\exp(-b_iD_{CSF})\right) $

    Where $S_i$ is the resultant signal along the diffusion-sensitizing
    gradient unit vector $\bold{v_i}; i = 1, ..., N$ with a b-value of $b_i$.
    $f_j; j = 1, ..., M$ is the volume fraction of the $j^{th}$ fiber
    population with an anisotropic diffusion tensor $\bold{D_j}$.

    $f_{GM}$ and $f_{CSF}$ are the volume fractions and $D_{GM}$ and $D_{CSF}$
    are the mean diffusivities of GM and CSF respectively.

    This equation is linear in $f_j, f_{GM}, f_{CSF}$ and can be simplified to
    a single matrix multiplication:

    $\bold{S} = \bold{Hf}$

    Where $\bold{S}$ is the signal vector at a certain voxel, $\bold{H}$ is
    the deconvolution kernel, and $\bold{f}$ is the vector of volume fractions
    for each compartment.

    Modern MRI scanners produce noise following a Rician or Noncentral Chi
    distribution, depending on their signal reconstruction technique [2]_.
    Using this linear model, it can be shown that the likelihood of a signal
    under a Noncentral Chi noise model is:

    $ P(\bold{S}|\bold{H}, \bold{f}, \sigma^2, n) = \prod_{i=1}^{N}\left(
      \frac{S_i}{\bar{S_i}}\right)^n\exp\left\{-\frac{1}{2\sigma^2}\left[
      S_i^2 + \bar{S}_i^2\right]\right\}I_{n-1}\left(\frac{S_i\bar{S}_i}
      {\sigma^2}\right)u(S_i) $

    Where $S_i$ and $\bar{S}_i = \bold{Hf}$ are the measured and expected
    signals respectively, and $n$ is the number of coils in the scanner, and
    $I_{n-1}$ is the modified Bessel function of first kind of order $n-1$.
    This gives the likelihood under a Rician distribution when $n$ is set to 1.

    By taking the negative log of this with respect to $\bold{f}$ and setting
    the derivative to 0, the $\bold{f}$ maxmizing likelihood is found to be:

    $ \bold{f} = \bold{f} \circ \frac{\bold{H}^T\left[\bold{S}\circ \frac{I_n(
      \bold{S}\circ \bold{Hf}/\sigma^2)} {I_{n-1}(\bold{S}\circ\bold{Hf}/
      \sigma^2)} \right ]} {\bold{H}^T\bold{Hf}} $

    The solution can be found using an iterative scheme, just as in the
    Richardson-Lucy algorithm:

    $ \bold{f}^{k+1} = \bold{f}^k \circ \frac{\bold{H}^T\left[\bold{S}\circ
      \frac{I_n(\bold{S}\circ\bold{Hf}^k/\sigma^2)} {I_{n-1}(\bold{S}\circ
      \bold{Hf}^k/\sigma^2)} \right ]} {\bold{H}^T\bold{Hf}^k} $

    In order to apply this, a reasonable estimate of $\sigma^2$ is required.
    To find this, a separate iterative scheme is found using the derivative
    of the negative log with respect to $\sigma^2$, and is run in parallel.
    This is shown here:

    $ \alpha^{k+1} = \frac{1}{nN}\left\{ \frac{\bold{S}^T\bold{S} + \bold{f}^T
      \bold{H}^T\bold{Hf}}{2} - \bold{1}^T_N\left[(\bold{S}\circ\bold{Hf})
      \circ\frac{I_n(\bold{S}\circ\bold{Hf}/\alpha^k)}{I_{n-1}(\bold{S}\circ
      \bold{Hf}/\alpha^k)} \right ]\right \} $

    For more details, see [1]_.

    References
    ----------
    .. [1] Canales-Rodríguez, E. J., Daducci, A., Sotiropoulos, S. N., Caruyer,
           E., Aja-Fernández, S., Radua, J., Mendizabal, J. M. Y.,
           Iturria-Medina, Y., Melie-García, L., Alemán-Gómez, Y., Thiran,
           J.-P.,Sarró, S., Pomarol-Clotet, E., & Salvador, R. (2015).
           Spherical Deconvolution of Multichannel Diffusion MRI Data with
           Non-Gaussian Noise Models and Spatial Regularization. PLOS ONE,
           10(10), e0138910. https://doi.org/10.1371/journal.pone.0138910

    .. [2] Constantinides, C. D., Atalar, E., & McVeigh, E. R. (1997).
           Signal-to-Noise Measurements in Magnitude Images from NMR Phased
           Arrays. Magnetic Resonance in Medicine : Official Journal of the
           Society of Magnetic Resonance in Medicine / Society of Magnetic
           Resonance in Medicine, 38(5), 852–857.
    '''

    n_comp = kernel.shape[1]  # number of compartments
    n_grad = len(data)  # gradient directions

    fodf = np.ones((n_comp, 1))  # initial guess is iso-probable
    fodf = fodf / np.sum(fodf, axis=0)  # normalize initial guess

    if recon_type == "smf":
        n_order = 1  # Rician noise (same as Noncentral Chi with order 1)
    elif recon_type == "sos":
        n_order = n_coils  # Noncentral Chi noise (order = # of coils)
    else:
        raise ValueError("Invalid recon_type. Should be 'smf' or 'sos', " +
                         f"received {recon_type}")

    data = data.reshape(-1, 1)
    reblurred = np.matmul(kernel, fodf)

    # For use later
    kernel_t = kernel.T
    f_zero = 0  # minimum value allowed in fODF

    # Initialize variance
    sigma0 = 1/15
    sigma2 = sigma0**2 * np.ones(data.shape)  # Expand into vector

    reblurred_s = data * reblurred / sigma2

    for _ in range(n_iter):
        fodf_i = fodf
        ratio = mbessel_ratio(n_order, reblurred_s)
        rl_factor = np.matmul(kernel_t, data*ratio) / \
            (np.matmul(kernel_t, reblurred) + _EPS)

        fodf = fodf_i * rl_factor  # result of iteration
        fodf = np.maximum(f_zero, fodf)  # apply positivity constraint

        # Update other variables
        reblurred = np.matmul(kernel, fodf)
        reblurred_s = data * reblurred / sigma2

        # Iterate variance
        sigma2_i = (1 / (n_grad * n_order)) * \
            np.sum((data**2 + reblurred**2) / 2 -
                   (sigma2 * reblurred_s) * ratio,
                   axis=0)
        sigma2_i = np.minimum((1 / 8)**2, np.maximum(sigma2_i, (1 / 80)**2))
        sigma2 = sigma2_i * np.ones(data.shape)  # Expand into vector

    fodf = fodf / (np.sum(fodf, axis=0) + _EPS)  # normalize final result

    f_gm = np.squeeze(fodf[n_comp-2])  # GM compartment
    f_csf = np.squeeze(fodf[n_comp-1])  # CSF compartment
    fodf = np.squeeze(fodf[:n_comp-2])  # white matter compartments
    f_wm = np.sum(fodf)  # white matter fraction
    combined = fodf + (f_gm + f_csf) / len(fodf)
    f_iso = f_csf + f_gm

    return fodf, f_gm, f_csf, f_wm, f_iso, combined


def mbessel_ratio(n, x):
    '''
    Fast computation of modified Bessel function ratio (first kind).

    Computes:

    $I_{n}(x) / I_{n-1}(x)$

    using Perron's continued fraction equation where $I_n$ is the modified
    Bessel function of first kind, order $n$ [1]_.

    Parameters
    ----------
    n : int
        Order of Bessel function in numerator (denominator is of order n-1).
        Must be a positive int.
    x : float or ndarray
        Value or array of values with which to compute ratio.

    Returns
    -------
    y : float or ndarray
        Result of ratio computation.

    References
    ----------
    .. [1] W. Gautschi and J. Slavik, “On the computation of modified Bessel
           function ratios,” Math. Comp., vol. 32, no. 143, pp. 865–875, 1978,
           doi: 10.1090/S0025-5718-1978-0470267-9
    '''

    y = x / ((2*n + x) - (2*x * (n + 1/2) / (2*n + 1 + 2*x - (2*x*(n + 3/2) / (
        2*n + 2 + 2*x - (2*x*(n + 5/2) / (2*n + 3 + 2*x)))))))

    return y


def generate_kernel(gtab, sphere, wm_response, gm_response, csf_response):
    '''
    Generate deconvolution kernel

    Compute kernel mapping orientation densities of white matter fiber
    populations (along each vertex of the sphere) and isotropic volume
    fractions to a diffusion weighted signal.

    Parameters
    ----------
    gtab : GradientTable
    sphere : Sphere
        Sphere with which to sample discrete fiber orientations in order to
        construct kernel
    wm_response : 1d ndarray or 2d ndarray or AxSymShResponse, optional
        Tensor eigenvalues as a (2,) ndarray, multishell eigenvalues as
        a (len(unique_bvals_tolerance(gtab.bvals))-1, 2) ndarray in
        order of smallest to largest b-value, or an AxSymShResponse.
    gm_response : float, optional
        Mean diffusivity for GM compartment. If `None`, then grey
        matter compartment set to all zeros.
    csf_response : float, optional
        Mean diffusivity for CSF compartment. If `None`, then CSF
        compartment set to all zeros.

    Returns
    -------
    kernel : 2d ndarray (N, M)
        Computed kernel; can be multiplied with a vector consisting of volume
        fractions for each of M-2 fiber populations as well as GM and CSF
        fractions to produce a diffusion weighted signal.
    '''

    # Coordinates of sphere vertices
    sticks = sphere.vertices

    n_grad = len(gtab.gradients)  # number of gradient directions
    n_wm_comp = sticks.shape[0]  # number of fiber populations
    n_comp = n_wm_comp + 2  # plus isotropic compartments

    kernel = np.zeros((n_grad, n_comp))

    # White matter compartments
    list_bvals = unique_bvals_tolerance(gtab.bvals)
    n_bvals = len(list_bvals) - 1  # number of unique b-values

    if isinstance(wm_response, AxSymShResponse):
        # Data-driven response
        where_dwi = lazy_index(~gtab.b0s_mask)
        gradients = gtab.gradients[where_dwi]
        gradients = gradients / np.linalg.norm(gradients, axis=1)[..., None]
        S0 = wm_response.S0
        for i in range(n_wm_comp):
            # Response oriented along [0, 0, 1], so must rotate sticks[i]
            rot_mat = vec2vec_rotmat(sticks[i], np.array([0, 0, 1]))
            rot_gradients = np.dot(rot_mat, gradients.T).T
            rot_sphere = Sphere(xyz=rot_gradients)
            # Project onto rotated sphere and scale
            rot_response = wm_response.on_sphere(rot_sphere) / S0
            kernel[where_dwi, i] = rot_response

        # Set b0 components
        kernel[gtab.b0s_mask, :] = 1

    elif wm_response.shape == (n_bvals, 2):
        # Multi-shell response
        bvals = gtab.bvals
        bvecs = gtab.bvecs
        for n, bval in enumerate(list_bvals[1:]):
            indices = get_bval_indices(bvals, bval)
            with warnings.catch_warnings():  # extract relevant b-value
                warnings.simplefilter("ignore")
                gtab_sub = gradient_table(bvals[indices], bvecs[indices])

            for i in range(n_wm_comp):
                # Signal generated by WM-fiber for each gradient direction
                evals = np.concatenate((wm_response[n], [wm_response[n, -1]]))
                S = single_tensor(gtab_sub,
                                  evals=evals,
                                  evecs=all_tensor_evecs(sticks[i]))
                kernel[indices, i] = S

        # Set b0 components
        b0_indices = get_bval_indices(bvals, list_bvals[0])
        kernel[b0_indices, :] = 1

    else:
        # Single-shell response
        for i in range(n_wm_comp):
            # Signal generated by WM-fiber for each gradient direction
            evals = np.concatenate((wm_response, [wm_response[-1]]))
            S = single_tensor(gtab, evals=evals,
                              evecs=all_tensor_evecs(sticks[i]))
            kernel[:, i] = S

    # GM compartment
    if gm_response is None:
        S_gm = np.zeros((n_grad))
    else:
        S_gm = \
            single_tensor(gtab, evals=np.array(
                [gm_response, gm_response, gm_response]))

    if csf_response is None:
        S_csf = np.zeros((n_grad))
    else:
        S_csf = \
            single_tensor(gtab, evals=np.array(
                [csf_response, csf_response, csf_response]))

    kernel[:, n_comp-2] = S_gm
    kernel[:, n_comp-1] = S_csf

    return kernel


def global_fit(model, data, sphere, mask=None, use_tv=True, verbose=False):
    '''
    Computes fODF for all voxels simultaneously.

    Simultaneous computation parallelizes estimation and also permits spatial
    regularization via total variation (RUMBA-SD + TV) [1]_.

    Parameters
    ----------
    model : RumbaSD
        RUMBA-SD model.
    data : 4d ndarray (x, y, z, N)
        Signal values for entire brain. None of the volume dimensions x, y, z
        can be 1 if TV regularization is required.
    sphere : Sphere
        Sphere on which to construct fODF.
    mask : 3d ndarray (x, y, z), optional
        Binary mask specifying voxels of interest with 1; fODF will only be
        fit at these voxel (0 elsewhere). `None` generates a mask of all 1s.
        Default: None
    use_tv : bool, optional
        If true, applies total variation regularization. This requires a brain
        volume with no singleton dimensions. Default: True
    verbose : bool, optional
        If true, logs updates on estimated signal-to-noise ratio after each
        iteration. Default: False

    Returns
    -------
    fodf : 4d ndarray (x, y, z, K)
        fODF computed for each voxel, where K is the vertices on `sphere`
    f_gm : 3d ndarray (x, y, z)
        GM volume fraction at each voxel.
    f_csf : 3d ndarray (x, y, z)
        CSF volume fraction at each voxel.
    f_wm : 3d ndarray (x, y, z)
        White matter volume fraction at each voxel.
    f_iso : 3d ndarray (x, y, z)
        Isotropic volume fraction at each voxel (GM + CSF).
    combined : 4d ndarray (x, y, z, K)
        fODF combined with isotropic compartment for each voxel.

    References
    ----------
    .. [1] Canales-Rodríguez, E. J., Daducci, A., Sotiropoulos, S. N., Caruyer,
           E., Aja-Fernández, S., Radua, J., Mendizabal, J. M. Y.,
           Iturria-Medina, Y., Melie-García, L., Alemán-Gómez, Y., Thiran,
           J.-P.,Sarró, S., Pomarol-Clotet, E., & Salvador, R. (2015).
           Spherical Deconvolution of Multichannel Diffusion MRI Data with
           Non-Gaussian Noise Models and Spatial Regularization. PLOS ONE,
           10(10), e0138910. https://doi.org/10.1371/journal.pone.0138910
    '''

    # Checking data and mask shapes

    if len(data.shape) != 4:
        raise ValueError(f"Data should be 4D, received shape f{data.shape}")

    # Signal repair, normalization

    # Normalize data to mean b0 image
    data = normalize_data(data, model.where_b0s, _EPS)
    # Rearrange data to match corrected gradient table
    data = np.concatenate(
        (np.ones([*data.shape[:3], 1]), data[..., model.where_dwi]), axis=3)
    data[data > 1] = 1  # clip values between 0 and 1
    # All arrays are converted to float32 to reduce memory load in global_fit
    data = data.astype(np.float32)

    if mask is None:  # default mask includes all voxels
        mask = np.ones(data.shape[:3])

    if data.shape[:3] != mask.shape:
        raise ValueError("Mask shape should match first 3 dimensions of data, "
                         + f"but data dimensions are f{data.shape} while mask"
                         + f"dimensions are f{mask.shape}")

    # Generate kernel
    kernel = generate_kernel(model.gtab, sphere, model.wm_response,
                             model.gm_response, model.csf_response
                             ).astype(np.float32)

    # Fit fODF
    fodf, f_gm, f_csf, f_wm, f_iso, combined = \
        rumba_deconv_global(data, kernel, mask,
                            model.n_iter,
                            model.recon_type,
                            model.n_coils,
                            model.R, use_tv,
                            verbose)

    return fodf, f_gm, f_csf, f_wm, f_iso, combined


def rumba_deconv_global(data, kernel, mask, n_iter=600, recon_type='smf',
                        n_coils=1, R=1, use_tv=True, verbose=False):
    '''
    Fit fODF for a all voxels simultaneously using RUMBA-SD.

    Deconvolves the kernel from the diffusion-weighted signal at each voxel by
    computing a maximum likelihood estimation of the fODF [1]_. Global fitting
    also permits the use of total variation regularization (RUMBA-SD + TV). The
    spatial dependence introduced by TV promotes smoother solutions (i.e.
    prevents oscillations), while still allowing for sharp discontinuities
    [2]_. This promots smoothness and continuity along individual tracts while
    preventing smoothing of adjacent tracts.

    Generally, global_fit will proceed more quickly than the voxelwise fit
    provided that the computer has adequate RAM (>= 16 GB will be more than
    sufficient.).

    Parameters
    ----------
    data : 4d ndarray (x, y, z, N)
        Signal values for entire brain. None of the volume dimensions x, y, z
        can be 1 if TV regularization is required.
    kernel : 2d ndarray (N, M)
        Deconvolution kernel mapping volume fractions of the M compartments to
        N-length signal. Last two columns should be for GM and CSF.
    mask : 3d ndarray(x, y, z)
        Binary mask specifying voxels of interest with 1; fODF will only be
        fit at these voxels (0 elsewhere).
    n_iter : int, optional
        Number of iterations for fODF estimation. Must be a positive int.
        Default: 600
    recon_type : {'smf', 'sos'}, optional
        MRI reconstruction method: spatial matched filter (SMF) or
        sum-of-squares (SoS). SMF reconstruction generates Rician noise while
        SoS reconstruction generates Noncentral Chi noise. Default: 'smf'
    n_coils : int, optional
        Number of coils in MRI scanner -- only relevant in SoS reconstruction.
        Must be a positive int. Default: 1
    use_tv : bool, optional
        If true, applies total variation regularization. This requires a brain
        volume with no singleton dimensions. Default: True
    verbose : bool, optional
        If true, logs updates on estimated signal-to-noise ratio after each
        iteration. Default: False

    Returns
    -------
    fodf : 4d ndarray (x, y, z, M-1)
        fODF computed for each voxel.
    f_gm : 3d ndarray (x, y, z)
        GM volume fraction at each voxel.
    f_csf : 3d ndarray (x, y, z)
        CSF volume fraction at each voxel.
    f_wm : 3d ndarray (x, y, z)
        White matter volume fraction at each voxel.
    f_iso : 3d ndarray (x, y, z)
        Isotropic volume fraction at each voxel (GM + CSF)
    combined : 4d ndarray (x, y, z, M-1)
        fODF combined with isotropic compartment for each voxel.

    Notes
    -----
    TV modifies our cost function as follows:

    $ J(\bold{f}) = -\log{P(\bold{S}|\bold{H}, \bold{f}, \sigma^2, n)}) +
      \alpha_{TV}TV(\bold{f}) $

    where the first term is the negative log likelihood described in the notes
    of `rumba_deconv`, and the second term is the TV energy, or the sum of
    gradient absolute values for the fODF across the entire brain. This results
    in a new multiplicative factor in the iterative scheme, now becoming:

    $ \bold{f}^{k+1} = \bold{f}^k \circ \frac{\bold{H}^T\left[\bold{S}\circ
      \frac{I_n(\bold{S}\circ\bold{Hf}^k/\sigma^2)} {I_{n-1}(\bold{S}\circ
      \bold{Hf}^k/\sigma^2)} \right ]} {\bold{H}^T\bold{Hf}^k}\circ\bold{R}^k $

    where $\bold{R}^k$ is computed voxelwise by:

    $ (\bold{R}^k)_j = \frac{1}{1 - \alpha_{TV}div\left(\frac{\triangledown[
        \bold{f}^k_{3D}]_j}{\lvert\triangledown[\bold{f}^k_{3D}]_j \rvert}
        \right)\biggr\rvert_{x, y, z}} $

    Here, $\triangledown$ is the symbol for the 3D gradient at any voxel.

    The regularization strength, $\alpha_{TV}$ is updated after each iteration
    by the discrepancy principle -- specifically, it is selected to match the
    estimated variance after each iteration [3]_.

    References
    ----------
    .. [1] Canales-Rodríguez, E. J., Daducci, A., Sotiropoulos, S. N., Caruyer,
           E., Aja-Fernández, S., Radua, J., Mendizabal, J. M. Y.,
           Iturria-Medina, Y., Melie-García, L., Alemán-Gómez, Y., Thiran,
           J.-P., Sarró, S., Pomarol-Clotet, E., & Salvador, R. (2015).
           Spherical Deconvolution of Multichannel Diffusion MRI Data with
           Non-Gaussian Noise Models and Spatial Regularization. PLOS ONE,
           10(10), e0138910. https://doi.org/10.1371/journal.pone.0138910

    .. [2] Rudin, L. I., Osher, S., & Fatemi, E. (1992). Nonlinear total
           variation based noise removal algorithms. Physica D: Nonlinear
           Phenomena, 60(1), 259–268.
           https://doi.org/10.1016/0167-2789(92)90242-F

    .. [3] Chambolle A. An algorithm for total variation minimization and
           applications. Journal of Mathematical Imaging and Vision. 2004;
           20:89–97.
    '''

    # Crop data to reduce memory consumption
    dim_orig = data.shape
    ixmin, ixmax = bounding_box(mask)
    data = crop(data, ixmin, ixmax)
    mask = crop(mask, ixmin, ixmax)

    if np.any(np.array(data.shape[:3]) == 1) and use_tv:
        raise ValueError("Cannot use TV regularization if any spatial" +
                         "dimensions are 1; " +
                         f"provided dimensions were {data.shape[:3]}")

    epsilon = 1e-7

    n_grad = kernel.shape[0]  # gradient directions
    n_comp = kernel.shape[1]  # number of compartments
    dim = data.shape
    n_v_tot = np.prod(dim[:3])  # total number of voxels

    # Initial guess is iso-probable
    fodf0 = np.ones((n_comp, 1), dtype=np.float32)
    fodf0 = fodf0 / np.sum(fodf0, axis=0)

    if recon_type == "smf":
        n_order = 1  # Rician noise (same as Noncentral Chi with order 1)
    elif recon_type == "sos":
        n_order = n_coils  # Noncentral Chi noise (order = # of coils)
    else:
        raise ValueError("Invalid recon_type. Should be 'smf' or 'sos', " +
                         f"received f{recon_type}")

    mask_vec = np.ravel(mask)
    # Indices of target voxels
    index_mask = np.atleast_1d(np.squeeze(np.argwhere(mask_vec)))
    n_v_true = len(index_mask)  # number of target voxels

    data_2d = np.zeros((n_v_true, n_grad), dtype=np.float32)
    for i in range(n_grad):
        data_2d[:, i] = np.ravel(data[:, :, :, i])[
            index_mask]  # only keep voxels of interest

    data_2d = data_2d.T
    fodf = np.tile(fodf0, (1, n_v_true))
    reblurred = np.matmul(kernel, fodf)

    # For use later
    kernel_t = kernel.T
    f_zero = 0

    # Initialize algorithm parameters
    sigma0 = 1/15
    sigma2 = sigma0**2
    tv_lambda = sigma2  # initial guess for TV regularization strength

    # Expand into matrix form for iterations
    sigma2 = sigma2 * np.ones(data_2d.shape, dtype=np.float32)
    tv_lambda_aux = np.zeros((n_v_tot), dtype=np.float32)

    reblurred_s = data_2d * reblurred / sigma2

    for i in range(n_iter):
        fodf_i = fodf
        ratio = mbessel_ratio(n_order, reblurred_s).astype(np.float32)
        rl_factor = np.matmul(kernel_t, data_2d*ratio) / \
            (np.matmul(kernel_t, reblurred) + _EPS)

        if use_tv:  # apply TV regularization
            tv_factor = np.ones(fodf_i.shape, dtype=np.float32)
            fodf_4d = _reshape_2d_4d(fodf_i.T, mask)
            # Compute gradient, divergence
            gr = _grad(fodf_4d)
            d_inv = 1 / np.sqrt(epsilon**2 + np.sum(gr**2, axis=3))
            gr_norm = (gr * d_inv[:, :, :, None, :])
            div_f = _divergence(gr_norm)
            g0 = np.abs(1 - tv_lambda * div_f)
            tv_factor_4d = 1 / (g0 + _EPS)

            for j in range(n_comp):
                tv_factor_1d = np.ravel(tv_factor_4d[:, :, :, j])[index_mask]
                tv_factor[j, :] = tv_factor_1d

            # Apply TV regularization to iteration factor
            rl_factor = rl_factor * tv_factor

        fodf = fodf_i * rl_factor  # result of iteration
        fodf = np.maximum(f_zero, fodf)  # positivity constraint

        # Update other variables
        reblurred = np.matmul(kernel, fodf)
        reblurred_s = data_2d * reblurred / sigma2

        # Iterate variance
        sigma2_i = (1 / (n_grad * n_order)) * \
            np.sum((data_2d**2 + reblurred**2) / 2 - (
                sigma2 * reblurred_s) * ratio, axis=0)
        sigma2_i = np.minimum((1 / 8)**2, np.maximum(sigma2_i, (1 / 80)**2))

        if verbose:
            logger.info("Iteration %d of %d", i+1, n_iter)

            snr_mean = np.mean(1 / np.sqrt(sigma2_i))
            snr_std = np.std(1 / np.sqrt(sigma2_i))
            logger.info(
                "Mean SNR (S0/sigma) estimated to be %.3f +/- %.3f",
                snr_mean, snr_std)
        # Expand into matrix
        sigma2 = np.tile(sigma2_i[None, :], (data_2d.shape[0], 1))

        # Update TV regularization strength using the discrepancy principle
        if use_tv:
            if R == 1:
                tv_lambda = np.mean(sigma2_i)

                if tv_lambda < (1/30)**2:
                    tv_lambda = (1/30)**2
            else:  # different factor for each voxel
                tv_lambda_aux[index_mask] = sigma2_i
                tv_lambda = np.reshape(tv_lambda_aux, (*dim[:3], 1))

    fodf = fodf.astype(np.float64)
    fodf = fodf / (np.sum(fodf, axis=0)[None, ...] + _EPS)  # normalize fODF

    # Extract compartments
    fodf_4d = np.zeros((*dim_orig[:3], n_comp))
    _reshape_2d_4d(fodf.T, mask, out=fodf_4d[ixmin[0]:ixmax[0],
                                             ixmin[1]:ixmax[1],
                                             ixmin[2]:ixmax[2]])
    fodf = fodf_4d[:, :, :, :-2]  # WM compartment
    f_gm = fodf_4d[:, :, :, -2]  # GM compartment
    f_csf = fodf_4d[:, :, :, -1]  # CSF compartment
    f_wm = np.sum(fodf, axis=3)  # white matter volume fraction
    combined = fodf + (f_gm[..., None] + f_csf[..., None]) \
        / fodf.shape[3]
    f_iso = f_gm + f_csf

    return fodf, f_gm, f_csf, f_wm, f_iso, combined


def _grad(M):
    '''
    Computes one way first difference
    '''
    x_ind = list(range(1, M.shape[0])) + [M.shape[0]-1]
    y_ind = list(range(1, M.shape[1])) + [M.shape[1]-1]
    z_ind = list(range(1, M.shape[2])) + [M.shape[2]-1]

    grad = np.zeros((*M.shape[:3], 3, M.shape[-1]), dtype=np.float32)
    grad[:, :, :, 0, :] = M[x_ind, :, :, :] - M
    grad[:, :, :, 1, :] = M[:, y_ind, :, :] - M
    grad[:, :, :, 2, :] = M[:, :, z_ind, :] - M

    return grad


def _divergence(F):
    '''
    Computes divergence of a 3-dimensional vector field (with one way
    first difference)
    '''
    Fx = F[:, :, :, 0, :]
    Fy = F[:, :, :, 1, :]
    Fz = F[:, :, :, 2, :]

    x_ind = [0] + list(range(F.shape[0]-1))
    y_ind = [0] + list(range(F.shape[1]-1))
    z_ind = [0] + list(range(F.shape[2]-1))

    fx = Fx - Fx[x_ind, :, :, :]
    fx[0, :, :, :] = Fx[0, :, :, :]  # edge conditions
    fx[-1, :, :, :] = -Fx[-2, :, :, :]

    fy = Fy - Fy[:, y_ind, :, :]
    fy[:, 0, :, :] = Fy[:, 0, :, :]
    fy[:, -1, :, :] = -Fy[:, -2, :, :]

    fz = Fz - Fz[:, :, z_ind, :]
    fz[:, :, 0, :] = Fz[:, :, 0, :]
    fz[:, :, -1, :] = -Fz[:, :, -2, :]

    return fx + fy + fz


def _reshape_2d_4d(M, mask, out=None):
    if out is None:
        out = np.zeros((*mask.shape, M.shape[-1]), dtype=M.dtype)
    n = 0
    for i, j, k in np.ndindex(mask.shape):
        if mask[i, j, k]:
            out[i, j, k, :] = M[n, :]
            n += 1
    return out
