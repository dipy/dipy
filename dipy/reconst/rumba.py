'''Tools for using Robust and Unbiased Model-BAsed Spherical Deconvolution (RUMBA-SD)'''
import numpy as np
from numpy.matlib import repmat

from dipy.sims.voxel import multi_tensor
from dipy.core.geometry import cart2sphere
from dipy.reconst.shm import lazy_index, normalize_data
from dipy.core.gradients import gradient_table
from dipy.reconst.multi_voxel import multi_voxel_fit
from dipy.reconst.odf import OdfModel, OdfFit
from dipy.reconst.cache import Cache


# Machine precision for numerical stability in division
EPS = np.finfo(float).eps


class RumbaSD(OdfModel, Cache):

    def __init__(self, gtab, lambda1=1.7e-3, lambda2=0.2e-3, lambda_csf=3.0e-3, lambda_gm=0.8e-4,
                 n_iter=600, recon_type='smf', n_coils=1, R=1):
        # TODO: optional isotropic compartments
        '''
        Robust and Unbiased Model-BAsed Spherical Deconvolution (RUMBA-SD)

        Modification of the Richardson-Lucy algorithm accounting for Rician and Noncentral Chi
        noise distributions, which more accurately represent MRI noise. Computes a maximum
        likelihood estimation of the fiber orientation density function (fODF) at each voxel [1]_.
        Includes compartments for cerbrospinal fluid (CSF) and grey matter (GM) to account for
        partial volume effects.

        Kernel for deconvolution constructed using a priori knowledge of white matter diffusivity
        parallel and perpendicular to the fiber, as well as the mean diffusivity of CSF and GM.


        Parameters
        ----------
        gtab : GradientTable
        lambda1 : float, optional
            white matter diffusivity parallel to fiber axis (first DTI eigenvalue). Default: 1.7e-3
        lambda2 : float, optional
            white matter diffusivity perpendicular to fiber axis (second/third DTI eigenvalues are
            assumed equal). Default: 0.2e-3
        lambda_csf : float, optional
            mean diffusivity for CSF. Default: 3.0e-3
        lambda_gm : float, optional
            mean diffusivity for grey matter. Default: 0.8e-4
        n_iter : int, optional
            Number of iterations for fODF estimation. Must be a positive int. Default: 600
        recon_type : {'smf', 'sos'}, optional
            MRI reconstruction method: spatial matched filter (SMF) or sum-of-squares (SoS). SMF
            reconstruction generates Rician noise while SoS reconstruction generates Noncentral Chi
            noise. Default: 'smf'
        n_coils : int, optional
            Number of coils in MRI scanner -- only relevant in SoS reconstruction. Must be a
            positive int. Default: 1
        R : int, optional
            Acceleration factor of the acquisition. For SIEMENS, R = iPAT factor. For GE, R = ASSET
            factor. For PHILIPS, R = SENSE factor. Typical values are 1 or 2. Must be a positive
            int. Default: 1

        References
        ----------
        .. [1] Canales-Rodríguez, E. J., Daducci, A., Sotiropoulos, S. N., Caruyer, E.,
                Aja-Fernández, S., Radua, J., Mendizabal, J. M. Y., Iturria-Medina, Y.,
                Melie-García, L., Alemán-Gómez, Y., Thiran, J.-P., Sarró, S., Pomarol-Clotet, E.,
                & Salvador, R. (2015). Spherical Deconvolution of Multichannel Diffusion MRI Data
                with Non-Gaussian Noise Models and Spatial Regularization. PLOS ONE, 10(10),
                e0138910. https://doi.org/10.1371/journal.pone.0138910


        '''

        ## Correct gradient table to contain b0 data at the beginning ##

        if not np.any(gtab.b0s_mask):
            raise ValueError("Gradient table has no b0 measurements")

        # Masks to extract b0/non-b0 measurements
        self._where_b0s = lazy_index(gtab.b0s_mask)
        self._where_dwi = lazy_index(~gtab.b0s_mask)

        # Reconstruct gradient table
        bvals_cor = np.concatenate(([0], gtab.bvals[self._where_dwi]))
        bvecs_cor = np.concatenate(([[0, 0, 0]], gtab.bvecs[self._where_dwi]))
        gtab_cor = gradient_table(bvals_cor, bvecs_cor)

        # Initializes self.gtab
        OdfModel.__init__(self, gtab_cor)

        ## Store diffusivities ##
        if lambda1 < 0 or lambda2 < 0:
            raise ValueError(f"lambda1 and lambda2 must be > 0, received lambda1={lambda1}" +
                             f", lambda2={lambda2}")
        if lambda_csf is not None and lambda_csf < 0:
            raise ValueError(
                f"lambda_csf must be None or > 0, received {lambda_csf}")
        if lambda_gm is not None and lambda_gm < 0:
            raise ValueError(
                f"lambda_gm must be None or > 0, received {lambda_gm}")

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda_csf = lambda_csf
        self.lambda_gm = lambda_gm

        ## Initializing remaining parameters ##
        if R < 1 or n_iter < 1 or n_coils < 1:
            raise ValueError(f"R, n_iter, and n_coils must be >= 1, but R={R}, n_iter={n_iter}" +
                             f", and n_coils={n_coils} ")
        if recon_type not in ['smf', 'sos']:
            raise ValueError(
                f"Invalid recon_type. Should be 'smf' or 'sos', received {recon_type}")

        self.R = R
        self.n_iter = n_iter
        self.recon_type = recon_type
        self.n_coils = n_coils

    @multi_voxel_fit
    def fit(self, data):

        ## Signal repair, normalization ##

        # Normalize data to mean b0 image
        data = normalize_data(data, self._where_b0s, min_signal=EPS)
        # Rearrange data to match corrected gradient table
        data = np.concatenate(([1], data[self._where_dwi]))
        data[data > 1] = 1  # Clip values between 0 and 1

        return RumbaFit(self, data)


class RumbaFit(OdfFit):

    def __init__(self, model, data):
        '''
        Computes fODF for a single voxel

        Parameters
        ----------
        model : RumbaSD
            RUMBA-SD model.
        data : 1d ndarray
            signal values for a single voxel.

        '''
        OdfFit.__init__(self, model, data)
        self.kernel = None
        self._f_csf = None
        self._f_gm = None

        # TODO: are these necessary?
        self._gfa = None
        self.npeaks = 5
        self._peak_values = None
        self._peak_indices = None
        self._qa = None

    def odf(self, sphere):
        '''
        Computes fODF at discrete vertices on sphere.

        Parameters
        ----------
        sphere : Sphere
            sphere on which to construct fODF.

        Returns
        -------
        fodf_wm : 1d ndarray
            fODF computed at each vertex on `sphere`.

        '''

        ## Creation of kernel ##

        # Check if previously constructed
        self.kernel = self.model.cache_get('kernel', key=sphere)

        if self.kernel is None:
            self.kernel = generate_kernel(self.model.gtab, sphere, self.model.lambda1,
                                          self.model.lambda2, self.model.lambda_csf,
                                          self.model.lambda_gm)
            self.model.cache_set('kernel', sphere, self.kernel)

        ## Fitting ##
        fodf_wm, f_csf, f_gm = rumba_deconv(self.data, self.kernel,
                                            self.model.n_iter, self.model.recon_type,
                                            self.model.n_coils)

        self._f_csf = f_csf
        self._f_gm = f_gm

        # TODO: there was a comment about how the isotropic compartments only really work with
        #       multi-shell data. should there be a condition here that only adds the isotropic
        #       compartments if the data is multi-shell?

        return fodf_wm

    @property
    def f_gm(self):
        if self._f_gm is None:
            raise RuntimeError(
                "No fODF generated yet; call odf to generate grey matter volume fraction maps")
        return self._f_gm

    @property
    def f_csf(self):
        if self._f_csf is None:
            raise RuntimeError(
                "No fODF generated yet; call odf to generate CSF volume fraction maps")
        return self._f_csf


def rumba_deconv(data, kernel, n_iter=600, recon_type='smf', n_coils=1):
    '''
    Fit fODF for a single voxel using RUMBA-SD.

    Deconvolves the kernel from the diffusion-weighted signal by computing a maximum likelihood
    estimation of the fODF. Minimizes the negative log-likelihood of the data under Rician or
    Noncentral Chi noise distributions by adapting the iterative technique developed in the
    Richardson-Lucy algorithm [1]_.

    Parameters
    ----------
    data : 1d ndarray (N,)
        signal values for a single voxel.
    kernel : 2d ndarray (N, M)
        Deconvolution kernel mapping volume fractions of the M compartments to N-length signal.
        Last two columns should be for cerebrospinal fluid (CSF) and grey matter (GM) compartments
        respectively.
    n_iter : int, optional
        Number of iterations for fODF estimation. Must be a positive int. Default: 600
    recon_type : {'smf', 'sos'}, optional
        MRI reconstruction method: spatial matched filter (SMF) or sum-of-squares (SoS). SMF
        reconstruction generates Rician noise while SoS reconstruction generates Noncentral Chi
        noise. Default: 'smf'
    n_coils : int, optional
        Number of coils in MRI scanner -- only relevant in SoS reconstruction. Must be a
        positive int. Default: 1

    Returns
    -------
    fodf_wm : ndarray (M-2,)
        fODF for white matter compartments.
    f_csf : float
        volume fraction of CSF.
    f_gm : float
        volume fraction of GM.

    Notes
    -----
    The diffusion MRI signal measured at a given voxel is a sum of contributions from each
    intra-voxel compartment, including parallel white matter (WM) fiber populations in a given
    orientation as well as effects from grey matter (GM) and cerebrospinal fluid (CSF). The
    equation governing these contributions is:

    $ S_i = S_0\left(\sum_{j=1}^{M}f_j\exp(-b_i\bold{v}_i^T\bold{D}_j\bold{v}_i) + f_{GM}
      \exp(-b_iD_{GM})+f_{CSF}\exp(-b_iD_{CSF})\right) $

    Where $S_i$ is the resultant signal along the diffusion-sensitizing gradient unit vector
    $\bold{v_i}; i = 1, ..., N$ with a b-value of $b_i$. $f_j; j = 1, ..., M$ is the volume
    fraction of the $j^{th}$ fiber population with an anisotropic diffusion tensor $\bold{D_j}$.

    $f_{GM}$ and $f_{CSF}$ are the volume fractions and D_{GM} and D_{CSF} are the mean diffusivity
    coefficients of GM and CSF respectively.

    This equation is linear in $f_j, f_{GM}, f_{CSF}$ and can be simplified to a single matrix
    multiplication:

    $\bold{S} = \bold{Hf}$

    Where $\bold{S}$ is the signal vector at a certain voxel, $\bold{H}$ is the deconvolution
    kernel, and $\bold{f}$ is the vector of volume fractions for each compartment.

    Modern MRI scanners produce noise following a Rician or Noncentral Chi distribution, depending
    on their signal reconstruction technique [2]_. Using this linear model, it can be shown that
    the likelihood of a signal under a Noncentral Chi noise model is:

    $ P(\bold{S}|\bold{H}, \bold{f}, \sigma^2, n) = \prod_{i=1}^{N}\left(\frac{S_i}{\bar{S_i}}
      \right)^n\exp\left\{-\frac{1}{2\sigma^2}\left[S_i^2 + \bar{S}_i^2\right]\right\}I_{n-1}\left(
      \frac{S_i\bar{S}_i}{\sigma^2}\right)u(S_i) $

    Where $S_i$ and $\bar{S}_i = \bold{Hf}$ are the measured and expected signals respectively, and
    $n$ is the number of coils in the scanner, and $I_{n-1}$ is the modified Bessel function of
    first kind of order $n-1$. This gives the likelihood under a Rician distribution when $n$ is
    set to 1.

    By taking the negative log of this with respect to $\bold{f}$ and setting the derivative to 0,
    the $\bold{f}$ maxmizing likelihood is found to be:

    $ \bold{f} = \bold{f} \circ \frac{\bold{H}^T\left[\bold{S}\circ \frac{I_n(\bold{S}\circ
      \bold{Hf}/\sigma^2)} {I_{n-1}(\bold{S}\circ\bold{Hf}/\sigma^2)} \right ]} {\bold{H}^T
      \bold{Hf}} $

    The solution can be found using an iterative scheme, just as in the Richardson-Lucy algorithm:

    $ \bold{f}^{k+1} = \bold{f}^k \circ \frac{\bold{H}^T\left[\bold{S}\circ\frac{I_n(\bold{S}\circ
      \bold{Hf}^k/\sigma^2)} {I_{n-1}(\bold{S}\circ\bold{Hf}^k/\sigma^2)} \right ]} {\bold{H}^T
      \bold{Hf}^k} $

    In order to apply this, a reasonable estimate of $\sigma^2$ is required. To find this, a
    separate iterative scheme is found using the derivative of the negative log with respect to
    $\sigma^2$, and is run in parallel. This is shown here:

    $ \alpha^{k+1} = \frac{1}{nN}\left\{ \frac{\bold{S}^T\bold{S} + \bold{f}^T\bold{H}^T\bold{Hf}}
    {2} - \bold{1}^T_N\left[(\bold{S}\circ\bold{Hf}) \circ\frac{I_n(\bold{S}\circ\bold{Hf}/\alpha^k
    )}{I_{n-1}(\bold{S}\circ\bold{Hf}/\alpha^k)} \right ]\right \} $

    For more details, see [1]_.

    References
    ----------
    .. [1] Canales-Rodríguez, E. J., Daducci, A., Sotiropoulos, S. N., Caruyer, E.,
            Aja-Fernández, S., Radua, J., Mendizabal, J. M. Y., Iturria-Medina, Y.,
            Melie-García, L., Alemán-Gómez, Y., Thiran, J.-P., Sarró, S., Pomarol-Clotet, E.,
            & Salvador, R. (2015). Spherical Deconvolution of Multichannel Diffusion MRI Data
            with Non-Gaussian Noise Models and Spatial Regularization. PLOS ONE, 10(10),
            e0138910. https://doi.org/10.1371/journal.pone.0138910

    .. [2] Constantinides, C. D., Atalar, E., & McVeigh, E. R. (1997). Signal-to-Noise
            Measurements in Magnitude Images from NMR Phased Arrays. Magnetic Resonance in
            Medicine : Official Journal of the Society of Magnetic Resonance in Medicine / Society
            of Magnetic Resonance in Medicine, 38(5), 852–857.
    '''

    n_c = kernel.shape[1]  # number of compartments
    n_g = len(data)  # gradient directions

    fodf = np.ones((n_c, 1))  # initial guess is iso-probable
    fodf = fodf / np.sum(fodf, axis=0)  # normalize initial guess

    if recon_type == "smf":
        n_order = 1  # Rician noise (same as Noncentral Chi with order 1)
    elif recon_type == "sos":
        n_order = n_coils  # Noncentral Chi noise (order = # of coils)
    else:
        raise ValueError(
            f"Invalid recon_type. Should be 'smf' or 'sos', received {recon_type}")

    data = data.reshape(-1, 1)
    reblurred = np.matmul(kernel, fodf)

    # For use later
    kernel_t = kernel.T
    f_zero = 0  # minimum value allowed in fODF

    # Initialize variance
    sigma0 = 1/15
    sigma2 = sigma0**2
    sigma2 = sigma2 * np.ones(data.shape)  # expand into vector

    reblurred_s = data * reblurred / sigma2

    for _ in range(n_iter):
        fodf_i = fodf
        ratio = mbessel_ratio(n_order, reblurred_s)
        rl_factor = np.matmul(kernel_t, data*ratio) / \
            (np.matmul(kernel_t, reblurred) + EPS)

        fodf = fodf_i * rl_factor  # result of iteration
        fodf = np.maximum(f_zero, fodf)  # apply positivity constraint

        # Update other variables
        reblurred = np.matmul(kernel, fodf)
        reblurred_s = data * reblurred / sigma2

        # Iterate variance
        sigma2_i = (1/(n_g * n_order)) * np.sum((data**2 + reblurred **
                                                 2) / 2 - (sigma2 * reblurred_s) * ratio, axis=0)
        sigma2_i = np.minimum((1/8)**2, np.maximum(sigma2_i, (1/80)**2))
        sigma2 = sigma2_i * np.ones(data.shape)  # Expand into vector

    fodf = fodf / (np.sum(fodf, axis=0) + EPS)  # normalize final result

    fodf_wm = np.squeeze(fodf[:n_c-2])  # white matter components
    f_csf = fodf[n_c-2]  # CSF component
    f_gm = fodf[n_c-1]  # grey matter component

    return fodf_wm, f_csf, f_gm


def mbessel_ratio(n, x):
    '''
    Fast computation of modified Bessel function ratio (first kind).

    Computes:

    $I_{n}(x) / I_{n-1}(x)$

    using Perron's continued fraction equation where $I_n$ is the modified Bessel function of first
    kind, order $n$ [1]_.

    Parameters
    ----------
    n : int
        Order of Bessel function in numerator (denominator is of order n-1). Must be a positive
        int.
    x : float or ndarray
        value or array of values with which to compute ratio

    Returns
    -------
    y : float or ndarray
        Result of ratio computation.

    References
    ----------
    .. [1] W. Gautschi and J. Slavik, “On the computation of modified Bessel function ratios,”
           Math. Comp., vol. 32, no. 143, pp. 865–875, 1978, doi: 10.1090/S0025-5718-1978-0470267-9
    '''

    y = x / ((2*n + x) - (2*x * (n + 1/2) / (2*n + 1 + 2*x - (2*x * (n + 3/2) / (
        2*n + 2 + 2*x - (2*x * (n + 5/2) / (2*n + 3 + 2*x)))))))

    return y


def generate_kernel(gtab, sphere, lambda1=1.7e-3, lambda2=0.2e-3, lambda_csf=3.0e-3,
                    lambda_gm=0.8e-4):
    '''
    Generate deconvolution kernel

    Compute kernel mapping volume fractions of white matter fiber populations (oriented along each
    vertex of a sphere), cerebrospinal fluid (CSF) and grey matter (GM) to a diffusion weighted
    signal.

    Parameters
    ----------
    gtab : GradientTable
    sphere : Sphere
        sphere with which to sample discrete fiber orientations in order to construct kernel
    lambda1 : float, optional
        white matter diffusivity parallel to fiber axis (first DTI eigenvalue). Default: 1.7e-3
    lambda2 : float
        white matter diffusivity perpendicular to fiber axis (second/third DTI eigenvalues are
        assumed equal). Default: 0.2e-3
    lambda_csf : float
        mean diffusivity for CSF. Default: 3.0e-3
    lambda_gm : float
        mean diffusivity for grey matter. Default: 0.8e-3

    Returns
    -------
    kernel : ndarray (N, M)
        computed kernel; can be multiplied with a vector consisting of volume fractions for each of
        M-2 fiber populations and fractions for CSF and GM to produce a diffusion weighted signal.
    '''

    # Coordinates of sphere vertices
    _, theta, phi = cart2sphere(
        sphere.x,
        sphere.y,
        sphere.z
    )

    n_grad = len(gtab.gradients)  # number of gradient directions
    n_wm_comp = len(theta)  # number of fiber populations
    n_comp = n_wm_comp + 2  # plus compartments for GM and for CSF

    kernel = np.zeros((n_grad, n_comp))

    S0 = 1  # S0 assumed to be 1
    fi = 100  # volume fraction assumed to be 100%

    # White matter components
    for i in range(n_wm_comp):
        # Convert angles to degrees
        angles = [theta[i] * 180 / np.pi, phi[i] * 180 / np.pi]

        # Signal generated by WM-fiber for each gradient direction
        S, _ = multi_tensor(gtab, np.array([[lambda1, lambda2, lambda2]]),
                            S0, [angles], [fi], None)
        kernel[:, i] = S

    # CSF and GM components
    S_csf, _ = multi_tensor(gtab, np.array([[lambda_csf, lambda_csf, lambda_csf]]),
                            S0, [[0, 0]], [100], None)
    S_gm, _ = multi_tensor(gtab, np.array([[lambda_gm, lambda_gm, lambda_gm]]),
                           S0, [[0, 0]], [100], None)

    kernel[:, n_wm_comp] = S_csf
    kernel[:, n_wm_comp + 1] = S_gm

    return kernel


'''
==============================================================
======================= GLOBAL FITTING =======================
==============================================================
'''


def global_fit(model, data, sphere, mask=None, use_tv=True, verbose=False):
    '''
    Computes fODF for all voxels simultaneously

    Simultaneous computation parallelizes estimation and also permits spatial regularization
    via total variation.

    Parameters
    ----------
    model : RumbaSD
        RUMBA-SD model.
    data : 4d ndarray (x, y, z, N)
        Signal values for entire brain. (x, y, z) represent brain volume, while N is the number of
        gradient directions.
    sphere : Sphere
        sphere on which to construct fODF.
    mask : 3d ndarray (x, y, z), optional
        Binary mask specifying voxels of interest with 1; fODF will only be fit at these voxel
        (0 elsewhere). `None` generates a mask of all 1s. Default: None
    use_tv : bool, optional
        if true, applies total variation regularization. Default: True
    verbose : bool, optional
        if true, fit prints updates on estimated signal-to-noise ratio after each iteration.
        Default: False

    Returns
    -------
    fodf_wm : ndarray (x, y, z, K)
        fODF computed for each voxel, where K is the number of vertices on `sphere`
    f_csf : 3d ndarray (x, y, z)
        volume fraction of CSF at each voxel.
    f_gm : 3d ndarray (x, y, z)
        volume fraction of GM at each voxel.
    '''

    ## Checking data and mask shapes ##

    if len(data.shape) != 4:
        raise ValueError(f"Data should be 4D, received shape f{data.shape}")

    ## Signal repair, normalization ##

    # Normalize data to mena b0 image
    data = normalize_data(data, model._where_b0s, EPS)
    # Rearrange data to match corrected gradient table
    data = np.concatenate(
        (np.ones([*data.shape[:3], 1]), data[..., model._where_dwi]), axis=3)
    data[data > 1] = 1  # clip values between 0 and 1

    if mask is None:  # default mask includes all voxels
        mask = np.ones(data.shape[:3])

    if data.shape[:3] != mask.shape:
        raise ValueError("Mask shape should match first 3 dimensions of data, but data " +
                         f"dimensions are f{data.shape} while mask dimensions are f{mask.shape}")

    # Generate kernel
    kernel = generate_kernel(model.gtab, sphere, model.lambda1, model.lambda2,
                             model.lambda_csf, model.lambda_gm)

    # Fit fODF
    fodf_wm, f_csf, f_gm = rumba_deconv_global(data, kernel, mask, model.n_iter, model.recon_type,
                                               model.n_coils, model.R, use_tv, verbose)

    return fodf_wm, f_csf, f_gm


def rumba_deconv_global(data, kernel, mask, n_iter=600, recon_type='smf',
                        n_coils=1, R=1, use_tv=True, verbose=False):
    '''
    Fit fODF for a all voxels simultaneously using RUMBA-SD.

    Deconvolves the kernel from the diffusion-weighted signal at each voxel by computing a
    maximum likelihood estimation of the fODF [1]_. Global fitting also permits the use of total
    variation (TV) regularization. The spatial dependence introduced by TV promotes smoother
    solutions (i.e. prevents oscillations), while still allowing for sharp discontinuities [2]_.
    This promots smoothness and continuity along individual tracts while preventing smoothing of
    adjacent tracts.

    Parameters
    ----------
    data : 4d ndarray (x, y, z, N)
        Signal values for entire brain. None of the volume dimensions x, y, z can be 1 if TV
        regularization is required.
    kernel : 2d ndarray (N, M)
        Deconvolution kernel mapping volume fractions of the M compartments to N-length signal.
        Last two columns should be for cerebrospinal fluid (CSF) and grey matter (GM) compartments
        respectively.
    mask : 3d ndarray(x, y, z)
        Binary mask specifying voxels of interest with 1; fODF will only be fit at these voxel
        (0 elsewhere).
    n_iter : int, optional
        Number of iterations for fODF estimation. Must be a positive int. Default: 600
    recon_type : {'smf', 'sos'}, optional
        MRI reconstruction method: spatial matched filter (SMF) or sum-of-squares (SoS). SMF
        reconstruction generates Rician noise while SoS reconstruction generates Noncentral Chi
        noise. Default: 'smf'
    n_coils : int, optional
        Number of coils in MRI scanner -- only relevant in SoS reconstruction. Must be a
        positive int. Default: 1
    use_tv : bool, optional
        if true, applies total variation regularization. Default: True
    verbose : bool, optional
        if true, fit prints updates on estimated signal-to-noise ratio after each iteration.
        Default: False

    Returns
    -------
    fodf_wm : 4d ndarray (x, y, z, M-2)
        fODF for white matter compartments at each voxel.
    f_csf : 3d ndarray (x, y, z)
        volume fraction of CSF at each voxel.
    f_gm : 3d ndarray (x, y, z)
        volume fraction of GM at each voxel.

    Notes
    -----
    TV modifies our cost function as follows:

    $ J(\bold{f}) = -\log{P(\bold{S}|\bold{H}, \bold{f}, \sigma^2, n)}) + \alpha_{TV}TV(\bold{f}) $

    where the first term is the negative log likelihood described in the notes of `rumba_deconv`,
    and the second term is the TV energy, or the sum of gradient absolute values for the fODF
    across the entire brain. This results in a new multiplicative factor in the iterative scheme,
    now becoming: 

    $ \bold{f}^{k+1} = \bold{f}^k \circ \frac{\bold{H}^T\left[\bold{S}\circ\frac{I_n(\bold{S}\circ
      \bold{Hf}^k/\sigma^2)} {I_{n-1}(\bold{S}\circ\bold{Hf}^k/\sigma^2)} \right ]} {\bold{H}^T
      \bold{Hf}^k}\circ\bold{R}^k $

    where $\bold{R}^k$ is computed voxelwise by:

    $ (\bold{R}^k)_j = \frac{1}{1 - \alpha_{TV}div\left(\frac{\triangledown[\bold{f}^k_{3D}]_j}
      {\lvert\triangledown[\bold{f}^k_{3D}]_j \rvert} \right )\biggr\rvert_{x, y, z}} $

    Here, $\triangledown$ is the gradient symbol for the 3D gradient at any voxel.

    The regularization strength, $\alpha_{TV}$ is updated after each iteration by the discrepancy
    principle -- specifically, it is selected to match the estimated variance after each iteration
    [3]_.

    References
    ----------
    .. [1] Canales-Rodríguez, E. J., Daducci, A., Sotiropoulos, S. N., Caruyer, E., Aja-Fernández,
           S., Radua, J., Mendizabal, J. M. Y., Iturria-Medina, Y., Melie-García, L., Alemán-Gómez,
           Y., Thiran, J.-P., Sarró, S., Pomarol-Clotet, E., & Salvador, R. (2015). Spherical
           Deconvolution of Multichannel Diffusion MRI Data with Non-Gaussian Noise Models and
           Spatial Regularization. PLOS ONE, 10(10), e0138910.
           https://doi.org/10.1371/journal.pone.0138910

    .. [2] Rudin, L. I., Osher, S., & Fatemi, E. (1992). Nonlinear total variation based noise
           removal algorithms. Physica D: Nonlinear Phenomena, 60(1), 259–268.
           https://doi.org/10.1016/0167-2789(92)90242-F

    .. [3] Chambolle A. An algorithm for total variation minimization and applications. Journal of
           Mathematical Imaging and Vision. 2004; 20:89–97.
    '''

    if np.any(data.shape[:3] == 1) and use_tv:
        raise ValueError("Cannot use TV regularization if any spatial dimensions are 1; " +
                         f"provided dimensions were {data.shape[:3]}")

    epsilon = 1e-7

    n_g = kernel.shape[0]  # gradient directions
    n_c = kernel.shape[1]  # number of components
    dim = data.shape
    n_v_tot = np.prod(dim[:3])  # total number of voxels

    fodf0 = np.ones((n_c, 1))  # initial guess is iso-probable
    fodf0 = fodf0 / np.sum(fodf0, axis=0)

    if recon_type == "smf":
        n_order = 1  # Rician noise (same as Noncentral Chi with order 1)
    elif recon_type == "sos":
        n_order = n_coils  # Noncentral Chi noise (order = # of coils)
    else:
        raise ValueError(
            f"Invalid recon_type. Should be 'smf' or 'sos', received f{recon_type}")

    mask_vec = np.ravel(mask, order='F')
    index_mask = np.squeeze(np.argwhere(mask_vec))  # indices of target voxels
    n_v_true = len(index_mask)  # number of target voxels

    data_2d = np.zeros((n_v_true, n_g), dtype=np.float32)
    for i in range(n_g):
        data_2d[:, i] = np.ravel(data[:, :, :, i], order='F')[
            index_mask]  # only keep voxels of interest

    data_2d = data_2d.T
    fodf = repmat(fodf0, 1, n_v_true)
    reblurred = np.matmul(kernel, fodf)

    # For use later
    kernel_T = kernel.T
    f_zero = 0

    # Initialize algorithm parameters
    sigma0 = 1/15
    sigma2 = sigma0**2
    tv_lambda = sigma2  # initial guess for TV regularization strength

    # Expand into matrix form for iterations
    sigma2 = sigma2 * np.ones(data_2d.shape)
    tv_lambda_aux = np.zeros((n_v_tot, 1), dtype=np.float32)

    reblurred_s = data_2d * reblurred / sigma2

    for i in range(n_iter):
        fodf_i = fodf
        ratio = mbessel_ratio(n_order, reblurred_s)
        rl_factor = np.matmul(kernel_T, data_2d*ratio) / \
            (np.matmul(kernel_T, reblurred) + EPS)

        if use_tv:  # apply TV regularization
            tv_factor = np.ones(fodf_i.shape, dtype=np.float32)

            for j in range(n_c):

                fodf_jv = np.zeros((n_v_tot, 1), dtype=np.float32)
                fodf_jv[index_mask, 0] = np.squeeze(
                    fodf_i[j, :])  # zeros at non-target voxels
                fodf_3d = fodf_jv.reshape((dim[0], dim[1], dim[2]), order='F')

                # stack x, y, and z gradients
                gr = np.stack(np.gradient(fodf_3d), axis=-1)
                d = np.sqrt(np.sum(gr**2, axis=3))
                d = np.sqrt(epsilon**2 + d**2)
                div_f = divergence(gr / np.stack((d, d, d), axis=-1))
                g0 = np.abs(1 - tv_lambda * div_f)
                tv_factor_3d = 1 / (g0 + EPS)
                tv_factor_1d = np.ravel(tv_factor_3d, order='F')[index_mask]
                tv_factor[j, :] = tv_factor_1d

            rl_factor = rl_factor * tv_factor  # apply TV regularization to iteration factor

        fodf = fodf_i * rl_factor  # result of iteration
        fodf = np.maximum(f_zero, fodf)  # positivity constraint

        # Update other variables
        reblurred = np.matmul(kernel, fodf)
        reblurred_s = data_2d * reblurred / sigma2

        # Iterate variance
        sigma2_i = (1/(n_g * n_order)) * np.sum((data_2d**2 + reblurred**2) / 2 - (
            sigma2 * reblurred_s) * ratio, axis=0)
        sigma2_i = np.minimum((1/8)**2, np.maximum(sigma2_i, (1/80)**2))

        if verbose:
            print(f"Iteration {i+1} of {n_iter}")

            snr_mean = np.mean(1 / np.sqrt(sigma2_i))
            snr_std = np.std(1 / np.sqrt(sigma2_i))
            print(
                f"Mean SNR (S0/sigma) estimated to be {snr_mean} +/- {snr_std}")

            # TODO: are these updates useful?
            print(f"Reconstruction using {n_order} coils")
            mean_fodf = np.mean(np.sum(fodf, axis=0))
            print(f"Mean fODF: {mean_fodf}")

        sigma2 = repmat(sigma2_i, data_2d.shape[0], 1)  # expand into matrix

        if use_tv:  # update TV regularization strength using the discrepancy principle
            if R == 1:
                tv_lambda = np.mean(sigma2_i)

                if tv_lambda < (1/30)**2:
                    tv_lambda = (1/30)**2
            else:  # different factor for each voxel
                tv_lambda_aux[index_mask] = sigma2_i
                tv_lambda = np.reshape(tv_lambda_aux, dim[:3], order='F')

    fodf = fodf / (np.sum(fodf, axis=0)[None, ...] + EPS)  # normalize fODF

    # Extract WM components
    fodf_wm = np.zeros([*dim[:3], n_c-2], np.float32)
    for i in range(n_c - 2):
        f_tmp = np.zeros((n_v_tot, 1), dtype=np.float32)
        f_tmp[index_mask, 0] = fodf[i, :]
        fodf_wm[:, :, :, i] = np.reshape(f_tmp, dim[:3], order='F')

    # Extract isotropic components
    f_tmp = np.zeros((n_v_tot, 1), dtype=np.float32)
    f_tmp[index_mask, 0] = fodf[n_c-2, :]
    f_csf = np.reshape(f_tmp, dim[:3], order='F')  # CSF volume fraction

    f_tmp = np.zeros((n_v_tot, 1), dtype=np.float32)
    f_tmp[index_mask, 0] = fodf[n_c-1, :]
    f_gm = np.reshape(f_tmp, dim[:3], order='F')  # GM volume fraction

    return fodf_wm, f_csf, f_gm


def divergence(F):
    '''
    Compute divergence of 3-dimensional vector field

    Parameters
    ----------
    F : 4d ndarray (x, y, z, 3)
        3-dimensional vector field split into x, y, and z components (i.e. stack of x-volume,
        y-volume, z-volume)

    Returns
    -------
    div : ndarray (x, y, z)
        divergence at each point in vector field

    Notes
    -----
    The formula for divergence of a vector field is as follows:

    $ \text{div} (\bold{F}) = \triangledown \cdot \bold{F} = \frac{\partial{F_x}}{\partial{x}}
      + \frac{\partial{F_y}}{\partial{y}} + \frac{\partial{F_z}}{\partial{z}} $
    '''
    Fx = F[:, :, :, 0]  # x-component of vector field
    Fy = F[:, :, :, 1]  # y-component
    Fz = F[:, :, :, 2]  # z-component

    fx = np.gradient(Fx, axis=0)
    fy = np.gradient(Fy, axis=1)
    fz = np.gradient(Fz, axis=2)

    div = fx + fy + fz

    return div
