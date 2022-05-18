"""Robust and Unbiased Model-BAsed Spherical Deconvolution (RUMBA-SD)"""
import logging
import warnings

import numpy as np

from dipy.core.geometry import vec2vec_rotmat
from dipy.core.gradients import gradient_table, unique_bvals_tolerance, \
    get_bval_indices
from dipy.core.onetime import auto_attr
from dipy.core.sphere import Sphere
from dipy.data import get_sphere
from dipy.reconst.shm import lazy_index, normalize_data
from dipy.reconst.odf import OdfModel, OdfFit
from dipy.reconst.csdeconv import AxSymShResponse
from dipy.segment.mask import bounding_box, crop
from dipy.sims.voxel import single_tensor, all_tensor_evecs

# Machine precision for numerical stability in division
_EPS = np.finfo(float).eps
logger = logging.getLogger(__name__)


class RumbaSDModel(OdfModel):

    def __init__(self, gtab, wm_response=np.array([1.7e-3, 0.2e-3, 0.2e-3]),
                 gm_response=0.8e-3, csf_response=3.0e-3, n_iter=600,
                 recon_type='smf', n_coils=1, R=1, voxelwise=True,
                 use_tv=False, sphere=None, verbose=False):
        """
        Robust and Unbiased Model-BAsed Spherical Deconvolution (RUMBA-SD) [1]_

        Modification of the Richardson-Lucy algorithm accounting for Rician
        and Noncentral Chi noise distributions, which more accurately
        represent MRI noise. Computes a maximum likelihood estimation of the
        fiber orientation density function (fODF) at each voxel. Includes
        white matter compartments alongside optional GM and CSF compartments
        to account for partial volume effects. This fit can be performed
        voxelwise or globally. The global fit will proceed more quickly than
        the voxelwise fit provided that the computer has adequate RAM (>= 16 GB
        should be sufficient for most datasets).

        Kernel for deconvolution constructed using a priori knowledge of white
        matter response function, as well as the mean diffusivity of GM and/or
        CSF. RUMBA-SD is robust against impulse response imprecision, and thus
        the default diffusivity values are often adequate [2]_.


        Parameters
        ----------
        gtab : GradientTable
        wm_response : 1d ndarray or 2d ndarray or AxSymShResponse, optional
            Tensor eigenvalues as a (3,) ndarray, multishell eigenvalues as
            a (len(unique_bvals_tolerance(gtab.bvals))-1, 3) ndarray in
            order of smallest to largest b-value, or an AxSymShResponse.
            Default: np.array([1.7e-3, 0.2e-3, 0.2e-3])
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
        voxelwise : bool, optional
            If true, performs a voxelwise fit. If false, performs a global fit
            on the entire brain at once. The global fit requires a 4D brain
            volume in `fit`. Default: True
        use_tv : bool, optional
            If true, applies total variation regularization. This only takes
            effect in a global fit (`voxelwise` is set to `False`). TV can only
            be applied to 4D brain volumes with no singleton dimensions.
            Default: False
        sphere : Sphere, optional
            Sphere on which to construct fODF. If None, uses `repulsion724`.
            Default: None
        verbose : bool, optional
            If true, logs updates on estimated signal-to-noise ratio after each
            iteration. This only takes effect in a global fit (`voxelwise` is
            set to `False`). Default: False

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


        """

        if not np.any(gtab.b0s_mask):
            raise ValueError("Gradient table has no b0 measurements")

        self.gtab_orig = gtab  # save for prediction

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

        if voxelwise and use_tv:
            raise ValueError("Total variation has no effect in voxelwise fit")
        if voxelwise and verbose:
            warnings.warn("Verbosity has no effect in voxelwise fit",
                          UserWarning)

        self.voxelwise = voxelwise
        self.use_tv = use_tv

        self.verbose = verbose

        if sphere is None:
            self.sphere = get_sphere('repulsion724')
        else:
            self.sphere = sphere

        if voxelwise:
            self.fit = self._voxelwise_fit
        else:
            self.fit = self._global_fit

        # Fitting parameters
        self.kernel = None

    def _global_fit(self, data, mask=None):
        """
        Fit fODF and GM/CSF volume fractions globally.

        Parameters
        ----------
        data : ndarray (x, y, z, N)
            Signal values for each voxel. Must be 4D.
        mask : ndarray (x, y, z), optional
            Binary mask specifying voxels of interest with 1; results will only
            be fit at these voxels (0 elsewhere). If `None`, fits all voxels.
            Default: None.

        Returns
        -------
        model_fit : RumbaFit
            Fit object storing model parameters.

        """

        # Checking data and mask shapes
        if len(data.shape) != 4:
            raise ValueError(
                f"Data should be 4D, received shape f{data.shape}")

        if mask is None:  # default mask includes all voxels
            mask = np.ones(data.shape[:3])

        if data.shape[:3] != mask.shape:
            raise ValueError("Mask shape should match first 3 dimensions of "
                             + f"data, but data dimensions are f{data.shape} "
                             + f"while mask dimensions are f{mask.shape}")

        # Signal repair, normalization

        # Normalize data to mean b0 image
        data = normalize_data(data, self.where_b0s, _EPS)
        # Rearrange data to match corrected gradient table
        data = np.concatenate(
            (np.ones([*data.shape[:3], 1]), data[..., self.where_dwi]), axis=3)
        data[data > 1] = 1  # clip values between 0 and 1

        # All arrays are converted to float32 to reduce memory load
        data = data.astype(np.float32)

        # Generate kernel
        self.kernel = generate_kernel(self.gtab, self.sphere, self.wm_response,
                                      self.gm_response, self.csf_response
                                      ).astype(np.float32)

        # Fit fODF
        model_params = rumba_deconv_global(data, self.kernel, mask,
                                           self.n_iter,
                                           self.recon_type,
                                           self.n_coils,
                                           self.R, self.use_tv,
                                           self.verbose)

        model_fit = RumbaFit(self, model_params)
        return model_fit

    def _voxelwise_fit(self, data, mask=None):
        """
        Fit fODF and GM/CSF volume fractions voxelwise.

        Parameters
        ----------
        data : ndarray ([x, y, z], N)
            Signal values for each voxel.
        mask : ndarray ([x, y, z]), optional
            Binary mask specifying voxels of interest with 1; results will only
            be fit at these voxels (0 elsewhere). If `None`, fits all voxels.
            Default: None.

        Returns
        -------
        model_fit : RumbaFit
            Fit object storing model parameters.

        """

        if mask is None:  # default mask includes all voxels
            mask = np.ones(data.shape[:-1])

        if data.shape[:-1] != mask.shape:
            raise ValueError("Mask shape should match first dimensions of "
                             + f"data, but data dimensions are f{data.shape} "
                             + f"while mask dimensions are f{mask.shape}")

        self.kernel = generate_kernel(self.gtab,
                                      self.sphere,
                                      self.wm_response,
                                      self.gm_response,
                                      self.csf_response)

        model_params = np.zeros(
            data.shape[:-1] + (len(self.sphere.vertices) + 2,))

        for ijk in np.ndindex(data.shape[:-1]):
            if mask[ijk]:

                vox_data = data[ijk]
                # Normalize data to mean b0 image
                vox_data = normalize_data(vox_data, self.where_b0s,
                                          min_signal=_EPS)
                # Rearrange data to match corrected gradient table
                vox_data = np.concatenate(([1], vox_data[self.where_dwi]))
                vox_data[vox_data > 1] = 1  # clip values between 0 and 1

                # Fitting
                model_param = rumba_deconv(vox_data,
                                           self.kernel,
                                           self.n_iter,
                                           self.recon_type,
                                           self.n_coils)

                model_params[ijk] = model_param

        model_fit = RumbaFit(self, model_params)
        return model_fit


class RumbaFit(OdfFit):

    def __init__(self, model, model_params):
        """
        Constructs fODF, GM/CSF volume fractions, and other derived results.

        fODF and GM/CSF fractions are normalized to collectively sum to 1 for
        each voxel.

        Parameters
        ----------
        model : RumbaSDModel
            RumbaSDModel-SD model.
        model_params : ndarray ([x, y, z], M)
            fODF and GM/CSF volume fractions for each voxel.

        """

        self.model = model
        self.model_params = model_params

    def odf(self, sphere=None):
        """
        Constructs fODF at discrete vertices on model sphere for each voxel.

        Parameters
        ----------
        sphere : Sphere, optional
            Sphere on which to construct fODF. If specified, must be the same
            sphere used by the `RumbaSDModel` model. Default: None.

        Returns
        -------
        odf : ndarray ([x, y, z], M-2)
            fODF computed at each vertex on model sphere.

        """
        if sphere is not None and sphere != self.model.sphere:
            raise ValueError("Reconstruction sphere must be the same as used"
                             + " in the RUMBA-SD model.")

        odf = self.model_params[..., :-2]
        return odf

    @auto_attr
    def f_gm(self):
        """
        Constructs GM volume fraction for each voxel.

        Returns
        -------
        f_gm : ndarray ([x, y, z])
            GM volume fraction.
        """

        f_gm = self.model_params[..., -2]
        return f_gm

    @auto_attr
    def f_csf(self):
        """
        Constructs CSF volume fraction for each voxel.

        Returns
        -------
        f_csf : ndarray ([x, y, z])
            CSF volume fraction.
        """

        f_csf = self.model_params[..., -1]
        return f_csf

    @auto_attr
    def f_wm(self):
        """
        Constructs white matter volume fraction for each voxel.

        Equivalent to sum of fODF.

        Returns
        -------
        f_wm : ndarray ([x, y, z])
            White matter volume fraction.
        """

        f_wm = np.sum(self.odf(), axis=-1)
        return f_wm

    @auto_attr
    def f_iso(self):
        """
        Constructs isotropic volume fraction for each voxel.

        Equivalent to sum of GM and CSF volume fractions.

        Returns
        -------
        f_iso : ndarray ([x, y, z])
            Isotropic volume fraction.
        """

        f_iso = self.f_gm + self.f_csf
        return f_iso

    @auto_attr
    def combined_odf_iso(self):
        """
        Constructs fODF combined with isotropic volume fraction at discrete
        vertices on model sphere.

        Distributes isotropic compartments evenly along each fODF direction.
        Sums to 1.

        Returns
        -------
        combined : ndarray ([x, y, z], M-2)
            fODF combined with isotropic volume fraction.
        """

        odf = self.odf()
        combined = odf + self.f_iso[..., None] / odf.shape[-1]
        return combined

    def predict(self, gtab=None, S0=None):
        """
        Compute signal prediction on model gradient table given given fODF
        and GM/CSF volume fractions for each voxel.

        Parameters
        ----------
        gtab : GradientTable, optional
            The gradients for which the signal will be predicted. Use the
            model's gradient table if `None`. Default: None
        S0 : ndarray ([x, y, z]) or float, optional
            The non diffusion-weighted signal value for each voxel. If a float,
            the same value is used for each voxel. If `None`, 1 is used for
            each voxel. Default: None

        Returns
        -------
        pred_sig : ndarray ([x, y, z], N)
            The predicted signal.

        """
        model_params = self.model_params

        if gtab is None:
            gtab = self.model.gtab_orig

        pred_kernel = generate_kernel(gtab,
                                      self.model.sphere,
                                      self.model.wm_response,
                                      self.model.gm_response,
                                      self.model.csf_response)

        if S0 is None:
            S0 = np.ones(model_params.shape[:-1] + (1,))
        elif isinstance(S0, np.ndarray):
            S0 = S0[..., None]
        else:
            S0 = S0 * np.ones(model_params.shape[:-1] + (1,))

        pred_sig = np.empty(model_params.shape[:-1] + (len(gtab.bvals),))
        for ijk in np.ndindex(model_params.shape[:-1]):
            pred_sig[ijk] = S0[ijk] * np.dot(pred_kernel, model_params[ijk])

        return pred_sig


def rumba_deconv(data, kernel, n_iter=600, recon_type='smf', n_coils=1):
    r"""
    Fit fODF and GM/CSF volume fractions for a voxel using RUMBA-SD [1]_.

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
    fit_vec : 1d ndarray (M,)
        Vector containing fODF and GM/CSF volume fractions. First M-2
        components are fODF while last two are GM and CSF respectively.

    Notes
    -----
    The diffusion MRI signal measured at a given voxel is a sum of
    contributions from each intra-voxel compartment, including parallel white
    matter (WM) fiber populations in a given orientation as well as effects
    from GM and CSF. The equation governing these  contributions is:

    $S_i = S_0\left(\sum_{j=1}^{M}f_j\exp(-b_i\textbf{v}_i^T\textbf{D}_j
    \textbf{v}_i) + f_{GM}\exp(-b_iD_{GM})+f_{CSF}\exp(-b_iD_{CSF})\right)$

    Where $S_i$ is the resultant signal along the diffusion-sensitizing
    gradient unit vector $\textbf{v_i}; i = 1, ..., N$ with a b-value of $b_i$.
    $f_j; j = 1, ..., M$ is the volume fraction of the $j^{th}$ fiber
    population with an anisotropic diffusion tensor $\textbf{D_j}$.

    $f_{GM}$ and $f_{CSF}$ are the volume fractions and $D_{GM}$ and $D_{CSF}$
    are the mean diffusivities of GM and CSF respectively.

    This equation is linear in $f_j, f_{GM}, f_{CSF}$ and can be simplified to
    a single matrix multiplication:

    $\textbf{S} = \textbf{Hf}$

    Where $\textbf{S}$ is the signal vector at a certain voxel, $\textbf{H}$ is
    the deconvolution kernel, and $\textbf{f}$ is the vector of volume
    fractions for each compartment.

    Modern MRI scanners produce noise following a Rician or Noncentral Chi
    distribution, depending on their signal reconstruction technique [2]_.
    Using this linear model, it can be shown that the likelihood of a signal
    under a Noncentral Chi noise model is:

    $P(\textbf{S}|\textbf{H}, \textbf{f}, \sigma^2, n) = \prod_{i=1}^{N}\left(
    \frac{S_i}{\bar{S_i}}\right)^n\exp\left\{-\frac{1}{2\sigma^2}\left[
    S_i^2 + \bar{S}_i^2\right]\right\}I_{n-1}\left(\frac{S_i\bar{S}_i}
    {\sigma^2}\right)u(S_i)$

    Where $S_i$ and $\bar{S}_i = \textbf{Hf}$ are the measured and expected
    signals respectively, and $n$ is the number of coils in the scanner, and
    $I_{n-1}$ is the modified Bessel function of first kind of order $n-1$.
    This gives the likelihood under a Rician distribution when $n$ is set to 1.

    By taking the negative log of this with respect to $\textbf{f}$ and setting
    the derivative to 0, the $\textbf{f}$ maximizing likelihood is found to be:

    $\textbf{f} = \textbf{f} \circ \frac{\textbf{H}^T\left[\textbf{S}\circ
    \frac{I_n(\textbf{S}\circ \textbf{Hf}/\sigma^2)} {I_{n-1}(\textbf{S}
    \circ\textbf{Hf}\sigma^2)} \right ]} {\textbf{H}^T\textbf{Hf}}$

    The solution can be found using an iterative scheme, just as in the
    Richardson-Lucy algorithm:

    $\textbf{f}^{k+1} = \textbf{f}^k \circ \frac{\textbf{H}^T\left[\textbf{S}
    \circ\frac{I_n(\textbf{S}\circ\textbf{Hf}^k/\sigma^2)} {I_{n-1}(\textbf{S}
    \circ\textbf{Hf}^k/\sigma^2)} \right ]} {\textbf{H}^T\textbf{Hf}^k}$

    In order to apply this, a reasonable estimate of $\sigma^2$ is required.
    To find this, a separate iterative scheme is found using the derivative
    of the negative log with respect to $\sigma^2$, and is run in parallel.
    This is shown here:

    $\alpha^{k+1} = \frac{1}{nN}\left\{ \frac{\textbf{S}^T\textbf{S} +
    \textbf{f}^T\textbf{H}^T\textbf{Hf}}{2} - \textbf{1}^T_N\left[(\textbf{S}
    \circ\textbf{Hf})\circ\frac{I_n(\textbf{S}\circ\textbf{Hf}/\alpha^k)}
    {I_{n-1}(\textbf{S}\circ\textbf{Hf}/\alpha^k)} \right ]\right \}$

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
           Arrays. Magnetic Resonance in Medicine: Official Journal of the
           Society of Magnetic Resonance in Medicine / Society of Magnetic
           Resonance in Medicine, 38(5), 852–857.
    """

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
    sigma0 = 1 / 15
    sigma2 = sigma0**2 * np.ones(data.shape)  # Expand into vector

    reblurred_s = data * reblurred / sigma2

    for _ in range(n_iter):
        fodf_i = fodf
        ratio = mbessel_ratio(n_order, reblurred_s)
        rl_factor = np.matmul(kernel_t, data * ratio) / \
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

    # Normalize final result
    fit_vec = np.squeeze(fodf / (np.sum(fodf, axis=0) + _EPS))

    return fit_vec


def mbessel_ratio(n, x):
    r"""
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
    """

    y = x / ((2 * n + x) - (2 * x * (n + 1 / 2) / (2 * n + 1 + 2 * x - (
        2 * x * (n + 3 / 2) / (2 * n + 2 + 2 * x - (2 * x * (n + 5 / 2) / (
            2 * n + 3 + 2 * x)))))))

    return y


def generate_kernel(gtab, sphere, wm_response, gm_response, csf_response):
    """
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
        Tensor eigenvalues as a (3,) ndarray, multishell eigenvalues as
        a (len(unique_bvals_tolerance(gtab.bvals))-1, 3) ndarray in
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
    """

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

    elif wm_response.shape == (n_bvals, 3):
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
                S = single_tensor(gtab_sub,
                                  evals=wm_response[n],
                                  evecs=all_tensor_evecs(sticks[i]))
                kernel[indices, i] = S

        # Set b0 components
        b0_indices = get_bval_indices(bvals, list_bvals[0])
        kernel[b0_indices, :] = 1

    else:
        # Single-shell response
        for i in range(n_wm_comp):
            # Signal generated by WM-fiber for each gradient direction
            S = single_tensor(gtab, evals=wm_response,
                              evecs=all_tensor_evecs(sticks[i]))
            kernel[:, i] = S

        # Set b0 components
        kernel[gtab.b0s_mask, :] = 1

    # GM compartment
    if gm_response is None:
        S_gm = np.zeros(n_grad)
    else:
        S_gm = \
            single_tensor(gtab, evals=np.array(
                [gm_response, gm_response, gm_response]))

    if csf_response is None:
        S_csf = np.zeros(n_grad)
    else:
        S_csf = \
            single_tensor(gtab, evals=np.array(
                [csf_response, csf_response, csf_response]))

    kernel[:, n_comp - 2] = S_gm
    kernel[:, n_comp - 1] = S_csf

    return kernel


def rumba_deconv_global(data, kernel, mask, n_iter=600, recon_type='smf',
                        n_coils=1, R=1, use_tv=True, verbose=False):
    r"""
    Fit fODF for all voxels simultaneously using RUMBA-SD.

    Deconvolves the kernel from the diffusion-weighted signal at each voxel by
    computing a maximum likelihood estimation of the fODF [1]_. Global fitting
    also permits the use of total variation regularization (RUMBA-SD + TV). The
    spatial dependence introduced by TV promotes smoother solutions (i.e.
    prevents oscillations), while still allowing for sharp discontinuities
    [2]_. This promotes smoothness and continuity along individual tracts while
    preventing smoothing of adjacent tracts.

    Generally, global_fit will proceed more quickly than the voxelwise fit
    provided that the computer has adequate RAM (>= 16 GB should be more than
    sufficient).

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
    fit_array : 4d ndarray (x, y, z, M)
        fODF and GM/CSF volume fractions computed for each voxel. First M-2
        components are fODF, while last two are GM and CSf respectively.

    Notes
    -----
    TV modifies our cost function as follows:

    $J(\textbf{f}) = -\log{P(\textbf{S}|\textbf{H}, \textbf{f}, \sigma^2, n)})+
    \alpha_{TV}TV(\textbf{f})$

    where the first term is the negative log likelihood described in the notes
    of `rumba_deconv`, and the second term is the TV energy, or the sum of
    gradient absolute values for the fODF across the entire brain. This results
    in a new multiplicative factor in the iterative scheme, now becoming:

    $\textbf{f}^{k+1} = \textbf{f}^k \circ \frac{\textbf{H}^T\left[\textbf{S}
    \circ\frac{I_n(\textbf{S}\circ\textbf{Hf}^k/\sigma^2)} {I_{n-1}(\textbf{S}
    \circ\textbf{Hf}^k/\sigma^2)} \right ]} {\textbf{H}^T\textbf{Hf}^k}\circ
    \textbf{R}^k$

    where $\textbf{R}^k$ is computed voxelwise by:

    $(\textbf{R}^k)_j = \frac{1}{1 - \alpha_{TV}div\left(\frac{\triangledown[
    \textbf{f}^k_{3D}]_j}{\lvert\triangledown[\textbf{f}^k_{3D}]_j \rvert}
    \right)\biggr\rvert_{x, y, z}}$

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
    """

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
    sigma0 = 1 / 15
    sigma2 = sigma0**2
    tv_lambda = sigma2  # initial guess for TV regularization strength

    # Expand into matrix form for iterations
    sigma2 = sigma2 * np.ones(data_2d.shape, dtype=np.float32)
    tv_lambda_aux = np.zeros(n_v_tot, dtype=np.float32)

    reblurred_s = data_2d * reblurred / sigma2

    for i in range(n_iter):
        fodf_i = fodf
        ratio = mbessel_ratio(n_order, reblurred_s).astype(np.float32)
        rl_factor = np.matmul(kernel_t, data_2d * ratio) / \
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
            logger.info("Iteration %d of %d", i + 1, n_iter)

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

                if tv_lambda < (1 / 30)**2:
                    tv_lambda = (1 / 30)**2
            else:  # different factor for each voxel
                tv_lambda_aux[index_mask] = sigma2_i
                tv_lambda = np.reshape(tv_lambda_aux, (*dim[:3], 1))

    fodf = fodf.astype(np.float64)
    fodf = fodf / (np.sum(fodf, axis=0)[None, ...] + _EPS)  # normalize fODF

    # Extract compartments
    fit_array = np.zeros((*dim_orig[:3], n_comp))
    _reshape_2d_4d(fodf.T, mask, out=fit_array[ixmin[0]:ixmax[0],
                                               ixmin[1]:ixmax[1],
                                               ixmin[2]:ixmax[2]])

    return fit_array


def _grad(M):
    """
    Computes one way first difference
    """
    x_ind = list(range(1, M.shape[0])) + [M.shape[0] - 1]
    y_ind = list(range(1, M.shape[1])) + [M.shape[1] - 1]
    z_ind = list(range(1, M.shape[2])) + [M.shape[2] - 1]

    grad = np.zeros((*M.shape[:3], 3, M.shape[-1]), dtype=np.float32)
    grad[:, :, :, 0, :] = M[x_ind, :, :, :] - M
    grad[:, :, :, 1, :] = M[:, y_ind, :, :] - M
    grad[:, :, :, 2, :] = M[:, :, z_ind, :] - M

    return grad


def _divergence(F):
    """
    Computes divergence of a 3-dimensional vector field (with one way
    first difference)
    """
    Fx = F[:, :, :, 0, :]
    Fy = F[:, :, :, 1, :]
    Fz = F[:, :, :, 2, :]

    x_ind = [0] + list(range(F.shape[0] - 1))
    y_ind = [0] + list(range(F.shape[1] - 1))
    z_ind = [0] + list(range(F.shape[2] - 1))

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
    """
    Faster reshape from 2D to 4D.
    """
    if out is None:
        out = np.zeros((*mask.shape, M.shape[-1]), dtype=M.dtype)
    n = 0
    for i, j, k in np.ndindex(mask.shape):
        if mask[i, j, k]:
            out[i, j, k, :] = M[n, :]
            n += 1
    return out
