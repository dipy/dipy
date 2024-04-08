"""Tools for fitting Bingham distributions to orientation distribution
functions (ODF), as described in Riffert et al [1]_. The resulting
distributions can further be used to compute ODF-lobe-specific measures such as
the fiber density (FD) and fiber spread (FS) [1]_ and the orientation
dispersion index (ODI) [2]_.

References
----------
.. [1] Riffert TW, Schreiber J, Anwander A, Knösche TR. Beyond fractional
        anisotropy: Extraction of bundle-specific structural metrics from
        crossing fiber models. NeuroImage. 2014 Oct 15;100:176-91.
.. [2] R. Neto Henriques, “Advanced methods for diffusion MRI data analysis
        and their application to the healthy ageing brain.” Apollo -
        University of Cambridge Repository, 2018. doi: 10.17863/CAM.29356.
"""

from warnings import warn
import numpy as np

from dipy.direction import peak_directions
from dipy.reconst.shm import sh_to_sf
from dipy.core.ndindex import ndindex
from dipy.core.onetime import auto_attr


def _bingham_fit_peak(sf, peak, sphere, max_angle):
    """
    Fit Bingham function on the ODF lobe aligned with peak.

    Parameters
    ----------
    sf: 1d ndarray
        The odf function - surface function (sf) - evaluated on the vertices 
        of `sphere`.
    peak: ndarray (3, 1)
        The peak direction of the lobe to fit.
    sphere: `Sphere` class instance
        The Sphere providing the odf's discrete directions
    max_angle: float
        The maximum angle in degrees of the neighbourhood around the
        peak to consider for fitting.

    Returns
    -------
    f0: float
        Maximum amplitude of the ODF peak.
    k1: tuple of floats
        Concentration parameter of Bingham's major axis k1.
    k2: tuple of floats
        Concentration parameter of Bingham's minor axis k2.
    mu0: ndarray (3,) of floats
        Bingham's main axis.
    mu1: ndarray (3,) of floats
        Bingham's major concentration axis.
    mu2: ndarray (3,) of floats
        Bingham's minor concentration axis.
    """
    # abs for twice the number of pts to fit
    dot_prod = np.abs(sphere.vertices.dot(peak))
    min_dot = np.cos(np.deg2rad(max_angle))

    # [p] are the selected ODF vertices (N, 3) around the peak of the lobe
    # within max_angle
    p = sphere.vertices[dot_prod > min_dot]
    # [v] are the selected ODF amplitudes (N, 1) around the peak of the lobe
    # within max_angle
    v = sf[dot_prod > min_dot]

    # Test that the surface along peak direction contains
    # at least 3 non-zero directions
    if np.count_nonzero(v) < 3:
        return 0, 0.0, 0.0,  np.zeros(3), np.zeros(3)

    x, y, z = (p[:, 0], p[:, 1], p[:, 2])

    # Create an orientation matrix (T) to approximate mu0, mu1 and mu2
    T = np.array([[x**2 * v, x * y * v, x * z * v],
                  [x * y * v, y**2 * v, y * z * v],
                  [x * z * v, y * z * v, z**2 * v]])

    T = np.sum(T, axis=-1) / np.sum(v)

    # eigh better than eig. T will always be symmetric, eigh is faster.
    evals, evecs = np.linalg.eigh(T)

    # Not ordering the evals, eigh orders by default.
    mu0 = evecs[:, 2]
    mu1 = evecs[:, 1]
    mu2 = evecs[:, 0]
    f0 = v.max()  # Maximum amplitude of the ODF peak

    # If no real fit is possible, return null
    if np.iscomplex(mu1).any() or np.iscomplex(mu2).any():
        return 0, 0.0, 0.0,  np.zeros(3), np.zeros(3)

    # Calculating the A matrix
    A = np.zeros((len(v), 2), dtype=float)  # (N, 2)
    A[:, 0] = p.dot(mu1)**2
    A[:, 1] = p.dot(mu2)**2

    # Test that AT.A is invertible for pseudo-inverse
    ATA = A.T.dot(A)
    if np.linalg.matrix_rank(ATA) != ATA.shape[0]:
        return 0, 0.0, 0.0,  np.zeros(3), np.zeros(3)

    # Calculating the Beta matrix
    B = np.zeros_like(v)
    B[v > 0] = np.log(v[v > 0] / f0)  # (N, 1)

    # Calculating the Kappas
    k = np.abs(np.linalg.pinv(ATA).dot(A.T).dot(B))
    k1 = k[0]
    k2 = k[1]
    if k1 > k2:
        k1, k2 = k2, k1
        mu1, mu2 = mu2, mu1

    return f0, k1, k2, mu0, mu1, mu2


def odf_to_bingham(odf, sphere, npeaks=5, max_search_angle=6,
                   min_sep_angle=60, rel_th=0.1):
    """
    Fit a Bingham distribution onto each principal ODF lobe.

    Parameters
    ----------
    odf: 1d ndarray
        The ODF function evaluated on the vertices of `sphere`
    sphere: `Sphere` class instance
        The Sphere providing the odf's discrete directions
    npeak: int, optional
        Maximum number of peaks found (default 5 peaks).
    max_search_angle: float, optional.
        Maximum angle between a peak and its neighbour directions
        for fitting the Bingham distribution. 6º according to [1]_.
    min_sep_angle: float, optional
        Minimum separation angle between two peaks for peak extraction.
    rel_th: float, optional
        Relative threshold used for peak extraction.

    Returns
    -------
    fits: list of tuples
        Bingham distribution parameters for each ODF peak.
    n: float
        Number of maximum peaks for the input ODF.

    Notes
    -----
    Lobes are first found by performing a peak extraction on the input ODF and
    Bingham distributions are then fitted around each of the extracted peaks
    using the method described by Riffert et al [1]_.

    References
    ----------
    .. [1] Riffert TW, Schreiber J, Anwander A, Knösche TR. Beyond fractional
            anisotropy: Extraction of bundle-specific structural metrics from
            crossing fiber models. NeuroImage. 2014 Oct 15;100:176-91.
    """
    # extract all maxima on the ODF
    dirs, vals, inds = peak_directions(odf, sphere,
                                       relative_peak_threshold=rel_th,
                                       min_separation_angle=min_sep_angle)

    # n becomes the new limit of peaks and sets a maximum of peaks in case
    # the ODF has more than npeaks.
    n = min(npeaks, vals.shape[0])

    # Calculate dispersion on all and each of the peaks up to 'n'
    if vals.shape[0] != 0:
        fits = []
        for i in range(n):
            fit = _bingham_fit_peak(odf, dirs[i], sphere,
                                    max_search_angle)
            fits.append(fit)

    return fits, n


def _bingham_to_odf(f0, k1, k2, major_axis, minor_axis, vertices):
    """
    Sample a Bingham function on the directions described by `vertices`.
    The function assumes that `vertices` are already normalized and no
    checks are performed to validate that this is the case.

    Parameters
    ----------
    f0: float
        Maximum value of the Bingham function.
    k1: float
        Concentration along major axis.
    k2: float
        Concentration along minor axis.
    major_axis: ndarray (3)
        Direction of major axis
    minor_axis: ndarray (3)
        Direction of minor axis
    vertices: ndarray (N, 3)
        Unit sphere directions along which the distribution
        is evaluated.

    Returns
    -------
    fn : array (N,)
        Sampled Bingham function values at each point on directions.

    Notes
    -----
    Refer to method `bingham_to_odf` for the definition of
    the Bingham distribution.
    """
    fn = f0*np.exp(-k1*vertices.dot(major_axis)**2
                   - k2*vertices.dot(minor_axis)**2)

    return fn.T


def bingham_to_odf(f0, k1, k2, major_axis, minor_axis, vertices):
    """
    Sample a Bingham function on the directions described by `vertices`.

    Parameters
    ----------
    f0: float
        Maximum value of the Bingham function.
    k1: float
        Concentration along major axis.
    k2: float
        Concentration along minor axis.
    major_axis: ndarray (3)
        Direction of major axis
    minor_axis: ndarray (3)
        Direction of minor axis
    vertices: ndarray (N, 3)
        Unit sphere directions along which the distribution
        is evaluated.

    Returns
    -------
    fn : array (N,)
        Sampled Bingham function values at each point on directions.

    Notes
    -----
    The Bingham function is defined as [1]_,[2]_,[3]_:

    .. math::

        f(n) = f_0 \exp[-k_1(\mu_1^Tn)^2-k_2(\mu_2^Tn)^2]

    where $f(n)$ is the Bingham function value at a given direction $n$, $f_0$
    is the Bingham maximum peak value, $k_1$ and $k_2$ are the concentration
    constants parameters (large values correspond to lower dispersion values)
    along the two dispersion axes $\mu_1$ and $\mu_2$.

    References
    ----------
    .. [1] Bingham, C., 1974. An antipodally symmetric distribution on the
           sphere. Anna1 Stat. 2, 1201-1225.
    .. [2] Riffert, T.W., Schreiber, J., Anwander, A., Knösche, T.R., 2014.
           Beyond fractional Anisotropy: Extraction of Bundle-specific
           structural metrics from crossing fiber models.
           Neuroimage 100: 176-191. doi: 10.1016/j.neuroimage.2014.06.015
    .. [3] Henriques RN, 2018. Advanced Methods for Diffusion MRI Data Analysis
           and their Application to the Healthy Ageing Brain (Doctoral thesis).
           Downing College, University of Cambridge. doi: 10.17863/CAM.29356
    """
    if not (np.linalg.norm(vertices, axis=-1) == 1).any():
        warn("Some sphere directions are not normalized. Normalizing.",
             UserWarning)
        vertices /= np.linalg.norm(vertices, axis=-1, keepdims=True)

    return _bingham_to_odf(f0, k1, k2, major_axis, minor_axis, vertices)


def bingham_multi_voxel_odf(bingham_params, sphere, mask=None):
    """
    Reconstruct ODFs from fitted Bingham parameters on multiple voxels.

    Parameters
    ----------
    bingham_params : ndarray (...., nl, 12)
        ndarray containing the model parameters of Binghams fitted to ODFs in
        the following order:
            Maximum value of the Bingham function (f0, index 0);
            concentration parameters k1 and k2 (indexes 1 and 2);
            elements of Bingham's main direction (indexes 3-5);
            elements of Bingham's dispersion major axis (indexes 6-8);
            elements of Bingham's dispersion minor axis (indexes 9-11).
    sphere: `Sphere` class instance
         The Sphere providing the odf's discrete directions
    mask: ndarray, optional
        Map marking the coordinates in the data that should be analyzed

    Returns
    -------
    ODF : ndarray (..., n_directions)
            The value of the odf on each point of `sphere`.
    """
    n_directions = sphere.vertices.shape[0]

    shape = bingham_params.shape[0:-2]
    if mask is None:
        mask = np.ones(shape)

    odf = np.zeros(shape + (n_directions,))
    for idx in ndindex(shape):
        if not mask[idx]:
            continue
        bpars = bingham_params[idx]
        f0s = bpars[..., 0]
        npeaks = np.sum(f0s > 0)

        this_odf = 0

        for li in range(npeaks):
            f0 = f0s[li]
            k1 = bpars[li, 1]
            k2 = bpars[li, 2]
            mu1 = bpars[li, 6:9]
            mu2 = bpars[li, 9:12]

            this_odf += _bingham_to_odf(f0, k1, k2, mu1, mu2, sphere.vertices)
        odf[idx] = this_odf

    return odf


def bingham_fiber_density(bingham_params, mask=None, n_thetas=50, n_phis=100):
    """
    Compute fiber density for each lobe for a given Bingham ODF.

    Parameters
    ----------
    bingham_params : ndarray (...., nl, 12)
        ndarray containing the model parameters of Bingham's fitted to ODFs in
        the following order:
            Maximum value of the Bingham function (f0, index 0);
            concentration parameters k1 and k2 (indexes 1 and 2);
            elements of Bingham's main direction (indexes 3-5);
            elements of Bingham's dispersion major axis (indexes 6-8);
            elements of Bingham's dispersion minor axis (indexes 9-11).
    mask: ndarray
        Map marking the coordinates in the data that should be analyzed
    n_thetas: unsigned int, optional
        Number of steps along theta axis for the integration.
    n_phis: unsigned int, optional
        Number of steps along phi axis for the integration.

    Returns
    -------
    fd: ndarray (...., nl)
        Fiber density for each Bingham function.

    Notes
    -----
    Fiber density (fd) is given by the integral of the Bingham function [1]_.

    References
    ----------
    .. [1] Riffert TW, Schreiber J, Anwander A, Knösche TR. Beyond fractional
           anisotropy: Extraction of bundle-specific structural metrics from
           crossing fiber models. NeuroImage. 2014 Oct 15;100:176-91.
    """
    # Define directions for the integral
    phi = np.linspace(0, 2 * np.pi, n_phis, endpoint=False)  # [0, 2pi]
    theta = np.linspace(0, np.pi, n_thetas)  # [0, pi]
    coords = np.array([[p, t] for p in phi for t in theta]).T
    dphi = phi[1] - phi[0]
    dtheta = theta[1] - theta[0]
    sin_theta = np.sin(coords[1])

    # these directions are normalized
    u = np.array([np.cos(coords[0]) * np.sin(coords[1]),
                  np.sin(coords[0]) * np.sin(coords[1]),
                  np.cos(coords[1])]).T

    shape = bingham_params.shape[0:-2]
    if mask is None:
        mask = np.ones(shape)

    # loop integral calculation for each image voxel
    fd = np.zeros(bingham_params.shape[0:-1])
    for idx in ndindex(shape):
        if not mask[idx]:
            continue

        bpars = bingham_params[idx]
        f0s = bpars[..., 0]
        npeaks = np.sum(f0s > 0)

        for li in range(npeaks):
            f0 = f0s[li]
            k1 = bpars[li, 1]
            k2 = bpars[li, 2]
            mu1 = bpars[li, 6:9]
            mu2 = bpars[li, 9:12]

            bingham_eval = _bingham_to_odf(f0, k1, k2, mu1, mu2, u)
            fd[idx + (li,)] = np.sum(bingham_eval * sin_theta * dtheta * dphi)

    return fd


def bingham_fiber_spread(f0, fd):
    """
    Compute fiber spread for each lobe for a given Bingham volume.

    Parameters
    ----------
    f0: ndarray
        Peak amplitude (f0) of each Bingham function.
    fd: ndarray
        Fiber density (fd) of each Bingham function.

    Returns
    -------
    fs: list of floats
        Fiber spread (fs) of each each Bingham function.

    Notes
    -----
    Fiber spread (fs) is defined as fs = fd/f0 and characterizes the spread of
    the lobe, i.e. the higher the fs, the wider the lobe [1]_.

    References
    ----------
    .. [1] Riffert TW, Schreiber J, Anwander A, Knösche TR. Beyond fractional
           anisotropy: Extraction of bundle-specific structural metrics from
           crossing fiber models. NeuroImage. 2014 Oct 15;100:176-91.
    """
    fs = np.zeros(f0.shape)
    fs[f0 > 0] = fd[f0 > 0] / f0[f0 > 0]

    return fs


def k2odi(k):
    """
    Convert the Bingham/Watson concentration parameter k to the orientation
    dispersion index (ODI).

    Parameters
    ----------
    k: ndarray
        Watson/Bingham concentration parameter

    Returns
    -------
    ODI: float or ndarray
        Orientation Dispersion Index

    Notes
    -----
    Orientation Dispersion Index for Watson/Bingham functions are defined as
    [1]_,[2]_:

    .. math::

        ODI = \frac{2}{pi} \arctan{( \frac{1}{k})}

    References
    ----------
    .. [1] Zhang H, Schneider T, Wheeler-Kingshott CA, Alexander DC.
            NODDI: practical in vivo neurite orientation dispersion and
            density imaging of the human brain. Neuroimage. 2012; 61(4),
            1000-1016. doi: 10.1016/j.neuroimage.2012.03.072
    .. [2] R. Neto Henriques, “Advanced methods for diffusion MRI data analysis
            and their application to the healthy ageing brain.” Apollo -
            University of Cambridge Repository, 2018. doi: 10.17863/CAM.29356.
    """
    odi = np.zeros(k.shape)
    odi[k > 0] = 2/np.pi * np.arctan(1 / k[k > 0])
    return odi


def odi2k(odi):
    """
    Convert the orientation dispersion index (ODI) to the Bingham/Watson
    concentration parameter k.

    Parameters
    ----------
    ODI: ndarray
        Orientation Dispersion Index

    Returns
    -------
    k: float or ndarray
        Watson/Bingham concentration parameter

    Notes
    -----
    Orientation Dispersion Index for Watson/Bingham functions are defined as
    [1]_,[2]_:

    .. math::

        ODI = \frac{2}{pi} \arctan ( \frac{1}{k} )

    References
    ----------
    .. [1] Zhang H, Schneider T, Wheeler-Kingshott CA, Alexander DC.
            NODDI: practical in vivo neurite orientation dispersion and
            density imaging of the human brain. Neuroimage. 2012; 61(4),
            1000-1016. doi: 10.1016/j.neuroimage.2012.03.072
    .. [2] R. Neto Henriques, “Advanced methods for diffusion MRI data analysis
            and their application to the healthy ageing brain.” Apollo -
            University of Cambridge Repository, 2018. doi: 10.17863/CAM.29356.
    """
    k = np.zeros(odi.shape)
    k[odi > 0] = 1/np.tan(np.pi/2 * odi[odi > 0])
    return k


def _convert_bingham_pars(fits, npeaks):
    """
    Convert list of tuples output of the Bingham fit to ndarray.

    Parameters
    ----------
    fits : tuple
        Tuple of nl elements containing the Bingham function parameters
        in the following order:
            Maximum value of the Bingham function (f0);
            concentration parameters (k1 and k2);
            elements of Bingham's main direction (mu0);
            elements of Bingham's dispersion major axis (mu1);
            and elements of Bingham's dispersion minor axis (mu2).
    npeaks: int
        Maximum number of fitted Bingham functions, by number of peaks.

    Returns
    -------
    bingham_params : ndarray (nl, 12)
        ndarray containing the model parameters of Bingham fitted to ODFs in
        the following order:
            Maximum value of the Bingham function (f0, index 0);
            concentration parameters k1 and k2 (indexes 1 and 2);
            elements of Bingham's main direction (indexes 3-5);
            elements of Bingham's dispersion major axis (indexes 6-8);
            elements of Bingham's dispersion minor axis (indexes 9-11).
    """
    n = len(fits)
    bpars = np.zeros((npeaks, 12))
    for ln in range(n):
        bpars[ln, 0] = fits[ln][0]
        bpars[ln, 1] = fits[ln][1]
        bpars[ln, 2] = fits[ln][2]
        bpars[ln, 3:6] = fits[ln][3]
        bpars[ln, 6:9] = fits[ln][4]
        bpars[ln, 9:12] = fits[ln][5]
    return bpars


def global_voxel_metric(bmetric, bfd):
    """
    Compute global scalar maps for metrics of Bingham functions
    fitted to multiple ODF lobes.

    Parameters
    ----------
    bmetric: ndarray(..., nl)
        An arbitrary metric with values for nl ODF lobes.
    bfd: ndarray(..., nl)
        Bingham's fiber density estimates for the nl ODF lobes

    Returns
    -------
    gmetric: ndarray(...)
        Global weighted averaged Bingham metric

    Notes
    -----
    The global metric is computed as the weighted average of a given metric
    across the multiple ODF lobes (weights are defined by the Bingham fiber
    density estimates)
    """
    return np.sum(bmetric * bfd, axis=-1) / np.sum(bfd, axis=-1)


class BinghamMetrics:
    """
    Class for Bingham Metrics."""

    def __init__(self, model_params):
        """ Initialization of the Bingham Metrics Class.

        Parameters
        ----------
        model_params : ndarray (..., nl, 12)
            ndarray containing Bingham's model parameters fitted to ODFs
            in the following order:
            Maximum value of the Bingham function (f0, index 0);
            concentration parameters k1 and k2 (indexes 1 and 2);
            elements of Bingham's main direction (indexes 3-5);
            elements of Bingham's dispersion major axis (indexes 6-8);
            elements of Bingham's dispersion minor axis (indexes 9-11).
        """
        self.model_params = model_params

        self.peak_dirs = model_params[..., 3:6]
        self.peak_values = model_params[..., 0]
        

    @auto_attr
    def afd(self):
        """ Maximum Bingham Amplitude for each ODF lobe."""
        return self.peak_values

    @auto_attr
    def kappa_1(self):
        """ Concentration parameter k1 for each ODF lobe."""
        return self.model_params[..., 1]

    @auto_attr
    def kappa_2(self):
        """ Concentration parameter k2 for each ODF lobe."""
        return self.model_params[..., 2]

    @auto_attr
    def kappa_total(self):
        """ Overall concentration parameters for an ODF peak.

        Note:
        ----
        Overall (combined) concentration parameters for each lobe as defined
        by equation 19 in [4]_ as:

        .. math::
            k_{total} = sqrt{(k_1 * k_2)}
            
        References
        ----------
        .. [4] Tariq M, Schneider T, Alexander DC, Wheeler-Kingshott CAG,
            Zhang H. Bingham–NODDI: Mapping anisotropic orientation dispersion
            of neurites using diffusion MRI NeuroImage. 2016; 133:207-223.
        """
        return np.sqrt(self.kappa_1 * self.kappa_2)

    @auto_attr
    def odi_1(self):
        """ Orientation Dispersion index 1 computed for each ODF lobe from
        concentration parameter kappa_1."""
        return k2odi(self.kappa_1)

    @auto_attr
    def odi_2(self):
        """ Orientation Dispersion index 2 computed for each ODF lobe from
        concentration parameter kappa_2."""
        return k2odi(self.kappa_2)

    @auto_attr
    def odi_total(self):
        """ Overall Orientation Dispersion Index (ODI) computed for am
        ODF peak from the overall concentration parameter (k_total).
        Defined by equation 20 in [4]_.
            
        References
        ----------
        .. [4] Tariq M, Schneider T, Alexander DC, Wheeler-Kingshott CAG,
            Zhang H. Bingham–NODDI: Mapping anisotropic orientation dispersion
            of neurites using diffusion MRI NeuroImage. 2016; 133:207-223.
        """
        return k2odi(self.kappa_total)

    @auto_attr
    def fd(self):
        """ Fiber Density computed as the integral of the Bingham functions
        fitted for each ODF lobe. """
        return bingham_fiber_density(self.model_params)

    @auto_attr
    def godi_1(self):
        """ Global Orientation Dispersion Index 1 (weighted average of odi1
        across all lobes where the weights are each lobe's fd estimate).
        """
        return global_voxel_metric(self.odi_1, self.fd)

    @auto_attr
    def godi_2(self):
        """ Global Orientation Dispersion Index 2 (weighted average of odi2
        across all lobes where the weights are each lobe's fd estimate).
        """
        return global_voxel_metric(self.odi_2, self.fd)

    @auto_attr
    def godi_total(self):
        """ Global Total Orientation Dispersion Index (weighted average of
        odf_total across all lobes where the weights are each lobe's
        fd estimate)."""
        return global_voxel_metric(self.odi_total, self.fd)

    @auto_attr
    def gfd(self):
        """ Global fiber density (sum of fd estimates of all ODF lobes)."""
        return np.sum(self.fd, axis=-1)

    @auto_attr
    def fs(self):
        """ Fiber spread computed for each ODF lobe.

        Notes
        -----
        Fiber spread (fs) is defined as fs = fd/f0 and characterizes the
        spread of the lobe, i.e. the higher the fs, the wider the lobe [1]_.

        References
        ----------
        .. [1] Riffert TW, Schreiber J, Anwander A, Knösche TR. Beyond
               fractional anisotropy: Extraction of bundle-specific structural
               metrics from crossing fiber models. NeuroImage. 2014 Oct 15;
               100:176-91.
        """
        return bingham_fiber_spread(self.afd, self.fd)
    
    @auto_attr
    def gfs(self):
        """ Global fiber spread (weighted average of fiber spread across all
        lobes where the weights are each lobe's fd estimate)."""
        return global_voxel_metric(self.fs, self.fd)

    def odf(self, sphere):
        """ Reconstruct ODFs from fitted Bingham parameters on multiple voxels.

        Parameters
        ----------
        sphere: `Sphere` class instance
            The Sphere providing the discrete directions for ODF
            reconstruction.

        Returns
        -------
        ODF : ndarray (..., n_directions)
            The value of the odf on each point of `sphere`.
        """
        mask = self.gfd > 0
        return bingham_multi_voxel_odf(self.model_params, sphere, mask=mask)


def bingham_from_odf(odf, sphere, mask=None, npeaks=5, max_search_angle=6,
                     min_sep_angle=60, rel_th=0.1):
    """
    Fit the Bingham function from an ODF.

    Parameters
    ----------
    odf: ndarray
        Orientation Distribution Function sampled on the vertices of a sphere.
    sphere: `Sphere` class instance
        The Sphere providing the odf's discrete directions.
    mask: ndarray
        Map marking the coordinates in the data that should be analyzed.
    npeak: int
        Maximum number of peaks found (default 5 peaks).
    max_search_angle: float, optional.
        Maximum angle between a peak and its neighbour directions
        for fitting the Bingham distribution.
    min_sep_angle: float, optional
        Minimum separation angle between two peaks for peak extraction.
    rel_th: float, optional
        Relative threshold used for peak extraction.

    Return
    ------
    BinghamMetrics: class instance
        Class instance containing metrics computed from Bingham functions
        fitted to ODF lobes.
    """
    shape = odf.shape[0:-1]
    
    if mask is None:
        mask = np.ones(shape)

    # Bingham parameters saved in an ndarray with shape:
    # (Nx, Ny, Nz, n_max_peak, 12).
    bpars = np.zeros(shape + (npeaks,) + (12,))

    for idx in ndindex(shape):
        if not mask[idx]:
            continue

        [fits, npeaks_final] = odf_to_bingham(
            odf[idx], sphere, npeaks, max_search_angle=max_search_angle,
            min_sep_angle=min_sep_angle, rel_th=rel_th)

        bpars[idx] = _convert_bingham_pars(fits, npeaks)
        
    return BinghamMetrics(bpars)


def bingham_from_sh(sh, sphere, sh_order_max, mask=None, npeaks=5,
                    max_search_angle=6, min_sep_angle=60, rel_th=0.1):
    """
    Fit the Bingham function from an ODF's spherical harmonics (SH)
    representation.

    Parameters
    ----------
    sh : ndarray
        SH coefficients representing a spherical function.
    sphere : Sphere
        The points on which to sample the spherical function.
    sh_order_max: int
        Maximum order used for the SH reconstruction.
    mask: ndarray
        Map marking the coordinates in the data that should be analyzed.
    npeak: int
        Maximum number of peaks found (default 5 peaks).
    max_search_angle: float, optional.
        Maximum angle between a peak and its neighbour directions
        for fitting the Bingham distribution.
    min_sep_angle: float, optional
        Minimum separation angle between two peaks for peak extraction.
    rel_th: float, optional
        Relative threshold used for peak extraction.

    Return
    ------
    BinghamMetrics: class instance
        Class instance containing metrics computed from Bingham functions
        fitted to ODF lobes.
    """
    shape = sh.shape[0:-1]
    
    if mask is None:
        mask = np.ones(shape)

    # Bingham parameters saved in an ndarray with shape:
    # (Nx, Ny, Nz, n_max_peak, 12).
    bpars = np.zeros(shape + (npeaks,) + (12,))

    for idx in ndindex(shape):
        if not mask[idx]:
            continue

        odf = sh_to_sf(sh[idx], sphere, sh_order_max=sh_order_max)
        
        [fits, npeaks_final] = odf_to_bingham(
            odf, sphere, npeaks, max_search_angle=max_search_angle,
            min_sep_angle=min_sep_angle, rel_th=rel_th)

        bpars[idx] = _convert_bingham_pars(fits, npeaks)
        
    return BinghamMetrics(bpars)
