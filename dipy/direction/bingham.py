r"""Tools for fitting Bingham distributions to orientation distribution
functions (ODF), as described in Riffert et al [1]_. The resulting distributions
can further be used to compute ODF-lobe-specific measures such as the 
fiber density (FD) and fiber spread (FS) [1]_ and the orientation dispersion
index (ODI) [2]_.

References
----------
.. [1] Riffert TW, Schreiber J, Anwander A, Knösche TR. Beyond fractional
        anisotropy: Extraction of bundle-specific structural metrics from
        crossing fiber models. NeuroImage. 2014 Oct 15;100:176-91.
.. [2] R. Neto Henriques, “Advanced methods for diffusion MRI data analysis
        and their application to the healthy ageing brain.” Apollo - 
        University of Cambridge Repository, 2018. doi: 10.17863/CAM.29356.
"""

from math import cos, radians
from warnings import warn
import numpy as np
import nibabel as nib

from dipy.direction import peak_directions
from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf
from dipy.core.ndindex import ndindex


def bingham_fit_odf(odf, sphere, npeaks=5, max_search_angle=6,
                    min_sep_angle=60, rel_th=0.1):
    r"""
    Fit a Bingham distribution onto each principal ODF lobe.

    Parameters
    ----------
    odf: 1d ndarray
        The ODF function evaluated on the vertices of `sphere`
    sphere: `Sphere` class instance
        The Sphere providing the odf's discrete directions
    npeak: int
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
    directions, values, _ = peak_directions(odf, sphere,
                                            relative_peak_threshold=rel_th,
                                            min_separation_angle=min_sep_angle)

    # n becomes the new limit of peaks and sets a maximum of peaks in case
    # the voxel has more than npeaks.
    n = min(npeaks, values.shape[0])

    # Calculate dispersion on all and each of the peaks up to 'n'
    if values.shape[0] != 0:
        fits = []
        for i in range(n):
            fit = _bingham_fit_peak(odf, directions[i], sphere,
                                    max_search_angle)
            fits.append(fit)

    return fits, n


def _bingham_fit_peak(sf, peak, sphere, max_angle):
    r"""
    Fit Bingham function on the ODF lobe aligned with peak.

    Parameters
    ----------
    odf: 1d ndarray
        The odf function evaluated on the vertices of `sphere`
    peak: ndarray (3, 1)
        The peak direction of the lobe to fit.
    sphere: `Sphere` class instance
        The Sphere providing the odf's discrete directions
    max_angle: float
        The maximum angle in degrees of the neighbourhood around
        peak to consider for fitting.

    Returns
    -------
    f0: float
        Maximum amplitude of the distribution/peak.
    k1: tuple of floats
        Concentration parameter of major axis k1.
    k2: tuple of floats
        Concentration parameter of minor axis k2.
    mu0: ndarray (3,) of floats
        Main axis of ODF peak.
    mu1: ndarray (3,) of floats
        Major concentration axis.
    mu2: ndarray (3,) of floats
        Minor concentration axis.
    """
    # abs for twice the number of pts to fit
    dot_prod = np.abs(sphere.vertices.dot(peak))
    min_dot = cos(radians(max_angle))

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
    f0 = v.max()  # Maximum amplitude of the ODF

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


def bingham_odf(f0, k1, k2, major_axis, minor_axis, vertices):
    r"""
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

    where $f(n)$ is the Bingham function value at a given direction $n$, $f_0%
    is the Bingham maximum peak value, $k_1$ and $k_2$ are the concentration
    constants parameters (large values correspond to lower dispersion values)
    along the two dispersion axes $\mu_1$ and $\mu_2$-

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

    fn = f0*np.exp(-k1*vertices.dot(major_axis)**2
                   - k2*vertices.dot(minor_axis)**2)

    return fn.T


def bingham_fiber_density(bingham_fits, n_thetas=50, n_phis=100):
    r"""
    Compute fiber density for each lobe for a given Bingham ODF.

    Parameters
    ----------
    bingham_fits: list of tuples
        Bingham distributions. Each tuple describes a lobe of the
        initially fitted function.
    n_thetas: unsigned int, optional
        Number of steps along theta axis for the integration.
    n_phis: unsigned int, optional
        Number of steps along phi axis for the integration.

    Returns
    -------
    fd: list of floats
        Fiber density for each Bingham distribution.

    Notes
    -----
    Fiber density (FD) is given by the integral of the Bingham function [1]_.

    References
    ----------
    .. [1] Riffert TW, Schreiber J, Anwander A, Knösche TR. Beyond fractional
           anisotropy: Extraction of bundle-specific structural metrics from
           crossing fiber models. NeuroImage. 2014 Oct 15;100:176-91.
    """
    phi = np.linspace(0, 2 * np.pi, n_phis, endpoint=False)  # [0, 2pi]
    theta = np.linspace(0, np.pi, n_thetas)  # [0, pi]
    coords = np.array([[p, t] for p in phi for t in theta]).T
    dphi = phi[1] - phi[0]
    dtheta = theta[1] - theta[0]
    sin_theta = np.sin(coords[1])

    u = np.array([np.cos(coords[0]) * np.sin(coords[1]),
                  np.sin(coords[0]) * np.sin(coords[1]),
                  np.cos(coords[1])]).T

    fd = []
    for f0, k1, k2, mu0, mu1, mu2 in bingham_fits:
        bingham_eval = bingham_odf(f0, k1, k2, mu1, mu2, u)
        fd.append(np.sum(bingham_eval * sin_theta * dtheta * dphi))

    return fd


def bingham_fiber_spread(bingham_fits, fd=None):
    r"""
    Compute fiber spread for each lobe for a given Bingham volume.

    Fiber spread (FS) characterizes the spread of the lobe.
    The higher the FS, the wider the lobe. Equation (7) of [1]_.

    Parameters
    ----------
    bingham_fits: list of tuples
        Bingham distributions. Each tuple describes a lobe of the
        initially fitted function.
    fd: list of floats or None
        Fiber density (fd) of each Bingham distribution. If None, fd
        will be computed by the method.

    Returns
    -------
    fs: list of floats
        Fiber spread for each Bingham distribution in the input Bingham fit.

    References
    ----------
    .. [1] Riffert TW, Schreiber J, Anwander A, Knösche TR. Beyond fractional
           anisotropy: Extraction of bundle-specific structural metrics from
           crossing fiber models. NeuroImage. 2014 Oct 15;100:176-91.
    """
    
    f0 = np.array([x for x, _, _, _, _, _ in bingham_fits])

    if fd is None:
        fd = bingham_fiber_density(bingham_fits)
    fd = np.asarray(fd)

    fs = np.zeros((len(f0),))
    fs[f0 > 0] = fd[f0 > 0] / f0[f0 > 0]

    return fs.tolist()


def bingham_orientation_dispersion(bingham_fits):
    r"""
    Compute the orientation dispersion indexes (ODI) from the
    concentration parameters (k1, k2) as described in [2]_ and [3]_.
    
    Parameters
    ----------
    bingham_fits : list of tuples
        Bingham distributions. Each tuple describes a lobe of the
        initially fitted function.

    Returns
    -------
    odi: list of tuples
        Each tuple contains the ODIs 1 & 2 (for k1 & k2, respectively).

    References
    ----------
    .. [2] R. Neto Henriques, “Advanced methods for diffusion MRI data analysis
            and their application to the healthy ageing brain.” Apollo - 
            University of Cambridge Repository, 2018. doi: 10.17863/CAM.29356.
    .. [3] Zhang H, Schneider T, Wheeler-Kingshott CA, Alexander DC.
            NODDI: practical in vivo neurite orientation dispersion and
            density imaging of the human brain. Neuroimage. 2012; 61(4), 
            1000-1016. doi: 10.1016/j.neuroimage.2012.03.072

    """
    
    odi = []
    
    for f0, k1, k2, mu0, mu1, mu2 in bingham_fits:
        
        odi_1 = (2/np.pi) * np.arctan(1/k1)    
        odi_2 = (2/np.pi) * np.arctan(1/k2)
        
        odi.append([odi_1, odi_2]) # Appending for each lobe both odi's

    return odi


def _convert_bingham_pars(fits):
    """ Convert tuple output of Bingham fit functions to ndarray"""
    n = len(fits)
    bpars = np.zeros((n, 12))
    for ln in range(n):
        bpars[ln, 0] = fits[ln][0]
        bpars[ln, 1] = fits[ln][1]
        bpars[ln, 2] = fits[ln][2]

        v1 = fits[ln][3].T
        v2 = fits[ln][4].T
        v0 = np.cross(v1, v2)
        bpars[ln, 3:6] = v0
        bpars[ln, 6:9] = v1
        bpars[ln, 9:12] = v2

    return bpars


def bingham_from_sh_new(odf, sphere, mask=None, npeaks=5, max_search_angle=6,
                        min_sep_angle=60, rel_th=0.1):
    """ Documentation missing """
    shape = odf.shape
    if mask is None:
        mask = np.ones(shape[0:-1])

    # For my later functions I need the Bingham parameters saved in an ndarray
    # with dimensions (Nx, Ny, Nz, n_max_peak, 12).
    bpars = np.zeros(shape[0:-1] + (npeaks,) + (12,))

    for idx in ndindex(shape):
        if not mask[idx]:
            continue

        [fits, npeaks_final] = bingham_fit_odf(
            odf[idx], sphere, npeaks, max_search_angle=max_search_angle,
            min_sep_angle=min_sep_angle, rel_th=rel_th)

        bpars[idx, :npeaks_final, :] = _convert_bingham_pars(fits)
    return bpars


def bingham_from_sh(sh, mask, sh_order, npeaks, sphere, max_angle,
                    min_sep_angle, rel_th):
    r"""
    Function that calls in the 4D Spherical Harmonics file of a brain image
    and loops through the image to fit the Bingham distribution at each voxel.

    Parameters
    ----------
    sh: 4D nifti image
        Image holding the spherical harmonics.
    mask: 3D nifti image
        binary mask of the brain tissue to fit the Bingham distribution.
    sh_order: int
        The order of the spherical harmonics.
    npeaks: int
        number of maximum peaks to be estimated per voxel.
    sphere: string
        DIPY Sphere on which the ODF is defined.
        'symmetric642' gives 10242 vertices after subdividing by (2)
        as in [1]_.
        'repulsion724' gives 11554 vertices after subdividing by (2)
    max_angle: float
        The maximum angle in degrees of the neighbourhood around
        peak to consider for fitting.
    min_sep_angle: float
        Minimum separation angle between two ODF peaks for peak extraction.
    rel_th: float
        Relative threshold used for peak extraction.

    Returns
    -------
    afd: 4D ndarray
        Maximum amplitude of ODF peaks (f0) or axonal fiber density.
    kappa1: 4D ndarray
        First concentration parameter for all ODF peaks.
    kappa2
    mu_0: 5D ndarray
        Main axis direction fort all ODF peaks.
    mu_1: 5D ndarray
        Axis direction along major concentration parameter for all ODF peaks.
    mu_3: 5D ndarray
        Axis direction along minor concentration parameter for all ODF peaks.
    f_d: 4D ndarray
        Fiber density as in equation (6) of [1]_.
    f_s: 4D ndarray
        Fiber spread as in equation (7) of [1]_.
    odi: 5D ndarray
        Orientation Dispersion Index for k1 and k2 for all ODF peaks
        as in [2]_ and [3]_.

    References
    ----------
    .. [1] Riffert TW, Schreiber J, Anwander A, Knösche TR. Beyond fractional
           anisotropy: Extraction of bundle-specific structural metrics from
           crossing fiber models. NeuroImage. 2014 Oct 15;100:176-91.
    .. [2] R. Neto Henriques, “Advanced methods for diffusion MRI data analysis
            and their application to the healthy ageing brain.” Apollo -
            University of Cambridge Repository, 2018. doi: 10.17863/CAM.29356.
    .. [3] Zhang H, Schneider T, Wheeler-Kingshott CA, Alexander DC.
            NODDI: practical in vivo neurite orientation dispersion and
            density imaging of the human brain. Neuroimage. 2012; 61(4),
            1000-1016. doi: 10.1016/j.neuroimage.2012.03.072

    """
    # Reading in the spherical harmonics and mask
    sphe_har = nib.load(sh)
    datash = sphe_har.get_fdata()
    print('datash.shape (%d, %d, %d, %d)' % datash.shape)

    mask = nib.load(mask)
    datamask = mask.get_fdata()

    sh_order = int(sh_order)
    npeaks = int(npeaks)

    sphere = get_sphere(sphere)
    sphere = sphere.subdivide(2)

    shape = datash.shape[:3]
    afd = np.zeros((shape + (npeaks,)))
    kappa1 = np.zeros((shape + (npeaks,)))
    kappa2 = np.zeros((shape + (npeaks,)))
    mu_0 = np.zeros((shape + (npeaks, 3)))
    mu_1 = np.zeros((shape + (npeaks, 3)))
    mu_2 = np.zeros((shape + (npeaks, 3)))
    f_d = np.zeros((shape + (npeaks,)))
    f_s = np.zeros((shape + (npeaks,)))
    odi = np.zeros((shape + (npeaks, 2)))

    print('Fitting the Bingham distribution for the input brain volume.')
    for idx in ndindex(shape):
        if not datamask[idx]:
            continue

        odf = sh_to_sf(datash[idx], sphere, sh_order=sh_order)

        [fits, npeaks_final] = bingham_fit_odf(odf, sphere, npeaks,
                                               max_search_angle=max_angle,
                                               min_sep_angle=min_sep_angle,
                                               rel_th=rel_th)

        f0 = np.array([i[0] for i in fits])
        k1 = np.array([i[1] for i in fits])
        k2 = np.array([i[2] for i in fits])
        mu0 = np.squeeze(np.array([i[3] for i in fits]))
        mu1 = np.squeeze(np.array([i[4] for i in fits]))
        mu2 = np.squeeze(np.array([i[5] for i in fits]))

        fd = bingham_fiber_density(fits)
        fs = bingham_fiber_spread(fits)
        od = bingham_orientation_dispersion(fits)

        afd[idx][:npeaks_final] = f0[:npeaks_final]
        kappa1[idx][:npeaks_final] = k1[:npeaks_final]
        kappa2[idx][:npeaks_final] = k2[:npeaks_final]
        mu_0[idx][:npeaks_final] = mu0[:npeaks_final]
        mu_1[idx][:npeaks_final] = mu1[:npeaks_final]
        mu_2[idx][:npeaks_final] = mu2[:npeaks_final]
        f_d[idx][:npeaks_final] = fd[:npeaks_final]
        f_s[idx][:npeaks_final] = fs[:npeaks_final]
        odi[idx][:npeaks_final] = od[:npeaks_final]

    # Returning f0 (AFD), Kappas, mu's, FD, FS and ODI in 4D/5D volumes.
    # They include the number of peaks in the ODF.
    return afd, kappa1, kappa2, mu_0, mu_1, mu_2, f_d, f_s, odi
