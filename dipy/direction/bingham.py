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


def bingham_fit_odf(odf, sphere, npeaks, max_search_angle=6,
                    min_sep_angle=60, rel_th=0.1):
    r"""
    Fit a Bingham distribution onto each principal ODF lobe. Lobes
    are first found by performing a peak extraction on the input
    ODF and Bingham distributions are then fitted around each of
    the extracted peaks using the method described by Riffert et
    al [1]_.

    Parameters
    ----------
    odf: ndarray
        ODF evaluated on the sphere `sphere`.
    sphere: DIPY Sphere
        Sphere on which the ODF is defined.
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

    References
    ----------
    .. [1] Riffert TW, Schreiber J, Anwander A, Knösche TR. Beyond fractional
            anisotropy: Extraction of bundle-specific structural metrics from
            crossing fiber models. NeuroImage. 2014 Oct 15;100:176-91.
    """
    
    # extract all maxima on the ODF
    directions, values, indices = peak_directions(odf, sphere,
                                                  relative_peak_threshold=rel_th,
                                                  min_separation_angle=min_sep_angle)
    
    # n becomes the new limit of peaks and sets a maximum of peaks in case
    # the voxel has more than npeaks.
    n = min(npeaks, values.shape[0])  

    # Calculate dispersion on all and each of the peaks up to 'n'
    if values.shape[0] != 0:
        
        fits = []
        
        for i in range (n):
            fit = _bingham_fit_peak(odf, directions[i], sphere, max_search_angle)
            fits.append(fit)
        
        # This is an array of size: npeaks * 12
        # 12 = [f0, kappa1, kappa2, mu0*3, mu1*3, mu2*3] 
        # bingham_fits = np.array(fits)
        
        
    return fits, n


def _bingham_fit_peak(sf, peak, sphere, max_angle):
    r"""
    Fit Bingham function on the ODF lobe aligned with peak.

    Parameters
    ----------
    sf: ndarray
        Spherical function evaluated on sphere (ODF).
    peak: ndarray (3, 1)
        The peak direction of the lobe to fit.
    sphere: DIPY Sphere
        The sphere used to project SH to the ODF.
    max_angle: float
        The maximum angle in degrees of the neighbourhood around
        peak to consider for fitting.

    Returns
    ------
    f0: float
        Maximum amplitude of the distribution/peak.
    concentration: tuple (2,) of floats
        Concentration parameters of principal axes (kappa 1 and kappa 2).
    mu_1: ndarray (3,) of floats
        Major concentration axis.
    mu_2: ndarray (3,) of floats
        Minor concentration axis.
    """
    
    # abs for twice the number of pts to fit
    dot_prod = np.abs(sphere.vertices.dot(peak))
    min_dot = cos(radians(max_angle))

    # [p] are the selected vertices (N, 3) around the peak of the lobe
    # within max_angle
    p = sphere.vertices[dot_prod > min_dot]
    # [v] are the selected ODF values (N, 1) around the peak of the lobe 
    # within max_angle
    v = sf[dot_prod > min_dot].reshape((-1, 1)) 

    # Test that the surface along peak direction contains
    # at least 3 non-zero directions
    if np.count_nonzero(v) < 3:
        return 0, 0.0, 0.0,  np.zeros(3), np.zeros(3)

    x, y, z = (p[:, 0:1], p[:, 1:2], p[:, 2:])

    # Create an orientation matrix (T) to approximate mu0, mu1 and mu2
    T = np.array([[x**2 * v, x * y * v, x * z * v],
                  [x * y * v, y**2 * v, y * z * v],
                  [x * z * v, y * z * v, z**2 * v]])
    
    T = np.sum(np.squeeze(T), axis=-1) / np.sum(v)

    # eig vs. eigh? T will always be symmetric, eigh is faster.
    evals, evecs = np.linalg.eigh(T)

    # ordered = np.argsort(evals) # Not ordering the evals, eigh orders by default.
    mu0 = evecs[:, 2].reshape((3, 1))
    mu1 = evecs[:, 1].reshape((3, 1))
    mu2 = evecs[:, 0].reshape((3, 1))
    f0 = v.max() # Maximum amplitude of the ODF

    # If no real fit is possible, return null
    if np.iscomplex(mu1).any() or np.iscomplex(mu2).any():
        return 0, 0.0, 0.0,  np.zeros(3), np.zeros(3)
    
    # Calculating the A matrix
    A = np.zeros((len(v), 2), dtype=float)  # (N, 2)
    A[:, 0:1] = p.dot(mu1)**2
    A[:, 1:] = p.dot(mu2)**2

    # Test that AT.A is invertible for pseudo-inverse
    ATA = A.T.dot(A)
    if np.linalg.matrix_rank(ATA) != ATA.shape[0]:
        return 0, 0.0, 0.0,  np.zeros(3), np.zeros(3)
    
    # Calculating the Beta matrix
    B = np.zeros_like(v)
    B[v > 0] = np.log(v[v > 0] / f0)  # (N, 1)
    
    # Calculating the Kappas
    k = np.abs(np.linalg.inv(ATA).dot(A.T).dot(B))
    k1 = k[0, 0]
    k2 = k[1, 0]
    if k1 > k2:
        k1, k2 = k2, k1
        mu1, mu2 = mu2, mu1
    
    return f0, k1, k2, mu0, mu1, mu2


def bingham_odf(f0, k1, k2, major_axis, minor_axis, vertices):
    r"""
    Evaluate Bingham distribution on the sphere
    described by `vertices`.

    Parameters
    ----------
    f0: float
        Maximum amplitude of the distribution.
    k1: float
        Concentration along major axis.
    k2: float
        Concentration along minor axis.
    vertices: ndarray (N, 3)
        Unit sphere directions along which the distribution
        is evaluated.
    
    Returns
    -------
    sf: Spherical function of Bingham distribution evaluated in sphere.
        
    """
    
    if not (np.linalg.norm(vertices, axis=-1) == 1).any():
        warn("Some sphere directions are not normalized. Normalizing.",
             UserWarning)
        vertices /= np.linalg.norm(vertices, axis=-1, keepdims=True)

    sf = f0*np.exp(-k1*vertices.dot(major_axis)**2
                   - k2*vertices.dot(minor_axis)**2)
    
    return sf.T


def bingham_fiber_density(bingham_fits, n_thetas=50, n_phis=100):
    r"""
    Compute fiber density for each lobe for a given Bingham ODF.

    Fiber density (FD) is given by the integral of the Bingham
    distribution over the sphere and describes the apparent
    quantity of fibers passing through an ODF lobe.

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

    References
    ----------
    .. [1] Riffert TW, Schreiber J, Anwander A, Knösche TR. Beyond fractional
           anisotropy: Extraction of bundle-specific structural metrics from
           crossing fiber models. NeuroImage. 2014 Oct 15;100:176-91.
    """
    
    phi = np.linspace(0, 2 * np.pi, n_phis, endpoint=False)  # [0, 2pi[
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
    The higher the FS, the wider the lobe.

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


def bingham_from_sh(sh, mask, sh_order, npeaks, sphere):
    r"""
    Funtion that calls in the 4D Spherical Harmonics file of a brain image
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
    
    Returns
    -----------

    """
    
    # Reading in the spherical harmonics and mask
    sphe_har = nib.load(sh)
    datash = sphe_har.get_fdata()
    print('datash.shape (%d, %d, %d, %d)' % datash.shape)
    # datash = datash[:,:,25:26,:]
    
    mask = nib.load(mask)
    datamask = mask.get_fdata()
    # datamask = datamask[:,:,25:26]
    
    sh_order = int(sh_order)
    npeaks = int(npeaks)
        
    sphere = get_sphere(sphere)
    sphere = sphere.subdivide(2)
 
    shape = datash.shape[:3]
    afd = np.zeros((shape + (npeaks,)))
    kappa1 = np.zeros((shape + (npeaks,)))
    kappa2 = np.zeros((shape + (npeaks,)))
    mu_0 =  np.zeros((shape + (npeaks, 3)))
    mu_1 =  np.zeros((shape + (npeaks, 3)))
    mu_2 =  np.zeros((shape + (npeaks, 3)))
    f_d = np.zeros((shape + (npeaks,)))
    f_s = np.zeros((shape + (npeaks,)))
    odi = np.zeros((shape + (npeaks, 2)))
    
    for idx in ndindex(shape):
        if not datamask[idx]:
            continue
        
        odf = sh_to_sf(datash[idx], sphere, sh_order=sh_order)

        [bingham_fits, npeaks_final] = bingham_fit_odf(odf, sphere, npeaks,
                                                       max_search_angle=6,
                                                       min_sep_angle=60,
                                                       rel_th=0.1)
        
        f0 = np.array([i[0] for i in bingham_fits])
        k1 = np.array([i[1] for i in bingham_fits])
        k2 = np.array([i[2] for i in bingham_fits])
        mu0 = np.squeeze(np.array([i[3] for i in bingham_fits]))
        mu1 = np.squeeze(np.array([i[4] for i in bingham_fits]))
        mu2 = np.squeeze(np.array([i[5] for i in bingham_fits]))
        
        fd = bingham_fiber_density(bingham_fits)
        fs = bingham_fiber_spread(bingham_fits)
        od = bingham_orientation_dispersion(bingham_fits)
           
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
