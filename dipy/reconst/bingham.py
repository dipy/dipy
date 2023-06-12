"""Tools for fitting Bingham distributions to orientation distributions,
as described in Riffert et al [1]_. The resulting distributions can further
be used to compute ODF-lobe-specific measures such as the fiber density
and fiber spread [1]_.

References
----------
.. [1] Riffert TW, Schreiber J, Anwander A, Knösche TR. Beyond fractional
           anisotropy: Extraction of bundle-specific structural metrics from
           crossing fiber models. NeuroImage. 2014 Oct 15;100:176-91.
"""

from math import cos, radians
from warnings import warn
import numpy as np

from dipy.direction import peak_directions


def bingham_to_sf(f0, k1, k2 , major_axis, minor_axis, vertices):
    """
    Evaluate Bingham distribution for a sphere.

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
    """
    if not (np.linalg.norm(vertices, axis=-1) == 1).any():
        warn("Some sphere directions are not normalized. Normalizing.",
             UserWarning)
        vertices /= np.linalg.norm(vertices, axis=-1, keepdims=True)

    sf = f0*np.exp(-k1*vertices.dot(major_axis)**2
                   -k2*vertices.dot(minor_axis)**2)
    return sf.T


def bingham_fit_sf(sf, sphere, max_search_angle=15,
                      min_sep_angle=15, rel_th=0.1):
    """
    Fit a Bingham distribution onto each principal SF lobe. Lobes
    are first found by performing a peak extraction on the input
    SF, and Bingham distributions are then fitting around each of
    the extracted peaks using the method described in Riffert et
    al [1]_.

    Parameters
    ----------
    sf: ndarray
        Spherical function (SF) evaluated on the sphere `sphere`.
    sphere: DIPY Sphere
        Sphere on which the SF is defined.
    max_search_angle: float, optional
        Maximum angle between a peak and its neighbour directions
        for fitting the Bingham distribution.
    min_sep_angle: float, optional
        Minimum separation angle between two peaks for peak extraction.
    rel_th: float, optional
        Relative threshold used for peak extraction.

    Returns
    -------
    fits: list of tuples
        Bingham distribution parameters for each SF peak.

    References
    ----------
    .. [1] Riffert TW, Schreiber J, Anwander A, Knösche TR. Beyond fractional
           anisotropy: Extraction of bundle-specific structural metrics from
           crossing fiber models. NeuroImage. 2014 Oct 15;100:176-91.
    """
    # extract all maximum on the SF
    peaks, _, _ = peak_directions(sf, sphere,
                                  relative_peak_threshold=rel_th,
                                  min_separation_angle=min_sep_angle)

    fits = []
    for peak in peaks:
        fit = _bingham_fit_peak(sf, peak, sphere, max_search_angle)
        fits.append(fit)

    return fits


def _bingham_fit_peak(sf, peak, sphere, max_angle):
    """
    Fit Bingham function on the lobe aligned with peak.

    Parameters
    ----------
    sf: ndarray
        Spherical function evaluated on sphere.
    peak: ndarray (3, 1)
        The direction of the lobe to fit.
    sphere: DIPY Sphere
        The sphere used to project SH to SF.
    max_angle: float
        The maximum angle in degrees of the neighbourhood around
        peak to consider for fitting.

    Return
    ------
    f0: float
        Maximum amplitude of the distribution.
    concentration: tuple (2,) of floats
        Concentration parameter of principal axes.
    mu_1: ndarray (3,) of floats
        Major concentration axis.
    mu_2: ndarray (3,) of floats
        Minor concentration axis.
    """
    # abs for twice the number of pts to fit
    dot_prod = np.abs(sphere.vertices.dot(peak))
    min_dot = cos(radians(max_angle))

    p = sphere.vertices[dot_prod > min_dot]
    v = sf[dot_prod > min_dot].reshape((-1, 1))  # (N, 1)

    # test that the sf along peak direction contains
    # at least 3 non-zero directions
    if np.count_nonzero(v) < 3:
        return 0, 0.0, 0.0,  np.zeros(3), np.zeros(3)

    x, y, z = (p[:, 0:1], p[:, 1:2], p[:, 2:])

    # create an orientation matrix to approximate mu0, mu1 and mu2
    T = np.array([[x**2 * v, x * y * v, x * z * v],
                  [x * y * v, y**2 * v, y * z * v],
                  [x * z * v, y * z * v, z**2 * v]])
    T = np.sum(np.squeeze(T), axis=-1) / np.sum(v)

    eval, evec = np.linalg.eig(T)

    ordered = np.argsort(eval)
    mu1 = evec[:, ordered[1]].reshape((3, 1))
    mu2 = evec[:, ordered[0]].reshape((3, 1))
    f0 = v.max()

    # If no real fit is possible, return null
    if np.iscomplex(mu1).any() or np.iscomplex(mu2).any():
        return 0, 0.0, 0.0,  np.zeros(3), np.zeros(3)

    A = np.zeros((len(v), 2), dtype=float)  # (N, 2)
    A[:, 0:1] = p.dot(mu1)**2
    A[:, 1:] = p.dot(mu2)**2

    # Test that AT.A is invertible for pseudo-inverse
    ATA = A.T.dot(A)
    if np.linalg.matrix_rank(ATA) != ATA.shape[0]:
        return 0, 0.0, 0.0,  np.zeros(3), np.zeros(3)

    B = np.zeros_like(v)
    B[v > 0] = np.log(v[v > 0] / f0)  # (N, 1)
    k = np.abs(np.linalg.inv(ATA).dot(A.T).dot(B))
    k1 = k[0, 0]
    k2 = k[1, 0]
    if k1 > k2:
        k1, k2 = k2, k1
        mu1, mu2 = mu2, mu1

    return f0, k1, k2, mu1, mu2


def bingham_to_fiber_density(bingham_fits, n_thetas=50, n_phis=100):
    """
    Compute fiber density for each lobe for a given Bingham ODF.

    Fiber density (FD) is given by the integral of the Bingham
    distribution over the sphere and describes the apparent
    quantity of fibers passing through an ODF lobe.

    Parameters
    ----------
    bingham_fits: list of tuples
        Bingham distributions. Each tuple describes a lobe of the
        initial fitted function.
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
    for f0, k1, k2, mu1, mu2 in bingham_fits:
        bingham_eval = bingham_to_sf(f0, k1, k2, mu1, mu2, u)
        fd.append(np.sum(bingham_eval * sin_theta * dtheta * dphi))

    return fd


def bingham_to_fiber_spread(bingham_fits, fd=None):
    """
    Compute fiber spread for each lobe for a given Bingham volume.

    Fiber spread (FS) characterizes the spread of the lobe.
    The higher FS is, the wider the lobe.

    Parameters
    ----------
    bingham_fits: list of tuples
        Bingham distributions. Each tuple describes a lobe of the
        initial fitted function.
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
    f0 = np.array([x for x, _,_,_,_ in bingham_fits])

    if fd is None:
        fd = bingham_to_fiber_density(bingham_fits)
    fd = np.asarray(fd)

    fs = np.zeros((len(f0),))
    fs[f0 > 0] = fd[f0 > 0] / f0[f0 > 0]

    return fs.tolist()
