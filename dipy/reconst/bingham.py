# -*- coding: utf-8 -*-

import itertools
import multiprocessing

from math import cos, radians
from dipy.data import get_sphere
import numpy as np

from dipy.direction import peak_directions
from dipy.reconst.shm import sh_to_sf_matrix
from scilpy.reconst.utils import get_sh_order_and_fullness


# Constants
NB_PARAMS = 7


class BinghamDistribution(object):
    """
    Scaled Bingham distribution, given by:
        B(u) = f0 * e^(- k1 * (mu1 * u)**2 - k2 * (mu2 * u)**2),
    mu1 and mu2 are unit vectors.

    Params
    ------
    f0: float
        Scaling parameter of the distribution.
    mu_prime1: ndarray (3,)
        Axis with highest concentration scaled by the
        concentration parameter k1.
    mu_prime2: ndarray (3,)
        Axis with lowest concentration scaled by the
        concentration parameter k2.
    """
    def __init__(self, f0, mu_prime1, mu_prime2):
        self.f0 = f0  # scaling factor
        self.k1 = np.linalg.norm(mu_prime1)
        self.k2 = np.linalg.norm(mu_prime2)
        if self.k1 != 0:
            self.mu1 = mu_prime1 / self.k1
        else:
            self.mu1 = mu_prime1

        if self.k2 != 0:
            self.mu2 = mu_prime2 / self.k2
        else:
            self.mu2 = mu_prime2

    def evaluate(self, vertices):
        bu = self.f0 * np.exp(- self.k1 * self.mu1.dot(vertices.T)**2.
                              - self.k2 * self.mu2.dot(vertices.T)**2.)

        return bu.reshape((-1))  # (1, N)

    def peak_direction(self):
        v = np.cross(self.mu1, self.mu2)
        return v

    def get_flatten(self):
        mu_prime1 = self.mu1 * self.k1
        mu_prime2 = self.mu2 * self.k2
        ret = np.array([self.f0,
                        *mu_prime1.reshape((-1)),
                        *mu_prime2.reshape((-1))])
        return ret


def bingham_to_sf(bingham_volume, vertices):
    """
    Convert a Bingham distributions volume to a spherical function.

    Parameters
    ----------
    bingham_volume: Bingham parameters volume.
        A Bingham distributions volume.
    vertices: ndarray (n_vertices, 3)
        Sampling directions.

    Returns
    -------
    sf: ndarray (..., n_lobes, N_PARAMS)
        Bingham distribution evaluated at vertices.
    """
    f0s = bingham_volume[..., 0]  # (X, Y, Z, 5)
    mu1s = bingham_volume[..., 1:4]  # (X, Y, Z, 5, 3)
    mu2s = bingham_volume[..., 4:7]  # (X, Y, Z, 5, 3)
    k1s = np.linalg.norm(mu1s, axis=-1)  # (X, Y, Z, 5)
    k2s = np.linalg.norm(mu2s, axis=-1)  # (X, Y, Z, 5)
    # normalize mu1 and mu2
    mu1s[k1s != 0] /= k1s[k1s != 0][..., None]
    mu2s[k2s != 0] /= k2s[k2s != 0][..., None]

    # transpose vertices to (3, N)
    if len(vertices.shape) == 1:
        vertices = vertices.reshape((3, 1))
    elif vertices.shape[-1] == 3:
        vertices = vertices.T

    # compute the SF
    sf = k1s[..., None] * mu1s.dot(vertices)**2 +\
        k2s[..., None] * mu2s.dot(vertices)**2
    sf = f0s[..., None] * np.exp(-sf)
    return sf


def bingham_to_peak_direction(bingham_volume):
    """
    Compute peak direction for each lobe for a given Bingham volume.

    Parameters
    ----------
    binghams: ndarray (..., max_lobes, 9)
        Bingham volume.

    Returns
    -------
    peak_dir: ndarray (..., max_lobes, 3)
        Peak direction image.
    """
    mu1s = bingham_volume[..., 1:4]  # (X, Y, Z, 5, 3)
    mu2s = bingham_volume[..., 4:7]  # (X, Y, Z, 5, 3)
    k1s = np.linalg.norm(mu1s, axis=-1)  # (X, Y, Z, 5)
    k2s = np.linalg.norm(mu2s, axis=-1)  # (X, Y, Z, 5)

    # normalize mu1 and mu2
    mu1s[k1s != 0] /= k1s[k1s != 0][..., None]
    mu2s[k2s != 0] /= k2s[k2s != 0][..., None]

    # compute peak direction
    peak_dir = np.cross(mu1s, mu2s)
    return peak_dir


def bingham_fit_sh(sh, max_lobes=5, abs_th=0.,
                   rel_th=0., min_sep_angle=25.,
                   max_fit_angle=15, mask=None,
                   nbr_processes=None):
    """
    Approximate SH field by fitting Bingham distributions to
    up to ``max_lobes`` lobes per voxel, sorted in descending order
    by the amplitude of their peak direction.

    Parameters
    ----------
    sh: ndarray (X, Y, Z, ncoeffs)
        SH coefficients array.
    max_lobes: unsigned int, optional
        Maximum number of lobes to fit per voxel.
    abs_th: float, optional
        Absolute threshold for peak extraction.
    rel_th: float, optional
        Relative threshold for peak extraction in the range [0, 1].
    min_sep_angle: float, optional
        Minimum separation angle between two adjacent peaks in degrees.
    max_fit_angle: float, optional
        The maximum distance in degrees around a peak direction for
        fitting the Bingham function.
    mask: ndarray (X, Y, Z), optional
        Mask to apply to the data.
    nbr_processes: unsigned int, optional
        The number of processes to use. If None, than
        ``multiprocessing.cpu_count()`` processes are executed.

    Returns
    -------
    out: ndarray (X, Y, Z, max_lobes*9)
        Bingham functions array.
    """
    order, full_basis = get_sh_order_and_fullness(sh.shape[-1])
    shape = sh.shape

    sphere = get_sphere('symmetric724').subdivide(2)
    B_mat = sh_to_sf_matrix(sphere, order,
                            full_basis=full_basis,
                            return_inv=False)

    nbr_processes = multiprocessing.cpu_count()\
        if nbr_processes is None \
        or nbr_processes < 0 \
        or nbr_processes > multiprocessing.cpu_count() \
        else nbr_processes

    if mask is not None:
        sh = sh[mask]
    else:
        sh = sh.reshape((-1, shape[-1]))

    sh = np.array_split(sh, nbr_processes)
    pool = multiprocessing.Pool(nbr_processes)
    out = pool.map(_bingham_fit_sh_chunk, zip(sh, itertools.repeat(B_mat),
                                              itertools.repeat(sphere),
                                              itertools.repeat(abs_th),
                                              itertools.repeat(min_sep_angle),
                                              itertools.repeat(rel_th),
                                              itertools.repeat(max_lobes),
                                              itertools.repeat(max_fit_angle)))
    pool.close()
    pool.join()

    out = np.concatenate(out, axis=0)
    if mask is not None:
        bingham = np.zeros(shape[:3] + (max_lobes, NB_PARAMS))
        bingham[mask] = out
        return bingham

    out = out.reshape(shape[:3] + (max_lobes, NB_PARAMS))
    return out


def _bingham_fit_sh_chunk(args):
    """
    Fit Bingham functions on a (N, ncoeffs) chunk taken from a SH field.
    """
    sh_chunk = args[0]
    B_mat = args[1]
    sphere = args[2]
    abs_th = args[3]
    min_sep_angle = args[4]
    rel_th = args[5]
    max_lobes = args[6]
    max_angle = args[7]

    out = np.zeros((len(sh_chunk), max_lobes, NB_PARAMS))
    for i, sh in enumerate(sh_chunk):
        odf = sh.dot(B_mat)
        odf[odf < abs_th] = 0.
        if (odf > 0.).any():
            lobes = \
                _bingham_fit_multi_peaks(odf, sphere, max_angle,
                                         min_sep_angle, rel_th)
            for ll in range(min(len(lobes), max_lobes)):
                lobe = lobes[ll]
                out[i, ll, :] = lobe.get_flatten()
    return out


def _bingham_fit_multi_peaks(odf, sphere, max_angle,
                             min_sep_angle, rel_th):
    """
    Peak extraction followed by Bingham fit for each peak.

    Parameters
    ----------
    odf: ndarray
        ODF expressed as a spherical function evaluated on sphere.
    sphere: DIPY Sphere
        Sphere on which odf is defined.
    max_angle: float
        Maximum angle between a peak and its neighbouring directions
        to be included when fitting the Bingham distribution.
    min_sep_angle: float
        Minimum separation angle between two peaks for peak extraction.
    rel_th: float
        Relative threshold used for peak extraction.
    """
    peaks, _, _ = peak_directions(odf, sphere,
                                  relative_peak_threshold=rel_th,
                                  min_separation_angle=min_sep_angle)

    lobes = []
    for peak in peaks:
        peak_fit = _bingham_fit_peak(odf, peak, sphere, max_angle)
        lobes.append(peak_fit)

    return lobes


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
    res: BinghamDistribution
        The Bingham distribution approximating the lobe aligned with peak
        on the SF.
    """
    # abs for twice the number of pts to fit
    dot_prod = np.abs(sphere.vertices.dot(peak))
    min_dot = cos(radians(max_angle))

    p = sphere.vertices[dot_prod > min_dot]
    v = sf[dot_prod > min_dot].reshape((-1, 1))  # (N, 1)

    # test that the peak contains at least 3 non-zero directions
    if np.count_nonzero(v) < 3:
        return BinghamDistribution(0, np.zeros(3), np.zeros(3))

    x, y, z = (p[:, 0:1], p[:, 1:2], p[:, 2:])

    # create an orientation matrix to approximate mu0, mu1 and mu2
    T = np.zeros((3, 3))
    T[0, 0] = np.sum(x**2 * v)
    T[1, 1] = np.sum(y**2 * v)
    T[2, 2] = np.sum(z**2 * v)
    T[1, 0] = np.sum(x * y * v)
    T[2, 0] = np.sum(x * z * v)
    T[2, 1] = np.sum(y * z * v)
    T[0, 1] = T[1, 0]
    T[0, 2] = T[2, 0]
    T[1, 2] = T[2, 1]
    T = T / np.sum(v)

    eval, evec = np.linalg.eig(T)

    ordered = np.argsort(eval)
    mu1 = evec[:, ordered[1]].reshape((3, 1))
    mu2 = evec[:, ordered[0]].reshape((3, 1))
    f0 = v.max()

    if np.iscomplex(mu1).any() or np.iscomplex(mu2).any():
        return BinghamDistribution(0, np.zeros(3), np.zeros(3))

    A = np.zeros((len(v), 2), dtype=float)  # (N, 2)
    A[:, 0:1] = p.dot(mu1)**2
    A[:, 1:] = p.dot(mu2)**2

    # Test that AT.A is invertible for pseudo-inverse
    ATA = A.T.dot(A)
    if np.linalg.matrix_rank(ATA) != ATA.shape[0]:
        return BinghamDistribution(0, np.zeros(3), np.zeros(3))

    B = np.zeros_like(v)
    B[v > 0] = np.log(v[v > 0] / f0)  # (N, 1)
    k = np.abs(np.linalg.inv(ATA).dot(A.T).dot(B))
    k1 = k[0]
    k2 = k[1]
    if k[0] > k[1]:
        k1 = k[1]
        k2 = k[0]
        mu2 = evec[:, ordered[1]].reshape((3, 1))
        mu1 = evec[:, ordered[0]].reshape((3, 1))

    return BinghamDistribution(f0, k1 * mu1, k2 * mu2)


def compute_fiber_density(bingham, m=50, mask=None, nbr_processes=None):
    """
    Compute fiber density for each lobe for a given Bingham volume.

    Fiber density (FD) is given by integrating
    the Bingham function over the sphere. Its unit is
    in 1/mm**3.

    Parameters
    ----------
    bingham: ndarray (X, Y, Z, max_lobes*9)
        Input Bingham volume.
    m: unsigned int, optional
        Number of steps along theta axis for the integration. The number of
        steps along the phi axis is 2*m.
    mask: ndarray (X, Y, Z), optional
        Mask to apply to the computation.
    nbr_processes: unsigned int, optional
        The number of processes to use. If None, then
        ``multithreading.cpu_count()`` processes are launched.

    Returns
    -------
    res: ndarray (X, Y, Z, max_lobes)
        FD per lobe for each voxel.
    """
    shape = bingham.shape

    phi = np.linspace(0, 2 * np.pi, 2 * m, endpoint=False)  # [0, 2pi[
    theta = np.linspace(0, np.pi, m)  # [0, pi]
    coords = np.array([[p, t] for p in phi for t in theta]).T
    dphi = phi[1] - phi[0]
    dtheta = theta[1] - theta[0]

    nbr_processes = multiprocessing.cpu_count()\
        if nbr_processes is None \
        or nbr_processes < 0 \
        or nbr_processes > multiprocessing.cpu_count() \
        else nbr_processes

    if mask is not None:
        bingham = bingham[mask]

    bingham = bingham.reshape((-1, np.prod(shape[-2:])))
    bingham = np.array_split(bingham, nbr_processes)
    pool = multiprocessing.Pool(nbr_processes)
    res = pool.map(_compute_fiber_density_chunk,
                   zip(bingham,
                       itertools.repeat(coords),
                       itertools.repeat(dphi),
                       itertools.repeat(dtheta)))
    pool.close()
    pool.join()

    res = np.concatenate(res, axis=0)
    nbr_lobes = shape[-2]

    if mask is not None:
        fd = np.zeros(shape[:3] + (nbr_lobes,))
        fd[mask] = res
        return fd

    res = np.reshape(np.array(res), shape[:3] + (nbr_lobes,))
    return res


def _compute_fiber_density_chunk(args):
    """
    Compute fiber density for a chunk taken from a Bingham volume.
    """
    binghams_chunk = args[0]
    coords = args[1]
    dphi = args[2]
    dtheta = args[3]
    theta = coords[1]
    u = np.array([np.cos(coords[0]) * np.sin(coords[1]),
                  np.sin(coords[0]) * np.sin(coords[1]),
                  np.cos(coords[1])]).T

    nbr_lobes = binghams_chunk.shape[1] // NB_PARAMS
    out = np.zeros((len(binghams_chunk), nbr_lobes))
    for i, binghams in enumerate(binghams_chunk):
        for lobe_i in range(nbr_lobes):
            params = binghams[lobe_i * NB_PARAMS:(lobe_i + 1) * NB_PARAMS]
            lobe = BinghamDistribution(params[0], params[1:4], params[4:7])
            if lobe.f0 > 0:
                fd = np.sum(lobe.evaluate(u) * np.sin(theta) * dtheta * dphi)
                out[i, lobe_i] = fd
    return out


def compute_fiber_spread(binghams, fd):
    """
    Compute fiber spread for each lobe for a given Bingham volume.

    Fiber spread (FS) characterizes the spread of the lobe.
    The higher FS is, the wider the lobe. The unit of the
    FS is radians.

    Parameters
    ----------
    binghams: ndarray (X, Y, Z, max_lobes*9)
        Bingham volume.
    fd: ndarray (X, Y, Z, max_lobes)
        Fiber density image.

    Returns
    -------
    fs: ndarray (X, Y, Z, max_lobes)
        Fiber spread image.
    """
    f0 = binghams[..., :, 0]
    fs = np.zeros_like(fd)
    fs[f0 > 0] = fd[f0 > 0] / f0[f0 > 0]

    return fs


def compute_fiber_fraction(fd):
    """
    Compute the fiber fraction for each lobe at each voxel.

    The fiber fraction (FF) represents the fraction of the current lobe's
    FD on the total FD for all lobes. For each voxel, the FF sums to 1.

    Parameters
    ----------
    fd: ndarray (X, Y, Z, max_lobes)
        Fiber density image.

    Returns
    -------
    ff: ndarray (X, Y, Z, max_lobes)
        Fiber fraction image.
    """
    ff = np.zeros_like(fd)
    sum = np.sum(fd, axis=-1)
    mask = sum > 0
    for ll in range(ff.shape[-1]):
        ff[..., ll][mask] = fd[..., ll][mask] / sum[mask]

    return ff
