from __future__ import division, print_function, absolute_import

from multiprocessing import cpu_count, Pool
from itertools import repeat
from warnings import warn, catch_warnings, simplefilter

from ..utils.six.moves import xrange

import numpy as np
import scipy.optimize as opt

from .recspeed import local_maxima, remove_similar_vertices, search_descending
from ..core.onetime import auto_attr
from dipy.core.sphere import HemiSphere, Sphere
from dipy.data import get_sphere
from dipy.core.ndindex import ndindex


# Classes OdfModel and OdfFit are using API ReconstModel and ReconstFit from
# .base

default_sphere = HemiSphere.from_sphere(get_sphere('symmetric724'))


def peak_directions_nl(sphere_eval, relative_peak_threshold=.25,
                       min_separation_angle=45, sphere=default_sphere,
                       xtol=1e-7):
    """Non Linear Direction Finder

    Parameters
    ----------
    sphere_eval : callable
        A function which can be evaluated on a sphere.
    relative_peak_threshold : float
        Only return peaks greater than ``relative_peak_threshold * m`` where m
        is the largest peak.
    min_separation_angle : float in [0, 90]
        The minimum distance between directions. If two peaks are too close only
        the larger of the two is returned.
    sphere : Sphere
        A discrete Sphere. The points on the sphere will be used for initial
        estimate of maximums.
    xtol : float
        Relative tolerance for optimization.

    Returns
    -------
    directions : array (N, 3)
        Points on the sphere corresponding to N local maxima on the sphere.
    values : array (N,)
        Value of sphere_eval at each point on directions.

    """
    # Find discrete peaks for use as seeds in non-linear search
    discrete_values = sphere_eval(sphere)
    values, indices = local_maxima(discrete_values, sphere.edges)

    seeds = np.column_stack([sphere.theta[indices], sphere.phi[indices]])

    # Helper function
    def _helper(x):
        sphere = Sphere(theta=x[0], phi=x[1])
        return -sphere_eval(sphere)

    # Non-linear search
    num_seeds = len(seeds)
    theta = np.empty(num_seeds)
    phi = np.empty(num_seeds)
    for i in xrange(num_seeds):
        peak = opt.fmin(_helper, seeds[i], xtol=xtol, disp=False)
        theta[i], phi[i] = peak

    # Evaluate on new-found peaks
    small_sphere = Sphere(theta=theta, phi=phi)
    values = sphere_eval(small_sphere)

    # Sort in descending order
    order = values.argsort()[::-1]
    values = values[order]
    directions = small_sphere.vertices[order]

    # Remove directions that are too small
    n = search_descending(values, relative_peak_threshold)
    directions = directions[:n]

    # Remove peaks too close to each-other
    directions, idx = remove_similar_vertices(directions, min_separation_angle,
                                              return_index=True)
    values = values[idx]
    return directions, values


class OdfModel(object):
    """An abstract class to be sub-classed by specific odf models

    All odf models should provide a fit method which may take data as it's
    first and only argument.
    """
    def fit(self, data):
        """To be implemented by specific odf models"""
        raise NotImplementedError("To be implemented in sub classes")


class OdfFit(object):
    def odf(self, sphere):
        """To be implemented but specific odf models"""
        raise NotImplementedError("To be implemented in sub classes")


def peak_directions(odf, sphere, relative_peak_threshold=.25,
                    min_separation_angle=45):
    """Get the directions of odf peaks

    Parameters
    ----------
    odf : 1d ndarray
        The odf function evaluated on the vertices of `sphere`
    sphere : Sphere
        The Sphere providing discrete directions for evaluation.
    relative_peak_threshold : float
        Only return peaks greater than ``relative_peak_threshold * m`` where m
        is the largest peak.
    min_separation_angle : float in [0, 90] The minimum distance between
        directions. If two peaks are too close only the larger of the two is
        returned.

    Returns
    -------
    directions : (N, 3) ndarray
        N vertices for sphere, one for each peak
    values : (N,) ndarray
        peak values
    indices : (N,) ndarray
        peak indices of the directions on the sphere

    """
    odf = np.ascontiguousarray(odf)
    values, indices = local_maxima(odf, sphere.edges)
    # If there is only one peak return
    if len(indices) == 1:
        return sphere.vertices[indices], values, indices

    n = search_descending(values, relative_peak_threshold)
    indices = indices[:n]
    directions = sphere.vertices[indices]
    directions, uniq = remove_similar_vertices(directions,
                                               min_separation_angle,
                                               return_index=True)
    values = values[uniq]
    indices = indices[uniq]
    return directions, values, indices


class PeaksAndMetrics(object):
    pass

from tempfile import mkdtemp
import os.path as path


def peaks_from_model_parallel(model, data, sphere, relative_peak_threshold,
                     min_separation_angle, mask=None, return_odf=False,
                     return_sh=True, gfa_thr=0, normalize_peaks=False,
                     sh_order=8, sh_basis_type=None, ravel_peaks=False,
                     npeaks=5, nbr_process=None):
    """
    Fits the model to data and computes peaks and metrics using multiprocessing

    Parameters
    ----------
    See peaks_from_model function

    nbr_process: int
        Number of subprocess to use (default multiprocessing.cpu_count()).

    """
    if nbr_process is None:
        nbr_process = cpu_count()


    if nbr_process < 2 :
        return peaks_from_model(model, data, sphere, relative_peak_threshold, min_separation_angle, mask, return_odf, return_sh, gfa_thr, normalize_peaks, sh_order, sh_basis_type, ravel_peaks, npeaks)

    shape = list(data.shape)
    data = np.reshape(data,(-1,shape[-1]))

    n = data.shape[0]
    nbr_chunks = nbr_process**2
    chunk_size = int(np.ceil(n / nbr_chunks))
    data_chunks = [data[i:i + chunk_size] for i in range(0, n, chunk_size)]

    if mask is not None:
        mask = mask.flatten()
        mask_chunks = [mask[i:i + chunk_size] for i in range(0, n, chunk_size)]
    else:
        mask_chunks = [None] * nbr_chunks

    pool = Pool(nbr_process)

    pam_res = pool.map(__peaks_from_model_parallel_sub,
            zip(repeat(model),
                data_chunks,
                repeat(sphere),
                repeat(relative_peak_threshold),
                repeat(min_separation_angle),
                mask_chunks,
                repeat(return_odf),
                repeat(return_sh),
                repeat(gfa_thr),
                repeat(normalize_peaks),
                repeat(sh_order),
                repeat(sh_basis_type),
                repeat(ravel_peaks),
                repeat(npeaks)))
    pool.close()

    data_chunks = None
    pam = PeaksAndMetrics()
    #memmap are used to reduce de memory usage
    temp_dir = mkdtemp()
    pam.gfa = np.memmap(path.join(temp_dir, 'gfa.dat'), dtype=pam_res[0].gfa.dtype, mode='w+', shape=(data.shape[0]))
    pam.peak_dirs = np.memmap(path.join(temp_dir, 'peak_dirs.dat'), dtype=pam_res[0].peak_dirs.dtype, mode='w+', shape=(data.shape[0], npeaks, 3))
    pam.peak_values = np.memmap(path.join(temp_dir, 'peak_values.dat'), dtype=pam_res[0].peak_values.dtype, mode='w+', shape=(data.shape[0], npeaks))
    pam.peak_indices = np.memmap(path.join(temp_dir, 'peak_indices.dat'), dtype=pam_res[0].peak_indices.dtype, mode='w+', shape=(data.shape[0], npeaks))
    pam.qa =  np.memmap(path.join(temp_dir, 'qa.dat'), dtype=pam_res[0].qa.dtype, mode='w+', shape=(data.shape[0], npeaks))
    if return_odf:
        pam.odf = np.memmap(path.join(temp_dir, 'odf.dat'), dtype=pam_res[0].odf.dtype, mode='w+', shape=(data.shape[0], len(sphere.vertices)))
    else:
        pam.odf = None
    if return_sh:
        n_shm_coeff = (sh_order + 2) * (sh_order + 1) / 2
        pam.shm_coeff = np.memmap(path.join(temp_dir, 'shm.dat'), dtype=pam_res[0].shm_coeff.dtype, mode='w+', shape=(data.shape[0], n_shm_coeff))
        pam.invB = pam_res[0].invB
    else:
        pam.shm_coeff = None
        pam.invB = None

    #copy sub process result arrays to a single result array
    for i in range(len(pam_res)):
        start_pos = i * chunk_size
        end_pos = (i+1) * chunk_size
        if start_pos >= data.shape[0]:
            break
        pam.gfa[start_pos : end_pos] = pam_res[i].gfa[:]
        pam.peak_dirs[start_pos : end_pos] = pam_res[i].peak_dirs[:]
        pam.peak_values[start_pos : end_pos] = pam_res[i].peak_values[:]
        pam.peak_indices[start_pos : end_pos] = pam_res[i].peak_indices[:]
        pam.qa[start_pos : end_pos] = pam_res[i].qa[:]

        if return_sh:
            pam.shm_coeff[start_pos : end_pos] = pam_res[i].shm_coeff[:]
        if return_odf:
            pam.odf[start_pos : end_pos] = pam_res[i].odf[:]

    #reshape the metric to the original shape
    pam.peak_dirs = np.reshape(pam.peak_dirs, shape[:-1] + [npeaks, 3])
    pam.peak_values = np.reshape(pam.peak_values, shape[:-1] + [npeaks])
    pam.peak_indices = np.reshape(pam.peak_indices, shape[:-1] + [npeaks])
    pam.qa = np.reshape(pam.qa, shape[:-1] + [npeaks])
    pam.gfa = np.reshape(pam.gfa, shape[:-1])
    return pam


def __peaks_from_model_parallel_sub(args):
    return peaks_from_model(*args)

def peaks_from_model(model, data, sphere, relative_peak_threshold,
                     min_separation_angle, mask=None, return_odf=False,
                     return_sh=True, gfa_thr=0, normalize_peaks=False,
                     sh_order=8, sh_basis_type=None, ravel_peaks=False,
                     npeaks=5):
    """Fits the model to data and computes peaks and metrics

    Parameters
    ----------
    model : a model instance
        `model` will be used to fit the data.
    sphere : Sphere
        The Sphere providing discrete directions for evaluation.
    relative_peak_threshold : float
        Only return peaks greater than ``relative_peak_threshold * m`` where m
        is the largest peak.
    min_separation_angle : float in [0, 90] The minimum distance between
        directions. If two peaks are too close only the larger of the two is
        returned.
    mask : array, optional
        If `mask` is provided, voxels that are False in `mask` are skipped and
        no peaks are returned.
    return_odf : bool
        If True, the odfs are returned.
    return_sh : bool
        If True, the odf as spherical harmonics coefficients is returned
    gfa_thr : float
        Voxels with gfa less than `gfa_thr` are skipped, no peaks are returned.
    normalize_peaks : bool
        If true, all peak values are calculated relative to `max(odf)`.
    sh_order : int, optional
        Maximum SH order in the SH fit.  For `sh_order`, there will be
        ``(sh_order + 1) * (sh_order + 2) / 2`` SH coefficients (default 8).
    sh_basis_type : {None, 'mrtrix', 'fibernav'}
        ``None`` for the default dipy basis which is the fibernav basis,
        ``mrtrix`` for the MRtrix basis, and
        ``fibernav`` for the FiberNavigator basis
    ravel_peaks : bool
        If True, the peaks are returned as [x1, y1, z1, ..., xn, yn, zn] instead
        of Nx3. Set this flag to True if you want to visualize the peaks in the
        fibernavigator or in mrtrix.
    npeaks : int
        Maximum number of peaks found (default 5 peaks).

    Returns
    -------
    pam : PeaksAndMetrics
        an object with ``gfa``, ``peak_directions``, ``peak_values``,
        ``peak_indices``, ``odf``,``shm_coeffs`` as attributes

    """

    shape = data.shape[:-1]
    if mask is None:
        mask = np.ones(shape, dtype='bool')
    else:
        if mask.shape != shape:
            raise ValueError("Mask is not the same shape as data.")

    sh_smooth = 0
    gfa_array = np.zeros(shape)
    qa_array = np.zeros((shape + (npeaks,)))

    peak_dirs = np.zeros((shape + (npeaks, 3)))
    peak_values = np.zeros((shape + (npeaks,)))
    peak_indices = np.zeros((shape + (npeaks,)), dtype='int')
    peak_indices.fill(-1)

    if return_sh:
        #import here to avoid circular imports
        from dipy.reconst.shm import sph_harm_lookup, smooth_pinv

        sph_harm_basis = sph_harm_lookup.get(sh_basis_type)
        if sph_harm_basis is None:
            raise ValueError("Invalid basis name.")
        B, m, n = sph_harm_basis(sh_order, sphere.theta, sphere.phi)
        L = -n * (n + 1)
        invB = smooth_pinv(B, np.sqrt(sh_smooth) * L)
        n_shm_coeff = (sh_order + 2) * (sh_order + 1) / 2
        shm_coeff = np.zeros((shape + (n_shm_coeff,)))
        invB = invB.T

    if return_odf:
        odf_array = np.zeros((shape + (len(sphere.vertices),)))

    global_max = -np.inf
    for idx in ndindex(shape):
        if not mask[idx]:
            continue

        odf = model.fit(data[idx]).odf(sphere)

        if return_sh:
            shm_coeff[idx] = np.dot(odf, invB)

        if return_odf:
            odf_array[idx] = odf

        gfa_array[idx] = gfa(odf)
        if gfa_array[idx] < gfa_thr:
            global_max = max(global_max, odf.max())
            continue

        # Get peaks of odf
        direction, pk, ind = peak_directions(odf, sphere, relative_peak_threshold,
                                             min_separation_angle)

        # Calculate peak metrics
        global_max = max(global_max, pk[0])
        n = min(npeaks, len(pk))
        qa_array[idx][:n] = pk[:n] - odf.min()

        peak_dirs[idx][:n] = direction[:n]
        peak_indices[idx][:n] = ind[:n]
        peak_values[idx][:n] = pk[:n]

        if normalize_peaks:
            peak_values[idx][:n] /= pk[0]
            peak_dirs[idx] *= peak_values[idx][:, None]

    #gfa_array = gfa_array
    qa_array /= global_max
    #peak_values = peak_values
    #peak_indices = peak_indices

    # The fibernavigator only supports float32. Since this form is mainly
    # for external visualisation, we enforce float32.
    if ravel_peaks:
        peak_dirs = peak_dirs.reshape(shape + (3*npeaks,)).astype('float32')

    pam = PeaksAndMetrics()
    pam.peak_dirs = peak_dirs
    pam.peak_values = peak_values
    pam.peak_indices = peak_indices
    pam.gfa = gfa_array
    pam.qa = qa_array

    if return_sh:
        pam.shm_coeff = shm_coeff
        pam.invB = invB
    else:
        pam.shm_coeff = None
        pam.invB = None

    if return_odf:
        pam.odf = odf_array
    else:
        pam.odf = None

    return pam


def gfa(samples):
    """The general fractional anisotropy of a function evaluated on the unit sphere"""
    diff = samples - samples.mean(-1)[..., None]
    n = samples.shape[-1]
    numer = n*(diff*diff).sum(-1)
    denom = (n-1)*(samples*samples).sum(-1)
    return np.sqrt(numer/denom)


def minmax_normalize(samples, out=None):
    """Min-max normalization of a function evaluated on the unit sphere

    Normalizes samples to ``(samples - min(samples)) / (max(samples) -
    min(samples))`` for each unit sphere.

    Parameters
    ----------
    samples : ndarray (..., N)
        N samples on a unit sphere for each point, stored along the last axis
        of the array.
    out : ndrray (..., N), optional
        An array to store the normalized samples.

    Returns
    -------
    out : ndarray, (..., N)
        Normalized samples.

    """
    if out is None:
        dtype = np.common_type(np.empty(0, 'float32'), samples)
        out = np.array(samples, dtype=dtype, copy=True)
    else:
        out[:] = samples

    sample_mins = np.min(samples, -1)[..., None]
    sample_maxes = np.max(samples, -1)[..., None]
    out -= sample_mins
    out /= (sample_maxes - sample_mins)
    return out
