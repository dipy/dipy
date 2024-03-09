import tempfile

from multiprocessing import Pool
from itertools import repeat
from os import path


import numpy as np
import scipy.optimize as opt

from dipy.reconst.odf import gfa
from dipy.reconst.recspeed import (local_maxima, remove_similar_vertices,
                                   search_descending)
from dipy.core.sphere import Sphere
from dipy.data import default_sphere
from dipy.core.ndindex import ndindex
from dipy.reconst.shm import sh_to_sf_matrix
from dipy.reconst.eudx_direction_getter import EuDXDirectionGetter
from dipy.utils.deprecator import deprecated_params
from dipy.utils.multiproc import determine_num_processes


def peak_directions_nl(sphere_eval, relative_peak_threshold=.25,
                       min_separation_angle=25, sphere=default_sphere,
                       xtol=1e-7):
    """Non Linear Direction Finder.

    Parameters
    ----------
    sphere_eval : callable
        A function which can be evaluated on a sphere.
    relative_peak_threshold : float
        Only return peaks greater than ``relative_peak_threshold * m`` where m
        is the largest peak.
    min_separation_angle : float in [0, 90]
        The minimum distance between directions. If two peaks are too close
        only the larger of the two is returned.
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
    for i in range(num_seeds):
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


def peak_directions(odf, sphere, relative_peak_threshold=.5,
                    min_separation_angle=25, is_symmetric=True):
    """Get the directions of odf peaks.

    Peaks are defined as points on the odf that are greater than at least one
    neighbor and greater than or equal to all neighbors. Peaks are sorted in
    descending order by their values then filtered based on their relative size
    and spacing on the sphere. An odf may have 0 peaks, for example if the odf
    is perfectly isotropic.

    Parameters
    ----------
    odf : 1d ndarray
        The odf function evaluated on the vertices of `sphere`
    sphere : Sphere
        The Sphere providing discrete directions for evaluation.
    relative_peak_threshold : float in [0., 1.]
        Only peaks greater than ``min + relative_peak_threshold * scale`` are
        kept, where ``min = max(0, odf.min())`` and
        ``scale = odf.max() - min``.
    min_separation_angle : float in [0, 90]
        The minimum distance between directions. If two peaks are too close
        only the larger of the two is returned.
    is_symmetric : bool, optional
        If True, v is considered equal to -v.

    Returns
    -------
    directions : (N, 3) ndarray
        N vertices for sphere, one for each peak
    values : (N,) ndarray
        peak values
    indices : (N,) ndarray
        peak indices of the directions on the sphere

    Notes
    -----
    If the odf has any negative values, they will be clipped to zeros.

    """
    values, indices = local_maxima(odf, sphere.edges)

    # If there is only one peak return
    n = len(values)
    if n == 0 or (values[0] < 0.):
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0, dtype=int)
    elif n == 1:
        return sphere.vertices[indices], values, indices

    odf_min = np.min(odf)
    odf_min = max(odf_min, 0.)
    # because of the relative threshold this algorithm will give the same peaks
    # as if we divide (values - odf_min) with (odf_max - odf_min) or not so
    # here we skip the division to increase speed
    values_norm = (values - odf_min)

    # Remove small peaks
    n = search_descending(values_norm, relative_peak_threshold)
    indices = indices[:n]
    directions = sphere.vertices[indices]

    # Remove peaks too close together
    directions, uniq = remove_similar_vertices(directions,
                                               min_separation_angle,
                                               return_index=True,
                                               remove_antipodal=is_symmetric)
    values = values[uniq]
    indices = indices[uniq]
    return directions, values, indices


def _pam_from_attrs(klass, sphere, peak_indices, peak_values, peak_dirs,
                    gfa, qa, shm_coeff, B, odf):
    """
    Construct PeaksAndMetrics object (or subclass) from its attributes.

    This is also useful for pickling/unpickling of these objects (see also
    :func:`__reduce__` below).

    Parameters
    ----------
    klass : class
        The class of object to be created.
    sphere : `Sphere` class instance.
        Sphere for discretization.
    peak_indices : ndarray
        Indices (in sphere vertices) of the peaks in each voxel.
    peak_values : ndarray
        The value of the peaks.
    peak_dirs : ndarray
        The direction of each peak.
    gfa : ndarray
        The Generalized Fractional Anisotropy in each voxel.
    qa : ndarray
        Quantitative Anisotropy in each voxel.
    shm_coeff : ndarray
        The coefficients of the spherical harmonic basis for the ODF in
        each voxel.
    B : ndarray
        The spherical harmonic matrix, for multiplication with the
        coefficients.
    odf : ndarray
        The orientation distribution function on the sphere in each voxel.

    Returns
    -------
    pam : Instance of the class `klass`.
    """
    this_pam = klass()
    this_pam.sphere = sphere
    this_pam.peak_dirs = peak_dirs
    this_pam.peak_values = peak_values
    this_pam.peak_indices = peak_indices
    this_pam.gfa = gfa
    this_pam.qa = qa
    this_pam.shm_coeff = shm_coeff
    this_pam.B = B
    this_pam.odf = odf
    return this_pam


class PeaksAndMetrics(EuDXDirectionGetter):
    def __reduce__(self): return _pam_from_attrs, (self.__class__,
                                                   self.sphere,
                                                   self.peak_indices,
                                                   self.peak_values,
                                                   self.peak_dirs,
                                                   self.gfa,
                                                   self.qa,
                                                   self.shm_coeff,
                                                   self.B,
                                                   self.odf)


def _peaks_from_model_parallel(model, data, sphere, relative_peak_threshold,
                               min_separation_angle, mask, return_odf,
                               return_sh, gfa_thr, normalize_peaks, sh_order,
                               sh_basis_type, legacy, npeaks, B, invB,
                               num_processes):

    shape = list(data.shape)
    data = np.reshape(data, (-1, shape[-1]))
    n = data.shape[0]
    nbr_chunks = num_processes ** 2
    chunk_size = int(np.ceil(n / nbr_chunks))
    indices = list(zip(np.arange(0, n, chunk_size),
                       np.arange(0, n, chunk_size) + chunk_size))

    with tempfile.TemporaryDirectory() as tmpdir:
        data_file_name = path.join(tmpdir, 'data.npy')
        np.save(data_file_name, data)
        if mask is not None:
            mask = mask.flatten()
            mask_file_name = path.join(tmpdir, 'mask.npy')
            np.save(mask_file_name, mask)
        else:
            mask_file_name = None

        pool = Pool(num_processes)

        pam_res = pool.map(_peaks_from_model_parallel_sub,
                           zip(repeat((data_file_name, mask_file_name)),
                               indices,
                               repeat(model),
                               repeat(sphere),
                               repeat(relative_peak_threshold),
                               repeat(min_separation_angle),
                               repeat(return_odf),
                               repeat(return_sh),
                               repeat(gfa_thr),
                               repeat(normalize_peaks),
                               repeat(sh_order),
                               repeat(sh_basis_type),
                               repeat(legacy),
                               repeat(npeaks),
                               repeat(B),
                               repeat(invB)))
        pool.close()

        pam = PeaksAndMetrics()
        pam.sphere = sphere
        # use memmap to reduce the memory usage
        pam.gfa = np.memmap(path.join(tmpdir, 'gfa.npy'),
                            dtype=pam_res[0].gfa.dtype,
                            mode='w+',
                            shape=(data.shape[0]))

        pam.peak_dirs = np.memmap(path.join(tmpdir, 'peak_dirs.npy'),
                                  dtype=pam_res[0].peak_dirs.dtype,
                                  mode='w+',
                                  shape=(data.shape[0], npeaks, 3))
        pam.peak_values = np.memmap(path.join(tmpdir, 'peak_values.npy'),
                                    dtype=pam_res[0].peak_values.dtype,
                                    mode='w+',
                                    shape=(data.shape[0], npeaks))
        pam.peak_indices = np.memmap(path.join(tmpdir, 'peak_indices.npy'),
                                     dtype=pam_res[0].peak_indices.dtype,
                                     mode='w+',
                                     shape=(data.shape[0], npeaks))
        pam.qa = np.memmap(path.join(tmpdir, 'qa.npy'),
                           dtype=pam_res[0].qa.dtype,
                           mode='w+',
                           shape=(data.shape[0], npeaks))
        if return_sh:
            nbr_shm_coeff = (sh_order + 2) * (sh_order + 1) // 2
            pam.shm_coeff = np.memmap(path.join(tmpdir, 'shm.npy'),
                                      dtype=pam_res[0].shm_coeff.dtype,
                                      mode='w+',
                                      shape=(data.shape[0], nbr_shm_coeff))
            pam.B = pam_res[0].B
        else:
            pam.shm_coeff = None
            pam.invB = None
        if return_odf:
            pam.odf = np.memmap(path.join(tmpdir, 'odf.npy'),
                                dtype=pam_res[0].odf.dtype,
                                mode='w+',
                                shape=(data.shape[0], len(sphere.vertices)))
        else:
            pam.odf = None

        # copy subprocesses pam to a single pam (memmaps)
        for i, (start_pos, end_pos) in enumerate(indices):
            pam.gfa[start_pos: end_pos] = pam_res[i].gfa
            pam.peak_dirs[start_pos: end_pos] = pam_res[i].peak_dirs
            pam.peak_values[start_pos: end_pos] = pam_res[i].peak_values
            pam.peak_indices[start_pos: end_pos] = pam_res[i].peak_indices
            pam.qa[start_pos: end_pos] = pam_res[i].qa
            if return_sh:
                pam.shm_coeff[start_pos: end_pos] = pam_res[i].shm_coeff
            if return_odf:
                pam.odf[start_pos: end_pos] = pam_res[i].odf

        # load memmaps to arrays and reshape the metric
        shape[-1] = -1
        pam.gfa = np.reshape(np.array(pam.gfa), shape[:-1])
        pam.peak_dirs = np.reshape(np.array(pam.peak_dirs), shape + [3])
        pam.peak_values = np.reshape(np.array(pam.peak_values), shape)
        pam.peak_indices = np.reshape(np.array(pam.peak_indices), shape)
        pam.qa = np.reshape(np.array(pam.qa), shape)
        if return_sh:
            pam.shm_coeff = np.reshape(np.array(pam.shm_coeff), shape)
        if return_odf:
            pam.odf = np.reshape(np.array(pam.odf), shape)

        # Make sure all worker processes have exited before leaving context
        # manager in order to prevent temporary file deletion errors in windows
        pool.join()

    return pam


def _peaks_from_model_parallel_sub(args):
    (data_file_name, mask_file_name) = args[0]
    (start_pos, end_pos) = args[1]
    model = args[2]
    sphere = args[3]
    relative_peak_threshold = args[4]
    min_separation_angle = args[5]
    return_odf = args[6]
    return_sh = args[7]
    gfa_thr = args[8]
    normalize_peaks = args[9]
    sh_order = args[10]
    sh_basis_type = args[11]
    legacy = args[12]
    npeaks = args[13]
    B = args[14]
    invB = args[15]

    data = np.load(data_file_name, mmap_mode='r')[start_pos:end_pos]
    if mask_file_name is not None:
        mask = np.load(mask_file_name, mmap_mode='r')[start_pos:end_pos]
    else:
        mask = None

    return peaks_from_model(model, data, sphere, relative_peak_threshold,
                            min_separation_angle, mask, return_odf,
                            return_sh, gfa_thr, normalize_peaks,
                            sh_order, sh_basis_type, legacy, npeaks, B, invB,
                            parallel=False, num_processes=None)


@deprecated_params('sh_order', 'sh_order_max', since='1.9', until='2.0')
def peaks_from_model(model, data, sphere, relative_peak_threshold,
                     min_separation_angle, mask=None, return_odf=False,
                     return_sh=True, gfa_thr=0, normalize_peaks=False,
                     sh_order_max=8, sh_basis_type=None, legacy=True, npeaks=5,
                     B=None, invB=None, parallel=False, num_processes=None):
    """Fit the model to data and computes peaks and metrics

    Parameters
    ----------
    model : a model instance
        `model` will be used to fit the data.
    data : ndarray
        Diffusion data.
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
    sh_order_max : int, optional
        Maximum SH order (l) in the SH fit.  For `sh_order_max`, there
        will be
        ``(sh_order_max + 1) * (sh_order_max + 2) / 2``
        SH coefficients (default 8).
    sh_basis_type : {None, 'tournier07', 'descoteaux07'}
        ``None`` for the default DIPY basis,
        ``tournier07`` for the Tournier 2007 [2]_ basis, and
        ``descoteaux07`` for the Descoteaux 2007 [1]_ basis
        (``None`` defaults to ``descoteaux07``).
    legacy: bool, optional
        True to use a legacy basis definition for backward compatibility
        with previous ``tournier07`` and ``descoteaux07`` implementations.
    npeaks : int
        Maximum number of peaks found (default 5 peaks).
    B : ndarray, optional
        Matrix that transforms spherical harmonics to spherical function
        ``sf = np.dot(sh, B)``.
    invB : ndarray, optional
        Inverse of B.
    parallel: bool
        If True, use multiprocessing to compute peaks and metric
        (default False). Temporary files are saved in the default temporary
        directory of the system. It can be changed using ``import tempfile``
        and ``tempfile.tempdir = '/path/to/tempdir'``.
    num_processes: int, optional
        If `parallel` is True, the number of subprocesses to use
        (default multiprocessing.cpu_count()). If < 0 the maximal number of
        cores minus ``num_processes + 1`` is used (enter -1 to use as many
        cores as possible). 0 raises an error.

    Returns
    -------
    pam : PeaksAndMetrics
        An object with ``gfa``, ``peak_directions``, ``peak_values``,
        ``peak_indices``, ``odf``, ``shm_coeffs`` as attributes

    References
    ----------
    .. [1] Descoteaux, M., Angelino, E., Fitzgibbons, S. and Deriche, R.
           Regularized, Fast, and Robust Analytical Q-ball Imaging.
           Magn. Reson. Med. 2007;58:497-510.
    .. [2] Tournier J.D., Calamante F. and Connelly A. Robust determination
           of the fibre orientation distribution in diffusion MRI:
           Non-negativity constrained super-resolved spherical deconvolution.
           NeuroImage. 2007;35(4):1459-1472.

    """
    if return_sh and (B is None or invB is None):
        B, invB = sh_to_sf_matrix(
            sphere, sh_order_max, sh_basis_type, return_inv=True, legacy=legacy)

    num_processes = determine_num_processes(num_processes)

    if parallel and num_processes > 1:
        # It is mandatory to provide B and invB to the parallel function.
        # Otherwise, a call to np.linalg.pinv is made in a subprocess and
        # makes it timeout on some system.
        # see https://github.com/dipy/dipy/issues/253 for details
        return _peaks_from_model_parallel(model,
                                          data, sphere,
                                          relative_peak_threshold,
                                          min_separation_angle,
                                          mask, return_odf,
                                          return_sh,
                                          gfa_thr,
                                          normalize_peaks,
                                          sh_order_max,
                                          sh_basis_type,
                                          legacy,
                                          npeaks,
                                          B,
                                          invB,
                                          num_processes)

    shape = data.shape[:-1]
    if mask is None:
        mask = np.ones(shape, dtype='bool')
    else:
        if mask.shape != shape:
            raise ValueError("Mask is not the same shape as data.")

    gfa_array = np.zeros(shape)
    qa_array = np.zeros((shape + (npeaks,)))

    peak_dirs = np.zeros((shape + (npeaks, 3)))
    peak_values = np.zeros((shape + (npeaks,)))
    peak_indices = np.zeros((shape + (npeaks,)), dtype=np.int32)
    peak_indices.fill(-1)

    if return_sh:
        n_shm_coeff = (sh_order_max + 2) * (sh_order_max + 1) // 2
        shm_coeff = np.zeros((shape + (n_shm_coeff,)))

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
        direction, pk, ind = peak_directions(odf, sphere,
                                             relative_peak_threshold,
                                             min_separation_angle)

        # Calculate peak metrics
        if pk.shape[0] != 0:
            global_max = max(global_max, pk[0])

            n = min(npeaks, pk.shape[0])
            qa_array[idx][:n] = pk[:n] - odf.min()

            peak_dirs[idx][:n] = direction[:n]
            peak_indices[idx][:n] = ind[:n]
            peak_values[idx][:n] = pk[:n]

            if normalize_peaks:
                peak_values[idx][:n] /= pk[0]
                peak_dirs[idx] *= peak_values[idx][:, None]

    qa_array /= global_max

    return _pam_from_attrs(PeaksAndMetrics,
                           sphere,
                           peak_indices,
                           peak_values,
                           peak_dirs,
                           gfa_array,
                           qa_array,
                           shm_coeff if return_sh else None,
                           B if return_sh else None,
                           odf_array if return_odf else None)


def reshape_peaks_for_visualization(peaks):
    """Reshape peaks for visualization.

    Reshape and convert to float32 a set of peaks for visualisation with mrtrix
    or the fibernavigator.

    Parameters
    ----------
    peaks: nd array (..., N, 3) or PeaksAndMetrics object
        The peaks to be reshaped and converted to float32.

    Returns
    -------
    peaks : nd array (..., 3*N)
    """
    if isinstance(peaks, PeaksAndMetrics):
        peaks = peaks.peak_dirs

    return peaks.reshape(np.append(peaks.shape[:-2], -1)).astype('float32')
