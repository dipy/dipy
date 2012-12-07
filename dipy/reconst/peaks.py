from __future__ import division

import functools

import numpy as np
import scipy.optimize as opt
from .recspeed import local_maxima, remove_similar_vertices, search_descending
from .odf import gfa
from dipy.core.sphere import HemiSphere, Sphere, unit_icosahedron
from dipy.core.ndindex import ndindex
from .cache import Cache

default_sphere = HemiSphere.from_sphere(unit_icosahedron.subdivide(3))


class ODFSampler(Cache):
    def __init__(self, fits):
        self._fits = fits

    def odf(self, sphere, voxel=None):
        odfs = self.cache_get("odfs", sphere)
        if odfs is None:
            odfs = np.array(self._fits.odf(sphere), copy=False, ndmin=4)
            self.cache_set("odfs", sphere, odfs)

        if voxel is None or voxel == ():
            voxel = (0, 0, 0)

        return odfs[voxel]


class DirectionFinder(object):
    """Abstract class for direction finding"""

    def __init__(self,
                 relative_peak_threshold=None,
                 min_separation_angle=None):

        if relative_peak_threshold is None:
            self.relative_peak_threshold = 0.25
        else:
            self.relative_peak_threshold = relative_peak_threshold

        if min_separation_angle is None:
            self.min_separation_angle = 45
        else:
            self.min_separation_angle = min_separation_angle

    def __call__(self, fits):
        shape = getattr(fits, "shape", None)
        if shape is None or np.prod(shape) == 1:
            shape = (1, 1, 1)

        mask = getattr(fits, "mask", None)
        if mask is None:
            mask = np.ones(shape)

        result = np.empty(shape, dtype=object)

        volume_sampler = ODFSampler(fits)

        for ijk in ndindex(shape):
            if mask[ijk]:
                voxel_sampler = functools.partial(volume_sampler.odf,
                                                  voxel=ijk)
                result[ijk] = self._directions_from_sphere(voxel_sampler)

        if result.shape == (1, 1, 1):
            result = result[0, 0, 0]

        return result

    def _directions_from_sphere(self, sphere_eval):
        """To be impemented by subclasses"""
        raise NotImplementedError()


class DiscreteDirectionFinder(DirectionFinder):
    """Discrete direction finder.

    Parameters
    ----------
    sphere : Sphere
        The sphere providing discrete directions for evaluation.
    relative_peak_threshold : float
        Only return peaks greater than ``relative_peak_threshold * m`` where m
        is the largest peak.
    min_separation_angle : float in [0, 90]
        The minimum distance between directions. If two peaks are too close
        only the larger of the two is returned.

    """
    def __init__(self, sphere=default_sphere, relative_peak_threshold=None,
                 min_separation_angle=None):

        super(DiscreteDirectionFinder, self).__init__(
            relative_peak_threshold=relative_peak_threshold,
            min_separation_angle=min_separation_angle)

        if sphere is None:
            sphere = default_sphere

        self.sphere = sphere

    def _directions_from_sphere(self, sphere_eval):
        """Find directions of a function evaluated on a discrete sphere.

        Parameters
        ----------
        sphere_eval : callable
            The function to maximize, should evaluate over a sphere.

        Returns
        -------
        directions : ndarray (N, 3)
            The directions of the N peaks.

        """
        sphere = self.sphere
        discrete_values = sphere_eval(self.sphere)
        return peak_directions(discrete_values, sphere,
                               self.relative_peak_threshold,
                               self.min_separation_angle)


def _nl_peak_finder(sphere_eval, seeds, xtol):
    """Non-linear search for peaks from each seed
    """
    # Helper function
    def _helper(x):
        sphere = Sphere(theta=x[0], phi=x[1])
        return -sphere_eval(sphere)

    # Non-linear search
    theta = np.zeros(len(seeds))
    phi = np.zeros(len(seeds))
    for i in xrange(len(seeds)):
        peak = opt.fmin(_helper, seeds[i], xtol=xtol, disp=False)
        theta[i], phi[i] = peak
    # Return one peak for each seed
    return theta, phi


class NonLinearDirectionFinder(DirectionFinder):
    """Non Linear Direction Finder

    Parameters
    ----------
    sphere : Sphere
        The Sphere providing discrete directions for evaluation.
    relative_peak_threshold : float
        Only return peaks greater than ``relative_peak_threshold * m`` where m
        is the largest peak.
    min_separation_angle : float in [0, 90]
        The minimum distance between directions. If two peaks are too close
        only the larger of the two is returned.
    xtol : float
        Relative tolerance for optimization.

    """
    def __init__(self, sphere=None, relative_peak_threshold=None,
                 min_separation_angle=None, xtol=1e-7):

        super(NonLinearDirectionFinder, self).__init__(
            relative_peak_threshold=relative_peak_threshold,
            min_separation_angle=min_separation_angle)

        if sphere is None:
            sphere = default_sphere

        self.sphere = sphere
        self.xtol = xtol

    def _directions_from_sphere(self, sphere_eval):
        """Find directions of a function evaluated on a discrete sphere

        Parameters
        ----------
        sphere_eval : callable
            The function to maximize, should evaluate over a sphere.

        Returns
        -------
        directions : ndarray (N, 3)
            The directions of the N peaks.

        """
        # Find discrete peaks for use as seeds in lon-linear search
        discrete_values = sphere_eval(self.sphere)
        values, indices = local_maxima(discrete_values, self.sphere.edges)
        n = search_descending(values, self.relative_peak_threshold)
        indices = indices[:n]
        seeds = np.column_stack([self.sphere.theta[indices],
                                 self.sphere.phi[indices]])

        # Non-linear search
        peak_theta, peak_phi = _nl_peak_finder(sphere_eval, seeds, self.xtol)

        # Evaluate on new-found peaks
        small_sphere = Sphere(theta=peak_theta, phi=peak_phi)
        values = sphere_eval(small_sphere)
        # Sort in descending order
        order = values.argsort()[::-1]
        values = values[order]
        directions = small_sphere.vertices[order]
        # Remove peaks too close to each-other
        directions = remove_similar_vertices(directions,
                                             self.min_separation_angle,)
        return directions


def peak_directions(odf, sphere, relative_peak_threshold,
                    min_separation_angle):
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

    """
    values, indices = local_maxima(odf, sphere.edges)
    # If there is only one peak return
    if len(indices) == 1:
        return sphere.vertices[indices]

    n = search_descending(values, relative_peak_threshold)
    indices = indices[:n]
    directions = sphere.vertices[indices]
    directions = remove_similar_vertices(directions, min_separation_angle)
    return directions


class PeaksAndMetrics(object):
    pass


def peaks_from_model(model, data, sphere, relative_peak_threshold,
                     min_separation_angle, mask=None, return_odf=False,
                     gfa_thr=0.02, normalize_peaks=False):
    """Fits the model to data and computes peaks and metrics.

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
    gfa_thr : float
        Voxels with gfa less than `gfa_thr` are skipped, no peaks are returned.
    normalize_peaks : bool
        If true, all peak values are calculated relative to `max(odf)`.

    """

    data_flat = data.reshape((-1, data.shape[-1]))
    size = len(data_flat)
    if mask is None:
        mask = np.ones(size, dtype='bool')
    else:
        mask = mask.ravel()
        if len(mask) != size:
            raise ValueError("mask is not the same size as data")

    npeaks = 5
    gfa_array = np.zeros(size)
    qa_array = np.zeros((size, npeaks))
    peak_values = np.zeros((size, npeaks))
    peak_indices = np.zeros((size, npeaks), dtype='int')
    peak_indices.fill(-1)

    if return_odf:
        odf_array = np.zeros((size, len(sphere.vertices)))

    global_max = -np.inf
    for i, sig in enumerate(data_flat):
        if not mask[i]:
            continue
        odf = model.fit(sig).odf(sphere)
        if return_odf:
            odf_array[i] = odf

        gfa_array[i] = gfa(odf)
        if gfa_array[i] < gfa_thr:
            global_max = max(global_max, odf.max())
            continue
        pk, ind = local_maxima(odf, sphere.edges)

        # Remove small peaks.
        gt_threshold = pk >= (relative_peak_threshold * pk[0])
        pk = pk[gt_threshold]
        ind = ind[gt_threshold]

        # Keep peaks which are unique, which means remove peaks that are too
        # close to a larger peak.
        _, where_uniq = remove_similar_vertices(sphere.vertices[ind],
                                                min_separation_angle,
                                                return_index=True)
        pk = pk[where_uniq]
        ind = ind[where_uniq]

        # Calculate peak metrics
        global_max = max(global_max, pk[0])
        n = min(npeaks, len(pk))
        qa_array[i, :n] = pk[:n] - odf.min()
        if normalize_peaks:
            peak_values[i, :n] = pk[:n] / pk[0]
        else:
            peak_values[i, :n] = pk[:n]
        peak_indices[i, :n] = ind[:n]

    shape = data.shape[:-1]
    gfa_array = gfa_array.reshape(shape)
    qa_array = qa_array.reshape(shape + (npeaks,)) / global_max
    peak_values = peak_values.reshape(shape + (npeaks,))
    peak_indices = peak_indices.reshape(shape + (npeaks,))

    pam = PeaksAndMetrics()
    pam.peak_values = peak_values
    pam.peak_indices = peak_indices
    pam.gfa = gfa_array
    pam.qa = qa_array
    if return_odf:
        pam.odf = odf_array.reshape(shape + odf_array.shape[-1:])
    else:
        pam.odf = None

    return pam
