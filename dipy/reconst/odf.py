from __future__ import division
from warnings import warn
import numpy as np
from .recspeed import local_maxima, remove_similar_vertices
from ..core.onetime import auto_attr
from dipy.core.sphere import unique_edges, unit_icosahedron, HemiSphere
#Classes OdfModel and OdfFit are using API ReconstModel and ReconstFit from .base 

default_sphere = HemiSphere.from_sphere(unit_icosahedron.subdivide(3))

class DirectionFinder(object):
    """Abstract class for direction finding"""

    def __init__(self):
        self._config = {}

    def __call__(self, sphere_eval):
        """To be impemented by subclasses"""
        raise NotImplementedError()

    def config(self, **kwargs):
        """Update direction finding parameters"""
        for i in kwargs:
            if i not in self._config:
                warn("{} is not a known parameter".format(i))
        self._config.update(kwargs)


class DiscreteDirectionFinder(DirectionFinder):
    """Discrete Direction Finder

    Parameters
    ----------
    sphere : Sphere
        The Sphere providing discrete directions for evaluation.
    relative_peak_threshold : float
        Only return peaks greater than ``relative_peak_threshold * m`` where m
        is the largest peak.
    min_separation_angle : float in [0, 90]
        The minimum distance between directions. If two peaks are too close only
        the larger of the two is returned.

    Returns
    -------
    directions : ndarray (N, 3)
        The directions of the N peaks.
    """

    def __init__(self, sphere=default_sphere, relative_peak_threshold=.25,
                 min_separation_angle=45):
        self._config = {"sphere": sphere,
                        "relative_peak_threshold": relative_peak_threshold,
                        "min_separation_angle": min_separation_angle}

    def __call__(self, sphere_eval):
        """Find directions of a function evaluated on a discrete sphere"""
        sphere = self._config["sphere"]
        relative_peak_threshold = self._config["relative_peak_threshold"]
        min_separation_angle = self._config["min_separation_angle"]
        discrete_values = sphere_eval(sphere)
        return peak_directions(discrete_values, sphere, 
                               relative_peak_threshold, min_separation_angle)

class OdfModel(object):
    """An abstract class to be sub-classed by specific odf models

    All odf models should provide a fit method which may take data as it's
    first and only argument.
    """
    direction_finder = DiscreteDirectionFinder()
    def fit(self, data):
        """To be implemented by specific odf models"""
        raise NotImplementedError("To be implemented in sub classes")

class OdfFit(object):
    def odf(self, sphere):
        """To be implemented but specific odf models"""
        raise NotImplementedError("To be implemented in sub classes")

    @auto_attr
    def directions(self):
        return self.model.direction_finder(self.odf)


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

    # Here we use the fact that we know values is sorted descending
    # This is like indices = indices[values > threshold] but faster
    threshold = values[0] * relative_peak_threshold
    too_small = values < threshold
    first_too_small = too_small.argmax()
    if first_too_small > 0:
        indices = indices[:first_too_small]

    directions = sphere.vertices[indices]
    directions = remove_similar_vertices(directions, min_separation_angle)
    return directions

class PeaksAndMetrics(object):
    pass

def peaks_from_model(model, data, sphere, relative_peak_threshold,
                     min_separation_angle, mask=None, return_odf=False,
                     gfa_thr=0.02, normalize_peaks=False):
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

def gfa(samples):
    """The general fractional anisotropy of a function evaluated on the unit sphere"""
    diff = samples - samples.mean(-1)[..., None]
    n = samples.shape[-1]
    numer = n*(diff*diff).sum(-1)
    denom = (n-1)*(samples*samples).sum(-1)
    return np.sqrt(numer/denom)
