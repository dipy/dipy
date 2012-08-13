from __future__ import division
import numpy as np
from .recspeed import local_maxima, remove_similar_vertices
from dipy.core.sphere import unique_edges
# Classes OdfModel and OdfFit are using API ReconstModel and ReconstFit from
# .base

class OdfModel(object):
    """An abstract class to be sub-classed by specific odf models

    All odf models should provide a fit method which may take data as it's
    first and only argument.
    """
    sphere = None
    def fit(self, data):
        """To be implemented but specific odf models"""
        raise NotImplementedError("To be implemented in sub classes")


class OdfFit(object):
    def odf(self):
        """To be implemented but specific odf models"""
        raise NotImplementedError("To be implemented in sub classes")


def peak_directions(odf, sphere, relative_peak_threshold,
                    peak_separation_angle):
    """Get the directions of odf peaks

    Parameters
    ----------
    odf : 1d ndarray
        The odf function evaluated on the vertices of `sphere`
    sphere :
        The sphere on which odf was evaluated
    relative_peak_threshold : float
        A relative threshold for excluding small peaks
    peak_separation_angle : float
        An angle threshold in degrees. Peaks too close to a larger peak are
        excluded.

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
    directions, mappiing = remove_similar_vertices(directions,
                                                   peak_separation_angle)
    return directions

class PeaksAndMetrics(object):
    pass

def peaks_from_model(model, data, mask=None, return_odf=False, gfa_thr=0.02, 
                     normalize_peaks=False):
    """Fits the model to data and computes peaks and metrics"""

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
        odf_array = np.zeros((size, len(model.sphere.vertices)))

    global_max = -np.inf
    for i, sig in enumerate(data_flat):
        if not mask[i]:
            continue
        odf = model.fit(sig).odf()
        if return_odf:
            odf_array[i] = odf

        gfa_array[i] = gfa(odf)
        if gfa_array[i] < gfa_thr:
            global_max = max(global_max, odf.max())
            continue
        pk, ind = local_maxima(odf, model.sphere.edges)
        """
        # Will update this later when filter_peaks is nailed down
        pk, ind = remove_similar_vertices(pk, ind,
                                          model._distance_matrix,
                                          model.relative_peak_threshold,
                                          model._cos_distance_threshold)
        """
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
