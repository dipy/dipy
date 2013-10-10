from __future__ import division, print_function, absolute_import
from .base import ReconstModel, ReconstFit
import numpy as np

# Classes OdfModel and OdfFit are using API ReconstModel and ReconstFit from
# .base

class OdfModel(ReconstModel):

    """An abstract class to be sub-classed by specific odf models

    All odf models should provide a fit method which may take data as it's
    first and only argument.
    """

    def fit(self, data):
        """To be implemented by specific odf models"""
        raise NotImplementedError("To be implemented in sub classes")


class OdfFit(ReconstFit):

    def odf(self, sphere):
        """To be implemented but specific odf models"""
        raise NotImplementedError("To be implemented in sub classes")


def gfa(samples):
    """The general fractional anisotropy of a function evaluated
    on the unit sphere"""
    diff = samples - samples.mean(-1)[..., None]
    n = samples.shape[-1]
    numer = n * (diff * diff).sum(-1)
    denom = (n - 1) * (samples * samples).sum(-1)
    return np.sqrt(numer / denom)


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


def reshape_peaks_for_visualisation(peaks):
    """Reshape peaks for visualisation.

    Reshape and convert to float32 a set of peaks for visualisation with mrtrix
    or the fibernavigator.

    Parameters:
    -----------
    peaks: nd array (..., N, 3) or PeaksAndMetrics object
        The peaks to be reshaped and converted to float32.

    Returns:
    --------
    peaks : nd array (..., 3*N) or PeaksAndMetrics object
    """

    if isinstance(peaks, PeaksAndMetrics):
        peaks.peak_dirs = np.reshape(peaks.peak_dirs,
                                     peaks.peak_dirs.shape[:-2], -1).astype('float32')
    else:
        peaks = np.reshape(peaks, peaks.shape[:-2], -1).astype('float32')

    return peaks
