from __future__ import division, print_function, absolute_import
from dipy.reconst.base import ReconstModel, ReconstFit
import numpy as np

# Classes OdfModel and OdfFit are using API ReconstModel and ReconstFit from
# .base


class OdfModel(ReconstModel):

    """An abstract class to be sub-classed by specific odf models

    All odf models should provide a fit method which may take data as it's
    first and only argument.
    """
    def __init__(self, gtab):
        ReconstModel.__init__(self, gtab)

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
