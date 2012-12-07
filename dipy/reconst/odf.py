from __future__ import division
import numpy as np


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


def gfa(samples):
    """The general fractional anisotropy of a function evaluated on the unit
    sphere."""
    diff = samples - samples.mean(-1)[..., None]
    n = samples.shape[-1]
    numer = n * (diff * diff).sum(-1)
    denom = (n - 1) * (samples * samples).sum(-1)
    return np.sqrt(numer / denom)
