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
    r"""The general fractional anisotropy of a function evaluated
    on the unit sphere

    Parameters
    ----------
    samples : ndarray
        Values of data on the unit sphere.

    Returns
    -------
    gfa : ndarray
        GFA evaluated in each entry of the array, along the last dimension.
        An `np.nan` is returned for coordinates that contain all-zeros in
        `samples`.

    Notes
    -----
    The GFA is defined as [1]_ ::

        \sqrt{\frac{n \sum_i{(\Psi_i - <\Psi>)^2}}{(n-1) \sum{\Psi_i ^ 2}}}

    Where $\Psi$ is an orientation distribution function sampled discretely on
    the unit sphere and angle brackets denote average over the samples on the
    sphere.

    .. [1] Quality assessment of High Angular Resolution Diffusion Imaging
           data using bootstrap on Q-ball reconstruction. J. Cohen Adad, M.
           Descoteaux, L.L. Wald. JMRI 33: 1194-1208.
    """
    diff = samples - samples.mean(-1)[..., None]
    n = samples.shape[-1]
    numer = np.array([n * (diff ** 2).sum(-1)])
    denom = np.array([(n - 1) * (samples ** 2).sum(-1)])
    result = np.ones_like(denom) * np.nan
    idx = np.where(denom > 0)
    result[idx] = np.sqrt(numer[idx] / denom[idx])
    return result.squeeze()


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
