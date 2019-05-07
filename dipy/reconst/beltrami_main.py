"""
Implements the main gradient descent function to estimate
the Free Water parameter from single-shell diffusion data.
"""

from __future__ import division
import numpy as np
import dipy.reconst.dti as dti
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import design_matrix
import beltrami as blt  # importing the functions from beltrami.py


def get_atten(data, gtab):
    """
    Preprocessing, get S0 and Ak

    Parameters
    ----------
    data : (X, Y, Z, K) ndarray
        Diffusion data acquired for K directions.
    gtab : (K, 7)
        Gradients table class instance.
    
    Returns
    -------
    S0 : (X, Y, Z) ndarray
        Non diffusion-weighted volume (bval = 0).
    Ak : (X, Y, Z, K) ndarray
        Normalized attenuations (Ak = data / S0).
    bvals : (K) ndarray
        Vector containing the bvalues.
    bvecs : (K, 3) ndarray
        Normalized gradient directions.

    Notes
    -----
    If multiple S0 volumes are present, the mean volume is returned,
    the bvals and bvecs returned are cropped to exclude the S0 volumes.
    """

    ind = gtab.b0s_mask
    bvals = gtab.bvals[~ind]
    bvecs = gtab.bvecs[~ind, :]
    bval = bvals[0]

    # getting S0 and Sk
    ind = gtab.b0s_mask
    S0s = data[..., ind]
    S0 = np.mean(S0s, axis=-1)[..., np.newaxis]
    S0[S0 < 0.0001] = 0.0001
    Sk = data[..., ~ind]

    # getting Ak
    D_min = 0.01
    D_max = 5
    A_min = np.exp(-bval * D_max)
    A_max = np.exp(-bval * D_min)
    Ak = Sk / S0
    Ak = np.clip(Ak, A_min, A_max)

    return (S0, Ak, bvals, bvecs)
