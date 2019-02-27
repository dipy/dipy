""" Functions that implement a beltrami framework algorithm to
fit the Free Water elimination model for single shell datasets """
from __future__ import division
import numpy as np
import dipy.reconst.dti as dti
from dipy.core.gradients import gradient_table


def x_manifold(D, mask, out):
    """
    Converts the diffusion tensor components
    into Iwasawa coordinates, according to [1], [2]

    Parameters
    ----------
    D : (..., 6) ndarray
        The lower triangular components of the diffusion tensor,
        in the following order: Dxx, Dxy, Dyy, Dxz, Dyz, Dzz
    mask : boolean array
        boolean mask that marks indices of the data that
        should be converted, with shape X.shape[:-1]
    out : (..., 6)
        Pre-allocated array to store the output
    
    Returns
    -------
    out : (..., 6) ndarray
        The six independent Iwasawa coordinates
    
    References
    ----------
    .. [1] Pasternak, O., Sochen, N., Gur, Y., Intrator, N., & Assaf, Y. (2009).
        Free water elimination and mapping from diffusion MRI.
        Magnetic Resonance in Medicine: An Official Journal of 
        the International Society for Magnetic Resonance in Medicine, 62(3), 717-730.

    .. [2] Gur, Y., & Sochen, N. (2007, October).
        Fast invariant Riemannian DT-MRI regularization.
        In 2007 IEEE 11th International Conference on Computer Vision (pp. 1-7). IEEE.

    """
    Dxx = D[..., 0]
    Dxy = D[..., 1]
    Dyy = D[..., 2]
    Dxz = D[..., 3]
    Dyz = D[..., 4]
    Dzz = D[..., 5]

    out[mask, 0] = Dxx[mask]  # X1
    out[mask, 3] = Dxy[mask] / Dxx[mask]  # X4
    out[mask, 4] = Dxz[mask] / Dxx[mask]  # X5
    out[mask, 1] = Dyy[mask] - out[mask, 0] * out[mask, 3]**2  # X2
    out[mask, 5] = (Dyz[mask] - out[mask, 0] * out[mask, 3] * out[mask, 4]) / out[mask, 1]  # X6
    out[mask, 2] = Dzz[mask] - out[mask, 0] * out[mask, 4]**2 - out[mask, 1] * out[mask, 5]**2  # X3



def d_manifold(X, mask, out):
    """
    Converts Iwasawa coordinates back to
    tensor components
    Parameters
    ----------
    X : (..., 6) ndarray
        The Iwasawa coordinetes, in the following order:
        X1, X2, X3, X4, X5, X6
    mask : boolean array
        boolean mask that marks indices of the data that
        should be converted, with shape X.shape[:-1]
    out : (..., 6)
        Pre-allocated array to store the output
    
    Returns
    -------
    out : (..., 6) ndarray
        The six diffuion tensor components
    
    References
    ----------
    See referneces section of the fucntion "x_manifold"

    """
    X1 = X[..., 0]
    X2 = X[..., 1]
    X3 = X[..., 2]
    X4 = X[..., 3]
    X5 = X[..., 4]
    X6 = X[..., 5]

    out[mask, 0] = X1[mask] # Dxx
    out[mask, 1] = X1[mask] * X4[mask] # Dxy
    out[mask, 2] = X2[mask] + X1[mask] * X4[mask]**2 # Dyy
    out[mask, 3] = X1[mask] * X5[mask] # Dxz
    out[mask, 4] = X1[mask] * X4[mask] * X5[mask] + X2[mask] * X6[mask] # Dyz
    out[mask, 5] = X3[mask] + X1[mask] * X5[mask]**2 + X2[mask] * X6[mask]**2 # Dzz
