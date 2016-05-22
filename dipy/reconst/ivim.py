#!/usr/bin/python
""" Classes and functions for fitting ivim model """
from __future__ import division, print_function, absolute_import

import warnings

import functools

import numpy as np

import scipy.optimize as op
import scipy.stats as st
import scipy.ndimage as ndim
import scipy.integrate as si

from dipy.core.gradients import gradient_table
from base import ReconstModel

import matplotlib.pylab as pl


def ivim_prediction(ivim_params, gtab, S0=1.0):
    """
    Predict a signal given the parameters of the IVIM model.

    Parameters
    ----------
    ivim_params : ndarray (N, 3)
        Ivim parameters f, D, D_star for all N voxels.
        The last dimension should have the 3 parameters
        f: Perfusion factor
        D: Diffusion coefficient 
        D_star: Pseudo diffusion coefficient

    gtab : a GradientTable class instance
        The gradient table for this prediction

    S0 : float or ndarray
        The non diffusion-weighted signal in every voxel, or across all
        voxels. Default: 1.0

    Returns
    -------
    pred_signal : ndarray (N, len(bvals))
        The predicted signal values. There will be N sets of Ivim parameters
        and each set will give as many signal values as the b-values.  

    Notes
    -----
    The predicted signal is given by:
    $S(b) = S_0*{f*e^{-b * D_star} + (1-f)* e^{-b * D}}$,

    References
    ----------
    .. [1]

    """
    # Can we better vectorize the equation ?
    bvals = gtab.bvals
    S = np.vectorize(lambda f, D, D_star, b: S0 *
                     (f * np.exp(-b * D_star) + (1 - f) * np.exp(-b * D)))
    N = len(ivim_params)
    pred_signal = []
    # Can we write a better code to iterate
    for i in range(N):
        f, D, D_star = ivim_params[i]
        pred_signal.append(S(f, D, D_star, bvals))
    return pred_signal
