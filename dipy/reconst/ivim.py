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
from dti import _min_positive_signal

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


class IvimModel(ReconstModel):
    """Ivim model
    """

    def __init__(self, gtab, fit_method="one_stage_fit", *args, **kwargs):
        """ A Ivim Model
        Parameters
        ----------
        gtab : GradientTable class instance
        fit_method : str or callable
                str can be one of the following:
            'one_stage_fit'
                Try to fit for all b values together

            'two_stage_fit'
                Fit for bvalues > 200 first and then
                for less than 200

        min_signal : float
            The minimum signal value. Needs to be a strictly positive
            number. Default: minimal signal in the data provided to `fit`.

        References
        ----------
        .. [1]
        .. [2]
        .. [3]

        """
        ReconstModel.__init__(self, gtab)

        if not callable(fit_method):
            try:
                fit_method = common_fit_methods[fit_method]
            except KeyError:
                e_s = '"' + str(fit_method) + '" is not a known fit '
                e_s += 'method, the fit method should either be a '
                e_s += 'function or one of the common fit methods'
                raise ValueError(e_s)
        self.fit_method = fit_method
        self.args = args
        self.kwargs = kwargs
        self.min_signal = self.kwargs.pop('min_signal', None)
        if self.min_signal is not None and self.min_signal <= 0:
            e_s = "The `min_signal` key-word argument needs to be strictly"
            e_s += " positive."
            raise ValueError(e_s)
