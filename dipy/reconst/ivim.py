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

    def __init__(self, gtab, fit_method="OLS", *args, **kwargs):
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

    def fit(self, data, mask=None):
        """ Fit method of the Ivim model class

        Parameters
        ----------
        data : array
            The measured signal from one voxel.

        mask : array
            A boolean array used to mark the coordinates in the data that
            should be analyzed that has the shape data.shape[:-1]

        """
        if mask is None:
                # Flatten it to 2D either way:
            data_in_mask = np.reshape(data, (-1, data.shape[-1]))
        else:
            # Check for valid shape of the mask
            if mask.shape != data.shape[:-1]:
                raise ValueError("Mask is not the same shape as data.")
            mask = np.array(mask, dtype=bool, copy=False)
            data_in_mask = np.reshape(data[mask], (-1, data.shape[-1]))

        if self.min_signal is None:
            min_signal = _min_positive_signal(data)
        else:
            min_signal = self.min_signal

        data_in_mask = np.maximum(data_in_mask, min_signal)
        # Change this according to the fit_method
        # params_in_mask = self.fit_method(self.design_matrix, data_in_mask,
        #                                  *self.args, **self.kwargs)

        if mask is None:
            out_shape = data.shape[:-1] + (-1, )
            ivim_params = params_in_mask.reshape(out_shape)
        else:
            ivim_params = np.zeros(data.shape[:-1] + (3,))
            ivim_params[mask, :] = params_in_mask

        return IvimFit(self, ivim_params)

    def predict(self, ivim_params, S0=1):
        """
        Predict a signal for this IvimModel class instance given parameters.

        Parameters
        ----------
        ivim_params : ndarray
            The last dimension should have 3 parameters: f, D, D_star

        S0 : float or ndarray
            The non diffusion-weighted signal in every voxel, or across all
            voxels. Default: 1
        """
        return ivim_prediction(ivim_params, self.gtab, S0)


common_fit_methods = {'OLS': ols_fit,
                      'NLLS': nlls_fit_tensor,
                      }
