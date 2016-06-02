#!/usr/bin/python
""" Classes and functions for fitting ivim model """
from __future__ import division, print_function, absolute_import
import warnings
import functools
import numpy as np
from scipy.optimize import leastsq

from dipy.core.gradients import gradient_table
from dipy.reconst.base import ReconstModel
from dipy.reconst.dti import _min_positive_signal


def ivim_function(params, bvals):
    """The ivim function
    Parameters
    ----------
    params : array
             The IVIM parameters [S0, f, D_star, D]

    bvals  : array
             bvalues
    """
    S0, f, D_star, D = params
    S = S0 * (f * np.exp(-bvals * D_star) + (1 - f) * np.exp(-bvals * D))
    return S


def _ivim_error(params, bvals, signal):
    """Error function to be used in fitting the model
    """
    return signal - ivim_function(params, bvals)


class IvimModel(ReconstModel):
    """Ivim model
    """

    def __init__(self, gtab, split_b=200.0, min_signal=None):
        """
        An IVIM model

        Parameters
        ----------
        gtab : GradientTable class instance
        split_b : split the data at this b-value for a two stage fit

        References
        ----------
        .. [1] Le Bihan, Denis, et al. "Separation of diffusion
                   and perfusion in intravoxel incoherent motion MR
                   imaging." Radiology 168.2 (1988): 497-505.
        .. [2] Federau, Christian, et al. "Quantitative measurement
                   of brain perfusion with intravoxel incoherent motion
                   MR imaging." Radiology 265.3 (2012): 874-881.
        """
        ReconstModel.__init__(self, gtab)

        self.min_signal = min_signal
        if self.min_signal is not None and self.min_signal <= 0:
            e_s = "The `min_signal` key-word argument needs to be strictly"
            e_s += " positive."
            raise ValueError(e_s)

    def fit(self, data, mask=None, x0=None, fit_method="one_stage"):
        """ Fit method of the Ivim model class

        Parameters
        ----------
        data : array
            The measured signal from one voxel.
        mask : array
            A boolean array used to mark the coordinates in the data that
            should be analyzed that has the shape data.shape[:-1]
        x0 : array
            Initial guesses for the parameters S0, f, D_star and D
            Default x0 = [1.0, 0.10, 0.001, 0.0009]
            These guess parameters are taken from the Federau paper
            Dimension can either be 1 or (N, 4) where N is the number of
            voxels in the data.

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

        # Generate guess_parameters for all voxels
        if x0 is None:
            x0 = np.ones(
                (data.shape[0], 4)) * [1.0, 0.10, 0.001, 0.0009]
        else:
            if x0.shape != (data.shape[0], 4):
                raise ValueError(
                    "Guess params should be of shape (num_voxels,4)")

        data_in_mask = np.maximum(data_in_mask, min_signal)

        if fit_method == "one_stage":
            params_in_mask = one_stage(data_in_mask, self.gtab,
                                           x0)
        elif fit_method == "two_stage":
            pass
        else:
            raise ValueError("""Fit method must be either
                             'one_stage' or 'two_stage'""")

        if mask is None:
            out_shape = data.shape[:-1] + (-1, )
            ivim_params = params_in_mask.reshape(out_shape)
        else:
            ivim_params = np.zeros(data.shape[:-1] + (4,))
            ivim_params[mask, :] = params_in_mask

        return IvimFit(self, ivim_params)


class IvimFit(object):

    def __init__(self, model, model_params):
        """ Initialize a IvimFit class instance.
            Parameters
            ----------
            The model parameters are S0, f, D_star, D
        """
        self.model = model
        self.model_params = model_params

    def __getitem__(self, index):
        model_params = self.model_params
        N = model_params.ndim
        if type(index) is not tuple:
            index = (index,)
        elif len(index) >= model_params.ndim:
            raise IndexError("IndexError: invalid index")
        index = index + (slice(None),) * (N - len(index))
        return type(self)(self.model, model_params[index])

    @property
    def shape(self):
        return self.model_params.shape[:-1]


def one_stage(data, gtab, x0, jac=False, bounds=None):
    """
    Fit the ivim params using least-squares.

    Parameters
    ----------
    data : array ([X, Y, Z, ...], g)
        Data or response variables holding the data. Note that the last
        dimension should contain the data. It makes no copies of data.

    jac : bool
        Use the Jacobian? Default: False

    Returns
    -------
    ivim_params: the S0, f, D_star, D value for each voxel.

    """
    flat_data = data.reshape((-1, data.shape[-1]))
    # Flatten for the iteration over voxels:
    bvals = gtab.bvals
    ivim_params = np.empty((flat_data.shape[0], 4))
    for vox in range(flat_data.shape[0]):
        res = leastsq(_ivim_error,
                      x0[vox],
                      args=(bvals, flat_data[vox]))
        ivim_params[vox, :4] = res[0]

    ivim_params.shape = data.shape[:-1] + (4,)
    return ivim_params
