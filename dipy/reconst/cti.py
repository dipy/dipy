#!/usr/bin/python
""" Classes and functions for fitting the correlation tensor model """

import warnings
import functools
import numpy as np
import scipy.optimize as opt

from dipy.reconst.base import ReconstModel
from dipy.reconst.dki import (
                                DiffusionKurtosisFit,

                            )

class CorrelationTensorModel(ReconstModel):
    """ Class for the Correlation Tensor Model
    """
    def __init__(self, gtab, fit_method="", *args, **kwargs): #not sure about the fit method yet
        """ Diffusion Kurtosis Tensor Model [1]

        Parameters
        ----------
        gtab : GradientTable class instance

        fit_method : str or callable


        args, kwargs : arguments and key-word arguments passed to the 
        fit_method.

        """

    def fit(self, data, mask=None):
        """ Fit method of the CTI model class

        Parameters
        ----------
        data : array
            The measured signal from one voxel.

        mask : array
            A boolean array used to mark the coordinates in the data that
            should be analyzed that has the shape data.shape[-1]

        """

class CorrelationTensorFit(DiffusionKurtosisFit):
    """ Class for fitting the Diffusion Kurtosis Model"""

    def __init__(self, model, model_params):
        """ Initialize a CorrelationTensorFit class instance.

        Since CTI is an extension of DKI, class instance is defined as subclass
        of the DiffusionKurtosis from dki.py

        Parameters
        ----------
        model : CorrelationTensorModel Class instance
            Class instance containing the Correlation Tensor Model for the fit
        model_params : ndarray (x, y, z, 27) or (n, 27)
            All parameters estimated from the correlation tensor model.
            Parameters are ordered as follows:

        """
        DiffusionKurtosisFit.__init__(self, model, model_params)

