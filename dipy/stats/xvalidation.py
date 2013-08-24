"""
Cross-validation analysis of diffusion models




"""
from __future__ import division, print_function, absolute_import
import numpy as np


def kfold_xval(model, data, folds):
    """
    Given a Model object perform iterative k-fold cross-validation of fitting
    that model

    Parameters
    ----------
    model : class instance of a Model

    data : ndarray
        Diffusion MRI data acquired with the gtab of the model

    folds: int
        The number of divisions to apply to the data

    Notes
    -----
    This function assumes that a prediction API is implemented in the Model
    class for which prediction is conducted. That is, the Fit object that gets
    generated upon fitting the model needs to have a `predict` method, which
    receives a GradientTable class instance as input and produces a predicted
    signal as output.

    """
    modder =  np.mod(data.shape[-1], folds)
    if modder!= 0:
        msg = "The number of folds must divide the data equally, "
        msg = "but np.mod(%s, %s) is %s"%(data.shape[-1], folds, modder)
        raise ValueError()


    # We are going to leave out some random samples in each iteration:
    order = np.random.permutation(data.shape[-1])


    for k in folds:
        pass
        ## fold_idx =
        ## this_data =
        ## this_fit = model.fit(data[])
        ## fit.predict()

    return prediction
