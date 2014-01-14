"""
Utility functions for the stats module

"""
from __future__ import division, print_function, absolute_import

import numpy as np

def coeff_of_determination(data, model, axis=-1):
    """
    Parameters
    ----------
    data : ndarray
        The real data

    model : ndarray
        The predictions of a model

    axis: int, optional
        The axis along which samples are organized (default: -1)

    Returns
    -------
    COD : ndarray
       The coefficient of determination. This has shape data.shape[:-1]


    Notes
    -----

    See: http://en.wikipedia.org/wiki/Coefficient_of_determination

    The coefficient of determination is calculated as:

    .. math::

        R^2 = 100 * (1 - \frac{SSE}{SSD})

    where SSE is the sum of the squared error between the model and the data
    (sum of the squared residuals) and SSD is the sum of the squares of the
    deviations of the (variance * N)
    """

    residuals = data - model
    ss_err = np.sum(residuals ** 2, axis=axis)

    demeaned_data = data - np.mean(data, axis=axis)[..., np.newaxis]
    ss_tot = np.sum(demeaned_data **2, axis=axis)

    # Don't divide by 0:
    if np.all(ss_tot==0.0):
        return np.nan

    return 100 * (1 - (ss_err/ss_tot))
