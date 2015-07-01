from __future__ import division, print_function, absolute_import
import numpy as np


def seg_stats(input_image, seg_image, nclass):
    r""" Mean and standard variation for 3 tissue classes

    1 is CSF
    2 is gray matter
    3 is white matter

    Parameters
    ----------
    input_image : ndarray
        blah blah



    Returns
    -------
    mu, std : float
        Mean and standard deviation for every class


    """
    mu = np.zeros(nclass)
    std = np.zeros(nclass)
    var = np.zeros(nclass)

    for i in range(1, nclass + 1):

        H = input_image[seg_image == i]

        mu[i - 1] = np.mean(H, -1)
        std[i - 1] = np.std(H, -1)
        var[i - 1] = np.var(H, -1)


    return mu, std, var
