from __future__ import division, print_function, absolute_import
import numpy as np


def seg_stats(input_image, seg_image, nclass):
    r""" Mean and standard variation for 3 tissue classes

    1 is CSF
    2 is grey matter
    3 is white matter

    Parameters
    ----------
    input_image : ndarray of grey level T1 image
    seg_image : ndarray of initital segmentation, also an image
    nclass : float numeber of classes (three in most cases)

    Returns
    -------
    mu, std, var : ndarray of dimensions 1x3
        Mean, standard deviation and variance for every class

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
