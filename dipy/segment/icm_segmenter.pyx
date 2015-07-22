#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
import numpy as np
cimport cython
cimport numpy as cnp
cdef extern from "dpy_math.h" nogil:
    cdef double NPY_PI
    double sqrt(double)
    double log(double)

#from __future__ import division, print_function, absolute_import
import numpy as np
from dipy.core.ndindex import ndindex


def icm(mu, var, masked_img, seg_img, classes, beta):
    r"""Use ICM to segment T1 image with MRF
    Parameters
    -----------

    mu : 1x3 ndarray - mean of each tissue
    var : 1x3 ndarray - variance of each tissue
    masked_img : 3D ndarray - masked T1 structural image
    seg_img : 3D ndarray - initial segmentation provided as an input
    classes : integer - number of tissue classes
    beta : float - the weight of the neighborhood

    Returns
    --------

    segmented : 3D ndarray - segmentation of the input T1 structural image
    totalE : 1x3 ndarrray - total energy per tissue class

    """

    totalE = np.zeros(classes)
    L = range(1, classes + 1)
    segmented = np.zeros(masked_img.shape)
    totalenergy = np.zeros(masked_img.shape)
    shape = masked_img.shape[:3]

#    masked_img = masked_img.copy(order='C')
#    masked_img_pad = add_padding_reflection(masked_img, 1)
#    seg_img = seg_img.copy(order='C')
#    seg_img_pad = add_padding_reflection(seg_img, 1)

    for idx in ndindex(shape):
        if not masked_img[idx]:
            continue
        for l in range(0, classes):

            totalE[l] = total_energy(masked_img, seg_img, mu, var, idx, l,
                                     beta)

        segmented[idx] = L[np.argmin(totalE)]
        totalenergy[idx] = np.argmin(totalE)

    return segmented, totalenergy


def total_energy(masked_image, masked_segmentation,
                 mu, var, index, label, beta):
    r""" Computes the total energy for the ICM model

    Parameters
    -----------
    masked_image : 3D ndarray - T1 structural image.
    masked_segmentation : 3D ndarray - initial segmentation.
    mu : This is a 1X3 ndarray with mean of each each tissue type
         [CSF, Grey Matter, White Matter]
    var : Also a 1x3 ndarray with the variance of each tissue type
    index : The voxel index
    label : The index of the label that is looping through (0, 1 or 2)
    beta : float - scalar value of the weight given to the neighborhood

    Returns
    --------

    energytotal : float scalar corresponding to the total energy in that voxel

    """

    energytotal = neg_log_likelihood(masked_image, mu, var, index, label)
    energytotal += gibbs_energy(masked_segmentation, index, label, beta)

    return energytotal


def neg_log_likelihood(img, mu, var, index, label):
    r""" Computes the negative log-likelihood

    Parameters
    -----------

    img : 3D ndarray. The grey scale T1 image. Must be masked.
    mu : This is a 1X3 ndarray with mean of each each tissue type
        [CSF, Grey Matter, White Matter]
    var : Also a 1x3 ndarray with the variance of each tissue type
    index : tuple - The voxel index
    label : The index of the label that is looping through (0, 1 or 2)

    Returns
    --------

    loglike : float - the negative log-likelihood

    """
    if var[label] == 0:
        loglike = 0
    else:
        loglike = ((img[index] - mu[label]) ** 2) / (2 * var[label])
        loglike += np.log(np.sqrt(var[label]))

    return loglike


def gibbs_energy(seg, index, label, beta):
    r""" Computes the Gibbs energy

    Parameters
    -----------
    seg : 3D ndarray - initial segmentation. Must be masked and zero padded
    index : tuple - The voxel index
    label : The index of the label that is looping through (0, 1 or 2)
    beta : float - scalar value of the weight given to the neighborhood

    Returns
    --------

    energy : float - Gibbs energy

    """

    label = label + 1
    energy = 0

    if label == seg[index[0] - 1, index[1], index[2]]:
        energy = energy
    else:
        if seg[index[0] - 1, index[1], index[2]] == 0:
            energy = energy
        else:
            energy = energy + beta

    if label == seg[index[0] + 1, index[1], index[2]]:
        energy = energy
    else:
        if seg[index[0] + 1, index[1], index[2]] == 0:
            energy = energy
        else:
            energy = energy + beta

    if label == seg[index[0], index[1] - 1, index[2]]:
        energy = energy
    else:
        if seg[index[0], index[1] - 1, index[2]] == 0:
            energy = energy
        else:
            energy = energy + beta

    if label == seg[index[0], index[1] + 1, index[2]]:
        energy = energy
    else:
        if seg[index[0], index[1] + 1, index[2]] == 0:
            energy = energy
        else:
            energy = energy + beta

    if label == seg[index[0], index[1], index[2] - 1]:
        energy = energy
    else:
        if seg[index[0], index[1], index[2] - 1] == 0:
            energy = energy
        else:
            energy = energy + beta

    if label == seg[index[0], index[1], index[2] + 1]:
        energy = energy
    else:
        if seg[index[0], index[1], index[2] + 1] == 0:
            energy = energy
        else:
            energy = energy + beta

    return energy
