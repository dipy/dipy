from __future__ import division, print_function, absolute_import

import numpy as np
from dipy.segment.mask import applymask
from dipy.core.ndindex import ndindex


def prob_neigh(nclass, masked_img, segmented_pad, beta):
    r""" Conditional probability of the label given the neighborhood
    Equation 2.18 of the Stan Z. Li book.

    Parameters
    -----------
    nclass : int - number of tissue classes
    masked_img : 3D ndarray - masked T1 structural image
    segmented_pad : 3D ndarray - tissue segmentation derived from the ICM
                                 model. Must be padded with zeros
    beta : float - value of th importance of the neighborhood

    Returns
    --------

    P_L_N : 4D ndarray - Probability of the label given the neighborhood of
                         the voxel

    """
    # probability of the tissue label (from the 3 classes) given the
    # neighborhood of each voxel
    P_L_N = np.zeros(masked_img.shape + (nclass,))
    # Normalization term for P_L_N
    P_L_N_norm = np.zeros_like(masked_img)
    shape = masked_img.shape[:3]

    # Notes clear implementation of P_L_N addition
    # ising = np.array([[0, 1, 0], [1, -1, 1], [0, 1, 0]])
    # a = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    # ising_kernel = np.array([a, ising, a])
    # P_L_N[i-1:i+2, j-1:j+2, k-1:k+2] += ising_kernel

    for l in range(0, nclass):
        for idx in ndindex(shape):
            if not masked_img[idx]:
                continue

            Label = l + 1
            P_L_N[idx[0], idx[1], idx[2], l] += ising(Label, segmented_pad[idx[0] + 1 - 1, idx[1] + 1, idx[2] + 1], beta)
            P_L_N[idx[0], idx[1], idx[2], l] += ising(Label, segmented_pad[idx[0] + 1 + 1, idx[1] + 1, idx[2] + 1], beta)
            P_L_N[idx[0], idx[1], idx[2], l] += ising(Label, segmented_pad[idx[0] + 1, idx[1] + 1 - 1, idx[2] + 1], beta)
            P_L_N[idx[0], idx[1], idx[2], l] += ising(Label, segmented_pad[idx[0] + 1, idx[1] + 1 + 1, idx[2] + 1], beta)
            P_L_N[idx[0], idx[1], idx[2], l] += ising(Label, segmented_pad[idx[0] + 1, idx[1] + 1, idx[2] + 1 - 1], beta)
            P_L_N[idx[0], idx[1], idx[2], l] += ising(Label, segmented_pad[idx[0] + 1, idx[1] + 1, idx[2] + 1 + 1], beta)

        # Eq 2.18
        P_L_N[:, :, :, l] = np.exp(- P_L_N[:, :, :, l])
        P_L_N_norm[:, :, :] += P_L_N[:, :, :, l]

    for l in range(0, nclass):
        P_L_N[:, :, :, l] = P_L_N[:, :, :, l]/P_L_N_norm
        # P_L_N[np.isnan(P_L_N)] = 0

    return P_L_N


def prob_image(nclass, masked_img, mu_upd, var_upd, P_L_N):
    r""" Conditional probability of the label given the image
    This is for equation 27 of the Zhang paper

    Parameters
    -----------
    nclass : int - number of tissue classes
    masked_img : 3D ndarray - masked T1 structural image
    mu_upd : 1x3 ndarray - current estimate of mean of each tissue type
    var_upd : 1x3 ndarray - current estimate of the variance of each tissue
                            type
    P_L_N : 4D ndarray - probability of the label given the neighborhood.
                         Previously computed by function prob_neigh

    Returns
    --------

    P_L_Y : 4D ndarray - Probability of the label given the input image

    """
    # probability of the tissue label (from the 3 classes) given the
    # voxel
    P_L_Y = np.zeros_like(P_L_N)
    P_L_Y_norm = np.zeros_like(masked_img)
    # normal density equation 11 of the Zhang paper
    g = np.zeros_like(masked_img)
    shape = masked_img.shape[:3]

    for l in range(0, nclass):
        for idx in ndindex(shape):
            if not masked_img[idx]:
                continue

            g[idx] = np.exp(-((masked_img[idx] - mu_upd[l]) ** 2 / 2 * var_upd[l])) / np.sqrt(2*np.pi*var_upd[l])
            P_L_Y[idx[0], idx[1], idx[2], l] = g[idx] * P_L_N[idx[0], idx[1], idx[2], l]

        P_L_Y_norm[:, :, :] += P_L_Y[:, :, :, l]

    for l in range(0, nclass):
        P_L_Y[:, :, :, l] = P_L_Y[:, :, :, l]/P_L_Y_norm

    P_L_Y[np.isnan(P_L_Y)] = 0

    return P_L_Y


def update_param(nclass, masked_img, datamask, mu_upd, P_L_Y):
    r""" Updates the mean and the variance in each iteration
    This is for equations 25 and 26 of the Zhang paper

    Parameters
    -----------
    nclass : int - number of tissue classes
    masked_img : 3D ndarray - masked T1 structural image
    datamask : 3D ndarray - mask of the T1 structural image
    mu_upd : 1x3 ndarray - current estimate of mean of each tissue type
    P_L_Y : Probability of the label given the input image

    Returns
    --------
    mu_upd : 1x3 ndarray - mean of each tissue class
    var_upd : 1x3 ndarray - variance of each tissue class

    """
    # temporary mu and var files to compute the update
    mu_num = np.zeros(masked_img.shape + (nclass,))
    var_num = np.zeros(masked_img.shape + (nclass,))
    denm = np.zeros(masked_img.shape + (nclass,))
    var_upd = np.zeros(3)
    shape = masked_img.shape[:3]

    for l in range(0, nclass):
        for idx in ndindex(shape):
            if not masked_img[idx]:
                continue

            mu_num[idx[0], idx[1], idx[2], l] = (P_L_Y[idx[0], idx[1], idx[2], l] * masked_img[idx])
            var_num[idx[0], idx[1], idx[2], l] = (P_L_Y[idx[0], idx[1], idx[2], l] * (masked_img[idx] - mu_upd[l])**2)
            denm[idx[0], idx[1], idx[2], l] = P_L_Y[idx[0], idx[1], idx[2], l]

        mu_upd[l] = np.sum(applymask(mu_num[:, :, :, l], datamask)) / np.sum(applymask(denm[:, :, :, l], datamask))
        var_upd[l] = np.sum(applymask(var_num[:, :, :, l], datamask)) / np.sum(applymask(denm[:, :, :, l], datamask))

        print('class: ', l)
        print('mu_num_sum:', np.sum(applymask(mu_num[:, :, :, l], datamask)))
        print('var_num_sum:', np.sum(applymask(var_num[:, :, :, l], datamask)))
        print('denominator_sum:', np.sum(applymask(denm[:, :, :, l], datamask)))
        print('updated_mu:', mu_upd[l])
        print('updated_var:', var_upd[l])

    return mu_upd, var_upd


def ising(l, voxel, beta):
    r""" Ising model

    Parameters
    -----------
    l : The value of the label that is being tested in each voxel (1, 2 or 3)
    voxel : The value of the voxel
    beta : the value of the weight given to the neighborhood

    Returns
    --------
    beta : Negative or positive float number

    """

    if l == voxel:
        return - beta
    else:
        return beta