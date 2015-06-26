from __future__ import division, print_function, absolute_import
import numpy as np
import math as m

def total_energy(masked_image, masked_segmentation, mu, var, index, label, beta):

    energytotal = loglikelihood(masked_image, mu, var, index, label) + gibbsenergy(masked_segmentation, mu, var, index, label)

    return energytotal

def loglikelihood(img, mu, var, index, label):

    loglike = img[index] - mu[label]/m.sqrt(var) + (m.log(2*m.pi*var[label]))/2

    return loglike

def gibbsenergy(img, index, label, beta):

    energy = 0

    if label == img[index[0]-1, index[1], index[2]]:
        energy = energy - beta
    else:
        energy = energy + beta

    if label == img[index[0]+1, index[1], index[2]]:
        energy = energy - beta
    else:
        energy = energy + beta

    if label == img[index[0], index[1]-1, index[2]]:
        energy = energy - beta
    else:
        energy = energy + beta

    if label == img[index[0], index[1]+1, index[2]]:
        energy = energy - beta
    else:
        energy = energy + beta

    if label == img[index[0], index[1], index[2]-1]:
        energy = energy - beta
    else:
        energy = energy + beta

    if label == img[index[0], index[1], index[2]+1]:
        energy = energy - beta
    else:
        energy = energy + beta

    return energy
