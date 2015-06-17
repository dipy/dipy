from __future__ import division, print_function, absolute_import
import numpy as np

def seg_stats(input_image, seg_image, nclass):

    '''
    To compute the mean and standard variation of each segmented area.

    '''
    mu = np.zeros(3)    
    std = np.zeros(3)    
    
    for i in range(0, nclass-1):

        H = input_image[seg_image == i]

        mu[i] = np.mean(H, -1)
        std[i] = np.std(H, -1)


    return mu, std