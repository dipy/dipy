"""
===========================================
Estimate Noise Levels for LocalPCA Datasets
===========================================

"""

import numpy as np
import nibabel as nib
import scipy as sp
import matplotlib.pyplot as plt
from time import time
from scipy import ndimage
from dipy.data import fetch_sherbrooke_3shell, read_sherbrooke_3shell
from dipy.data import fetch_stanford_hardi, read_stanford_hardi


def estimate_sigma_localpca(data,gtab):
    
    # first identify the number of the b0 images
    K = gtab.b0s_mask[gtab.b0s_mask].size

    if(K>1):
        print "MUBE Estimate"
        # If multiple b0 values then 
        data0 = data[...,gtab.b0s_mask]
        # MUBE Noise Estimate

        # Form their matrix
        X = data0.reshape(data0.shape[0] * data0.shape[1] * data0.shape[2], data0.shape[3])
        # X = NxK
        # Now to subtract the mean
        M = np.mean(X,axis = 0)
        X = X - M
        C = np.transpose(X).dot(X)
        C = C/data0.shape[3]
        # Do PCA and find the lowest principal component
        [d,W] = np.linalg.eigh(C)
        d[d < 0] = 0;
        d_new = d[d != 0]
        # we want eigen vectors of XX^T so we do
        V = X.dot(W)
        # unit normalize the clolumns of V
        # V = V / np.linalg.norm(V, ord=2, axis=0)
        # As the W is sorted in increasing eignevalue order, so is V
        # choose the smallest positive principle component
        I = V[:,d.shape[0] - d_new.shape[0]].reshape(data.shape[0], data.shape[1], data.shape[2])
        

    else:
        print "SIBE Estimate"
        # if only one b0 value then
        # SIBE Noise Estimate
        data0 = data[...,~gtab.b0s_mask]
        # MUBE Noise Estimate

        # Form their matrix
        X = data0.reshape(data0.shape[0] * data0.shape[1] * data0.shape[2], data0.shape[3])
        # X = NxK
        # Now to subtract the mean
        M = np.mean(X,axis = 0)
        X = X - M
        C = np.transpose(X).dot(X)
        C = C/data0.shape[3]
        # Do PCA and find the lowest principal component
        [d,W] = np.linalg.eigh(C)
        d[d < 0] = 0;
        d_new = d[d != 0]
        # we want eigen vectors of XX^T so we do
        V = X.dot(W)
        # unit normalize the clolumns of V
        # V = V / np.linalg.norm(V, ord=2, axis=0)
        # As the W is sorted in increasing eignevalue order, so is V
        # choose the smallest principle component
        I = V[:,d.shape[0] - d_new.shape[0]].reshape(data.shape[0], data.shape[1], data.shape[2])



    sigma = np.zeros(I.shape, dtype = np.float64)
    count = np.zeros_like(data)
    mean = np.zeros_like(data).astype(np.float64)

    # Compute the local noise variance of the I image
    for i in range(1,I.shape[0] - 1):
        print(i)
        for j in range(1, I.shape[1] - 1):
            for k in range(1, I.shape[2] - 1):
                    
                temp = I[i-1:i+2, j-1:j+2, k-1:k+2]
                temp = (temp - np.mean(temp)) * (temp - np.mean(temp))
                sigma[i-1:i+2, j-1:j+2, k-1:k+2] += temp  
                temp = data[i-1:i+2, j-1:j+2, k-1:k+2, :]
                mean[i-1:i+2, j-1:j+2, k-1:k+2, :] +=  np.mean(np.mean(np.mean(temp,axis = 0),axis=0),axis=0)
                count[i-1:i+2, j-1:j+2, k-1:k+2,:] += 1

    sigma = sigma / count[...,0]             

    print "Initial estimate of local noise variance"
    # Compute the local mean of for the data                    
    mean = mean / count

    snr = np.zeros_like(I).astype(np.float64)
    sigma_corr = np.zeros_like(data).astype(np.float64)

    # find the SNR and make the correction
    # SNR Correction

    for l in range(data.shape[3]):
        print(l)
        snr = mean[...,l] / np.sqrt(sigma)
        eta = 2 + snr**2 - (np.pi / 8) * np.exp(-0.5 * (snr**2)) * ((2 + snr**2) * sp.special.j0(0.25 *(snr**2)) + (snr**2) * sp.special.j1(0.25 *(snr**2)))**2
        sigma_corr[...,l] = sigma / eta

    print "SNR correction done"
    # smoothing by lpf
    sigma_corrr = ndimage.gaussian_filter(sigma_corr,5)
    sigmar = ndimage.gaussian_filter(sigma,5)

    return [sigmar,sigma_corrr]