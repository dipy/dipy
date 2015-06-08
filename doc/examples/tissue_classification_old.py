"""
============================
Tissue Classification for T1
============================
"""

import numpy as np
import nibabel as nib

from dipy.segment.mask import applymask
from dipy.core.ndindex import ndindex
from dipy.segment.rois_stats import seg_stats
from dipy.segment.energy_mrf import ising
from dipy.denoise.denspeed import add_padding_reflection
from dipy.segment.icm_map import icm
import matplotlib.pyplot as plt

dname = '/Users/jvillalo/Documents/GSoC_2015/Code/Data/T1_coronal/'
# dname = '/home/eleftherios/Dropbox/DIPY_GSoC_2015/T1_coronal/'

img = nib.load(dname + 't1_coronal_stack.nii.gz')
dataimg = img.get_data()

mask_image = nib.load(dname + 't1mask_coronal_stack.nii.gz')
datamask = mask_image.get_data()

seg = nib.load(dname + 't1seg_coronal_stack.nii.gz')
seg_init = seg.get_data()

masked_img = applymask(dataimg, datamask)
seg_init_masked = applymask(seg_init, datamask)

print('masked_img.shape (%d, %d, %d)' % masked_img.shape)
shape = masked_img.shape[:3]

masked_img = masked_img.copy(order='C')
seg_init_masked = seg_init_masked.copy(order='c')
masked_img_pad = add_padding_reflection(masked_img, 1)
seg_init_masked_pad = add_padding_reflection(seg_init_masked, 1)

print("computing the statistics of the ROIs (CSF, GM, WM)")
mu, std, var = seg_stats(masked_img, seg_init_masked, 3)
print("Intitial estimates of mu and var")
print(mu)
print(var)

nclass = 3

# probability of the tissue label (from the 3 classes) given the
# neighborhood of each voxel
P_L_N = np.zeros(masked_img.shape + (nclass,))
# Normalization term for P_L_N
P_L_N_norm = np.zeros_like(masked_img)
# probability of the tissue label (from the 3 classes) given the
# voxel
P_L_Y = np.zeros_like(P_L_N)
P_L_Y_norm = np.zeros_like(masked_img)
# normal density equation 11 of the Zhang paper
g = np.zeros_like(masked_img)
# temporary mu and var files to compute the update
mu_num = np.zeros(masked_img.shape + (nclass,))
var_num = np.zeros(masked_img.shape + (nclass,))
denm = np.zeros(masked_img.shape + (nclass,))

# weight of the neighborhood in ICM
beta = 1.5
# update variables of mu and variance
mu_upd = mu
var_upd = var
# number of iterations of the mu/var updates
niter = 5
# update segmented image after each ICM
seg_upd = seg_init_masked

# Notes clear implementation of P_L_N addition
# ising = np.array([[0, 1, 0], [1, -1, 1], [0, 1, 0]])
# a = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
# ising_kernel = np.array([a, ising, a])
# P_L_N[i-1:i+2, j-1:j+2, k-1:k+2] += isign_kernel

for i in range(0, niter):

    segmented = icm(mu_upd, var_upd, masked_img, seg_upd, nclass, beta)
    seg_upd = segmented
    segmented = segmented.copy(order='C')
    segmented_pad = add_padding_reflection(segmented, 1)

    # This loop is for equation 2.18 of the Stan Z. Li book.
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

    # This is for equation 27 of the Zhang paper
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

    # This is for equations 25 and 26 of the Zhang paper
    for l in range(0, nclass):
        for idx in ndindex(shape):
            if not masked_img[idx]:
                continue

            mu_num[idx[0], idx[1], idx[2], l] = (P_L_Y[idx[0], idx[1], idx[2], l] * masked_img[idx])
            var_num[idx[0], idx[1], idx[2], l] = (P_L_Y[idx[0], idx[1], idx[2], l] * (masked_img[idx] - mu_upd[l])**2)
            denm[idx[0], idx[1], idx[2], l] = P_L_Y[idx[0], idx[1], idx[2], l]

        mu_upd[l] = np.sum(applymask(mu_num[:, :, :, l], datamask)) / np.sum(applymask(denm[:, :, :, l], datamask))
        var_upd[l] = np.sum(applymask(var_num[:, :, :, l], datamask)) / np.sum(applymask(denm[:, :, :, l], datamask))

        print('class ', l)
        print(np.sum(applymask(mu_num[:, :, :, l], datamask)))
        print(np.sum(applymask(var_num[:, :, :, l], datamask)))
        print(np.sum(applymask(denm[:, :, :, l], datamask)))
        print(mu_upd[l], mu[l])
        print(var_upd[l], var[l])


print('Show results')
plt.figure()
plt.imshow(seg_init_masked[:, :, 1])
plt.figure()
plt.imshow(P_L_Y[:, :, 1, 0])
plt.figure()
plt.imshow(P_L_Y[:, :, 1, 1])
plt.figure()
plt.imshow(P_L_Y[:, :, 1, 2])
plt.figure()
plt.imshow(segmented[:, :, 1])
