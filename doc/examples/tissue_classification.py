"""
============================
Tissue Classification for T1
============================
"""

# import numpy as np
import nibabel as nib

from dipy.segment.mask import applymask
from dipy.segment.rois_stats import seg_stats
from dipy.denoise.denspeed import add_padding_reflection
from dipy.segment.icm_map import icm
from dipy.segment.mrf_em import prob_neigh, prob_image, update_param
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

print("computing the statistics of the ROIs [CSF, GM, WM]")
mu, std, var = seg_stats(masked_img, seg_init_masked, 3)
print("Intitial estimates of mu and var")
print('mean', mu)
print('variance', var)

# number of tissue classes
nclass = 3
# weight of the neighborhood in ICM
beta = 1.5
# number of iterations of the mu/var updates
niter = 5
# update variables of mean and variance
mu_upd = mu
var_upd = var
# update segmented image after each ICM
seg_upd = seg_init_masked

for i in range(0, niter):

    print('Iteration', i)

    # Calls the ICM function
    segmented, totalenergy = icm(mu_upd, var_upd, masked_img, seg_upd, nclass, beta)
    seg_upd = segmented
    segmented = segmented.copy(order='C')
    segmented_pad = add_padding_reflection(segmented, 1)

    # This is for equation 2.18 of the Stan Z. Li book.
    P_L_N = prob_neigh(nclass, masked_img, segmented_pad, beta)

    # This is for equation 27 of the Zhang paper
    P_L_Y = prob_image(nclass, masked_img, mu_upd, var_upd, P_L_N)

    # This is for equations 25 and 26 of the Zhang paper
    mu_upd, var_upd = update_param(nclass, masked_img, datamask, mu_upd, P_L_Y)

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
plt.imshow(seg_upd[:, :, 1])
