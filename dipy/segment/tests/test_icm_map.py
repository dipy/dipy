# Use ICM to segment T1 image with MRF

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from dipy.segment.mask import applymask
from dipy.core.ndindex import ndindex
from dipy.segment.rois_stats import seg_stats
from dipy.segment.energy_mrf import total_energy
from dipy.segment.energy_mrf import Ising
from dipy.denoise.denspeed import add_padding_reflection
from dipy.denoise.denspeed import remove_padding


dname = '/Users/jvillalo/Documents/GSoC_2015/Code/Data/T1_coronal/'
#dname = '/home/eleftherios/Dropbox/DIPY_GSoC_2015/'

#img = nib.load(dname + '3587_BL_T1_to_MNI_Linear_6p.nii.gz')
#dataimg = img.get_data()

#mask_image = nib.load(dname + '3587_mask.nii.gz')
#datamask = mask_image.get_data()

#masked_img = applymask(dataimg, datamask)
#print('masked_img.shape (%d, %d, %d)' % masked_img.shape)
#shape = masked_img.shape[:3]

#seg = nib.load(dname + '3587_BL_T1_to_MNI_Linear_6p_seg.nii.gz')
#seg_init = seg.get_data()
#seg_init_masked = applymask(seg_init, datamask)


img = nib.load(dname + 't1_coronal_stack.nii.gz')
dataimg = img.get_data()

mask_image = nib.load(dname + 't1mask_coronal_stack.nii.gz')
datamask = mask_image.get_data()

masked_img = applymask(dataimg, datamask)
print('masked_img.shape (%d, %d, %d)' % masked_img.shape)
shape = masked_img.shape[:3]

seg = nib.load(dname + 't1seg_coronal_stack.nii.gz')
seg_init = seg.get_data()
seg_init_masked = applymask(seg_init, datamask)

masked_img = masked_img.copy(order='C')
seg_init_masked = seg_init_masked.copy(order='c')
masked_img_pad = add_padding_reflection(masked_img, 1)
seg_init_masked_pad = add_padding_reflection(seg_init_masked, 1)

print("computing the statistics of the ROIs (CSF, GM, WM)")
mu, std = seg_stats(masked_img, seg_init_masked, 3)

print(mu)
print(std)

nclass = 3
L = range(1, nclass + 1)
niter = 2
beta = 100
totalE = np.zeros(nclass)
N = 0

segmented = np.zeros(dataimg.shape)


mu, std, var = seg_stats(masked_img, seg_init_masked, nclass)

while True:   

    for idx in ndindex(shape):
        if not masked_img[idx]:
            continue
        for l in range(0, nclass):

            totalE[l] = total_energy(masked_img_pad, seg_init_masked_pad,
                                     mu, var, idx, l, beta)

        segmented[idx] = L[np.argmin(totalE)]
        
        
        
    N = N + 1
    masked_img = segmented
    if N == niter:
        break


#print('Show results')
#plt.figure()
#plt.imshow(seg_init_masked[:,:,1])
#plt.figure()
#plt.imshow(segmented[:,:,1])


 
P_L_N = np.zeros((256, 256, 3, 3))
normalization = 0
P_L_Y = np.zeros((256, 256, 3, 3))
mu_upd = np.zeros(nclass)
var_upd = np.zeros(nclass)

# This loop is for equation 2.18 of the Stan Z. Li book.
for l in range(0, nclass):
    for idx in ndindex(shape):
        
        P_L_N[idx[0], idx[1], idx[2], l] += Ising(l, segmented[idx[0] - 1, idx[1], idx[2]], beta)
        P_L_N[idx[0], idx[1], idx[2], l] += Ising(l, segmented[idx[0] + 1, idx[1], idx[2]], beta)
        P_L_N[idx[0], idx[1], idx[2], l] += Ising(l, segmented[idx[0], idx[1] - 1, idx[2]], beta)
        P_L_N[idx[0], idx[1], idx[2], l] += Ising(l, segmented[idx[0], idx[1] + 1, idx[2]], beta)
        P_L_N[idx[0], idx[1], idx[2], l] += Ising(l, segmented[idx[0], idx[1], idx[2] - 1], beta)
        P_L_N[idx[0], idx[1], idx[2], l] += Ising(l, segmented[idx[0], idx[1], idx[2] + 1], beta)
        
    P_L_N[:,:,:,l] = np.exp(P_L_N[:,:,:,l])
    normalization += P_L_N[:,:,:,l]
P_L_N = P_L_N/normalization

# This is for equation 27 of the Zhang paper
for l in range(0, nclass):
    for idx in ndindex(shape):
        g[idx] = np.exp(-((masked_img[idx] - mu[l])**2/2*var[l]))/np.sqrt(2*pi*var[l])
        P_L_Y[idx[0], idx[1], idx[2], l] = g[idx] * P_L_Y[idx[0], idx[1], idx[2], l]
        
# This is for equations 25 and 26 of the Zhang paper 
for l in range(0, nclass):
    for idx in ndindex(shape):
        
        mu_upd[l] += (P_L_Y[idx[0], idx[1], idx[2], l] * masked_img[idx])/P_L_Y[idx[0], idx[1], idx[2], l]
        var_upd[l] += (P_L_Y[idx[0], idx[1], idx[2], l] * (masked_img[idx] - mu_upd[l])) /P_L_Y[idx[0], idx[1], idx[2], l]
        

        



 
        
        
            







