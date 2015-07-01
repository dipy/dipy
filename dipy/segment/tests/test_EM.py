import numpy as np
import nibabel as nib

from dipy.segment.mask import applymask
from dipy.core.ndindex import ndindex
from dipy.segment.rois_stats import seg_stats
from dipy.segment.energy_mrf import Ising
from dipy.denoise.denspeed import add_padding_reflection
from dipy.denoise.denspeed import remove_padding
from dipy.segment.tests.test_icm_map import icm
import matplotlib.pyplot as plt

dname = '/Users/jvillalo/Documents/GSoC_2015/Code/Data/T1_coronal/'
#dname = '/home/eleftherios/Dropbox/DIPY_GSoC_2015/'

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
print("Intitial estimates of mu and std")
print(mu)
print(std)

nclass = 3
P_L_N = np.zeros((256, 256, 3, nclass))
P_L_N_norm = np.zeros((256, 256, 3, nclass))
P_L_Y = np.zeros((256, 256, 3, nclass))
P_L_Y_norm = np.zeros((256, 256, 3, nclass))
g = np.zeros((256, 256, 3))

beta = 100
mu_upd = mu
var_upd = var

for i in range(0, 3):
    
    segmented = icm(mu_upd, var_upd, masked_img, seg_init_masked, nclass, beta)
    segmented = segmented.copy(order='C')
    segmented_pad = add_padding_reflection(segmented, 1)
    
    # This loop is for equation 2.18 of the Stan Z. Li book.
    for l in range(0, nclass):
        for idx in ndindex(shape):
        
            P_L_N[idx[0], idx[1], idx[2], l] += Ising(l, segmented_pad[idx[0] + 1 - 1, idx[1] + 1, idx[2] + 1], beta)
            P_L_N[idx[0], idx[1], idx[2], l] += Ising(l, segmented_pad[idx[0] + 1 + 1, idx[1] + 1, idx[2] + 1], beta)
            P_L_N[idx[0], idx[1], idx[2], l] += Ising(l, segmented_pad[idx[0] + 1, idx[1] + 1 - 1, idx[2] + 1], beta)
            P_L_N[idx[0], idx[1], idx[2], l] += Ising(l, segmented_pad[idx[0] + 1, idx[1] + 1 + 1, idx[2] + 1], beta)
            P_L_N[idx[0], idx[1], idx[2], l] += Ising(l, segmented_pad[idx[0] + 1, idx[1] + 1, idx[2] + 1 - 1], beta)
            P_L_N[idx[0], idx[1], idx[2], l] += Ising(l, segmented_pad[idx[0] + 1, idx[1] + 1, idx[2] + 1 + 1], beta)
        
        P_L_N[:,:,:,l] = np.exp(P_L_N[:,:,:,l])
        P_L_N_norm[:,:,:,l] += P_L_N[:,:,:,l]
    P_L_N = P_L_N/P_L_N_norm

    # This is for equation 27 of the Zhang paper
    for l in range(0, nclass):
        for idx in ndindex(shape):
            g[idx] = np.exp(-((masked_img[idx] - mu[l])**2/2*var[l]))/np.sqrt(2*3.14159*var[l])
            P_L_Y[idx[0], idx[1], idx[2], l] = g[idx] * P_L_Y[idx[0], idx[1], idx[2], l]
        P_L_Y_norm[:,:,:,l] += P_L_Y[:,:,:,l]
    P_L_Y = P_L_Y/P_L_Y_norm
        
    # This is for equations 25 and 26 of the Zhang paper
    for l in range(0, nclass):
        for idx in ndindex(shape):
            
            mu_upd[l] += (P_L_Y[idx[0], idx[1], idx[2], l] * masked_img[idx])/P_L_Y[idx[0], idx[1], idx[2], l]
            var_upd[l] += (P_L_Y[idx[0], idx[1], idx[2], l] * (masked_img[idx] - mu_upd[l])) /P_L_Y[idx[0], idx[1], idx[2], l]
            
    
print('Show results')
plt.figure()
plt.imshow(seg_init_masked[:,:,1])
plt.figure()
plt.imshow(segmented[:,:,1])

