

# Use ICM to segment T1 image with HMRF

import numpy as np
import nibabel as nib

img = nib.load('/Users/jvillalo/Documents/GSoC_2015/Code/Data/3587_BL_T1_to_MNI_Linear_6p.nii.gz')
dataimg = img.get_data()

mask = nib.load('/Users/jvillalo/Documents/GSoC_2015/Code/Data/3587_mask.nii.gz')
datamask = mask.get_data()

from dipy.segment.mask import applymask
masked_img = applymask(dataimg,datamask)
print('masked_img.shape (%d, %d, %d)' % masked_img.shape)
shape=masked_img.shape[:3]

# Still must do Otsu for 3 classes. 
#from init_param import otsu_param
#mu_tissue1, sig_tissue1, mu_tissue2, sig_tissue2 = otsu_param(masked_img, 4, 4)

seg = nib.load('/Users/jvillalo/Documents/GSoC_2015/Code/Data/FAST/3587_BL_T1_to_MNI_Linear_6p_seg.nii.gz')
seg_initial = seg.get_data()

nclass = 3
nh = 6   #neighborhood
niter = 1
totalE = np.zeros((shape[0],shape[1],shape[2],nclass))

from dipy.segment.ROI_stats import seg_stats
seg_stats(masked_img, seg_initial, 3)

from dipy.core.ndindex import ndindex

for idx in np.ndindex(shape):
    if not masked_img[idx]:
        continue
    
    while True:
        
        mu, std = seg_stats(masked_img, seg_initial, 3)
    
        for l in range(0,nclass-3):
            totalE = 1/(2*vars[l])*(masked_img-mus[l])^2 + np.log(sigs[l]) - beta*Himg[:,:,:,l]
        np.amin(totalE[-1])
        
    N = niter+1        
    if N = 1:
        break



    
    
    
    
    
    