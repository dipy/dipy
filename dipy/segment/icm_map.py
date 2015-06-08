

# Use ICM to segment T1 image with HMRF

import numpy as np
import nibabel as nib
#import scipy as io
#import dipy

img = nib.load('/Users/jvillalo/Documents/GSoC_2015/Code/Data/3587_BL_T1_to_MNI_Linear_6p.nii.gz')
dataimg = img.get_data()

mask = nib.load('/Users/jvillalo/Documents/GSoC_2015/Code/Data/3587_mask.nii.gz')
datamask = mask.get_data()

from dipy.segment.mask import applymask
masked_img = applymask(dataimg,datamask)
print('masked_img.shape (%d, %d, %d)' % masked_img.shape)
shape=masked_img.shape[:3]

from init_param import otsu_param
mu_tissue1, sig_tissue1, mu_tissue2, sig_tissue2 = otsu_param(masked_img, 4, 4)


nclass = 2
niter = 1000
nh = 6 #neighborhood

totalE = np.zeros((shape[0],shape[1],shape[2],nclass))

#mu=
#sigma=

for idx in np.ndindex(shape):
    if not masked_img[idx]:
        continue
    
    totalE = 1/sqrt(2*3.1416)
    



    
    
    
    
    
    