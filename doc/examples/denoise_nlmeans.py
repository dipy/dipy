"""
==============================================
Denoise images using Non-Local Means (NLMEANS)
==============================================
"""
import numpy as np
from time import time
import nibabel as nib
from dipy.denoise.nlmeans import nlmeans

dname = '/home/eleftherios/Downloads/'
img = nib.load(dname + 't1.nii')
data = img.get_data().astype('float64')
data = np.ascontiguousarray(data, dtype='f8')
aff = img.get_affine()

mask = data > 30

print("vol size", data.shape)

t = time()

den = nlmeans(data, mask, sigma=19.88)

print("total time", time() - t)
print("vol size", den.shape)

nib.save(nib.Nifti1Image(den, aff), dname + 't1_denoised.nii.gz')

# img = nib.load(dname + 't1_denoised.nii.gz')
# old = img.get_data()
# print(np.sum(np.abs(old-den)))