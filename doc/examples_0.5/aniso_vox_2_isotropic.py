
""" 

===============================
Anisotropic Voxels to Isotropic 
===============================

Overview
========


"""

import nibabel as nib

"""
resample
"""

from dipy.align.aniso2iso import resample
from dipy.data import get_data    

"""
replace with your nifti filename
"""

fimg=get_data('aniso_vox')    
img=nib.load(fimg)
data=img.get_data()
data.shape

"""
(58, 58, 24)
"""

affine=img.get_affine()
zooms=img.get_header().get_zooms()[:3]
zooms

"""
(4.0, 4.0, 5.0)
"""

new_zooms=(3.,3.,3.)
new_zooms

"""
(3.0, 3.0, 3.0)
"""

data2,affine2=resample(data,affine,zooms,new_zooms)
data2.shape

"""
(77, 77, 40)

Save the result as a nifti
"""

img2=nib.Nifti1Image(data2,affine2)
nib.save(img2,'iso_vox.nii.gz')

"""
Or as analyze format
"""

img3=nib.Spm2AnalyzeImage(data2,affine2)
nib.save(img3,'iso_vox.img')


