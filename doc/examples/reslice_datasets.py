
"""

==========================
Reslice diffusion datasets
==========================

Often in imaging it is common to reslice images in different resolutions.
Especially in dMRI we usually want images with isotropic voxel size as they
facilitate most tractography algorithms. In this example we show how you
can reslice a dMRI dataset to have isotropic voxel size.

Let's start by importing the necessary modules.
"""

import nibabel as nib

from dipy.align.reslice import reslice
from dipy.data import get_fnames
from dipy.io.image import load_nifti, save_nifti

###############################################################################
# We use here a very small dataset to show the basic principles but you can
# replace the following line with the path of your image.

fimg = get_fnames('aniso_vox')

###############################################################################
# We load the image, the affine of the image and the voxel size. The affine is
# the transformation matrix which maps image coordinates to world (mm)
# coordinates. Then, we print the shape of the volume

data, affine, voxel_size = load_nifti(fimg, return_voxsize=True)

print(f"Data size: {data.shape}")
print(f"Voxel size: {voxel_size}")

###############################################################################
# Set the required new voxel size.

new_voxel_size = (3., 3., 3.)
print(f"New Voxel size: {new_voxel_size}")

###############################################################################
# Start resampling (reslicing). Trilinear interpolation is used by default.

data2, affine2 = reslice(data, affine, voxel_size, new_voxel_size)
print(f"New data size: {data2.shape}")

###############################################################################
# Save the result as a new Nifti file.

save_nifti('iso_vox.nii.gz', data2, affine2)

###############################################################################
# Or as analyze format or any other supported format.

img3 = nib.Spm2AnalyzeImage(data2, affine2)
nib.save(img3, 'iso_vox.img')

###############################################################################
# Done. Check your datasets. As you may have already realized the same code
# can be used for general reslicing problems not only for dMRI data.
