import numpy as np
import scipy as sp
import nibabel as nib 
import matplotlib.pyplot as plt
from dipy.core.histeq import histeq
from dipy.segment.mask import median_otsu
from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)

from dipy.data import fetch_stanford_hardi, read_stanford_hardi

# fetch_stanford_hardi()
# read the test data
img = nib.load('/Users/Riddhish/Documents/GSOC/DIPY/data/Brain Extraction/forElef_GSoC_brain_extraction/fancyt1/s1_t1.nii.gz')
# img, gtab = read_stanford_hardi()
input_data = img.get_data()
input_affine = img.get_affine()
input_data = input_data[30:210,20:220,30:140]
# input_data = input_data[30:220, 30:220, 10:15]
print(input_data.shape)
# input_data = input_data[...,0]
# read the template 
img = nib.load('/Users/Riddhish/Documents/GSOC/DIPY/data/Brain Extraction/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c.nii')
template_data = img.get_data()
template_affine = img.get_affine()
# img = nib.load('mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_eye_mask.nii')
# template_data_eyemask = img.get_data()
# img = nib.load('mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_face_mask.nii')
# template_data_facemask = img.get_data()
img = nib.load('/Users/Riddhish/Documents/GSOC/DIPY/data/Brain Extraction/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_mask.nii')
template_data_mask = img.get_data()
# img = nib.load('mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_csf_tal_nlin_asym_09c.nii')
# template_data_csf = img.get_data()
# img = nib.load('mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_gm_tal_nlin_asym_09c.nii')
# template_data_gm = img.get_data()
# img = nib.load('mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_wm_tal_nlin_asym_09c.nii')
# template_data_wm = img.get_data()
# img = nib.load('mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_pd_tal_nlin_asym_09c.nii')
# template_data_pd = img.get_data()


# first do the pure brain extraction for the data using
# median otsu

b0_mask, mask = median_otsu(input_data, 2, 1)

sli = input_data.shape[2] / 2 

plt.figure('Brain segmentation')
plt.subplot(1, 4, 1).set_axis_off()
plt.imshow(input_data[:, :, sli].astype('float'),
           cmap='gray', origin='lower')
plt.title("Input Data")

plt.subplot(1, 4, 2).set_axis_off()
plt.imshow(mask[:, :, sli].astype('float'),
           cmap='gray', origin='lower')
plt.title("Median Otsu Output")
# plt.show()

# plot what all things are there in the template data
# fig, ax = plt.subplots(2,3)
# ax[0,0].imshow(template_data[...,sli], cmap='gray')
# ax[0,0].set_title("template data")
# ax[0,1].imshow(template_data_mask[...,sli], cmap='gray')
# ax[0,1].set_title("template data mask")
# ax[0,2].imshow(template_data_gm[...,sli], cmap='gray')
# ax[0,2].set_title("template gray matter")
# ax[1,0].imshow(template_data_wm[...,sli], cmap='gray')
# ax[1,0].set_title("template white matter")
# ax[1,1].imshow(template_data_pd[...,sli], cmap='gray')
# ax[1,1].set_title("template pd")
# ax[1,2].imshow(template_data_csf[...,sli], cmap='gray')
# ax[1,2].set_title("template csf")

# Use the template to align it to the data
c_of_mass = transform_centers_of_mass(input_data, input_affine,
                                      template_data, template_affine)

nbins = 32
sampling_prop = None
metric = MutualInformationMetric(nbins, sampling_prop)

level_iters = [10000, 1000, 100]
sigmas = [3.0, 1.0, 0.0]
factors = [4, 2, 1]
affreg = AffineRegistration(metric=metric,
                            level_iters=level_iters,
                            sigmas=sigmas,
                            factors=factors)
transform = TranslationTransform3D()
params0 = None
starting_affine = c_of_mass.affine
translation = affreg.optimize(input_data, template_data, transform, params0,
                              input_affine, template_affine,
                              starting_affine=starting_affine)
transformed = translation.transform(template_data)
transformed_mask = translation.transform(template_data_mask)

plt.subplot(1, 4, 3).set_axis_off()
plt.imshow(transformed[:, :, sli].astype('float'),
           cmap='gray', origin='lower')
plt.title("Tranformed mask output")

# fig, ax = plt.subplots(2,3)
# ax[0,0].imshow(template_data[...,sli], cmap='gray')
# ax[0,0].set_title("template data")
# ax[0,1].imshow(template_data_mask[...,sli], cmap='gray')
# ax[0,1].set_title("template data mask")
# ax[0,2].imshow(template_data_gm[...,sli] + template_data_wm[:,:,sli] + template_data_csf[:,:,sli], cmap='gray')
# ax[0,2].set_title("template gray matter")
# ax[1,0].imshow(template_data_wm[...,sli], cmap='gray')
# ax[1,0].set_title("template white matter")
# ax[1,1].imshow(template_data_pd[...,sli], cmap='gray')
# ax[1,1].set_title("template pd")
# ax[1,2].imshow(template_data_csf[...,sli], cmap='gray')
# ax[1,2].set_title("template csf")

# Compare the patches from the center voxel of the input data
# to the patches in the neighbourhood of the corrosponding voxel in the
# transformed template image
# Do weighted averaging using the labels
# then perform median otsu

patch_radius = 1
block_radius = 2
patch_size = 2*patch_radius + 1
block_size = 2*block_radius + 1
total_radius = block_radius + patch_radius 
h = 1
avg_wt = 0.0
wt_sum = 0.0
output_data = np.zeros(input_data.shape, dtype=np.float64)

for i in range(total_radius, input_data.shape[0] - total_radius ):
	print(i)
	for j in range(total_radius, input_data.shape[1] - total_radius):
		for k in range(total_radius, input_data.shape[2] - total_radius):
			wt_sum = 0.0
			avg_wt = 0.0
			# find the patch centered around the voxel
			patch = input_data[i - patch_radius : i + patch_radius,
							   j - patch_radius : j + patch_radius,
							   k - patch_radius : k + patch_radius]
			patch = np.array(patch, dtype = np.float64)
			patch = patch / np.sum(patch)

			for i0 in range(i - block_radius, i + block_radius):
				for j0 in range(j - block_radius, j + block_radius):
					for k0 in range(k - block_radius, k + block_radius):

						# now find a patch centered around each of the voxels in neighbourhood
						# from the transformed template

						patch_template = transformed[i - patch_radius : i + patch_radius,
							   						 j - patch_radius : j + patch_radius,
							   						 k - patch_radius : k + patch_radius]
						patch_template = patch_template / np.sum(patch_template)
						# compute the patch difference and the weight
						weight = np.exp(-np.sum((patch - patch_template)**2) / h*h)
						wt_sum += weight
						avg_wt += weight * transformed_mask[i0, j0, k0]

			output_data[i,j,k] = avg_wt / wt_sum


plt.subplot(1, 4, 4).set_axis_off()
plt.imshow(output_data[:, :, sli].astype('float'),
           cmap='gray', origin='lower')
plt.title("The patch averaging label output")
plt.show()
# Use the direct mapping for all the labels and then do the comparision for the data