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
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.imwarp import DiffeomorphicMap
from dipy.viz import regtools
from dipy.align.metrics import CCMetric, EMMetric

# The filepaths (and their corroesponding dropbox location)
filename_isbr = '/Users/Riddhish/Documents/GSOC/DIPY/data/Brain Extraction/IBSR_nifti_stripped/IBSR_02/IBSR_02_ana.nii.gz'
# filename_isbr = '../brain_extraction/First tests/input_data/IBSR_02_ana.nii.gz'
filename_template = '/Users/Riddhish/Documents/GSOC/DIPY/data/Brain Extraction/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c.nii'
# filename_template = '../brain_extraction/First tests/template_data/mni_icbm152_t1_tal_nlin_asym_09c.nii'
filename_template_mask = '/Users/Riddhish/Documents/GSOC/DIPY/data/Brain Extraction/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_mask.nii'
# filename_template_mask = '../brain_extraction/First tests/template_data/mni_icbm152_t1_tal_nlin_asym_09c_mask.nii'
filename_output = '/Users/Riddhish/Documents/GSOC/DIPY/data/Brain Extraction/forElef_GSoC_brain_extraction/brain_extracted.nii.gz'
# filename_output = '../brain_extraction/First tests/output_data/brain_extracted.nii.gz'
filename_output_mask = '/Users/Riddhish/Documents/GSOC/DIPY/data/Brain Extraction/forElef_GSoC_brain_extraction/brain_extracted_mask.nii.gz'
# filename_output_mask = '../brain_extraction/First tests/output_data/brain_extracted_mask.nii.gz'

# img = nib.load(
#     '/Users/Riddhish/Documents/GSOC/DIPY/data/Brain Extraction/IBSR_nifti_stripped/IBSR_02/IBSR_02_ana.nii.gz')
img = nib.load(filename_isbr)
# img, gtab = read_stanford_hardi()
input_data = img.get_data()
input_affine = img.get_affine()
# input_data = input_data[:,:,70:110]
# input_data = input_data[30:220, 30:220, 10:15]
print(input_data.shape)
input_data = input_data[..., 0]
# input_data = input_data[40:120,40:130,70:90]
# read the template
img = nib.load(filename_template)
template_data = img.get_data()
template_affine = img.get_affine()
# img = nib.load('mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_eye_mask.nii')
# template_data_eyemask = img.get_data()
# img = nib.load('mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_face_mask.nii')
# template_data_facemask = img.get_data()
img = nib.load(filename_template_mask)
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

sli = input_data.shape[2] / 2 - 10

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
template_data_temp = np.zeros(template_data.shape)
template_data_temp[
    template_data_mask > 0] = template_data[
        template_data_mask > 0]
# input_data = b0_mask
# template_data = template_data_temp
########## AffineRegisteration #######################
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

# plt.subplot(1, 2, 2).set_axis_off()
# plt.imshow(transformed[:, :, sli].astype('float'),
#            cmap='gray', origin='lower')
# plt.title("Tranformed template output")


############# Non Linear Registeration ###################

# Use the masked template

pre_align = translation.affine

metric = CCMetric(3)
level_iters = [10, 10, 5]
sdr = SymmetricDiffeomorphicRegistration(
    metric, level_iters, ss_sigma_factor=1.7)
mapping = sdr.optimize(
    input_data,
    template_data,
    input_affine,
    template_affine,
    pre_align)
transformed1 = mapping.transform(template_data)
transformed_mask1 = mapping.transform(template_data_mask)

plt.subplot(1, 4, 3).set_axis_off()
plt.imshow(transformed1[:, :, sli].astype('float'),
           cmap='gray', origin='lower')
plt.title("Tranformed template output")

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
# regtools.overlay_slices(input_data, transformed, None, 1, 'input', 'Template whole', 'reg_exp5.png')
patch_radius = 1
block_radius = 2
patch_size = 2 * patch_radius + 1
block_size = 2 * block_radius + 1
total_radius = block_radius + patch_radius
h = 1
avg_wt = 0.0
wt_sum = 0.0
output_data = np.zeros(input_data.shape, dtype=np.float64)

for i in range(total_radius, input_data.shape[0] - total_radius):
    print(i)
    for j in range(total_radius, input_data.shape[1] - total_radius):
        for k in range(total_radius, input_data.shape[2] - total_radius):
            wt_sum = 0.0
            avg_wt = 0.0
            # find the patch centered around the voxel
            patch = input_data[i - patch_radius: i + patch_radius,
                               j - patch_radius: j + patch_radius,
                               k - patch_radius: k + patch_radius]
            patch = np.array(patch, dtype=np.float64)
            patch = patch / np.sum(patch)

            for i0 in range(i - block_radius, i + block_radius):
                for j0 in range(j - block_radius, j + block_radius):
                    for k0 in range(k - block_radius, k + block_radius):

                        # now find a patch centered around each of the voxels in neighbourhood
                        # from the transformed template

                        patch_template = transformed1[
                            i0 - patch_radius: i0 + patch_radius,
                            j0 - patch_radius: j0 + patch_radius,
                            k0 - patch_radius: k0 + patch_radius]
                        patch_template = patch_template / \
                            np.sum(patch_template)
                        # compute the patch difference and the weight
                        weight = np.exp(-np.sum((patch -
                                                 patch_template)**2) / h * h)
                        wt_sum += weight
                        avg_wt += weight * transformed_mask1[i0, j0, k0]

            output_data[i, j, k] = avg_wt / wt_sum

output_data[np.isnan(output_data) == 1] = 0
out = np.zeros(input_data.shape, dtype=np.float64)
out[output_data > 0.9] = input_data[output_data > 0.9]

output_data[output_data < 0.9] = 0
plt.subplot(1, 4, 4).set_axis_off()
plt.imshow(out[:, :, sli].astype('float'),
           cmap='gray', origin='lower')
plt.title("The patch averaging label output")

plt.savefig('exp1.png', bbox_inches='tight')
plt.show()

nib.save(
    nib.Nifti1Image(
        output_data,
        input_affine), filename_output)

nib.save(
    nib.Nifti1Image(
        out,
        input_affine), filename_output_mask)
