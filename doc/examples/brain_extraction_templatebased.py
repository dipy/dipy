import numpy as np
import scipy as sp
import nibabel as nib
from time import time
import matplotlib.pyplot as plt
from dipy.core.histeq import histeq
from dipy.segment.mask import median_otsu
from dipy.segment.mask import brain_extraction
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

# dname = '..'
# The filepaths (and their corroesponding dropbox location)
filename_isbr = '/Users/Riddhish/Documents/GSOC/DIPY/data/Brain Extraction/IBSR_nifti_stripped/IBSR_02/IBSR_02_ana.nii.gz'
# filename_isbr = dname + '/brain_extraction/First tests/input_data/IBSR_02_ana.nii.gz'
filename_template = '/Users/Riddhish/Documents/GSOC/DIPY/data/Brain Extraction/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c.nii'
# filename_template = dname + '../brain_extraction/First tests/template_data/mni_icbm152_t1_tal_nlin_asym_09c.nii'
filename_template_mask = '/Users/Riddhish/Documents/GSOC/DIPY/data/Brain Extraction/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_mask.nii'
# filename_template_mask = dname + '../brain_extraction/First tests/template_data/mni_icbm152_t1_tal_nlin_asym_09c_mask.nii'
filename_output = '/Users/Riddhish/Documents/GSOC/DIPY/data/Brain Extraction/forElef_GSoC_brain_extraction/brain_extracted.nii.gz'
# filename_output = dname + '../brain_extraction/First tests/output_data/brain_extracted.nii.gz'
filename_output_mask = '/Users/Riddhish/Documents/GSOC/DIPY/data/Brain Extraction/forElef_GSoC_brain_extraction/brain_extracted_mask.nii.gz'
# filename_output_mask = dname + '../brain_extraction/First tests/output_data/brain_extracted_mask.nii.gz'

img = nib.load(filename_isbr)
input_data = img.get_data()
input_affine = img.get_affine()

print(input_data.shape)
input_data = input_data[..., 0]
# input_data = input_data[40:210, 30:220, 70:90]

# read the template
img = nib.load(filename_template)
template_data = img.get_data()
template_affine = img.get_affine()
# read the template mask
img = nib.load(filename_template_mask)
template_data_mask = img.get_data()

# first do the pure brain extraction for the data using
# median otsu

b0_mask, mask = median_otsu(input_data, 2, 1)

sli = input_data.shape[2] / 2

plt.figure('Brain segmentation')
plt.subplot(1, 3, 1).set_axis_off()
plt.imshow(input_data[:, :, sli].astype('float'),
           cmap='gray', origin='lower')
plt.title("Input Data")

# plt.subplot(1, 3, 2).set_axis_off()
# plt.imshow(mask[:, :, sli].astype('float'),
#            cmap='gray', origin='lower')
# plt.title("Median Otsu Output")

# Now we do the template based brain extraction
t = time()
[output_data, output_mask] = brain_extraction(input_data,
                                              input_affine,
                                              template_data,
                                              template_affine,
                                              template_data_mask,
                                              patch_radius=1,
                                              block_radius=2,
                                              threshold=0.9)
print("Time taken", time() - t)
plt.subplot(1, 3, 2).set_axis_off()
plt.imshow(output_data[:, :, sli].astype('float'),
           cmap='gray', origin='lower')
plt.title("The patch averaging label output")

plt.subplot(1, 3, 3).set_axis_off()
plt.imshow(output_mask[:, :, sli].astype('float'),
           cmap='gray', origin='lower')
plt.title("The patch averaging mask output")
# plt.savefig('exp1.png', bbox_inches='tight')
plt.show()

# nib.save(
#     nib.Nifti1Image(
#         output_data,
#         input_affine), filename_output)

# nib.save(
#     nib.Nifti1Image(
#         output_mask,
#         input_affine), filename_output_mask)
