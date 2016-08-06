"""
====================================
Brain extraction using template data
====================================

We show how to extract the brain from data with different modalities
with the help of a template T1 data which already has it's brain
extracted. The method uses inputs from [Eskildsen11]_ and [Lutkenhoff14]_
to get a robust brain extraction. The major ideas which are being used in 
the algorithm are as follows

* Affine and non-linear registeration

* Patch based averaging for similar modality and voting for dissimilar modalities

First let us load the necessary modules
"""

import numpy as np
import scipy as sp
import nibabel as nib
from time import time
import matplotlib.pyplot as plt
from dipy.segment.mask import (median_otsu, 
                              jaccard_index,
                              brain_extraction)

# dname = '..'
# The filepaths (and their corroesponding dropbox location)
filename_isbr = '/Users/Riddhish/Documents/GSOC/DIPY/data/Brain Extraction/IBSR_nifti_stripped/IBSR_02/IBSR_02_ana.nii.gz'
# filename_isbr = '/Users/Riddhish/Documents/GSOC/DIPY/data/Brain Extraction/forElef_GSoC_brain_extraction/tumor_case/t1_0.nii.gz'
# filename_isbr = dname + '/brain_extraction/First tests/input_data/IBSR_02_ana.nii.gz'
filename_isbr_mask = '/Users/Riddhish/Documents/GSOC/DIPY/data/Brain Extraction/IBSR_nifti_stripped/IBSR_02/IBSR_02_mask.nii.gz'
# filename_isbr_mask = dname + '/brain_extraction/First tests/input_data/IBSR_02_mask.nii.gz'
filename_template = '/Users/Riddhish/Documents/GSOC/DIPY/data/Brain Extraction/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c.nii'
# filename_template = dname + '../brain_extraction/First tests/template_data/mni_icbm152_t1_tal_nlin_asym_09c.nii'
filename_template_mask = '/Users/Riddhish/Documents/GSOC/DIPY/data/Brain Extraction/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_mask.nii'
# filename_template_mask = dname + '../brain_extraction/First tests/template_data/mni_icbm152_t1_tal_nlin_asym_09c_mask.nii'
filename_output = '/Users/Riddhish/Documents/GSOC/DIPY/data/Brain Extraction/forElef_GSoC_brain_extraction/brain_extracted.nii.gz'
# filename_output = dname + '../brain_extraction/First tests/output_data/brain_extracted.nii.gz'
filename_output_mask = '/Users/Riddhish/Documents/GSOC/DIPY/data/Brain Extraction/forElef_GSoC_brain_extraction/brain_extracted_mask.nii.gz'
# filename_output_mask = dname + '../brain_extraction/First tests/output_data/brain_extracted_mask.nii.gz'

"""
Read the input data and the template data from the DIPY datasets
"""

img = nib.load(filename_isbr)
input_data = img.get_data()
input_affine = img.get_affine()
input_data = input_data[..., 0]

print("Input T1 volume", input_data.shape)

img = nib.load(filename_template)
template_data = img.get_data()
template_affine = img.get_affine()
# read the template mask
img = nib.load(filename_template_mask)
template_data_mask = img.get_data()

print("Template volume", template_data.shape)

"""
Let us see how the template data looks like
"""

sli = template_data.shape[2] / 2

plt.figure("Template Data")
plt.subplot(1, 2, 1).set_axis_off()
plt.imshow(template_data[...,sli], cmap = 'gray', origin = 'lower')
plt.subplot(1, 2, 2).set_axis_off()
plt.imshow(template_data_mask[...,sli], cmap = 'gray', origin = 'lower')
plt.savefig('template_data.png', bbox_inches='tight')

"""
.. figure:: template_data.png
   :align: center

   **Axial slice of template and it's corresponding mask**.
"""

"""
Now we apply the ``brain extraction`` function which takes the input, template data and
template mask as inputs, along with their affine information. There are five other 
parameters which can be given to the function. 

The same_modality takes boolean value true if the input and template are of same modality
and false if they are not, when it takes value false the only useful parameters are the
patch_radius and threshold, rest are only used when the modalities are same.

The patch_radius and block_radius are the inputs for block wise local averaging which 
is used after the registeration step in the ``brain extraction``. The parameter value 
which is set to 1 as defaults governs the weighing, the threshold value governs the 
eroded boundary coefficient of the extracted mask. For more info on how these parameters 
works please look at the ``fast_patch_averaging`` function in dipy.segment.
"""

t = time()

[output_data, output_mask] = brain_extraction(input_data,
                                              input_affine,
                                              template_data,
                                              template_affine,
                                              template_data_mask,
                                              patch_radius=1,
                                              block_radius=2,
                                              threshold=0.9)

"""
Follow it up by a median otsu for added robustness
"""

b0_mask, mask = median_otsu(output_data, 2, 2)

print("Time taken", time() - t)

img = nib.load(filename_isbr_mask)
input_manual_mask = img.get_data()

"""
The jaccard's index measures how similar are the two binary images, so we compare
our generated mask with the manually extracted brain mask. Perfect match will corroespond 
to the jaccard index of 1.
"""

mea = jaccard_index(mask.astype(np.bool), input_manual_mask.astype(np.bool))

print("Jaccard Index", mea)

"""
Let us plot the axial slice of the extraction
"""

sli = input_data.shape[2] / 2

plt.figure('Brain segmentation')
plt.subplot(1, 3, 1).set_axis_off()
plt.imshow(input_data[:, :, sli].astype('float'),
           cmap='gray', origin='lower')
plt.title("Input Data")

plt.subplot(1, 3, 2).set_axis_off()
plt.imshow(output_data[:, :, sli].astype('float'),
           cmap='gray', origin='lower')
plt.title("Extraction output")

plt.subplot(1, 3, 3).set_axis_off()
plt.imshow(output_mask[:, :, sli].astype('float'),
           cmap='gray', origin='lower')
plt.title("Extracted mask")
plt.savefig('brain_extraction_same.png', bbox_inches='tight')

"""
.. figure:: brain_extraction_same.png
   :align: center

   **Input data (T1), the extracted brain and the corresponding mask (axial slice shown)**.
"""

nib.save(nib.Nifti1Image(b0_mask, input_affine), 'brain_extraction_diff.nii.gz')

"""
Now considering the input of b0 modality while the template remains the same T1 modality
"""

from dipy.data import fetch_sherbrooke_3shell, read_sherbrooke_3shell

fetch_sherbrooke_3shell()
img, gtab = read_sherbrooke_3shell()
input_data = img.get_data()
input_affine = img.get_affine()
input_data = input_data[...,0]

print("Input b0 volume", input_data.shape)

t = time()

[output_data, output_mask] = brain_extraction(input_data,
                                              input_affine,
                                              template_data,
                                              template_affine,
                                              template_data_mask,
                                              patch_radius=1,
                                              same_modality = False)

b0_mask, mask = median_otsu(output_data, 2, 2)

print("Time taken", time() - t)

sli = input_data.shape[2]/ 2 

plt.figure('Brain segmentation 2')
plt.subplot(1, 3, 1).set_axis_off()
plt.imshow(input_data[:, :, sli].astype('float'),
           cmap='gray', origin='lower', interpolation='none')
plt.title("Input Data")

plt.subplot(1, 3, 2).set_axis_off()
plt.imshow(b0_mask[:, :, sli].astype('float'),
           cmap='gray', origin='lower',interpolation='none')
plt.title("Extraction output")

plt.subplot(1, 3, 3).set_axis_off()
plt.imshow(mask[:, :, sli].astype('float'),
           cmap='gray', origin='lower', interpolation='none')
plt.title("Extracted mask")
plt.savefig('brain_extraction_diff.png', bbox_inches='tight')
plt.show()

"""
.. figure:: brain_extraction_diff.png
   :align: center

   **Input data (B0), the extracted brain and the corresponding mask (axial slice shown)**.
"""

nib.save(nib.Nifti1Image(b0_mask, input_affine), 'brain_extraction_diff.nii.gz')

"""

.. [Lutkenhoff14] Evan S. Lutkenhoff et al., Optimized Brain Extraction for
   Pathological Brains (OptiBET), PLOS, 2014

.. [Eskildsen11]  Simon Fristed Eskildsen et al., BEaST : Brain extraction based on
                  nonlocal segmentation technique, NeuroImage, vol 59, 2011.

.. include:: ../links_names.inc

"""
