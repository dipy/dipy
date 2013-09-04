"""==============================
Corpus callosum segmentation 
==============================

This example shows how to extract voxels in the corpus callosum where the 
diffusion is mainly oriented in the left-right direction from a raw DWI. 
These voxels will have a high value of red in the 
Colored Fractional Anisotropy (cfa) map.
The method uses the Colored Fractional Anisotropy as a threshold reference. 

The purpose of this kind of segmentation is not to clearly separate the 
structure, but rather to compute an automatic mask in order to compute various 
SNR in the region of interest later.
This gives a way to quantify the quality of the signal amongst various 
diffusion orientation, which can change accordingly to the structure, the 
composition and the orientation of the studied tissues.

As a first step, import the necessary modules:
"""

from __future__ import division, print_function

import nibabel as nib
import numpy as np

from dipy.data import fetch_stanford_hardi, read_stanford_hardi

"""Download and read the data for this tutorial.
Let's first load the data. We will use a dataset with 10 b0s and 
150 non-b0s with b-value 2000.
"""

fetch_stanford_hardi()
img, gtab = read_stanford_hardi()

"""img contains a nibabel Nifti1Image object (data) and gtab contains a GradientTable
object (gradient information e.g. b-values). For example to read the b-values
it is possible to write print(gtab.bvals).

Load the raw diffusion data and the affine data.
"""

data = img.get_data()
affine = img.get_affine()
print('data.shape (%d, %d, %d, %d)' % data.shape)

"""data.shape ``(81, 106, 76, 160)`` 
"""

"""To reduce the computation time, we will only estimate the tensor model 
inside the brain region by creating a mask without the background.
(See the masking example for more details about this step)
"""

from dipy.segment.mask import median_otsu
b0_mask, mask = median_otsu(data[..., 0], 4, 4)

"""We can now do a first segmentation of the data using the Colored Fractional 
Anisotropy (or cfa). It encodes in a 3D volume the orientation of the diffusion.
Red means that the principal direction of the tensor is in x, Green is for the 
y direction and the z direction is encoded with blue.

We know that the corpus callosum should be in the middle of the brain 
and since the principal diffusion direction is the x axis, 
the red channel should be the highest in the cfa.

Let's pick a threshold of 0.7 in the x axis and 0.1 in the y and z axis as a 
segmentation threshold. 

We will also define a rough roi, since noisy pixels could be considered in the
mask if it's not bounded properly. Adjusting the cfa threshold and the roi 
location enables the function to segment any part of the brain based on 
an orientation and spatial location criterion.

HEY READ THIS BEFORE MERGING!!!!1111!!!11111!!!!1
Note : Just as the threshold, roi could be supplied as a tuple, then made as a mask.
although it means you can supply premade mask made by rough drawing for example,
but we could probably try to support both if it's not too much of an hassle.
"""

from dipy.segment.mask import segment_from_dwi

threshold = (0.7, 1, 0, 0.1, 0, 0.1)
roi = (30, 50, 35, 80, 25, 50)

CC_box = np.zeros_like(data[..., 0])
CC_box[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]] = 1

mask_corpus_callosum, cfa = segment_from_dwi(data, gtab, CC_box, 
                                             threshold, mask=mask, return_cfa=True)

print ("Size of the mask :", np.count_nonzero(mask_corpus_callosum), \
       "voxels out of", np.size(CC_box))

"""We can save the produced dataset with nibabel to visualize them later on.
This way, we can also redo a segmentation of another region using the cfa instead
of recomputing everything everytime.

Note that we save the cfa with values between 0 and 255 for visualization purpose
in the fibernavigator. Remember that the function works with values between
0 and 1, but it will warn you if it's not the case.
"""

cfa_img = nib.Nifti1Image((cfa*255).astype(np.uint8), affine)
mask_corpus_callosum_img = nib.Nifti1Image(mask_corpus_callosum, affine)

nib.save(cfa_img, 'cfa.nii.gz')
nib.save(mask_corpus_callosum_img, 'mask_corpus_callosum.nii.gz')


"""The mask is a bit small, so let's change the threshold and restart segmenting
from the cfa using the corresponding function. Both of them do the same thing,
but the segmentation from cfa is faster when you have it, since it doesn't need 
to recompute the tensors in order to compute the cfa.
"""

from dipy.segment.mask import segment_from_cfa

threshold2 = (0.6, 1, 0, 0.1, 0, 0.1)
mask_corpus_callosum_from_cfa = segment_from_cfa(cfa, CC_box, threshold2)

mask_corpus_callosum_from_cfa_img = nib.Nifti1Image(mask_corpus_callosum_from_cfa, affine)
nib.save(mask_corpus_callosum_from_cfa_img, 'mask_corpus_callosum_from_cfa.nii.gz')

print ("Size of the mask :", np.count_nonzero(mask_corpus_callosum_from_cfa), \
       "voxels out of", np.size(CC_box))

"""Let's check the result of the segmentation using matplotlib.
"""

import matplotlib.pyplot as plt

fig = plt.figure('Corpus callosum segmentation')
plt.subplot(1, 2, 1)
plt.title("Corpus callosum")
plt.imshow((cfa[..., 0])[40, ...])

plt.subplot(1, 2, 2)
plt.title("Corpus callosum mask with a threshold of (%.1f, %.1f, %.1f, %.1f, %.1f, %.1f)" % threshold2)
plt.imshow(mask_corpus_callosum_from_cfa[40, ...])
fig.savefig("Comparison_of_segmentation.png")
