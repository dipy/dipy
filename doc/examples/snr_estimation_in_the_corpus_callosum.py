"""============================================
Computing SNR estimation in the corpus callosum
===============================================

This example shows how to extract voxels in the splenium of the corpus
callosum where the diffusion is mainly oriented in the left-right
direction from a raw DWI.  These voxels will have a high value of red
in the Colored Fractional Anisotropy (cfa) map.  The method uses the
colored fractional anisotropy as a threshold reference.

The purpose of this kind of segmentation is not to clearly separate the
structure, but rather to compute an automatic mask in order to compute the
SNR in the region of interest later.
This gives a way to quantify the quality of the signal amongst various
diffusion orientations, which can change according to the structure, the
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

"""img contains a nibabel Nifti1Image object (data) and gtab contains a
GradientTable object (gradient information e.g. b-values). For example to read
the b-values it is possible to write print(gtab.bvals).

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
b0_mask, mask = median_otsu(data)

"""We also need to fit a tensor model on the data in order to compute the cfa.
"""

from dipy.reconst.dti import TensorModel
tenmodel = TensorModel(gtab)
tensorfit = tenmodel.fit(data, mask=mask)

"""We can now do a first segmentation of the data using the Colored Fractional
Anisotropy (or cfa). It encodes in a 3D volume the orientation of the diffusion.
Red means that the principal direction of the tensor is in x, green is for the
y direction and the z direction is encoded with blue.

We know that the corpus callosum should be in the middle of the brain
and since the principal diffusion direction is the x axis,
the red channel should be the highest in the cfa.

Let's pick a range of 0.7 to 1 in the x axis and 0 to 0.1 in the y and z axis
as a segmentation threshold.

We will also define a rough roi, since noisy pixels could be considered in the
mask if it's not bounded properly. Adjusting the cfa threshold and the roi
location enables the function to segment any part of the brain based on
an orientation and spatial location. For now, we will pick half of the
bounding box from the segmentation of the brain, just in case the subject was 
not centered properly.
"""

from dipy.segment.mask import segment_from_cfa
from dipy.segment.mask import bounding_box

threshold = (0.7, 1, 0, 0.1, 0, 0.1)
CC_box = np.zeros_like(data[..., 0])

mins, maxs = bounding_box(mask)
mins = np.array(mins)
maxs = np.array(maxs)
diff = (maxs - mins) // 4
bounds_min = mins + diff
bounds_max = maxs - diff

CC_box[bounds_min[0]:bounds_max[0],
       bounds_min[1]:bounds_max[1],
       bounds_min[2]:bounds_max[2]] = 1

mask_corpus_callosum, cfa = segment_from_cfa(tensorfit, CC_box,
                                             threshold, return_cfa=True)

print("Size of the mask :", np.count_nonzero(mask_corpus_callosum), \
       "voxels out of", np.size(CC_box))

"""We can save the produced dataset with nibabel to visualize them later on.

Note that we save the cfa with values between 0 and 255 for visualization 
purpose. Remember that the function works with values between
0 and 1, but it will warn you if the supplied values do not fall in this range.
"""

cfa_img = nib.Nifti1Image((cfa*255).astype(np.uint8), affine)
mask_corpus_callosum_img = nib.Nifti1Image(mask_corpus_callosum.astype(np.uint8), affine)


"""The mask has random voxels outside the splenium because of the noise, so
let's change the threshold, the bounding box and restart segmenting from
the cfa.
"""

threshold2 = (0.6, 1, 0, 0.1, 0, 0.1)

CC_box = np.zeros_like(CC_box)
CC_box[bounds_min[0]:50,
       bounds_min[1]:bounds_max[1],
       bounds_min[2]:bounds_max[2]] = 1

mask_corpus_callosum2 = segment_from_cfa(tensorfit, CC_box, threshold2)

mask_corpus_callosum2_img = nib.Nifti1Image(mask_corpus_callosum2.astype(np.uint8), affine)
nib.save(mask_corpus_callosum2_img, 'mask_corpus_callosum2.nii.gz')

print("Size of the mask :", np.count_nonzero(mask_corpus_callosum2), \
       "voxels out of", np.size(CC_box))

"""Let's check the result of the second segmentation using matplotlib.
"""

import matplotlib.pyplot as plt
region = 40
fig = plt.figure('Corpus callosum segmentation')
plt.subplot(1, 2, 1)
plt.title("Corpus callosum")
plt.imshow((cfa[..., 0])[region, ...])

plt.subplot(1, 2, 2)
plt.title("Corpus callosum segmentation")
plt.imshow(mask_corpus_callosum2[region, ...])
fig.savefig("Comparison_of_segmentation.png")

"""
.. figure:: Comparison_of_segmentation.png
"""

"""Now that we have a crude mask, we can use all the voxels to estimate the SNR
in this region. Since the corpus callosum is in the middle of the brain, the
signal should be weaker and will greatly vary according to the direction of
the b vector that is used for each DWI. The SNR should be low in the X
orientation and high in the Y and Z orientations. The SNR is usually defined as
the ratio of the mean of the signal divided by the standard deviation of the
noise, that is

.. math::

    SNR = \frac{\mu_{signal}}{\sigma_{noise}}

We will compute the mean of the signal in the mask we just created and
the standard deviation from the noise in the background.
"""

mean_signal = np.mean(data[mask_corpus_callosum2], axis=0)

"""In order to have a good background estimation, we will re-use the brain mask
computed before, but add the neck and shoulder part and then invert the
mask.
"""

from scipy.ndimage.morphology import binary_dilation
mask_noise = binary_dilation(mask, iterations=10)

mask_noise[..., :mask_noise.shape[-1]//2] = 1
mask_noise = ~mask_noise

mask_noise_img = nib.Nifti1Image(mask_noise.astype(np.uint8), affine)
nib.save(mask_noise_img, 'mask_noise.nii.gz')

noise_std = np.std(data[mask_noise, :])

"""We can now compute the SNR for each dwi using the formula above. Let's find 
the position of the gradient direction that lies the closest to the X, Y and Z 
axis.
"""

# Exclude null bvecs from the search
idx = np.sum(tenmodel.bvec, axis=-1) == 0
tenmodel.bvec[idx] = np.inf

axis_X = np.argmin(np.sum((tenmodel.bvec-np.array([1, 0, 0]))**2, axis=-1))
axis_Y = np.argmin(np.sum((tenmodel.bvec-np.array([0, 1, 0]))**2, axis=-1))
axis_Z = np.argmin(np.sum((tenmodel.bvec-np.array([0, 0, 1]))**2, axis=-1))

"""Now that we have the closest b-vectors to each of the cartesian axis, 
let's compute their respective SNR and compare them to a b0 image's SNR.
"""

for direction in [0, axis_X, axis_Y, axis_Z]:
	SNR = mean_signal[direction]/noise_std
	print("SNR for direction", direction, "is :", SNR)

"""SNR for direction 0 is : ``39.7490994429``"""
"""SNR for direction 58 is : ``4.84444879426``"""
"""SNR for direction 57 is : ``22.6156341499``"""
"""SNR for direction 126 is : ``23.1985563491``"""

"""Since the diffusion is strong in the X axis, it is the lowest SNR in all of 
the DWIs, while the Y and Z axis have almost no diffusion and as such a high
SNR. The b0 still exhibits the highest SNR, since there is no diffusion
(and as such no signal drop) at all.
"""

"""Now that we have the SNR in the splenium of the corpus callosum, let's now
start a new segmentation to create a mask of the entire corpus callosum.  We
start by loosening the restrictions on the threshold and running the
segmentation again with the same bounding box.
"""
threshold = (0.2, 1, 0, 0.3, 0, 0.3)

mask_corpus_callosum3, cfa = segment_from_cfa(tensorfit, CC_box,
                                             threshold, return_cfa=True)

"""Let's now clean up our mask by getting rid of any leftover voxels that are
not a part of the corpus callosum.
"""

from dipy.segment.mask import clean_cc_mask

cleaned_cc_mask = clean_cc_mask(mask_corpus_callosum3)

cleaned_cc_mask_img = nib.Nifti1Image(cleaned_cc_mask.astype(np.uint8), affine)
nib.save(cleaned_cc_mask_img, 'mask_corpus_callosum3_cleaned.nii.gz')

"""Now let's check our result by plotting our new mask alongside our old mask.
"""

fig = plt.figure('Corpus callosum segmentation2')
plt.subplot(1, 2, 1)
plt.title("Old segmentation")
plt.imshow(mask_corpus_callosum3[region, ...])

plt.subplot(1, 2, 2)
plt.title("New segmentation")
plt.imshow(cleaned_cc_mask[region, ...])

fig.savefig("Comparison_of_segmentation2.png")

"""
.. figure:: Comparison_of_segmentation2.png
"""

