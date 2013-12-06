"""==========================================
SNR estimation for Diffusion-Weighted Images
=============================================

Computing the Signal-to-Noise-Ratio (SNR) of DW images is still an open question,
as SNR depends on the white matter structure of interest as well as 
the gradient direction corresponding to each DWI.  

In classical MRI, SNR can be defined as the ratio of the mean 
of the signal divided by the standard deviation of the
noise, that is

.. math::

    SNR = \frac{\mu_{signal}}{\sigma_{noise}}

The noise standard deviation can be computed from the background in any of the DW
images. But how do we computed the mean of the signal, and what signal?

The strategy here is to compute a \emph{'worst-case'} SNR for DWI. Several white matter  
structures such as the corpus callosum (CC), corticospinal tract (CST), or
the superior longitudinal fasciculus (SLF) can be easily identified from
the colored-FA (cfa) map. In this example, we will use voxels from the CC, 
which have the characteristic of being highly RED in the cfa map because mainly oriented in  
the left-right direction.

Hence, the purpose here is \emph{not} to get a valid 
segmentation of the CC but voxels where we are very confident that
the underlying fiber population is in left-right (x-direction). These voxels
will be used to compute the mean signal of all DW images. We know that the DW image
closest to the x-direction will be the one with \emph{most attenuated} diffusion signal.
Therefore,  this will produce a worst-case SNR estimation for the given DWI dataset. This is
the strategy adopted in several recent papers (see [1]_ and [2]_). It gives a good
indication of the quality of the DWI data.

First, we compute the tensor model in a brain mask (see the DTI example for more explanation)

"""

from __future__ import division, print_function
import nibabel as nib
import numpy as np
from dipy.data import fetch_stanford_hardi, read_stanford_hardi
fetch_stanford_hardi()
img, gtab = read_stanford_hardi()
data = img.get_data()
affine = img.get_affine()
from dipy.segment.mask import median_otsu
b0_mask, mask = median_otsu(data, 3, 1, True,
                            vol_idx=range(10, 50), dilate=2)
from dipy.reconst.dti import TensorModel
tenmodel = TensorModel(gtab)
tensorfit = tenmodel.fit(data, mask=mask)

"""Next, we set our red-blue-green thresholds to (0.7, 1) in the x axis
and (0, 0.1) in the y and z axes respectively.
These values work well in practice to isolate the very RED voxels of the cfa map.

Then, as assurance, we want just RED voxels in the CC (there could be
noisy red voxels around the brain mask and we don't want those). Unless the brain
acquisition was badly aligned, the CC is always close to the mid-sagittal slice. 

The following lines perform these two operations and then saves the computed mask.
"""

from dipy.segment.mask import segment_from_cfa
from dipy.segment.mask import bounding_box

threshold = (0.6, 1, 0, 0.1, 0, 0.1)
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

print(bounds_max[0])
CC_box[bounds_min[0]:50,
       bounds_min[1]:bounds_max[1],
       bounds_min[2]:bounds_max[2]] = 1

mask_cc_part, cfa = segment_from_cfa(tensorfit, CC_box,
				     threshold, return_cfa=True)

print("Size of the mask :", np.count_nonzero(mask_cc_part), \
       "voxels out of", np.size(CC_box))
cfa_img = nib.Nifti1Image((cfa*255).astype(np.uint8), affine)
mask_cc_part_img = nib.Nifti1Image(mask_cc_part.astype(np.uint8), affine)
nib.save(mask_cc_part_img, 'mask_CC_part.nii.gz')


"""Let's check the result of the second segmentation using matplotlib.
"""

# import matplotlib.pyplot as plt
# region = 40
# fig = plt.figure('Corpus callosum segmentation')
# plt.subplot(1, 2, 1)
# plt.title("Corpus callosum")
# plt.imshow((cfa[..., 0])[region, ...])

# plt.subplot(1, 2, 2)
# plt.title("Corpus callosum segmentation")
# plt.imshow(mask_cc_part[region, ...])
# fig.savefig("Comparison_of_segmentation.png")

"""
.. figure:: Comparison_of_segmentation.png
"""

"""Now that we are happy with our crude mask that selected voxels in x-direction, 
we can use all the voxels to estimate the mean signal in this region.
(\emph{recall that we did not want a perfect CC segmentation but just several voxels in the
x-direction})

"""

mean_signal = np.mean(data[mask_cc_part], axis=0)

"""Now, we need a good background estimation. We will re-use the brain mask
computed before and invert it to catch the outside of the brain. This could
also be determined manually with a box ROI in the background.
(certain MR manufacturers mask out the outside of the brain with 0's, you then
have to carefully define a box ROI manually).
"""

from scipy.ndimage.morphology import binary_dilation
mask_noise = binary_dilation(mask, iterations=10)
mask_noise[..., :mask_noise.shape[-1]//2] = 1
mask_noise = ~mask_noise
mask_noise_img = nib.Nifti1Image(mask_noise.astype(np.uint8), affine)
nib.save(mask_noise_img, 'mask_noise.nii.gz')

noise_std = np.std(data[mask_noise, :])

"""We can now compute the SNR for each DWI using the formula defined above.
Let's find the position of the gradient direction that lies the closest to
the X, Y and Z axes.
"""

# Exclude null bvecs from the search
idx = np.sum(gtab.bvecs, axis=-1) == 0
gtab.bvecs[idx] = np.inf
axis_X = np.argmin(np.sum((gtab.bvecs-np.array([1, 0, 0]))**2, axis=-1))
axis_Y = np.argmin(np.sum((gtab.bvecs-np.array([0, 1, 0]))**2, axis=-1))
axis_Z = np.argmin(np.sum((gtab.bvecs-np.array([0, 0, 1]))**2, axis=-1))

"""Now let's compute their respective SNR and compare them to the SNR of the
b0 image SNR. 
"""

for direction in [0, axis_X, axis_Y, axis_Z]:
	SNR = mean_signal[direction]/noise_std
	print("SNR for direction", direction, "is :", SNR)
	print(gtab.bvecs[direction]
	      
"""SNR for direction 0 is : ``39.7490994429``"""
"""SNR for direction 58 is : ``4.84444879426``"""
"""SNR for direction 57 is : ``22.6156341499``"""
"""SNR for direction 126 is : ``23.1985563491``"""

"""Since the CC is aligned with the X axis, it is the lowest SNR in all of
the DWIs, where the DW signal is the most attenuated. In comparison, the DW images in
perpendical Y and Z axes have a high SNR. The b0 still exhibits the highest SNR,
since there is no signal attenuation.

Hence, we can say the Stanford diffusion data has a 'worst-case' SNR of approximately 5, a 
'best-case' SNR of approximately 23, and a SNR of 40 on the b0 image. 
"""
