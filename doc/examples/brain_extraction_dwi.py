"""
===================================
Brain segmentation with median_otsu
===================================

We show how to extract brain information and mask from a b0 image using dipy's
segment.mask module.

First import the necessary modules:
"""

import numpy as np
import nibabel as nib

def histeq(im,nbr_bins=256):

   #get image histogram
   imhist,bins = np.histogram(im.flatten(), nbr_bins, normed=True)
   cdf = imhist.cumsum() #cumulative distribution function
   cdf = 255 * cdf / cdf[-1] #normalize

   #use linear interpolation of cdf to find new pixel values
   im2 = np.interp(im.flatten(),bins[:-1],cdf)

   return im2.reshape(im.shape)

"""
Download and read the data for this tutorial.

The scil_b0 dataset contains different data from different companies and models.
For this example, the data comes from a 3 tesla GE MRI.
"""

from dipy.data import fetch_scil_b0, read_scil_b0
fetch_scil_b0()
img = read_scil_b0()
data = np.squeeze(img.get_data())

"""
``img`` contains a nibabel Nifti1Image object. Data is the actual brain data as
a numpy ndarray.

Segment the brain using dipy's mask module.

``median_otsu`` returns the segmented brain data and a binary mask of the brain.
It is possible to fine tune the parameters of ``median_otsu`` (``median_radius``
and ``num_pass``) if extraction yields incorrect results but the default
parameters work well on most volumes. For this example, default parameters (4,
4) will be used.
"""

from dipy.segment.mask import median_otsu
b0_mask, mask = median_otsu(data, 4, 4)

"""
Saving the segmentation results is very easy using nibabel. We need the b0_mask,
and the binary mask volumes. The affine matrix which transform the image's
coordinates to the world coordinates is also needed. Here, we choose to save
both images in float32.
"""

mask_img = nib.Nifti1Image(mask.astype(np.float32), img.get_affine())
b0_img = nib.Nifti1Image(b0_mask.astype(np.float32), img.get_affine())

fname = 'ge_3t'
nib.save(mask_img, fname + '_binary_mask.nii.gz')
nib.save(b0_img, fname + '_mask.nii.gz')

"""
Quick view of the results middle slice using matplotlib.
"""

import matplotlib.pyplot as plt
from matplotlib.pyplot import colormaps as cm

sli = data.shape[2] / 2
plt.figure('Brain segmentation')
plt.subplot(1, 2, 1).set_axis_off()

'''
# Compute new display scale to view removed background.
min = data[:, :, sli].min()
max = data[:, :, sli].max()
mean = (max - min) / 2.0
qmean = mean / 4.0


plt.imshow(data[:, :, sli], vmin=min, vmax=mean - qmean, cmap='jet')#cmap='jet')
plt.subplot(1, 2, 2).set_axis_off()
plt.imshow(b0_mask[:, :, sli], vmin=min, vmax=mean - qmean, cmap='jet') #cmap='jet')
plt.savefig('median_otsu.png')
'''

plt.imshow(histeq(data[:, :, sli].astype('float')).T, cmap='gray', origin='lower')
plt.subplot(1, 2, 2).set_axis_off()
plt.imshow(histeq(b0_mask[:, :, sli].astype('float')).T, cmap='gray', origin='lower')
plt.savefig('median_otsu.png')

"""
.. figure:: median_otsu.png
   :align: center

   **An application of median_otsu for brain segmentation**.

``median_otsu`` can also automatically crop the outputs to remove the largest
possible number of background voxels. This makes outputted data significantly
smaller.  auto cropping in ``median_otsu`` is activated by setting the
``autocrop`` parameter to True.
"""

b0_mask_crop, mask_crop = median_otsu(data, 4, 4, autocrop=True)


