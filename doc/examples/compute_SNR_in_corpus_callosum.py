"""
=================================================
Corpus callosum segmentation and SNR computation.
=================================================

This example shows how to segment the corpus callosum from a raw DWI
using the color fa as a threshold reference. 

First import the necessary modules:
"""

from __future__ import division, print_function

import nibabel as nib
import numpy as np

from dipy.data import fetch_stanford_hardi, read_stanford_hardi

"""
Download and read the data for this tutorial.
Lets first load the data. We will use a dataset with 10 b0s and 
150 non-b0s with b-value 2000.
"""

fetch_stanford_hardi()
img, gtab = read_stanford_hardi()

"""
img contains a nibabel Nifti1Image object (data) and gtab contains a GradientTable
object (gradient information e.g. b-values). For example to read the b-values
it is possible to write print(gtab.bvals).

Load the raw diffusion data and the affine data.
"""

data = img.get_data()
affine = img.get_affine()
print('data.shape (%d, %d, %d, %d)' % data.shape)

"""
data.shape ``(81, 106, 76, 160)``

The stanford dataset contains many b0s and diffusion images. In order to reduce
computational burden for the sake of the example, we will select only one b0 and
60 directions. Since we only want to do a threshold on the color fa using 
tensor estimation, this should be enough for our example. 
"""

subset_b0 = (0,)
subset_gd = tuple(range(10, 70))
data_small = data[..., subset_b0 + subset_gd]

bvals_small = gtab.bvals[:, np.newaxis]
bvals_small = bvals_small[subset_b0 + subset_gd, :].squeeze()
# bvals is a 1d vec, so there should probably be a better way to do this...
bvecs_small = gtab.bvecs[subset_b0 + subset_gd, :]

print('data_small.shape (%d, %d, %d, %d)' % data.shape)
print('bvals_small.shape (%d)' % bvals_small.shape)
print('bvecs_small.shape (%d, %d)' % bvecs_small.shape)

"""
data_small.shape ``(81, 106, 76, 61)``
bvals_small.shape ``(61, 3)``
bvecs_small.shape ``(61)``

We will now create a new gtab object for our smaller dataset using the functions
available in dipy.
"""

from dipy.core.gradients import gradient_table_from_bvals_bvecs
gtab_small = gradient_table_from_bvals_bvecs(bvals_small, bvecs_small)

"""
To further reduce the computation time, we will only estimate the tensor model 
inside the brain region by creating a mask without the background.
(See the masking example for more details about this step)
"""

from dipy.segment.mask import median_otsu
b0_mask, mask = median_otsu(data_small[..., 0], 4, 4)

"""
We can now do a first segmentation of the data using the cfa. We will also get 
the rgb this way so we can further improve the masking if needed.

We know that the corpus callosum should be in the middle of the brain 
and since the principal diffusion direction is the x axis, 
the red channel should be the highest in the cfa.

Let's pick a threshold of 0.7 in the x axis and 0.1 in the y and z axis as a 
segmentation threshold. 

We will also define a rough roi, since noisy pixels could be considered in the
maks if it's not bounded properly. Adjusting the cfa threshold and the roi 
location enables the function to segment any part of the brain based on 
an orientation and spatial location criterion.

Note : Just as the threshold, roi could be supplied as a tuple, then made as a mask.
although it means you can supply premade mask made by rough drawing for example,
but we could probably try to support both if it's not too much of an hassle.
"""

from dipy.segment.mask import segment_from_dwi

threshold = (0, 0.7, 0, 0.1, 0, 0.1)
roi = (30, 50, 35, 80, 25, 50)

CC_box = np.zeros_like(data[..., 0])
CC_box[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]] = 1

mask_corpus_callosum, cfa = segment_from_dwi(data_small, gtab_small, CC_box, 
                                             threshold, mask=mask, return_cfa=True)

"""
We can save the produced dataset with nibabel to visualize them later on.
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

"""
Let's check the result of the segmentation using fvtk.
"""

from dipy.data import get_sphere
sphere = get_sphere('symmetric724')

from dipy.viz import fvtk
ren = fvtk.ren()

cfa_crop = cfa[roi, ...]
evals = tenfit.evals[20:50,55:85, 38:39]
evecs = tenfit.evecs[20:50,55:85, 38:39]

#tensor_odfs = tenmodel.fit(data_small[roi]).odf(sphere)

fvtk.add(ren, cfa_crop)
fvtk.add(ren, mask_corpus_callosum)
fvtk.show(r)

"""
The mask is not very good, so let's change the threshold and restart segmenting
from the cfa using the corresponding function. Both of them do the same thing,
but the segmentation from cfa is faster when you have it, since it doesn't need 
to recompute the tensors in order to compute the cfa.
"""

from dipy.segment.mask import segment_from_cfa

threshold2 = (0, 0.6, 0, 0.1, 0, 0.1)
mask_corpus_callosum_from_cfa = segment_from_cfa(cfa, CC_box, threshold2)

mask_corpus_callosum_from_cfa_img = nib.Nifti1Image(mask_corpus_callosum_from_cfa, affine)
nib.save(mask_corpus_callosum_from_cfa_img, 'mask_corpus_callosum_from_cfa.nii.gz')


print('Saving illustration as cfa_mask.png')
fvtk.record(ren, n_frames=1, out_path='cfa_mask.png', size=(800, 800))
