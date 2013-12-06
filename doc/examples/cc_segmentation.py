"""====================================
Corpus callosum segmentation using DTI
=======================================

We can slightly modify the SNR example to have a better and anatomically
more robust segmentation of the corpus callosum (CC).

By loosening the restrictions on the red-blue-green channels of the
colored-FA (cfa) map, we can obtain a segmentation
of the entire CC.

Here, we assume that the tensorfit is already done (see DTI or SNR examples).

"""
threshold = (0.2, 1, 0, 0.3, 0, 0.3)

mask_corpus_callosum, cfa = segment_from_cfa(tensorfit, CC_box,
                                             threshold, return_cfa=True)

"""Let's now clean up our mask by getting rid of any leftover voxels that are
not a part of the corpus callosum.
"""

from dipy.segment.mask import clean_cc_mask

cleaned_cc_mask = clean_cc_mask(mask_corpus_callosum3)
cleaned_cc_mask_img = nib.Nifti1Image(cleaned_cc_mask.astype(np.uint8), affine)
nib.save(cleaned_cc_mask_img, 'mask_corpus_callosum_cleaned.nii.gz')

"""Now let's check our result by plotting our new mask alongside our old mask.
"""

# fig = plt.figure('Corpus callosum segmentation2')
# plt.subplot(1, 2, 1)
# plt.title("Old segmentation")
# plt.imshow(mask_corpus_callosum3[region, ...])

# plt.subplot(1, 2, 2)
# plt.title("New segmentation")
# plt.imshow(cleaned_cc_mask[region, ...])

#fig.savefig("Comparison_of_segmentation2.png")

"""
.. figure:: Comparison_of_segmentation2.png
"""




