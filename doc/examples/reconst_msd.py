import numpy as np
import matplotlib.pyplot as plt
# import nibabel as nib
from dipy.reconst.opt_msd import (MultiShellDeconvModel, MultiShellResponse, 
                                  MSDeconvFit)
from dipy.reconst.csdeconv import auto_response
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.segment.tissue import TissueClassifierHMRF
from dipy.io.image import load_nifti
from dipy.segment.mask import median_otsu

# static file paths for experiments
fbvals = '/home/shreyasfadnavis/Data/HCP/BL/sub-100408/598a2aa44258600aa3128fd0.neuro-dwi/dwi.bvals'
fbvecs = '/home/shreyasfadnavis/Data/HCP/BL/sub-100408/598a2aa44258600aa3128fd0.neuro-dwi/dwi.bvecs'
fdwi = '/home/shreyasfadnavis/Data/HCP/BL/sub-100408/598a2aa44258600aa3128fd0.neuro-dwi/dwi.nii.gz'
ft1 = '/home/shreyasfadnavis/Data/HCP/BL/sub-100408/598a2aa44258600aa3128fcf.neuro-anat-t1w.acpc_aligned/t1.nii.gz'

t1, t1_affine = load_nifti(ft1)

dwi, dwi_affine = load_nifti(fdwi)
b0_mask, mask = median_otsu(dwi)

t1[mask == 0] = 0

print("Data Loaded!")

"""
Now we will define the other two parameters for the segmentation algorithm.
We will segment three classes, namely corticospinal fluid (CSF), white matter
(WM) and gray matter (GM).
"""
nclass = 3
"""
Then, the smoothness factor of the segmentation. Good performance is achieved
with values between 0 and 0.5.
"""
beta = 0.1

hmrf = TissueClassifierHMRF()
initial_segmentation, final_segmentation, PVE = hmrf.classify(t1, nclass,
                                                              beta)


# segmentation using the HMRF
def plot_HMRF():
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    img_ax = np.rot90(final_segmentation[..., 89])
    imgplot = plt.imshow(img_ax)
    a.axis('off')
    a.set_title('Axial')
    a = fig.add_subplot(1, 2, 2)
    img_cor = np.rot90(final_segmentation[:, 128, :])
    imgplot = plt.imshow(img_cor)
    a.axis('off')
    a.set_title('Coronal')
    plt.savefig('final_seg.png', bbox_inches='tight', pad_inches=0)
    # maps for each tissue class
    fig = plt.figure()
    a = fig.add_subplot(1, 3, 1)
    img_ax = np.rot90(PVE[..., 89, 0])
    imgplot = plt.imshow(img_ax, cmap="gray")
    a.axis('off')
    a.set_title('CSF')
    a = fig.add_subplot(1, 3, 2)
    img_cor = np.rot90(PVE[:, :, 89, 1])
    imgplot = plt.imshow(img_cor, cmap="gray")
    a.axis('off')
    a.set_title('Gray Matter')
    a = fig.add_subplot(1, 3, 3)
    img_cor = np.rot90(PVE[:, :, 89, 2])
    imgplot = plt.imshow(img_cor, cmap="gray")
    a.axis('off')
    a.set_title('White Matter')
    plt.savefig('probabilities.png', bbox_inches='tight', pad_inches=0)
    plt.show()

bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
gtab = gradient_table(bvals, bvecs)

response, ratio = auto_response(gtab, t1, roi_radius=10, fa_thr=0.7)
msd_model = MultiShellDeconvModel(gtab, response)
