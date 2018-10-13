import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from dipy.segment.tissue import TissueClassifierHMRF

msd_data = '/home/shreyasfadnavis/Desktop/dwi/output/mri/wmparc.mgz'
img = nib.load(msd_data)
data = img.get_data()

# msd_mask = '/home/shreyasfadnavis/Desktop/dwi/cc-mask.nii'
#
# img_mask = nib.load(msd_mask)
# mask = img_mask.get_data()

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
initial_segmentation, final_segmentation, PVE = hmrf.classify(data, nclass,
                                                              beta)
# segmentation using the HMRF
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
