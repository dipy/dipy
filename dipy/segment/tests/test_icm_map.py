import numpy as np
import nibabel as nib

from dipy.segment.mask import applymask
from dipy.core.ndindex import ndindex
from dipy.segment.rois_stats import seg_stats
from dipy.segment.energy_mrf import total_energy

dname = '/Users/jvillalo/Documents/GSoC_2015/Code/Data/'
# dname = '/home/eleftherios/Dropbox/DIPY_GSoC_2015/'

img = nib.load(dname + '3587_BL_T1_to_MNI_Linear_6p.nii.gz')
dataimg = img.get_data()

mask_image = nib.load(dname + '3587_mask.nii.gz')
datamask = mask_image.get_data()

masked_img = applymask(dataimg, datamask)
print('masked_img.shape (%d, %d, %d)' % masked_img.shape)
shape = masked_img.shape[:3]

seg = nib.load(dname + '3587_BL_T1_to_MNI_Linear_6p_seg.nii.gz')
seg_init = seg.get_data()
seg_init_masked = applymask(seg_init, datamask)

print("computing the statistics of the ROIs (CSF, GM, WM)")
mu, std = seg_stats(masked_img, seg_init_masked, 3)

print(mu)
print(std)

nclass = 3
L = range(1, nclass + 1)
niter = 1
beta = 0.05
totalE = np.zeros(nclass)
N = 0

segmented = np.zeros(dataimg.shape)

while True:

    mu, std = seg_stats(masked_img, seg_init, 3)
    var = std ** 2

    for idx in ndindex(shape):
        # print(idx)
        if not masked_img[idx]:
            continue
        for l in range(0, nclass):

            totalE[l] = total_energy(masked_img, seg_init_masked,
                                     mu, var, idx, l, beta)

        segmented[idx] = L[np.argmin(totalE)]

    N = N + 1
    if N == niter:
        break


# print('Show results')
figure()
imshow(seg_init_masked[:, :, 95])
figure()
imshow(segmented[:, :, 95])
