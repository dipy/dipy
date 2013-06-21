import numpy as np
import nibabel as nib
from dipy.data import fetch_stanford_hardi, read_stanford_hardi

fetch_stanford_hardi()
img, gtab = read_stanford_hardi()
data = img.get_data()

b0 = data[:,:,:, 0]
print('b0.shape (%d, %d, %d)' % b0.shape)

from dipy.segment.dwi_brain_mask import dwi_bet_filter
print('Computing brain mask from b0 image')
b0_mask, mask = dwi_bet_filter(b0, 4, 4, autocrop=False)

mask_img = nib.Nifti1Image(mask.astype(np.float32), img.get_affine())
nib.save(mask_img, 'mask.nii.gz')
b0_mask_img = nib.Nifti1Image(b0_mask.astype(np.float32), img.get_affine())
nib.save(b0_mask_img, 'b0_mask.nii.gz')

print('Computing brain mask & crop data')
dwi_mask, mask2 = dwi_bet_filter(data, 4, 4, autocrop=True)
dwi_mask_img = nib.Nifti1Image(dwi_mask.astype(np.float32), img.get_affine())
nib.save(dwi_mask_img, 'dwi_mask_crop.nii.gz')

#mean_volume = np.mean(data[...,:], 3)
mean_volume = np.mean(data[...,0:9], axis=-1)
print mean_volume.shape

from scipy.ndimage import generate_binary_structure, binary_dilation
from dipy.segment.mask import hist_mask

epi_mask = hist_mask(mean_volume)
epi_mask_img = nib.Nifti1Image(epi_mask.astype(np.byte), img.get_affine())
nib.save(epi_mask_img, 'epi_mask.nii.gz')
nib.save(nib.Nifti1Image(data[...,0], img.get_affine()), 'b0.nii.gz')
