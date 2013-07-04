import os
import numpy as np
import nibabel as nib
from dipy.data import fetch_stanford_hardi, read_stanford_hardi
from scipy.ndimage import generate_binary_structure, binary_dilation
from dipy.segment.mask import dwi_bet

# These files can be fetched from this link:
# https://www.dropbox.com/s/rfa6qesrc8oun0d/test_datasets_multi-site_all_companies.zip
files = ['./test_datasets_multi-site_all_companies/Stanford/b0.nii.gz', 
         './test_datasets_multi-site_all_companies/3T/GE/b0.nii.gz', 
         './test_datasets_multi-site_all_companies/3T/Siemens/b0.nii.gz', 
         './test_datasets_multi-site_all_companies/3T/Philips/b0.nii.gz',
         './test_datasets_multi-site_all_companies/1.5T/GE/b0.nii.gz', 
         './test_datasets_multi-site_all_companies/1.5T/Siemens/b0.nii.gz']

for f in files :
    print(f)
    img = nib.load(f)
    data = img.get_data()
    fname = os.path.splitext(os.path.splitext(f)[0])[0]

    b0_mask, mask = dwi_bet(data, 4, 4, autocrop=False)
    mask_img = nib.Nifti1Image(mask.astype(np.float32), img.get_affine())
    b0_img = nib.Nifti1Image(b0_mask.astype(np.float32), img.get_affine())
    nib.save(mask_img, fname+'_binary_mask.nii.gz')
    nib.save(b0_img, fname+'_mask.nii.gz')
    
    b0_mask_crop, mask_crop = dwi_bet(data, 4, 4, autocrop=True)
    mask_img_crop = nib.Nifti1Image(mask_crop.astype(np.float32), img.get_affine())
    b0_img_crop = nib.Nifti1Image(b0_mask_crop.astype(np.float32), img.get_affine())
    nib.save(mask_img_crop, fname+'_binary_mask_crop.nii.gz')
    nib.save(b0_img_crop, fname+'_mask_crop.nii.gz')
