import nibabel as nib

fdwi = "dMRI_CMRR_dir98_AP_4.nii.gz"
fbval = "dMRI_CMRR_dir98_AP_4.bval"
fbvec = "dMRI_CMRR_dir98_AP_4.bvec"

img = nib.load(fdwi)
data = img.get_data()
