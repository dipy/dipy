import nibabel as nib
from dipy.data import fetch_stanford_hardi, read_stanford_hardi
from dipy.segment.mask import hist_mask

#img, gtab = read_stanford_hardi()


files = ['./test_bench/Stanford/b0.nii.gz', 
         './test_bench/3T/GE/b0.nii.gz', 
         './test_bench/3T/Siemens/b0.nii.gz', 
         './test_bench/3T/Philips/b0.nii.gz',
         './test_bench/1.5T/GE/b0.nii.gz', 
         './test_bench/1.5T/Siemens/b0.nii.gz']

for f in files:

    img = nib.load(f)

    b0 = img.get_data().squeeze()
    affine = img.get_affine()

    #b0 = np.mean(data[..., 0:9], axis=-1)

    mask = hist_mask(b0)

    #nib.save(nib.Nifti1Image(b0, affine), f + 'b0.nii.gz')
    nib.save(nib.Nifti1Image(mask.astype('byte'), affine), f + '_mask.nii.gz')



