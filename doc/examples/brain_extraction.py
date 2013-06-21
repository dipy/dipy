import nibabel as nib
from dipy.data import fetch_stanford_hardi, read_stanford_hardi
from dipy.segment.mask import hist_mask

img, gtab = read_stanford_hardi()

data = img.get_data()
affine = img.get_affine()

b0 = np.mean(data[..., 0:9], axis=-1)

mask = hist_mask(b0)

nib.save(nib.Nifti1Image(b0, affine), 'b0.nii.gz')
nib.save(nib.Nifti1Image(mask.astype('byte'), affine), 'mask.nii.gz')



