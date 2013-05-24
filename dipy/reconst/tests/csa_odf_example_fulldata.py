import numpy as np
import nibabel as nib
from dipy.data import fetch_stanford_hardi, read_stanford_hardi, get_sphere
from dipy.reconst.shm import CsaOdfModel, QballModel, OpdtModel, normalize_data
from dipy.reconst.odf import gfa, odf_remove_negative_values, minmax_normalize

fetch_stanford_hardi()
img, gtab = read_stanford_hardi()


data = img.get_data()
affine = img.get_affine()
print('data.shape (%d, %d, %d, %d)' % data.shape)

#mrconvert dwi.nii -coord 3 0 - | threshold - - | median3D - - | median3D - mask.nii 
#mask = data[..., 0] > 50

csamodel = CsaOdfModel(gtab, 4, smooth=0.006)
print 'Computing the CSA odf...'
coeff = csamodel._get_shm_coef(data)
print coeff.shape

print 'Computing GFA...'
print np.min(coeff[:,:,:,0]),np.max(coeff[:,:,:,0])
gfa_sh = np.sqrt(1.0 - (coeff[:,:,:,0] ** 2 / ( np.sum(np.square(coeff), axis=3) ) ) )
gfa_sh[np.isnan(gfa_sh)] = 0

print 'Saving nifti...'
nib.save(nib.Nifti1Image(gfa_sh.astype('float32'), affine), 'gfa_full_brain.nii.gz')    
nib.save(nib.Nifti1Image(coeff.astype('float32'), affine), 'csa_odf_sh.nii.gz')


qballmodel = QballModel(gtab, 4, smooth=0.006)
print 'Computing the QBALL odf...'
coeff = qballmodel._get_shm_coef(data)
print coeff.shape

print 'Computing GFA...'
print np.min(coeff[:,:,:,0]),np.max(coeff[:,:,:,0])
gfa_sh = np.sqrt(1.0 - (coeff[:,:,:,0] ** 2 / ( np.sum(np.square(coeff), axis=3) ) ) )
gfa_sh[np.isnan(gfa_sh)] = 0

print 'Saving nifti...'
nib.save(nib.Nifti1Image(gfa_sh.astype('float32'), affine), 'gfa_qball_full_brain.nii.gz')    
nib.save(nib.Nifti1Image(coeff.astype('float32'), affine), 'qball_odf_sh.nii.gz')


opdtmodel = OpdtModel(gtab, 4, smooth=0.006)
print 'Computing the OPDT odf...'
coeff = opdtmodel._get_shm_coef(data)
print coeff.shape

print 'Computing GFA...'
print np.min(coeff[:,:,:,0]),np.max(coeff[:,:,:,0])
gfa_sh = np.sqrt(1.0 - (coeff[:,:,:,0] ** 2 / ( np.sum(np.square(coeff), axis=3) ) ) )
gfa_sh[np.isnan(gfa_sh)] = 0

print 'Saving nifti...'
nib.save(nib.Nifti1Image(gfa_sh.astype('float32'), affine), 'gfa_opdt_full_brain.nii.gz')    
nib.save(nib.Nifti1Image(coeff.astype('float32'), affine), 'opdt_odf_sh.nii.gz')










