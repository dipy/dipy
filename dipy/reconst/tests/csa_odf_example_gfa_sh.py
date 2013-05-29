import numpy as np
import nibabel as nib
from dipy.data import fetch_stanford_hardi, read_stanford_hardi, get_sphere
from dipy.reconst.shm import CsaOdfModel, QballModel, normalize_data
from dipy.reconst.odf import gfa, minmax_normalize

fetch_stanford_hardi()
img, gtab = read_stanford_hardi()


data = img.get_data()
affine = img.get_affine()
print('data.shape (%d, %d, %d, %d)' % data.shape)


mask = data[..., 0] > 50

mask_small  = mask[20:50,55:85, 38:40]
data_small  = data[20:50,55:85, 38:40]

csamodel = CsaOdfModel(gtab, 4, smooth=0.006)
csa_fit = csamodel.fit(data_small)

sphere = get_sphere('symmetric724')
csa_odf = csa_fit.odf(sphere)
gfa_csa = gfa(csa_odf)

odfs = csa_odf.clip(0)
gfa_csa_wo_zeros = gfa(odfs)

csa_mm = minmax_normalize(odfs) 
gfa_csa_mm = gfa(csa_mm)

qballmodel = QballModel(gtab, 6, smooth=0.006)
qball_fit = qballmodel.fit(data_small)
qball_odf = qball_fit.odf(sphere)
gfa_qball = gfa(qball_odf)
gfa_qball_mm = gfa(minmax_normalize(qball_odf))


print 'Saving GFAs...'
nib.save(nib.Nifti1Image(gfa_qball.astype('float32'), affine), 'gfa.nii.gz')    
nib.save(nib.Nifti1Image(gfa_qball_mm.astype('float32'), affine), 'gfa_mm.nii.gz')    
nib.save(nib.Nifti1Image(gfa_csa.astype('float32'), affine), 'gfa_csa.nii.gz')    
nib.save(nib.Nifti1Image(gfa_csa_wo_zeros.astype('float32'), affine), 'gfa_csa_wo_neg.nii.gz')    
nib.save(nib.Nifti1Image(gfa_csa_mm.astype('float32'), affine), 'gfa_csa_mm.nii.gz')    



coeff = csa_fit._shm_coef
print 'Printing min-max of the zeroth order SH coeff of the CSA odf'
print np.min(coeff[:,:,:,0]),np.max(coeff[:,:,:,0])
gfa_sh = np.sqrt(1.0 - (csa_fit._shm_coef[:,:,:,0] ** 2 / ( np.sum(np.square(csa_fit._shm_coef), axis=3) ) ) )
gfa_sh[np.isnan(gfa_sh)] = 0
nib.save(nib.Nifti1Image(gfa_sh.astype('float32'), affine), 'gfa_sh_csa.nii.gz')    

coeff_qb = qball_fit._shm_coef
gfa_sh_qb = np.sqrt(1.0 - (coeff_qb[:,:,:,0] ** 2 / ( np.sum(np.square(coeff_qb), axis=3) ) ) )
gfa_sh_qb[np.isnan(gfa_sh_qb)] = 0
nib.save(nib.Nifti1Image(gfa_sh_qb.astype('float32'), affine), 'gfa_sh_qball.nii.gz')    


