import numpy as np
import nibabel as nib
from dipy.data import fetch_stanford_hardi, read_stanford_hardi, get_sphere
from dipy.reconst.shm import CsaOdfModel, normalize_data
from dipy.reconst.odf import peaks_from_model, gfa

fetch_stanford_hardi()
img, gtab = read_stanford_hardi()

data = img.get_data()
affine = img.get_affine()
print('data.shape (%d, %d, %d, %d)' % data.shape)

mask = data[..., 0] > 50
mask_small  = mask[20:50,55:85, 38:40]
data_small  = data[20:50,55:85, 38:40]

csamodel = CsaOdfModel(gtab, 4, smooth=0.006)
sphere = get_sphere('symmetric724')
csapeaks = peaks_from_model(model=csamodel,
                            data=data_small,
                            sphere=sphere,
                            relative_peak_threshold=.8,
                            min_separation_angle=45,
                            mask=mask_small,
                            return_odf=False,
                            normalize_peaks=True)


csa_fit = csamodel.fit(data_small)
odfs = csa_fit.odf(sphere)
from dipy.viz import fvtk
r = fvtk.ren()
fvtk.add(r, fvtk.sphere_funcs(odfs, sphere, colormap='jet'))
fvtk.show(r)
fvtk.clear(r)

# Three ways to get the GFA
GFA = csapeaks.gfa
GFA_sh = csa_fit.gfa
GFA_odf = gfa(odfs)
coeff = csa_fit._shm_coef

nib.save(nib.Nifti1Image(GFA.astype('float32'), affine), 'gfa_small.nii.gz')    
nib.save(nib.Nifti1Image(GFA_sh.astype('float32'), affine), 'gfa_sh_small.nii.gz')
nib.save(nib.Nifti1Image(GFA_sh.astype('float32'), affine), 'gfa_odf_small.nii.gz')    
nib.save(nib.Nifti1Image(coeff.astype('float32'), affine), 'csa_sh_small.nii.gz')    

# If you want the full brain SH coefs or GFA
csa_fit = csamodel.fit(data)
GFA_sh = csa_fit.gfa
csa_sh_coeffs = csa_fit._shm_coef
nib.save(nib.Nifti1Image(GFA_sh.astype('float32'), affine), 'gfa_fullbrain.nii.gz')
nib.save(nib.Nifti1Image(csa_sh_coeffs.astype('float32'), affine), 'csa_sh_fullbrain.nii.gz')    


