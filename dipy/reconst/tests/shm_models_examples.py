import numpy as np
import nibabel as nib
from dipy.data import fetch_stanford_hardi, read_stanford_hardi, get_sphere
from dipy.reconst.shm import CsaOdfModel, QballModel, OpdtModel, sh_to_sf

fetch_stanford_hardi()
img, gtab = read_stanford_hardi()

vizu = 1

data = img.get_data()
affine = img.get_affine()
print('data.shape (%d, %d, %d, %d)' % data.shape)

#mrconvert dwi.nii -coord 3 0 - | threshold - - | median3D - - | median3D - mask.nii 
#mask = data[..., 0] > 50

order = 4
csamodel = CsaOdfModel(gtab, order, smooth=0.006)
print 'Computing the CSA odf...'
csafit  = csamodel.fit(data) 
coeff   = csafit._shm_coef

# dipy-dev compatible
#GFA = csafit.gfa
#nib.save(nib.Nifti1Image(GFA.astype('float32'), affine), 'gfa_csa.nii.gz')    
nib.save(nib.Nifti1Image(coeff.astype('float32'), affine), 'csa_odf_sh.nii.gz')


sphere = get_sphere('symmetric724')
odfs = sh_to_sf(coeff[20:50,55:85, 38:39], sphere, order)
if vizu :
    from dipy.viz import fvtk
    r = fvtk.ren()
    fvtk.add(r, fvtk.sphere_funcs(odfs, sphere, colormap='jet'))
    fvtk.show(r)
    fvtk.clear(r)


qballmodel = QballModel(gtab, order, smooth=0.006)
print 'Computing the QBALL odf...'
qballfit  = qballmodel.fit(data) 
coeff   = qballfit._shm_coef

#GFA = qballfit.gfa
# dipy 0.6 compatible
GFA = np.sqrt(1.0 - (coeff[:,:,:,0] ** 2 / ( np.sum(np.square(coeff), axis=3) ) ) )
GFA[np.isnan(GFA)] = 0
nib.save(nib.Nifti1Image(GFA.astype('float32'), affine), 'gfa_qball.nii.gz')    
nib.save(nib.Nifti1Image(coeff.astype('float32'), affine), 'qball_odf_sh.nii.gz')


if vizu :
    odfs = sh_to_sf(coeff[20:50,55:85, 38:39], sphere, order)
    fvtk.add(r, fvtk.sphere_funcs(odfs, sphere, colormap='jet'))
    fvtk.show(r)
    fvtk.clear(r)

    #min-max normalize
    odfs = (odfs - np.min(odfs, -1)[..., None]) / (np.max(odfs, -1) - np.min(odfs, -1))[..., None]
    fvtk.add(r, fvtk.sphere_funcs(odfs, sphere, colormap='jet'))
    fvtk.show(r)
    fvtk.clear(r)


opdtmodel = OpdtModel(gtab, order, smooth=0.006)
print 'Computing the Opdt odf...'
opdtfit = opdtmodel.fit(data) 
coeff   = opdtfit._shm_coef
#GFA     = opdtfit.gfa
#nib.save(nib.Nifti1Image(GFA.astype('float32'), affine), 'gfa_opdt.nii.gz')    
nib.save(nib.Nifti1Image(coeff.astype('float32'), affine), 'opdt_odf_sh.nii.gz')

if vizu :
    odfs = sh_to_sf(coeff[20:50,55:85, 38:39], sphere, order)
    fvtk.add(r, fvtk.sphere_funcs(odfs, sphere, colormap='jet'))
    fvtk.show(r)
    fvtk.clear(r)

