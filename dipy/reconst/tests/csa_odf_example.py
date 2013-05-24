import numpy as np
import nibabel as nib
from dipy.data import fetch_stanford_hardi, read_stanford_hardi, get_sphere
from dipy.reconst.shm import CsaOdfModel, normalize_data
from dipy.reconst.odf import peaks_from_model, minmax_normalize

fetch_stanford_hardi()
img, gtab = read_stanford_hardi()


data = img.get_data()
affine = img.get_affine()
print('data.shape (%d, %d, %d, %d)' % data.shape)


mask = data[..., 0] > 50

mask_small  = mask[20:50,55:85, 38:40]
data_small  = data[20:50,55:85, 38:40]

csamodel = CsaOdfModel(gtab, 4, smooth=0.006)
#csa_fit = csa_model.fit(data)

sphere = get_sphere('symmetric724')
csapeaks = peaks_from_model(model=csamodel,
                            data=data_small,
                            sphere=sphere,
                            relative_peak_threshold=.8,
                            min_separation_angle=45,
                            mask=mask_small,
                            return_odf=False,
                            normalize_peaks=True)

GFA = csapeaks.gfa

print('GFA.shape (%d, %d, %d)' % GFA.shape)
nib.save(nib.Nifti1Image(GFA.astype('float32'), affine), 'gfa.nii.gz')



from dipy.data import get_sphere
sphere = get_sphere('symmetric724')

from dipy.viz import fvtk
r = fvtk.ren()

odfs = csamodel.fit(data_small[:,:,1:2]).odf(sphere)
fvtk.add(r, fvtk.sphere_funcs(odfs, sphere, colormap='jet', norm=False))
fvtk.show(r)
fvtk.clear(r)

odfs = odfs.clip(min=0)
r = fvtk.ren()
fvtk.add(r, fvtk.sphere_funcs(odfs, sphere, colormap='jet', norm=False))
fvtk.show(r)

# min-max normalization
csa_mm = minmax_normalize(odfs)
r = fvtk.ren()
fvtk.add(r, fvtk.sphere_funcs(csa_mm, sphere, colormap='jet', norm=False))
fvtk.show(r)
fvtk.clear(r)


