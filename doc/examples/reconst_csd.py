import numpy as np

from dipy.reconst.dti import TensorModel
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel

from dipy.data import fetch_stanford_hardi, read_stanford_hardi

fetch_stanford_hardi()
img, gtab = read_stanford_hardi()

data = img.get_data()

print('data.shape (%d, %d, %d, %d)' % data.shape)

affine = img.get_affine()
zooms = img.get_header().get_zooms()[:3]
new_zooms = (2., 2., 2.)

from dipy.align.aniso2iso import resample

data2, affine2 = resample(data, affine, zooms, new_zooms)

print('data2.shape (%d, %d, %d, %d)' % data2.shape)

mask2 = data2[..., 0] > 50
tenmodel = TensorModel(gtab)

ci, cj, ck = np.array(data2.shape[:3]) / 2

w = 10

data3 = data2[ci - w : ci + w, 
              cj - w : cj + w,
              ck - w : ck + w]

tenfit = tenmodel.fit(data3)

from dipy.reconst.dti import fractional_anisotropy

FA = fractional_anisotropy(tenfit.evals)
FA[np.isnan(FA)] = 0

indices = np.where(FA > 0.7)

lambdas = tenfit.evals[indices][:, :2]

S0s = data3[indices][:, 0]

S0 = np.mean(S0s)

l01 = np.mean(lambdas, axis = 0) 

evals = np.array([l01[0], l01[1], l01[1]])

csd_model = ConstrainedSphericalDeconvModel(gtab, (evals, S0))

csd_fit = csd_model.fit(data2[:, :, 30], mask2[:, :, 30])

from dipy.data import get_sphere

sphere = get_sphere('symmetric724')

csd_odf = csd_fit.odf(sphere)

from dipy.viz import fvtk
r = fvtk.ren()
fvtk.add(r, fvtk.sphere_funcs(csd_odf[50:60, 50:60], sphere))
fvtk.show(r)
