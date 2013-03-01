import numpy as np
import nibabel as nib

from dipy.reconst.dti import TensorModel
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel

from dipy.data import fetch_beijing_dti, read_beijing_dti

fetch_beijing_dti()
img, gtab = read_beijing_dti()

data = img.get_data()
print('data.shape (%d, %d, %d, %d)' % data.shape)

affine = img.get_affine()
zooms = img.get_header().get_zooms()[:3]
new_zooms = (2., 2., 2.)

from dipy.align.aniso2iso import resample

data2, affine2 = resample(data, affine, zooms, new_zooms)

print('data2.shape (%d, %d, %d, %d)' % data2.shape)

mask = data2[..., 0] > 50
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

