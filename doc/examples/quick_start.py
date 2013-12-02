"""
=========================
Getting started with Dipy
=========================
"""


from dipy.data import get_data
fimg, fbval, fbvec = get_data('small_101D')

import nibabel as nib
img = nib.load(fimg)
data = img.get_data()

from dipy.io import read_bvals_bvecs
bvals, bvecs = read_bvals_bvecs(fbval, fbvec)

from dipy.core.gradients import gradient_table
gtab = gradient_table(bvals, bvecs)

from dipy.reconst.dti import TensorModel
ten = TensorModel(gtab)
tenfit = ten.fit(data)

from dipy.reconst.dti import fractional_anisotropy
fa = fractional_anisotropy(tenfit.evals)

from dipy.reconst.dti import color_fa
cfa = color_fa(fa, tenfit.evecs)


