from __future__ import division
import numpy as np
import nibabel as nib
from dipy.data import get_data
from dipy.core.gradients import gradient_table
import dipy.reconst.NODDIx as noddix
from scipy.linalg import get_blas_funcs
import matplotlib.pyplot as plt
from dipy.data import get_sphere
from dipy.io import read_bvals_bvecs
sphere = get_sphere('repulsion724')
gemm = get_blas_funcs("gemm")

big_delta = 17.5 * pow(10, -3)
small_delta = 37.8 * pow(10, -3)

noddi_data = 'C:/Users/Shreyas/Desktop/35/03_NODDI_66DIR_B2400.nii.gz'
img = nib.load(noddi_data)
data = img.get_data()

axial_middle = data.shape[2] // 2
plt.figure('Showing the datasets')
plt.subplot(1, 2, 1).set_axis_off()
plt.imshow(data[:, :, axial_middle, 0].T, cmap='gray', origin='lower')
plt.subplot(1, 2, 2).set_axis_off()
plt.imshow(data[:, :, axial_middle, 10].T, cmap='gray', origin='lower')
plt.show()
plt.savefig('data.png', bbox_inches='tight')

fbval = 'C:/Users/Shreyas/Desktop/35/03_NODDI_66DIR_B2400.bvals'
fbvec = 'C:/Users/Shreyas/Desktop/35/03_NODDI_66DIR_B2400.bvecs'

bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
gtab = gradient_table(bvals, bvecs, big_delta=big_delta,
                      small_delta=small_delta, b0_threshold=0, atol=1e-2)

noddix_model = noddix.NODDIxModel(gtab, fit_method='MIX')
NODDIx_fit = noddix_model.fit(data)
