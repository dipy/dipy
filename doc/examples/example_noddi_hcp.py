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

fname, fscanner = get_data('small_NODDIx_data')
params = np.loadtxt(fscanner)

# getting the gtab, bvals and bvecs
fbval = 'C:/Users/Shreyas/Desktop/dwi_data_HCP/dwi/sub-158035_dwi.bvals'
fbvec = 'C:/Users/Shreyas/Desktop/dwi_data_HCP/dwi/sub-158035_dwi.bvecs'

bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
bvals = bvals

noddi_data = 'C:/Users/Shreyas/Desktop/dwi_data_HCP/dwi/sub-158035_dwi.nii.gz'
img = nib.load(noddi_data)
data = img.get_data()

noddi_mask = \
    'C:/Users/Shreyas/Desktop/dwi_data_HCP/dwi/sub-158035_dwi_brainmask.nii.gz'
img_mask = nib.load(noddi_mask)
mask = img_mask.get_data()


def show_data(data):
    axial_middle = data.shape[2] // 2
    plt.figure('Showing the datasets')
    plt.subplot(1, 2, 1).set_axis_off()
    plt.imshow(data[:, :, axial_middle, 0].T, cmap='seismic', origin='lower')
    plt.subplot(1, 2, 2).set_axis_off()
    plt.imshow(data[:, :, axial_middle, 10].T, cmap='seismic', origin='lower')
    plt.show()
    plt.savefig('data.png', bbox_inches='tight')


big_delta = params[:, 4]
small_delta = params[:, 5]
gamma = 2.675987 * 10 ** 8
G = params[:, 3] / 10 ** 6
bvals = gamma ** 2 * G ** 2 * small_delta ** 2 * (big_delta - small_delta / 3.)
gtab = gradient_table(bvals, bvecs, big_delta=big_delta,
                      small_delta=small_delta, mutation=(0, 1.05),
                      b0_threshold=0, atol=1e-2)

noddix_model = noddix.NODDIxModel(gtab, params, fit_method='MIX')
NODDIx_fit = noddix_model.fit(data, mask)
