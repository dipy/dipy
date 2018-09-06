from __future__ import division
import numpy as np
import nibabel as nib
from dipy.data import get_data
from dipy.core.gradients import gradient_table
import dipy.reconst.NODDIx as noddix
import matplotlib.pyplot as plt
from dipy.data import get_sphere
from dipy.io import read_bvals_bvecs
from dipy.io.image import save_nifti

sphere = get_sphere('repulsion724')

fname, fscanner = get_data('small_NODDIx_data')
params = np.loadtxt(fscanner)

# getting the gtab, bvals and bvecs
fbval = '/home/shreyasfadnavis/Desktop/dwi/sub-158035_dwi.bvals'
fbvec = '/home/shreyasfadnavis/Desktop/dwi/sub-158035_dwi.bvecs'

bvals, bvecs = read_bvals_bvecs(fbval, fbvec)

for i in range(bvals.shape[0]):
    if bvals[i] > 5:
        bvals[i] = round(bvals[i] / 500.0) * 500.0
    else:
        bvals[i] = bvals[i] - 5
        
bvals = bvals/ 10**6

noddi_data = '/home/shreyasfadnavis/Desktop/dwi/sub-158035_dwi.nii.gz'
img = nib.load(noddi_data)
data = img.get_data()

noddi_mask = \
    '/home/shreyasfadnavis/Desktop/dwi/cc-mask.nii'
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

gtab = gradient_table(bvals, bvecs, big_delta=big_delta,
                      small_delta=small_delta, b0_threshold=0, atol=1e-2)

# Normalizing the data
S0s = data[..., gtab.b0s_mask]
S0avg = np.mean(S0s, axis=3)
datan = data/S0avg[..., None]
datan[np.isnan(datan)] = 0
datan[np.isinf(datan)] = 0

noddix_model = noddix.NODDIxModel(gtab, params, fit_method='MIX')
NODDIx_fit = noddix_model.fit(datan, mask)
affine = img.affine.copy()
save_nifti('params.nii.gz', NODDIx_fit, affine)
