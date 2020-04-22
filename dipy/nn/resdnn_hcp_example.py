import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.ndimage import rotate

from dipy.reconst.shm import sf_to_sh, just_sh_basis
from dipy.nn.resdnn_histology import Histo_ResDNN
from dipy.core.sphere import Sphere, HemiSphere
from dipy.data import default_sphere
from dipy.viz import window, actor


# Build the ResDNN Network & Load Weights
# TODO The Hard-coded path needs a fetcher here
resdnn_model = Histo_ResDNN()
#print(resdnn_model.model.weights)
model_weights_path = r'/nfs/masi/nathv/dipy_data_2020/monkey_stuff/monkey_weights_resdnn_mri_2018.h5'
model_weights_path = os.path.normpath(model_weights_path)
resdnn_model.model.load_weights(model_weights_path)


# Load the HCP Data with bvals and bvecs
hcp_data_path = r'/nfs/HCP/retest/data/122317/T1w/Diffusion'
hcp_data_path = os.path.normpath(hcp_data_path)

hcp_nifti_obj = nib.load(os.path.join(hcp_data_path, 'data.nii.gz'))
hcp_nifti = hcp_nifti_obj.get_fdata()

hcp_mask_obj = nib.load(os.path.join(hcp_data_path, 'nodif_brain_mask.nii.gz'))
hcp_mask = hcp_mask_obj.get_fdata()

hcp_bvals = np.loadtxt(os.path.join(hcp_data_path, 'bvals'))
hcp_bvecs = np.loadtxt(os.path.join(hcp_data_path, 'bvecs'))

# Extract indices from bvalues where 1000 and 0 are present
idxs_2k = [i for i in range(len(hcp_bvals)) if (hcp_bvals[i] > 1900 and hcp_bvals[i] < 2100)]
idxs_b0 = [i for i in range(len(hcp_bvals)) if (hcp_bvals[i] < 50)]

b0_s = hcp_nifti[:, :, :, idxs_b0]
mean_b0 = np.nanmean(b0_s, axis=3)
data_2k = hcp_nifti[:, :, :, idxs_2k]

# Normalize the 2K data by the mean b0
for each_vol in range(len(idxs_2k)):
    data_2k[:, :, :, each_vol] = np.divide(data_2k[:, :, :, each_vol], mean_b0)

nan_chk = np.isnan(data_2k)
data_2k[nan_chk] = 0
bvecs_2k = hcp_bvecs[:, idxs_2k]

# HCP Data fit to Spherical Harmonics
hcp_hemi_sphere = HemiSphere(xyz=bvecs_2k.transpose())
hcp_full_sphere = Sphere(xyz=bvecs_2k.transpose())

hcp_data_dims = data_2k.shape
data_2k_slice = data_2k[:, :, 65, :]

data_2k_1d_slice = np.reshape(data_2k_slice, [hcp_data_dims[0] * hcp_data_dims[1] * 1, hcp_data_dims[3]])

print('Fitting HCP Data to SH ..')
hcp_sh_coef = sf_to_sh(data_2k_1d_slice,
                       hcp_full_sphere,
                       sh_order=8,
                       basis_type='tournier07',
                       smooth=0.005)

hcp_slice = data_2k[70:100, 70:100, 72, :]
hcp_slice = hcp_slice.reshape([30, 30, 1, 90])
hcp_spheres = actor.odf_slicer(hcp_slice, sphere=hcp_full_sphere, scale=0.9, norm=False)
ren = window.Renderer()
ren.add(hcp_spheres)
window.show(ren)

# Make HCP Predictions
hcp_preds = resdnn_model.predict(hcp_sh_coef)

sh_basis_full_sphere = just_sh_basis(sphere=hcp_full_sphere, sh_order=8, basis_type='tournier07')

hcp_preds_sig = np.matmul(hcp_preds, sh_basis_full_sphere.transpose())
hcp_sig_full_sphere = np.matmul(hcp_sh_coef, sh_basis_full_sphere.transpose())

hcp_preds_sig = hcp_preds_sig.reshape([hcp_data_dims[0], hcp_data_dims[1], 1, 90])
hcp_sig_full_sphere = hcp_sig_full_sphere.reshape([hcp_data_dims[0], hcp_data_dims[1], 1, 90])

dw_spheres = actor.odf_slicer(hcp_sig_full_sphere, sphere=hcp_full_sphere, scale=0.6, norm=True)
ren = window.Renderer()
ren.add(dw_spheres)
window.show(ren)

# slice_fod = fod_sig_full_sphere[rand_indx, :]
# slice_fod_re = slice_fod.reshape([8, 8, 1, 200])
fod_spheres = actor.odf_slicer(hcp_preds_sig, sphere=hcp_full_sphere, scale=0.6, norm=True)
ren = window.Renderer()
ren.add(fod_spheres)
window.show(ren)

