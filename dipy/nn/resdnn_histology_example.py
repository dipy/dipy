import os
import numpy as np
import matplotlib.pyplot as plt
from dipy.io.image import load_nifti, save_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.reconst.shm import sf_to_sh, just_sh_basis, sh_to_sf
from dipy.nn.resdnn_histology import Histo_ResDNN
from dipy.core.sphere import Sphere, HemiSphere
from dipy.data import default_sphere
from dipy.viz import window, actor


hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi')

data, affine = load_nifti(hardi_fname)

bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs)

# Extract B0's and obtain a mean B0
b0_indices = np.where(bvals == 0)
print('B0 indices found are: {}'.format(b0_indices))

mean_b0 = data[:, :, :, b0_indices]
mean_b0 = np.squeeze(np.nanmean(mean_b0, axis=4))
print(mean_b0.shape)

# Detect number of b-values and extract a single shell of DW-MRI Data
unique_shells = np.unique(bvals)
print('Number of b-values: {}'.format(np.unique(bvals)))

# Going with the assumption that the first shell would be b0's
# and the second would be Diffusion-Weighted
dw_indices = np.where(bvals == unique_shells[1])

dw_data = data[:, :, :, dw_indices]
dw_data = np.squeeze(dw_data)

dw_bvals = bvals[dw_indices]
dw_bvecs = bvecs[dw_indices, :]

# Normalize the DW-MRI Data with the mean b0
norm_dw_data = np.zeros(dw_data.shape)
for n in range(len(dw_indices)):
    norm_dw_data[:, :, :, n] = np.divide(np.squeeze(dw_data[:, :, :, n]), mean_b0)

nan_chk = np.isnan(norm_dw_data)
norm_dw_data[nan_chk] = 0

# Fit Spherical Harmonics to DW-MRI Signal
# Form the Sphere for sf_to_sh
hsphere = HemiSphere(xyz=dw_bvecs)
f_sphere = Sphere(xyz=dw_bvecs)
sph = Sphere(xyz=np.vstack((hsphere.vertices, -hsphere.vertices)))

dw_sh_coef = sf_to_sh(norm_dw_data,
                      f_sphere,
                      sh_order=8,
                      basis_type='tournier07',
                      smooth=0.0005)

# Build the ResDNN Network & Load Weights
# TODO The Hard-coded path needs a fetcher here
resdnn_model = Histo_ResDNN()
#print(resdnn_model.model.weights)
model_weights_path = r'/nfs/masi/nathv/dipy_data_2020/monkey_stuff/monkey_weights_resdnn_mri_2018.h5'
model_weights_path = os.path.normpath(model_weights_path)
resdnn_model.model.load_weights(model_weights_path)
#print(resdnn_model.model.weights)
#resdnn_model.load_model_weights(model_weights_path)

# Extract a Small Chunk of Data and Run Predictions
small_data = dw_sh_coef[13:43, 44:74, 34:35, :]
#small_data = dw_sh_coef
small_data_dims = small_data.shape
# Flattening dimensions for running predictions
small_data_re = np.reshape(small_data, [small_data_dims[0]*small_data_dims[1]*small_data_dims[2], 45])
small_data_preds = resdnn_model.predict(small_data_re)
# Reshaping to original shape
small_data_orig_shape = np.reshape(small_data_preds, [small_data_dims[0], small_data_dims[1], small_data_dims[2], 45])

# Visualization Mean FOD SH
'''
slice_num = 35
plt.figure(1)
plt.subplot(2, 3, 1)
plt.imshow(np.squeeze(small_data_orig_shape[:, :, slice_num, 0]))
plt.subplot(2, 3, 2)
plt.imshow(np.squeeze(small_data_orig_shape[:, :, slice_num, 1]))
plt.subplot(2, 3, 3)
plt.imshow(np.squeeze(small_data_orig_shape[:, :, slice_num, 2]))
plt.subplot(2, 3, 4)
plt.imshow(np.squeeze(small_data_orig_shape[:, :, slice_num, 3]))
plt.subplot(2, 3, 5)
plt.imshow(np.squeeze(small_data_orig_shape[:, :, slice_num, 4]))
plt.subplot(2, 3, 6)
plt.imshow(np.squeeze(small_data_orig_shape[:, :, slice_num, 5]))
plt.show()
'''

plt.imshow(np.squeeze(small_data_orig_shape[:, :, 0, 0]))
plt.colorbar()
plt.show()


small_data_preds_sf = sh_to_sf(sh=small_data_orig_shape,
                               sphere=sph,
                               sh_order=8
                               )

fod_spheres = actor.odf_slicer(small_data_preds_sf, sphere=sph, scale=0.6, norm=True)
ren = window.Renderer()
ren.add(fod_spheres)
window.show(ren)


print('Debug here')
