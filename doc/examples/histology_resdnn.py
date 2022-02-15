import numpy as np
import nibabel as nib
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.reconst.shm import sf_to_sh, sh_to_sf
from dipy.nn.histo_resdnn import HistoResDNN
from dipy.core.sphere import Sphere, HemiSphere
from dipy.data import default_sphere, get_sphere
from fury import window, actor

import tensorflow as tf
from dipy.segment.mask import median_otsu
from enum import Enum


class CamParams(Enum):
    """
    Enum containing camera parameters
    """
    VIEW_POS = 'view_position'
    VIEW_CENTER = 'view_center'
    VIEW_UP = 'up_vector'
    ZOOM_FACTOR = 'zoom_factor'


gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi')

dwi = nib.load(hardi_fname)
data = np.squeeze(dwi.get_fdata())
affine = dwi.affine

bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs)

# Extract B0's and obtain a mean B0
b0_indices = np.where(bvals == 0)[0]
print('B0 indices found are: {}'.format(b0_indices))

mean_b0 = data[..., b0_indices]
mean_b0 = np.mean(mean_b0, axis=-1)
mask = nib.load('mask.nii.gz').get_fdata().astype(int)

# Detect number of b-values and extract a single shell of DW-MRI Data
unique_shells = np.unique(bvals)
print('Number of b-values: {}'.format(np.unique(bvals)))

# Going with the assumption that the first shell would be b0's
# and the second would be Diffusion-Weighted
dw_indices = np.where(bvals == unique_shells[1])[0]
dw_data = data[..., dw_indices]

dw_bvals = bvals[dw_indices]
dw_bvecs = bvecs[dw_indices, :]

# Normalize the DW-MRI Data with the mean b0
norm_dw_data = np.zeros(dw_data.shape)
# norm_dw_data = dw_data / mean_b0
for n in range(len(dw_indices)):
    norm_dw_data[..., n] = np.divide(dw_data[..., n], mean_b0,
                                     where=np.abs(mean_b0) > 0.0000001)

nan_chk = np.isnan(norm_dw_data)
norm_dw_data[nan_chk] = 0

# Fit Spherical Harmonics to DW-MRI Signal
# Form the Sphere for sf_to_sh
h_sphere = HemiSphere(xyz=dw_bvecs)
f_sphere = Sphere(xyz=dw_bvecs)
# sph = Sphere(xyz=np.vstack((h_sphere.vertices, -h_sphere.vertices)))
sphere = get_sphere('repulsion724')

dw_sh_coef = sf_to_sh(norm_dw_data,
                      h_sphere,
                      sh_order=8,
                      basis_type='descoteaux07',
                      smooth=0.0006)

# Build the ResDNN Network & Load Weights
# TODO The Hard-coded path needs a fetcher here
resdnn_model = HistoResDNN()
# print(resdnn_model.model.weights)

# Using a fetcher to grab model weights
fetch_model_weights_path = get_fnames('fetch_resdnn_weights')

# model_weights_path = r'/nfs/masi/nathv/dipy_data_2020/monkey_stuff/monkey_weights_resdnn_mri_2018.h5'
# model_weights_path = os.path.normpath(model_weights_path)

resdnn_model.model.load_weights(fetch_model_weights_path)
# print(resdnn_model.model.weights)
# resdnn_model.load_model_weights(model_weights_path)

ori_shape = dw_sh_coef.shape
flat_dw_sh_coef = dw_sh_coef[mask > 0]
# flat_dw_sh_coef = flat_dw_sh_coef
flat_pred_sh_coef = np.zeros(flat_dw_sh_coef.shape)

count = len(flat_dw_sh_coef) // 1000
for i in range(count+1):
    print(i)
    flat_pred_sh_coef[(i)*1000:(i+1)*1000] = \
        resdnn_model.predict(flat_dw_sh_coef[(i)*1000:(i+1)*1000])

pred_sh_coef = np.zeros(ori_shape)
pred_sh_coef[mask > 0] = flat_pred_sh_coef
nib.save(nib.Nifti1Image(pred_sh_coef, affine), 'fodf_bef.nii.gz')


# Visualization Mean FOD SH
predicted_sf = sh_to_sf(sh=pred_sh_coef,
                        sphere=sphere,
                        basis_type='descoteaux07',
                        sh_order=8)
predicted_sf[predicted_sf < 0] = 0
pred_sh_coef = sf_to_sh(predicted_sf,
                        sphere,
                        sh_order=8,
                        basis_type='descoteaux07',
                        smooth=0.0006)
nib.save(nib.Nifti1Image(pred_sh_coef, affine), 'fodf_after.nii.gz')

fod_spheres = actor.odf_slicer(predicted_sf, sphere=sphere,
                               scale=0.6, norm=True)
min_val, max_val = np.min(predicted_sf[..., 0]), np.max(predicted_sf[..., 0])
background_img = actor.slicer(predicted_sf[..., 0],
                              value_range=(min_val, max_val),
                              interpolation='nearest')

slice_index = ori_shape[1] // 2
fod_spheres.display_extent(0, ori_shape[0],
                           slice_index, slice_index,
                           0, ori_shape[2])
background_img.display_extent(0, ori_shape[0],
                              slice_index, slice_index,
                              0, ori_shape[2])

scene = window.Scene()
scene.add(fod_spheres)
scene.add(background_img)

camera = {}
camera[CamParams.ZOOM_FACTOR] = 2.0 / max(ori_shape)
eye_distance = max(ori_shape)

camera[CamParams.VIEW_POS] = np.array([(ori_shape[0] - 1) / 2.0,
                                       eye_distance,
                                       (ori_shape[2] - 1) / 2.0])
camera[CamParams.VIEW_CENTER] = np.array([(ori_shape[0] - 1) / 2.0,
                                          slice_index,
                                          (ori_shape[2] - 1) / 2.0])
camera[CamParams.VIEW_UP] = np.array([0.0, 0.0, 1.0])
scene.set_camera(position=camera[CamParams.VIEW_POS],
                 focal_point=camera[CamParams.VIEW_CENTER],
                 view_up=camera[CamParams.VIEW_UP])
scene.zoom(camera[CamParams.ZOOM_FACTOR])
window.show(scene)


print('Debug here')
