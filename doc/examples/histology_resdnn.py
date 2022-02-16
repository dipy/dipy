import os

from dipy.core.gradients import gradient_table
from dipy.data import get_fnames, get_sphere
from dipy.io.gradients import read_bvals_bvecs
from dipy.nn.histo_resdnn import HistoResDNN
from dipy.reconst.shm import sh_to_sf
from dipy.segment.mask import median_otsu
from fury import window, actor
import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
import tensorflow as tf


# Disable oneDNN warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Allow TF to use as much memory as needed
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Fetch DWI and GTAB
hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi')
dwi_img = nib.load(hardi_fname)
data = np.squeeze(dwi_img.get_fdata())
bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs)

# Masking to accelerate processing
b0_indices = np.where(gtab.bvals == 0)[0]
mean_b0 = data[..., b0_indices]
mean_b0 = np.mean(mean_b0, axis=-1)
_, mask = median_otsu(mean_b0)
# Picking only the biggest 'blob'
mask_labeled, _ = ndi.label(mask)
unique, count = np.unique(mask_labeled, return_counts=True)
val = unique[np.argmax(count[1:])+1]
mask[mask_labeled != val] = 0

nib.save(nib.Nifti1Image(mask.astype(np.uint8), dwi_img.affine),
         'mask.nii.gz')

# Build the ResDNN Network & load Weights
resdnn_model = HistoResDNN(verbose=True)
fetch_model_weights_path = get_fnames('fetch_resdnn_weights')
resdnn_model.load_model_weights(fetch_model_weights_path)
predicted_sh = resdnn_model.fit(data, gtab, mask=mask)

nib.save(nib.Nifti1Image(predicted_sh, dwi_img.affine), 'predicted_sh.nii.gz')

# Visualize the prediction onto the S0 amplitude
print('----------- VISUALIZATION -----------')
sphere = get_sphere('repulsion724')
predicted_sf = sh_to_sf(sh=predicted_sh, sphere=sphere,
                        basis_type=resdnn_model.sh_basis,
                        sh_order=resdnn_model.sh_order)
fod_spheres = actor.odf_slicer(predicted_sf, sphere=sphere,
                               scale=0.6, norm=True, mask=mask)
mean_sh = np.mean(predicted_sh, axis=-1)
background_img = actor.slicer(mean_sh, opacity=0.5,
                              interpolation='nearest')

# Select the mid coronal slice for slicer
ori_shape = mask.shape[0:3]
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

# Adjust the camera
camera = {}
camera['zoom_factor'] = 1.0 / max(ori_shape)
camera['view_position'] = np.array([(ori_shape[0] - 1) / 2.0,
                                    max(ori_shape),
                                    (ori_shape[2] - 1) / 2.0])
camera['view_center'] = np.array([(ori_shape[0] - 1) / 2.0,
                                  slice_index,
                                  (ori_shape[2] - 1) / 2.0])
camera['up_vector'] = np.array([0.0, 0.0, 1.0])

scene.set_camera(position=camera['view_position'],
                 focal_point=camera['view_center'],
                 view_up=camera['up_vector'])
scene.zoom(camera['zoom_factor'])
window.show(scene)
