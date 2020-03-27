import os
import numpy as np
import pickle
import random
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.ndimage import rotate

from dipy.reconst.shm import sf_to_sh, just_sh_basis
from dipy.nn.resdnn_histology import Histo_ResDNN
from dipy.core.sphere import Sphere, HemiSphere
from dipy.data import default_sphere
from dipy.viz import window, actor

# Load The Gradient Directions and form a Sphere
grad_mat_path = r'/nfs/masi/nathv/dipy_data_2020/monkey_stuff/ng_100.mat'
grad_mat_path = os.path.normpath(grad_mat_path)
grad_matrix = loadmat(grad_mat_path)
grad_dirs = grad_matrix['ng_100']

# Form the Sphere for sf_to_sh
hsphere = HemiSphere(xyz=grad_dirs.transpose())
f_sphere = Sphere(xyz=grad_dirs.transpose())
sph = Sphere(xyz=np.vstack((hsphere.vertices, -hsphere.vertices)))

hcp_sig_pkl_path = r'/nfs/masi/nathv/hcp_dw_sig.pkl'
hcp_fod_pkl_path = r'/nfs/masi/nathv/hcp_fod_sig.pkl'

with open(hcp_sig_pkl_path, 'rb') as fin:
    dw_signal = pickle.load(fin)

with open(hcp_fod_pkl_path, 'rb') as fin2:
    fod_signal = pickle.load(fin2)

#sh_slice = dw_signal[:, :, 0, 0]
#plt.imshow(rotate(np.squeeze(dw_signal[:, :, 0, 1]), 90))
#plt.show()

dw_slice = dw_signal[44:60, 120:136, :, :]
fod_slice = fod_signal[44:60, 120:136, :, :]

dw_spheres = actor.odf_slicer(dw_signal, sphere=sph, scale=0.9, norm=True)
ren = window.Renderer()
ren.add(dw_spheres)
window.show(ren)

#slice = dw_sig_full_sphere[rand_indx, :]
#slice_re = slice.reshape([8, 8, 1, 200])
dw_spheres = actor.odf_slicer(dw_slice, sphere=sph, scale=0.9, norm=False)
ren = window.Renderer()
ren.add(dw_spheres)
window.show(ren)

fod_spheres = actor.odf_slicer(fod_slice, sphere=sph, scale=0.9, norm=False)
ren = window.Renderer()
ren.add(fod_spheres)
window.show(ren)
'''
#slice_fod = fod_sig_full_sphere[rand_indx, :]
slice_fod_re = slice_fod.reshape([8, 8, 1, 200])
fod_spheres = actor.odf_slicer(slice_fod_re, sphere=sph, scale=0.9, norm=False)
ren = window.Renderer()
ren.add(fod_spheres)
window.show(ren)
'''