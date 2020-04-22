"""
This script is intended for training
of ResDNN Histology Model & save the weights for the model.

The model was re-trained for usage with different basis function ('mrtrix') set
as per the proposed model from the paper:

[1] Nath, V., Schilling, K. G., Parvathaneni, P., Hansen,
C. B., Hainline, A. E., Huo, Y., ... & Stepniewska, I. (2019).
Deep learning reveals untapped information for local white-matter
fiber reconstruction in diffusion-weighted MRI.
Magnetic resonance imaging, 62, 220-227.

[2] Nath, V., Schilling, K. G., Hansen, C. B., Parvathaneni,
P., Hainline, A. E., Bermudez, C., ... & Stępniewska, I. (2019).
Deep learning captures more accurate diffusion fiber orientations
distributions than constrained spherical deconvolution.
arXiv preprint arXiv:1911.07927.

"""

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

def train():

    # Load The Gradient Directions and form a Sphere
    grad_mat_path = r'/nfs/masi/nathv/dipy_data_2020/monkey_stuff/ng_100.mat'
    grad_mat_path = os.path.normpath(grad_mat_path)
    grad_matrix = loadmat(grad_mat_path)
    grad_dirs = grad_matrix['ng_100']

    # Form the Sphere for sf_to_sh
    hsphere = HemiSphere(xyz=grad_dirs.transpose())
    f_sphere = Sphere(xyz=grad_dirs.transpose())
    sph = Sphere(xyz=np.vstack((hsphere.vertices, -hsphere.vertices)))

    # Load the Data
    monkey_data_path = r'/nfs/masi/nathv/dipy_data_2020/monkey_stuff/dwmri_sig.pkl'
    monkey_data_path = os.path.normpath(monkey_data_path)

    with open(monkey_data_path, 'rb') as fin:
        dw_signal = pickle.load(fin)

    monkey_fod_path = r'/nfs/masi/nathv/dipy_data_2020/monkey_stuff/fod_sig.pkl'
    monkey_fod_path = os.path.normpath(monkey_fod_path)

    with open(monkey_fod_path, 'rb') as fin2:
        fod_signal = pickle.load(fin2)

    #monkey_matrix = loadmat(monkey_data_path)
    # We will be operating on 8th order SH, 45 Coefficients are standard
    dw_sh_coef = sf_to_sh(dw_signal,
                          f_sphere,
                          sh_order=8,
                          basis_type='tournier07',
                          smooth=0.005)

    fod_sh_coef = sf_to_sh(fod_signal,
                           f_sphere,
                           sh_order=8,
                           basis_type='tournier07',
                           smooth=0.005)

    sh_basis_full_sphere = just_sh_basis(sphere=sph, sh_order=8, basis_type='tournier07')

    # Measurements on full scale
    #dw_sig_full_sphere = np.matmul(dw_sh_coef, sh_basis_full_sphere.transpose())
    #fod_sig_full_sphere = np.matmul(fod_sh_coef, sh_basis_full_sphere.transpose())

    # Call Deep learning on SH Coeffs
    dl_model = Histo_ResDNN()
    history_metrics = dl_model.fit(dw_sh_coef, fod_sh_coef, epochs=100)

    #### End of Training ####

    # Saving the weights file
    dl_model.model.save_weights('/nfs/masi/nathv/dipy_data_2020/monkey_stuff/monkey_weights_resdnn_mri_2018.h5')

if __name__ == '__main__':
    train()