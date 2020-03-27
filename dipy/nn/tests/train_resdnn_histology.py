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

def main():

    # Load The Gradient Directions and form a Sphere
    grad_mat_path = r'/nfs/masi/nathv/dipy_data_2020/monkey_stuff/ng_100.mat'
    grad_mat_path = os.path.normpath(grad_mat_path)
    grad_matrix = loadmat(grad_mat_path)
    grad_dirs = grad_matrix['ng_100']

    # Form the Sphere for sf_to_sh
    hsphere = HemiSphere(xyz=grad_dirs.transpose())
    f_sphere = Sphere(xyz=grad_dirs.transpose())
    sph = Sphere(xyz=np.vstack((hsphere.vertices, -hsphere.vertices)))

    '''
    interactive = True
    ren = window.Renderer()
    ren.SetBackground(1, 1, 1)

    ren.add(actor.point(hsphere.vertices, window.colors.red,
                        point_radius=0.05))

    ren.add(actor.point(f_sphere.vertices, window.colors.green,
                        point_radius=0.05))

    window.record(ren, out_path='initial_vs_updated.png', size=(300, 300))
    if interactive:
        window.show(ren)
    '''

    # Load the Data
    monkey_data_path = r'/nfs/masi/nathv/dipy_data_2020/monkey_stuff/dwmri_sig.pkl'
    monkey_data_path = os.path.normpath(monkey_data_path)

    with open(monkey_data_path, 'rb') as fin:
        dw_signal = pickle.load(fin)

    monkey_fod_path = r'/nfs/masi/nathv/dipy_data_2020/monkey_stuff/fod_sig.pkl'
    monkey_fod_path = os.path.normpath(monkey_fod_path)

    with open(monkey_fod_path, 'rb') as fin2:
        fod_signal = pickle.load(fin2)

    #fod_signal_v2 =

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

    data_2k_1d_slice = np.reshape(data_2k_slice, [hcp_data_dims[0]*hcp_data_dims[1]*1, hcp_data_dims[3]])

    print('Fitting HCP Data to SH ..')
    hcp_sh_coef = sf_to_sh(data_2k_1d_slice,
                          hcp_full_sphere,
                          sh_order=8,
                          basis_type='tournier07',
                          smooth=0.005)


    '''
    hcp_slice = data_2k[70:100, 70:100, 72, :]
    hcp_slice = hcp_slice.reshape([30, 30, 1, 90])
    hcp_spheres = actor.odf_slicer(hcp_slice, sphere=hcp_full_sphere, scale=0.9, norm=False)
    ren = window.Renderer()
    ren.add(hcp_spheres)
    window.show(ren)
    interactive = True
    ren = window.Renderer()
    ren.SetBackground(1, 1, 1)

    ren.add(actor.point(hcp_hemi_sphere.vertices, window.colors.red,
                        point_radius=0.05))

    ren.add(actor.point(hcp_full_sphere.vertices, window.colors.green,
                        point_radius=0.05))

    window.record(ren, out_path='initial_vs_updated.png', size=(300, 300))
    if interactive:
        window.show(ren)
    '''
    print('Done ..')
    # Call Deep learning on SH Coeffs
    dl_model = Histo_ResDNN()
    history_metrics = dl_model.fit(dw_sh_coef, fod_sh_coef, epochs=100)

    # Make HCP Predictions
    hcp_preds = dl_model.predict(hcp_sh_coef)

    hcp_preds_sig = np.matmul(hcp_preds, sh_basis_full_sphere.transpose())
    hcp_sig_full_sphere = np.matmul(hcp_sh_coef, sh_basis_full_sphere.transpose())

    hcp_preds_sig = hcp_preds_sig.reshape([hcp_data_dims[0], hcp_data_dims[1], 1, 200])
    hcp_sig_full_sphere = hcp_sig_full_sphere.reshape([hcp_data_dims[0], hcp_data_dims[1], 1, 200])

    hcp_sig_pkl_path = r'/nfs/masi/nathv/hcp_dw_sig.pkl'
    hcp_fod_pkl_path = r'/nfs/masi/nathv/hcp_fod_sig.pkl'

    f_dw_pi = open(hcp_sig_pkl_path, 'wb')
    f_fod_pi = open(hcp_fod_pkl_path, 'wb')

    pickle.dump(hcp_sig_full_sphere, f_dw_pi)
    pickle.dump(hcp_preds_sig, f_fod_pi)

    #hcp_preds_re = hcp_preds.reshape([hcp])
    print('Debug here')

    '''
    rand_indx = random.sample(range(0,3000),64)
    slice = dw_sig_full_sphere[rand_indx, :]
    slice_re = slice.reshape([8, 8, 1, 200])
    dw_spheres = actor.odf_slicer(slice_re, sphere=sph, scale=0.9, norm=False)
    ren = window.Renderer()
    ren.add(dw_spheres)
    window.show(ren)

    slice_fod = fod_sig_full_sphere[rand_indx, :]
    slice_fod_re = slice_fod.reshape([8, 8, 1, 200])
    fod_spheres = actor.odf_slicer(slice_fod_re, sphere=sph, scale=0.9, norm=False)
    ren = window.Renderer()
    ren.add(fod_spheres)
    window.show(ren)
    '''



    '''
    # interactive = True
    ren = window.Renderer()
    ren.SetBackground(1, 1, 1)

    ren.add(actor.point(hsphere.vertices, window.colors.red,
                        point_radius=0.05))

    #window.record(ren, out_path='initial_vs_updated.png', size=(300, 300))
    #if interactive:
    #    window.show(ren)
    '''




    return None

if __name__ == '__main__':
    main()