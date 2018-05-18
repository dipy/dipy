# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 12:42:55 2017

@author: mafzalid
"""

# -*- coding: utf-8 -*-
from time import time
import numpy as np
import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.data import get_data
# import dipy.reconst.activeax as activeax
import dipy.reconst.NODDIx as NODDIx
# from dipy.segment.mask import median_otsu
from scipy.linalg import get_blas_funcs
gemm = get_blas_funcs("gemm")
# t1 = time()

# fname = get_data('mask_CC')
# img = nib.load(fname)
# mask = img.get_data()

fname, fscanner = get_data('NODDIx_example')
params = np.loadtxt(fscanner)
img = nib.load(fname)
data = img.get_data()

affine = img.affine
bvecs = params[:, 0:3]
G = params[:, 3] / 10 ** 6  # gradient strength
big_delta = params[:, 4]
small_delta = params[:, 5]
gamma = 2.675987 * 10 ** 8
bvals = gamma ** 2 * G ** 2 * small_delta ** 2 * (big_delta - small_delta / 3.)
gtab = gradient_table(bvals, bvecs, big_delta=big_delta,
                      small_delta=small_delta,
                      b0_threshold=0, atol=1e-2)
# signal_param = mix.make_signal_param(signal, bvals, bvecs, G, small_delta,
#                                     big_delta)
# am = np.array([1.84118307861360])


#def norm_meas_Aax(signal):
#
#    """
#    normalizing the signal based on the b0 values of each shell
#    """
#    y = signal
#    y01 = (y[0] + y[1] + y[2]) / 3
#    y02 = (y[93] + y[94] + y[95]) / 3
#    y03 = (y[186] + y[187] + y[188]) / 3
#    y04 = (y[279] + y[280] + y[281]) / 3
#    y1 = y[0:93] / y01
#    y2 = y[93:186] / y02
#    y3 = y[186:279] / y03
#    y4 = y[279:372] / y04
#    f = np.concatenate((y1, y2, y3, y4))
#    return f


def norm_meas_HCP(ydatam, b):

    """
    calculates std of the b0 measurements and normalizes all ydatam
    """
    b1 = np.where(b > 1e-5)
    b2 = range(b.shape[0])
    C = np.setdiff1d(b2, b1)
    b_zero_all = ydatam[C]
#    %sigma = std(b_zero_all, 1)
    b_zero_norm = sum(b_zero_all) / C.shape[0]
    y = ydatam / b_zero_norm
    b_zero_all1 = y[C]
#    sigma = np.std(b_zero_all1)
    return y

fit_method = 'MIX'
NODDIx_model = NODDIx.NODDIxModel(gtab, params,
                                  fit_method=fit_method)
NODDIx_fit = np.zeros((data.shape[0], data.shape[1], data.shape[2], 11))

t1 = time()
for i in range(1,2):  # range(data.shape[0]):
    for j in range(1):  # range(data.shape[1]):
        for k in range(data.shape[2]):  # in range(1):
#            if mask[i, j, k] > 0:
            signal = np.array(data[i, j, k])
            signal = norm_meas_HCP(signal, bvals)
#            signal = np.float64(signal)
#               signal_n = add_noise(signal, snr=20, noise_type='rician')
            NODDIx_fit[i, j, k, :] = NODDIx_model.fit(signal)
            print(i)
t2 = time()
fast_time = t2 - t1
print(fast_time)
# print(activeax_fit[0, 0])
affine = img.affine.copy()
nib.save(nib.Nifti1Image(NODDIx_fit[:, :, :, 0], affine),
         'f11_HCP_NODDIx.nii.gz')
nib.save(nib.Nifti1Image(NODDIx_fit[:, :, :, 1], affine),
         'f21_HCP_NODDIx.nii.gz')
nib.save(nib.Nifti1Image(NODDIx_fit[:, :, :, 2], affine),
         'f12_HCP_NODDIx.nii.gz')
nib.save(nib.Nifti1Image(NODDIx_fit[:, :, :, 3], affine),
         'f22_HCP_NODDIx.nii.gz')
nib.save(nib.Nifti1Image(NODDIx_fit[:, :, :, 4], affine),
         'f3_HCP_NODDIx.nii.gz')
nib.save(nib.Nifti1Image(NODDIx_fit[:, :, :, 5], affine),
         'OD1_HCP_NODDIx.nii.gz')
nib.save(nib.Nifti1Image(NODDIx_fit[:, :, :, 6], affine),
         'theta1_HCP_NODDIx.nii.gz')
nib.save(nib.Nifti1Image(NODDIx_fit[:, :, :, 7], affine),
         'phi1_HCP_NODDIx.nii.gz')
nib.save(nib.Nifti1Image(NODDIx_fit[:, :, :, 8], affine),
         'OD2_HCP_NODDIx.nii.gz')
nib.save(nib.Nifti1Image(NODDIx_fit[:, :, :, 9], affine),
         'theta2_HCP_NODDIx.nii.gz')
nib.save(nib.Nifti1Image(NODDIx_fit[:, :, :, 10], affine),
         'phi2_HCP_NODDIx.nii.gz')

# t2 = time()
# fast_time = t2 - t1
# print(fast_time)
# plt.imshow(data[:, :, 0], cmap='autumn', vmin=0, vmax=8); colorbar()
