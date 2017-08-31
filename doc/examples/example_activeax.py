# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 16:14:07 2017

@author: Maryam
"""


# -*- coding: utf-8 -*-
from time import time
import numpy as np
import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.data import get_data
import dipy.reconst.activeax as activeax
from dipy.sims.phantom import add_noise
from scipy.linalg import get_blas_funcs
gemm = get_blas_funcs("gemm")
#t1 = time()
fname, fscanner = get_data('ActiveAx_synth_2d')
params = np.loadtxt(fscanner)
img = nib.load(fname)
data = img.get_data()
affine = img.affine
bvecs = params[:, 0:3]
G = params[:, 3] / 10 ** 6  # gradient strength
big_delta = params[:, 4]
small_delta = params[:, 5]
te = params[:, 6]
gamma = 2.675987 * 10 ** 8
bvals = gamma ** 2 * G ** 2 * small_delta ** 2 * (big_delta - small_delta / 3.)
bvals = bvals
gtab = gradient_table(bvals, bvecs, big_delta=big_delta,
                      small_delta=small_delta,
                      b0_threshold=0, atol=1e-2)
# signal_param = mix.make_signal_param(signal, bvals, bvecs, G, small_delta,
#                                     big_delta)


def norm_meas_Aax(signal):

    """
    normalizing the signal based on the b0 values of each shell
    """
    y = signal
    y01 = (y[0] + y[1] + y[2]) / 3
    y02 = (y[93] + y[94] + y[95]) / 3
    y03 = (y[186] + y[187] + y[188]) / 3
    y04 = (y[279] + y[280] + y[281]) / 3
    y1 = y[0:93] / y01
    y2 = y[93:186] / y02
    y3 = y[186:279] / y03
    y4 = y[279:372] / y04
    f = np.concatenate((y1, y2, y3, y4))
    return f

fit_method = 'MIX'
activeax_model = activeax.ActiveAxModel(gtab, fit_method=fit_method)
activeax_fit = np.zeros((10, 10, 7))
t1 = time()
for i in range(10):
    for j in range(10):
        signal = np.array(data[i, j, 0])
        signal = norm_meas_Aax(signal)
        signal = np.float64(signal)
        signal_n = add_noise(signal, snr=1000, noise_type='rician')
        activeax_fit[i, j, :] = activeax_model.fit(signal_n)
t2 = time()
fast_time = t2 - t1
print(fast_time)
print(activeax_fit[0,0])
#affine = img.affine.copy()
#nib.save(nib.Nifti1Image(activeax_fit[:, :, 0], affine), 'f1_activeax_snr1000.nii.gz')
#nib.save(nib.Nifti1Image(activeax_fit[:, :, 1], affine), 'f2_activeax_snr1000.nii.gz')
#nib.save(nib.Nifti1Image(activeax_fit[:, :, 2], affine), 'f3_activeax_snr1000.nii.gz')
#nib.save(nib.Nifti1Image(activeax_fit[:, :, 3], affine), 'theta_activeax_snr1000.nii.gz')
#nib.save(nib.Nifti1Image(activeax_fit[:, :, 4], affine), 'phi_activeax_snr1000.nii.gz')
#nib.save(nib.Nifti1Image(activeax_fit[:, :, 5], affine), 'R_activeax_snr1000.nii.gz')
#nib.save(nib.Nifti1Image(activeax_fit[:, :, 6], affine), 'f4_activeax_snr1000.nii.gz')
#t2 = time()
#fast_time = t2 - t1
#print(fast_time)

#import dipy.reconst.mix as mix
#from dipy.reconst.recspeed import S2, S2_new, S1, Phi, Phi2, activeax_cost_one
#
#x = np.array([0.5, 0.5, 10, 0.8])
#phi = mix.activax_exvivo_model(x, bvals, bvecs, G, small_delta, big_delta)

#phi_dot = np.dot(phi.T, phi)
#
#phi_dot1 = np.zeros((N, M))
#for i in range(N):
#        for k in range(N):
#            for j in range(M):
#                phi_dot1[i, k] += phi[j, i] * phi[j, k]
#
#phi_inv = np.zeros((4, 4))
#func_inv(phi_dot, phi_inv)
#
#phi_mp = np.dot(phi_inv, phi.T)
#phi_sig = np.dot(phi_mp, signal)
#yhat = np.dot(phi, phi_sig)
#
#M = phi.shape[0]
#N = phi.shape[1]
#
#np.dot((signal - yhat).T, signal - yhat)
#norm_diff = 0
#for i in range(M):
#    norm_diff += (signal[i] - yhat[i]) * (signal[i] - yhat[i])
#
#activeax_cost_one(phi, signal)
#activeax_model.activeax_cost_one_slow(phi, signal)

