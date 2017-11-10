# -*- coding: utf-8 -*-
"""
Created on Wed Nov 08 09:37:32 2017

@author: mafzalid
"""
from time import time
import numpy as np
import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.data import get_data
# import dipy.reconst.activeax as activeax
import dipy.reconst.activeax_in_vivo_3compartments as activeax_in_vivo_3compartments
from scipy.linalg import get_blas_funcs
gemm = get_blas_funcs("gemm")
from dipy.segment.mask import median_otsu
# t1 = time()

#fname = get_data('mask_CC')
#img = nib.load(fname)
#mask = img.get_data()

fname, fscanner = get_data('ActiveAx_in_vivo1')
params = np.loadtxt(fscanner)
img = nib.load(fname)
data1 = img.get_data()

fname = get_data('ActiveAx_in_vivo2')
img = nib.load(fname)
data2 = img.get_data()

fname = get_data('ActiveAx_in_vivo3')
img = nib.load(fname)
data3 = img.get_data()

fname = get_data('ActiveAx_in_vivo4')
img = nib.load(fname)
data4 = img.get_data()

data = np.zeros((data1.shape[0], data1.shape[1], data1.shape[2], data1.shape[3]*4))

data[:, :, :, 0: data1.shape[3]] = data1
data[:, :, :, data1.shape[3]: data1.shape[3]*2] = data2
data[:, :, :, 2*data1.shape[3]: data1.shape[3]*3] = data3
data[:, :, :, 3*data1.shape[3]: data1.shape[3]*4] = data4

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
#am = np.array([1.84118307861360])


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
#    b_zero_all1 = y[C]
#    sigma = np.std(b_zero_all1)
#    y[C] = 1
    return y

#maskdata, mask = median_otsu(data, 3, 1, False,
#                             vol_idx=range(10, 50), dilate=2)
fname = get_data('mask_CC_in_vivo')
img = nib.load(fname)
mask = img.get_data()

Y = data[63,:,:,:]

fit_method = 'MIX'
activeax_model = activeax_in_vivo_3compartments.ActiveAxModel(gtab, params, fit_method=fit_method)
#activeax_model = activeax.ActiveAxModel(gtab, fit_method=fit_method)
activeax_fit = np.zeros((Y.shape[0], Y.shape[1], 1, 6))
mask[0:127,:] = mask[128:0:-1,:]

t1 = time()
for i in range(Y.shape[0]):
    for j in range(Y.shape[1]):
        for k in range(1):
            if mask[i, j, k] > 0:
                signal = np.array(Y[i, j])
                signal = norm_meas_HCP(signal, bvals)
                signal = np.float64(signal)
                signal[signal > 1] = 1
#               signal_n = add_noise(signal, snr=20, noise_type='rician')
                activeax_fit[i, j, k, :] = activeax_model.fit(signal)
                print(i)
t2 = time()
fast_time = t2 - t1
print(fast_time)
#print(activeax_fit[0, 0])
affine = img.affine.copy()
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 0], affine), 'f1_in_vivo_3comp_activeax_CC.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 1], affine), 'f2_in_vivo_3comp_activeax_CC.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 2], affine), 'f3_in_vivo_3comp_activeax_CC.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 3], affine), 'theta_in_vivo_3comp_activeax_CC.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 4], affine), 'phi_in_vivo_3comp_activeax_CC.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 5], affine), 'R_in_vivo_3comp_activeax_CC.nii.gz')
#nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 6], affine), 'f4_in_vivo_3comp_activeax_CC.nii.gz')
#t2 = time()
#fast_time = t2 - t1
#print(fast_time)
#plt.imshow(data[:, :, 0], cmap='autumn', vmin=0, vmax=8); colorbar()

#import matplotlib.pyplot as plt
#plt.plot(signal)
#plt.ylabel('some numbers')
#plt.show()

#%matplotlib inline
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#imgplot = plt.imshow(Y[:,:,0])
#imgplot = plt.imshow(mask[:,:,0])
