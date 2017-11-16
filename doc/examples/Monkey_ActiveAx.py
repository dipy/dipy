# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 10:57:58 2017

@author: mafzalid
"""
# -*- coding: utf-8 -*-
import os
from time import time
import numpy as np
import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.data import get_data
# import dipy.reconst.activeax as activeax
import dipy.reconst.activeax_fast as activeax_fast
from scipy.linalg import get_blas_funcs
from dipy.segment.mask import median_otsu
from dipy.data.fetcher import _make_fetcher
from os.path import join as pjoin

gemm = get_blas_funcs("gemm")

# t1 = time()

#fname = get_data('mask_CC')
#img = nib.load(fname)
#mask = img.get_data()

# Set a user-writeable file-system location to put files:
if 'DIPY_HOME' in os.environ:
    dipy_home = os.environ['DIPY_HOME']
else:
    dipy_home = pjoin(os.path.expanduser('~'), '.dipy')


fetch_Monkey_ActiveAx = _make_fetcher(
    "Monkey_ActiveAx",
    pjoin(dipy_home, 'Monkey_ActiveAx'),
    "https://osf.io/",
    ['he4aj/download', '9avhm/download', 'fkpm3/download', 'udc7v/download'],
    ['DRCMR_MAP_ActiveAx4CCfit_exvivo_E2503_Mbrain1_PGSE_b1925_3b0_N90.nii.gz',
     'DRCMR_MAP_ActiveAx4CCfit_exvivo_E2503_Mbrain1_PGSE_b1931_3b0_N90.nii.gz',
     'DRCMR_MAP_ActiveAx4CCfit_exvivo_E2503_Mbrain1_PGSE_b3091_3b0_N90.nii.gz',
     'DRCMR_MAP_ActiveAx4CCfit_exvivo_E2503_Mbrain1_PGSE_b13183_3b0_N90.nii.gz'],
    doc="Download a four shells ex-vivo ActiveAx dataset with 90 gradient directions",
    msg="For the complete datasets please visit : \
         https://osf.io/fkpm3/",
    data_size="49.8MB")


def read_Monkey_ActiveAx():
    """ Load Stanford HARDI dataset
    Returns
    -------
    img : obj,
        Nifti1Image
    gtab : obj,
        GradientTable
    """
    files, folder = fetch_Monkey_ActiveAx()
    fraw1 = pjoin(folder, 'DRCMR_MAP_ActiveAx4CCfit_exvivo_E2503_Mbrain1_PGSE_b13183_3b0_N90.nii.gz')
    fraw2 = pjoin(folder, 'DRCMR_MAP_ActiveAx4CCfit_exvivo_E2503_Mbrain1_PGSE_b13183_3b0_N90.nii.gz')
    fraw3 = pjoin(folder, 'DRCMR_MAP_ActiveAx4CCfit_exvivo_E2503_Mbrain1_PGSE_b13183_3b0_N90.nii.gz')
    fraw4 = pjoin(folder, 'DRCMR_MAP_ActiveAx4CCfit_exvivo_E2503_Mbrain1_PGSE_b13183_3b0_N90.nii.gz')
    img1 = nib.load(fraw1)
    img2 = nib.load(fraw2)
    img3 = nib.load(fraw3)
    img4 = nib.load(fraw4)
    return img1, img2, img3, img4

data1, data2, data3, data4 = read_Monkey_ActiveAx()

#fname, fscanner = get_data('ActiveAx_monkey1')
#params = np.loadtxt(fscanner)
#img = nib.load(fname)
#data1 = img.get_data()
#
#fname, fscanner = get_data('ActiveAx_monkey2')
#img = nib.load(fname)
#data2 = img.get_data()
#
#fname, fscanner = get_data('ActiveAx_monkey3')
#img = nib.load(fname)
#data3 = img.get_data()
#
#fname, fscanner = get_data('ActiveAx_monkey4')
#img = nib.load(fname)
#data4 = img.get_data()

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

maskdata, mask = median_otsu(data, 3, 1, False,
                             vol_idx=range(10, 50), dilate=2)

fit_method = 'MIX'
activeax_model = activeax_fast.ActiveAxModel(gtab, params, fit_method=fit_method)
#activeax_model = activeax.ActiveAxModel(gtab, fit_method=fit_method)
activeax_fit = np.zeros((data.shape[0], data.shape[1], data.shape[2], 7))

t1 = time()
for i in range(47, 63):  # range(data.shape[0]):
    for j in range(109, 177):  # range(data.shape[1]):
        for k in range(1, 2):  # in range(1):
            if mask[i, j, k] > 0:
                signal = np.array(maskdata[i, j, k])
                signal = norm_meas_Aax(signal)
                signal = np.float64(signal)
#               signal_n = add_noise(signal, snr=20, noise_type='rician')
                activeax_fit[i, j, k, :] = activeax_model.fit(signal)
                print(i)
t2 = time()
fast_time = t2 - t1
print(fast_time)
#print(activeax_fit[0, 0])
affine = img.affine.copy()
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 0], affine), 'f1_Monkey_activeax_CC_slice2_accurate.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 1], affine), 'f2_Monkey_activeax_CC_slice2_accurate.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 2], affine), 'f3_Monkey_activeax_CC_slice2_accurate.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 3], affine), 'theta_Monkey_activeax_CC_slice2_accurate.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 4], affine), 'phi_Monkey_activeax_CC_slice2_accurate.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 5], affine), 'R_Monkey_activeax_CC_slice2_accurate.nii.gz')
nib.save(nib.Nifti1Image(activeax_fit[:, :, :, 6], affine), 'f4_Monkey_activeax_CC_slice2_accurate.nii.gz')
#t2 = time()
#fast_time = t2 - t1
#print(fast_time)
#plt.imshow(data[:, :, 0], cmap='autumn', vmin=0, vmax=8); colorbar()
