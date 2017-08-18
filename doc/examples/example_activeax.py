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
import dipy.reconst.mix as mix
import dipy.reconst.activeax as activeax
from scipy.optimize import differential_evolution
from scipy.linalg import get_blas_funcs
gemm = get_blas_funcs("gemm")

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
signal = np.array(data[0, 0, 0])

signal_param = mix.make_signal_param(signal, bvals, bvecs, G, small_delta,
                                     big_delta)


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


signal = norm_meas_Aax(signal)
signal = np.float64(signal)
# jit decorator tells Numba to compile this function.
# The argument types will be inferred by Numba when function is called.

#t1 = time()
#res_one1 = mix_fast.dif_evol(signal, bvals, bvecs, G, small_delta, big_delta)
#t2 = time()
#res_one = mix.dif_evol(signal, bvals, bvecs, G, small_delta, big_delta)
#t3 = time()
#
#fast_time = t2 - t1
#slow_time = t3 - t2
#speedup = slow_time/fast_time
#print(speedup)
# print(mix_fast.cnt_stochastic)

t1 = time()
fit_method = 'MIX'
activeax_model = activeax.ActiveAxModel(gtab, fit_method=fit_method)
activeax_fit = activeax_model.fit(signal)
t2 = time()
#fit_method = 'MIX'
#activeax_model = activeax_fast.ActiveAxModel(gtab, fit_method=fit_method)
#activeax_fit2 = activeax_model.fit(data[0, 0, 0])
#t3 = time()

#t1 = time()
#bounds = [(0.01, np.pi), (0.01, np.pi), (0.1, 11), (0.1, 0.8)]
#res_one = differential_evolution(activeax_model.stoc_search_cost, bounds,
#                                 args=(signal,))
#t2 = time()

#res_one = mix.dif_evol(signal, bvals, bvecs, G, small_delta, big_delta)
#t3 = time()

fast_time = t2 - t1
print(fast_time)
#print(res_one.x)
print(activeax.overall_duration)
#slow_time = t3 - t2
#speedup = slow_time/fast_time
#print(speedup)
#x = res_one.x
#phi = activeax.Phi(x, gtab)
#fe = mix.estimate_f(np.array(signal), phi)

#yhat_zeppelin1 = np.zeros(small_delta.shape[0])
#S2_new(x_fe, bvals,  bvecs, yhat_zeppelin1)