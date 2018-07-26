from __future__ import division
import time as time
import numpy as np
# import nibabel as nib
from dipy.data import get_data
from dipy.reconst.shore import ShoreModel
from dipy.core.gradients import gradient_table
import dipy.reconst.NODDIx as noddix
from scipy.linalg import get_blas_funcs
from dipy.data import get_sphere
# from dipy.io import read_bvals_bvecs
sphere = get_sphere('repulsion724')
gemm = get_blas_funcs("gemm")

fname, fscanner = get_data('small_NODDIx_data')
params = np.loadtxt(fscanner)

# getting the gtab, bvals and bvecs
bvecs = params[:, 0:3]
G = params[:, 3] / 10 ** 6  # gradient strength
big_delta = params[:, 4]
small_delta = params[:, 5]
gamma = 2.675987 * 10 ** 8

bvals = gamma ** 2 * G ** 2 * small_delta ** 2 * (big_delta - small_delta / 3.)
gtab = gradient_table(bvals, bvecs, big_delta=big_delta,
                      small_delta=small_delta,
                      b0_threshold=0, atol=1e-2)

# instantiating the noddixmodel class
noddix_model = noddix.NODDIxModel(gtab, params, fit_method='MIX')

"""
Declare the parameters
"""
volfrac_ic1 = 0.1
volfrac_ec1 = 0.1
theta2 = 0.01745329  # 1 Degree
phi2 = 0.01745329  # 1 Degree

volfrac_ic2 = 0.3
volfrac_ec2 = 0.3
theta1 = 1.57079633  # 90 Degree
phi1 = 0.01745329  # 1 Degree

volfrac_csf = 0.6
OD1 = 0.1
OD2 = 0.1

"""
Lets contruct the signal now
"""
x_f_sig = [volfrac_ic1, volfrac_ic2, volfrac_ec1, volfrac_ec2, volfrac_csf,
           OD1, theta1, phi1, OD2, theta2, phi2]
f = x_f_sig[0:5]

phi = noddix_model.Phi2(x_f_sig)
reconst_signal = np.dot(phi, f)


def show_with_shore(gtab, reconst_signal):
    gtab.bvals = gtab.bvals * 10**6
    shore_model = ShoreModel(gtab)
    shore_fit = shore_model.fit(reconst_signal * 100)
    odf = shore_fit.odf(sphere)
    from dipy.viz import window, fvtk
    ren = window.Renderer()
    ren.add(fvtk.sphere_funcs(odf, sphere))
    window.show(ren)


def sim_voxel_fit(reconst_signal):
    """
    Fitting the Generated Signal
    Creating a List of Errors
    """
    t_start = time.time()
    NODDIx_fit = noddix_model.fit(reconst_signal)
    t_end = time.time()
    time_noddix = t_end - t_start
    volfic1_err = abs(min(abs(x_f_sig[0] - NODDIx_fit[0]), abs(NODDIx_fit[2] -
                          x_f_sig[0])))
    volfic2_err = abs(min(abs(x_f_sig[1] - NODDIx_fit[1]), abs(NODDIx_fit[3] -
                          x_f_sig[1])))
    volfec1_err = abs(min(abs(x_f_sig[2] - NODDIx_fit[2]), abs(NODDIx_fit[0] -
                          x_f_sig[2])))
    volfec2_err = abs(min(abs(x_f_sig[3] - NODDIx_fit[3]), abs(NODDIx_fit[1] -
                          x_f_sig[3])))
    volfiso_err = abs(x_f_sig[4] - NODDIx_fit[4])
    OD1_err = abs(x_f_sig[5] - NODDIx_fit[5])
    theta1_err = abs(min(abs(x_f_sig[6] - NODDIx_fit[6]), abs(NODDIx_fit[6] -
                         x_f_sig[9])))
    phi1_err = abs(min(abs(x_f_sig[7] - NODDIx_fit[7]), abs(NODDIx_fit[7] -
                       x_f_sig[10])))
    OD2_err = abs(x_f_sig[8] - NODDIx_fit[8])
    theta2_err = abs(min(abs(x_f_sig[9] - NODDIx_fit[9]), abs(NODDIx_fit[9] -
                         x_f_sig[6])))
    phi2_err = abs(min(abs(x_f_sig[10] - NODDIx_fit[10]),
                   abs(NODDIx_fit[10] - x_f_sig[7])))

    errors_list = [volfic1_err, volfic2_err, volfec1_err, volfec2_err,
                   volfiso_err, OD1_err, theta1_err, phi1_err, OD2_err,
                   theta2_err, phi2_err]

    time_noddix = t_end - t_start
    print('Time Taken to Fit: ', time_noddix)
    print('Actual Signal: ', x_f_sig)
    print('Estimation Result: ', NODDIx_fit)
    print('Errors: ', errors_list)
    print('Sum of Errors: ', sum(errors_list))

    show_with_shore(gtab, reconst_signal)
    return errors_list, time_noddix
