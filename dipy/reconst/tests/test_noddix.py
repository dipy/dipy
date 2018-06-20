from __future__ import division
import time as time
import numpy as np
import nibabel as nib
from dipy.core.gradients import gradient_table
import dipy.reconst.NODDIx as noddix
from scipy.linalg import get_blas_funcs
from dipy.data import get_sphere
sphere = get_sphere('repulsion724')
gemm = get_blas_funcs("gemm")

fname = 'C:/Users/Shreyas/dipy/dipy/data/files/small_NODDIx_HCP.hdr'
fscanner = 'C:/Users/Shreyas/dipy/dipy/data/files/HCP_scheme.txt'
params = np.loadtxt(fscanner)
img = nib.load(fname)

# getting the gtab, bvals and bvecs
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


# instantiating the noddixmodel class
noddix_model = noddix.NODDIxModel(gtab, params, fit_method='MIX')

"""
Declare the parameters
"""
volfrac_ic1 = 0.39
volfrac_ic2 = 0.39
volfrac_ec1 = 0.1
volfrac_ec2 = 0.1
volfrac_csf = 0.02
OD1 = 0.1
OD2 = 0.1
theta1 = 0.017453293
phi1 = 0.017453293
theta2 = 1.57079633
phi2 = 0.017453293

"""
Lets contruct the signal now
"""
x_fe_sig = [volfrac_ic1, volfrac_ic2, volfrac_ec1, volfrac_ec2, volfrac_csf,
            OD1, theta1, phi1, OD2, theta2, phi2]
fe = x_fe_sig[0:5]

phi = noddix_model.Phi2(x_fe_sig)
reconst_signal = np.dot(phi, fe)


def show_with_shore(gtab, reconst_signal):
    from dipy.reconst.shore import ShoreModel
    gtab.bvals = gtab.bvals * 10**6
    shore_model = ShoreModel(gtab)
    shore_fit = shore_model.fit(reconst_signal * 100)
    odf = shore_fit.odf(sphere)
    from dipy.viz import window, fvtk
    ren = window.Renderer()
    ren.add(fvtk.sphere_funcs(odf, sphere))
    window.show(ren)


"""

Fitting the Generated Signal
"""
t_start = time.time()
NODDIx_fit = noddix_model.fit(reconst_signal)
t_end = time.time()

"""
Creating a List of Errors
"""
errors_list = [abs(min(abs(x_fe_sig[0] - NODDIx_fit[0]), abs(NODDIx_fit[2] -
               x_fe_sig[0]))), abs(min(abs(x_fe_sig[1] - NODDIx_fit[1]),
               abs(NODDIx_fit[3] - x_fe_sig[1]))), abs(min(abs(x_fe_sig[2] -
               NODDIx_fit[2]), abs(NODDIx_fit[0] - x_fe_sig[2]))),
               abs(min(abs(x_fe_sig[3] - NODDIx_fit[3]), abs(NODDIx_fit[1] -
               x_fe_sig[3]))), abs(x_fe_sig[4] - NODDIx_fit[4]),
               abs(x_fe_sig[5] - NODDIx_fit[5]), abs(min(abs(x_fe_sig[6] -
               NODDIx_fit[6]), abs(NODDIx_fit[6] - x_fe_sig[9]))),
               abs(min(abs(x_fe_sig[7] - NODDIx_fit[7]), abs(NODDIx_fit[7] -
               x_fe_sig[10]))),abs(x_fe_sig[8] - NODDIx_fit[8]),
               abs(min(abs(x_fe_sig[9] - NODDIx_fit[9]), abs(NODDIx_fit[9] -
               x_fe_sig[6]))), abs(min(abs(x_fe_sig[10] - NODDIx_fit[10]),
               abs(NODDIx_fit[10] - x_fe_sig[7])))]


time_noddix = t_end - t_start
print('Time Taken to Fit: ', time_noddix)

if sum(errors_list) <= 5:
    print('Test Passed')
else:
    print('The Signal is not Fitting Properly..')
