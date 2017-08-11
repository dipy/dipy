# -*- coding: utf-8 -*-
"""
Created on Fri Aug 04 16:42:33 2017

@author: Maryam
"""

from dipy.reconst.base import ReconstModel
import numpy as np
from scipy.optimize import least_squares
import dipy.reconst.mix as mix
import dipy.reconst.mix_fast as mix_fast
from scipy.optimize import differential_evolution
# import cvxpy as cvx
from dipy.data import get_data
import nibabel as nib
from numba import jit
from scipy.linalg import get_blas_funcs
gemm = get_blas_funcs("gemm")

gamma = 2.675987 * 10 ** 8
D_intra = 0.6 * 10 ** 3
fname, fscanner = get_data('ActiveAx_synth_2d')
params = np.loadtxt(fscanner)
img = nib.load(fname)
data = img.get_data()
affine = img.affine
# bvecs = params[:, 0:3]
G = params[:, 3] / 10 ** 6  # gradient strength
big_delta = params[:, 4]
small_delta = params[:, 5]
D_iso = 2 * 10 ** 3


@jit(nopython=True, nogil=True, cache=True)
def func_bvec(bvecs, n):
    M = bvecs.shape[0]
    g_per = np.zeros((M))
    for i in range(M):
        g_per[i] = bvecs[i, 0]**2 + bvecs[i, 1]**2 + bvecs[i, 2]**2 - \
                   (bvecs[i, 0]*n[0] + bvecs[i, 1]*n[1] + bvecs[i, 2]*n[2])**2

    return g_per


@jit(nopython=True, nogil=True, cache=True)
def func_mul(x, am2, small_delta, big_delta):
    M = am2.shape[0]
    N = small_delta.shape[0]
    summ = np.zeros((N, M))
    for i in range(M):
        am = am2[i]
        D_intra_am = D_intra * am
        num = (2 * D_intra * am * small_delta) - 2 + \
              (2 * np.exp(-(D_intra_am * small_delta))) + \
              (2 * np.exp(-(D_intra_am * big_delta))) - \
              (np.exp(-(D_intra_am * (big_delta - small_delta)))) - \
              (np.exp(-(D_intra_am * (big_delta + small_delta))))

        denom = (D_intra ** 2) * (am ** 3) * ((x[2]) ** 2 * am - 1)
        summ[:, i] = num / denom
        return summ


def x_to_xs(x):
    x1 = x[0:3]
    x2 = np.zeros((3))
    x2[0:2] = x[0:2]
    x2[2] = x[3]
    return x1, x2


def xs_to_x():
    pass


#@profile
def S1(x1, gtab):

    big_delta = gtab.big_delta
    small_delta = gtab.small_delta

    sinT = np.sin(x1[0])
    cosT = np.cos(x1[0])
    sinP = np.sin(x1[1])
    cosP = np.cos(x1[1])
    n = np.array([cosP * sinT, sinP * sinT, cosT])
    bvecs = gtab.bvecs
    # Cylinder
    L = gtab.bvals * D_intra
    L1 = L * np.dot(gtab.bvecs, n) ** 2
    am = np.array([1.84118307861360, 5.33144196877749,
                   8.53631578218074, 11.7060038949077,
                   14.8635881488839, 18.0155278304879,
                   21.1643671187891, 24.3113254834588,
                   27.4570501848623, 30.6019229722078,
                   33.7461812269726, 36.8899866873805,
                   40.0334439409610, 43.1766274212415,
                   46.3195966792621, 49.4623908440429,
                   52.6050411092602, 55.7475709551533,
                   58.8900018651876, 62.0323477967829,
                   65.1746202084584, 68.3168306640438,
                   71.4589869258787, 74.6010956133729,
                   77.7431620631416, 80.8851921057280,
                   84.0271895462953, 87.1691575709855,
                   90.3110993488875, 93.4530179063458,
                   96.5949155953313, 99.7367932203820,
                   102.878653768715, 106.020498619541,
                   109.162329055405, 112.304145672561,
                   115.445950418834, 118.587744574512,
                   121.729527118091, 124.871300497614,
                   128.013065217171, 131.154821965250,
                   134.296570328107, 137.438311926144,
                   140.580047659913, 143.721775748727,
                   146.863498476739, 150.005215971725,
                   153.146928691331, 156.288635801966,
                   159.430338769213, 162.572038308643,
                   165.713732347338, 168.855423073845,
                   171.997111729391, 175.138794734935,
                   178.280475036977, 181.422152668422,
                   184.563828222242, 187.705499575101])

    am2 = (am / x1[2]) ** 2

#    summ = np.zeros((len(gtab.bvals), len(am)))

    summ = func_mul(x1, am2, small_delta, big_delta)
#    for i in range(len(am)):
#        num = (2 * D_intra * am2[i] * small_delta) - 2 + \
#              (2 * np.exp(-(D_intra * am2[i] * small_delta))) + \
#              (2 * np.exp(-(D_intra * am2[i] * big_delta))) - \
#              (np.exp(-(D_intra * am2[i] * (big_delta - small_delta)))) - \
#              (np.exp(-(D_intra * am2[i] * (big_delta + small_delta))))
#
#        denom = (D_intra ** 2) * (am2[i] ** 3) * ((x[2]) ** 2 * am2[i] - 1)
#        summ[:, i] = num / denom

    summ_rows = np.sum(summ, axis=1)
#        g_per = np.zeros(gtab.bvals.shape)

    g_per = func_bvec(bvecs, n)

#        for i in range(len(bvecs)):
#            g_per[i] = np.dot(bvecs[i, :], bvecs[i, :]) - \
#                       np.dot(bvecs[i, :], n) ** 2

    L2 = 2 * (g_per * gamma ** 2) * summ_rows * G ** 2

    yhat_cylinder = L1 + L2

    return yhat_cylinder


#@profile
def S2(x2, gtab):
    x2_0 = x2[0]
    x2_1 = x2[1]
    x2_2 = x2[2]
    sinT = np.sin(x2_0)
    cosT = np.cos(x2_0)
    sinP = np.sin(x2_1)
    cosP = np.cos(x2_1)
    n = np.array([cosP * sinT, sinP * sinT, cosT])
    # zeppelin
    yhat_zeppelin = gtab.bvals * ((D_intra - (D_intra * (1 - x2_2))) *
                                  (np.dot(gtab.bvecs, n) ** 2) + (D_intra *
                                                                  (1 - x2_2)))
    return yhat_zeppelin


def S2_new(x_fe, gtab):
    x, fe = x_fe_to_x_and_fe(x_fe)
    fe0 = fe[0]
    x_0 = x[0]
    x_1 = x[1]
    sinT = np.sin(x_0)
    cosT = np.cos(x_0)
    sinP = np.sin(x_1)
    cosP = np.cos(x_1)
    n = np.array([cosP * sinT, sinP * sinT, cosT])
    # zeppelin
    v = fe0/(fe0 + fe[1])
    yhat_zeppelin = gtab.bvals * ((D_intra - (D_intra * (1 - v))) *
                                  (np.dot(gtab.bvecs, n) ** 2) + (D_intra *
                                                                  (1 - v)))
    return yhat_zeppelin


def S3(gtab):
    # ball
    yhat_ball = D_iso * gtab.bvals
    return yhat_ball


def S4(gtab):
    # dot
    yhat_dot = np.zeros(gtab.bvals.shape)
    return yhat_dot


def Phi(x, gtab):
    x1, x2 = x_to_xs(x)
    yhat_zeppelin = S2(x2, gtab)
    yhat_cylinder = S1(x1, gtab)
    phi = np.vstack([yhat_cylinder, yhat_zeppelin, S3(gtab),
                     S4(gtab)]).T
    phi = np.ascontiguousarray(phi)
    return np.exp(-phi)


def Phi2(x_fe, gtab):
    x, fe = x_fe_to_x_and_fe(x_fe)
    x1 = x[0:3]
    yhat_zeppelin = S2_new(x_fe, gtab)
    yhat_cylinder = S1(x1, gtab)
    phi = np.vstack([yhat_cylinder, yhat_zeppelin, S3(gtab),
                     S4(gtab)]).T
    phi = np.ascontiguousarray(phi)
    return np.exp(-phi)


def bounds(x):
    bound = ([0.01, 0.01,  0.01, 0.01, 0.01, 0.1, 0.01], [0.9,  0.9,  0.9,
             np.pi, np.pi, 11, 0.9])
    return bound


def bounds_x_fe(x_fe):

    pass


def x_fe_to_x_and_fe(x_fe):
    fe = np.zeros((1, 4))
    fe = np.squeeze(fe)
    fe[0:3] = x_fe[0:3]
    fe[3] = x_fe[6]
    x = x_fe[3:6]
    return x, fe


def x_and_fe_to_x_fe(x, fe):
    x_fe = np.zeros([7])
    fe = np.squeeze(fe)
    x_fe[:3] = fe[:3]
    x_fe[3:6] = x[:3]
    x_fe[6] = fe[3]
    return x_fe


def estimate_signal(x_fe):
    x, fe = x_fe_to_x_and_fe(x_fe)
    x1, x2 = x_to_xs(x)
    S = fe[0] * S1(x1) + fe[1] * S2(x2) + fe[2] * S3() + fe[3] * S4()
    return S


def stoc_search_cost(x, signal, gtab):

    """
    Aax_exvivo_nlin

    Cost function for genetic algorithm

    Parameters
    ----------
    x : array
        x.shape = 4x1
        x(0) theta (radian)
        x(1) phi (radian)
        x(2) R (micrometers)
        x(3) v=f1/(f1+f2) (0.1 - 0.8)

    bvals
    bvecs
    G: gradient strength
    small_delta
    big_delta
    gamma: gyromagnetic ratio (2.675987 * 10 ** 8 )
    D_intra= intrinsic free diffusivity (0.6 * 10 ** 3 mircometer^2/sec)
    D_iso= isotropic diffusivity, (2 * 10 ** 3 mircometer^2/sec)

    Returns
    -------
    (signal -  S)^T(signal -  S)

    Notes
    --------
    cost function for genetic algorithm:

    .. math::

        (signal -  S)^T(signal -  S)
    """
    phi = Phi(x, gtab)
    error_one = mix.activeax_cost_one(phi, signal)
    return error_one


def nls_cost(x_fe, signal, gtab):

    """
    Aax_exvivo_eval

    cost function for the least square problem

    Parameters
    ----------
    x_fe : array
        x_fe(0) x_fe(1) x_fe(2)  are f1 f2 f3
        x_fe(3) theta
        x_fe(4) phi
        x_fe(5) R
        x_fe(6) as f4

    signal_param : array
        signal_param.shape = number of data points x 7

        signal_param = np.hstack([signal[:, None], bvals[:, None], bvecs,
                        G[:, None], small_delta[:, None], big_delta[:, None]])

    Returns
    -------
    sum{(signal -  phi*fe)^2}

    Notes
    --------
    cost function for the least square problem

    .. math::

        sum{(signal -  phi*fe)^2}
    """

    x, fe = x_fe_to_x_and_fe(x_fe)
    phi = Phi2(x_fe, gtab)

    return np.sum((np.squeeze(np.dot(phi, fe)) - signal) ** 2)


class ActiveAxModel(ReconstModel):

    def __init__(self, gtab, fit_method='MIX'):

        self.gtab = gtab
        self.big_delta = gtab.big_delta
        self.small_delta = gtab.small_delta
        self.gamma = gamma
        self.G = G
        D_iso = 2 * 10 ** 3
        self.yhat_ball = D_iso * self.gtab.bvals

    def fit(self, data, mask=None):

        bounds = [(0.01, np.pi), (0.01, np.pi), (0.1, 11), (0.1, 0.8)]

        res_one = differential_evolution(stoc_search_cost, bounds,
                                         args=(data, self.gtab))

        x = res_one.x
        phi = Phi(x, self.gtab)

        fe = mix_fast.estimate_f(np.array(data), phi)

        x_fe = x_and_fe_to_x_fe(x, fe)

        bounds = ([0.01, 0.01,  0.01, 0.01, 0.01, 0.1, 0.01], [0.9,  0.9,  0.9,
                  np.pi, np.pi, 11, 0.9])
        res = least_squares(nls_cost, x_fe, bounds=(bounds),
                            args=(data, self.gtab))

        result = res.x
        return result
# use numba to make the optimization faster
