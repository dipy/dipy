# -*- coding: utf-8 -*-

import numpy as np
# import nibabel as nib
# from dipy.core.gradients import gradient_table
from scipy.optimize import least_squares
import cvxpy as cvx
# from dipy.data import get_data
from scipy.optimize import differential_evolution, minimize
#import scipy
from numba import jit
from scipy.linalg import get_blas_funcs
gemm = get_blas_funcs("gemm")

gamma = 2.675987 * 10 ** 8
D_intra = 0.6 * 10 ** 3
D_iso = 2 * 10 ** 3


#@profile
def make_signal_param(signal, bvals, bvecs, G, small_delta, big_delta):

    signal_param = np.hstack([signal[:, None], bvals[:, None], bvecs,
                              G[:, None], small_delta[:, None],
                              big_delta[:, None]])

    return signal_param


#@profile
@jit(nopython=True, nogil=True, cache=True)
def func_bvec(bvecs, n):
    M = bvecs.shape[0]
    g_per = np.zeros((M))
    for i in range(M):
        g_per[i] = bvecs[i, 0]**2 + bvecs[i, 1]**2 + bvecs[i, 2]**2 - \
                   (bvecs[i, 0]*n[0] + bvecs[i, 1]*n[1] + bvecs[i, 2]*n[2])**2
#            g_per[i] = np.dot(bvecs[i, :], bvecs[i, :]) - \
#                       np.dot(bvecs[i, :], n) ** 2
    return g_per


#@profile
@jit(nopython=True, nogil=True, cache=True)
def func_mul(x, am2, small_delta, big_delta):
    M = am2.shape[0]
    N = small_delta.shape[0]
    summ = np.zeros((N, M))
    for i in range(M):
        am = am2[i]
        num = (2 * D_intra * am * small_delta) - 2 + \
              (2 * np.exp(-(D_intra * am * small_delta))) + \
              (2 * np.exp(-(D_intra * am * big_delta))) - \
              (np.exp(-(D_intra * am * (big_delta - small_delta)))) - \
              (np.exp(-(D_intra * am * (big_delta + small_delta))))

        denom = (D_intra ** 2) * (am ** 3) * ((x[2]) ** 2 * am - 1)
        summ[:, i] = num / denom
        return summ


#@profile
def activax_exvivo_compartments(x, bvals, bvecs, G, small_delta, big_delta,
                                gamma=gamma, D_intra=0.6 * 10 ** 3,
                                D_iso=2 * 10 ** 3, debug=False):

    """
    Aax_exvivo_nlin

    Parameters
    ----------
    x : array
        x.shape = 4x1
        x(0) theta (radian)
        x(1) phi (radian)
        x(2) R (micrometers)
        x(3) v=f1/(f1+f2) (ranges from 0.1 to 0.8)

    bvals : array
        bvals.shape = number of data points x 1
    bvecs : array
    G: gradient strength
    small_delta : array
    big_delta : array
    gamma: gyromagnetic ratio (2.675987 * 10 ** 8 )
    D_intra= intrinsic free diffusivity (0.6 * 10 ** 3 mircometer^2/sec)
    D_iso= isotropic diffusivity, (2 * 10 ** 3 mircometer^2/sec)

    Returns
    -------
    yhat_cylinder, yhat_zeppelin, yhat_ball, yhat_dot

    Notes
    --------
    The estimated dMRI normalized signal SActiveAx is assumed to be
    coming from the following four compartments:

    .. math::

        S_hat_ActiveAx = {f1}{exp(-yhat_cylinder)}+
                    {f2}{exp(-yhat_zeppelin)}+
                    {f3}{exp(-yhat_ball)}+
                    {f4}{exp(-yhat_dot)}

        where d_perp=D_intra*(1-v)

    """

    sinT = np.sin(x[0])
    cosT = np.cos(x[0])
    sinP = np.sin(x[1])
    cosP = np.cos(x[1])
    n = np.array([cosP * sinT, sinP * sinT, cosT])

    # Cylinder
    L = bvals * D_intra
#    print(L)
#    print(bvecs.shape)
#    print(n.shape)
    L1 = L * np.dot(bvecs, n) ** 2
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

    am2 = (am / x[2]) ** 2

#    summ = np.zeros((len(bvals), len(am)))

#    for i in range(len(am)):
#        num = (2 * D_intra * am2[i] * small_delta) - 2 + \
#              (2 * np.exp(-(D_intra * am2[i] * small_delta))) + \
#              (2 * np.exp(-(D_intra * am2[i] * big_delta))) - \
#              (np.exp(-(D_intra * am2[i] * (big_delta - small_delta)))) - \
#              (np.exp(-(D_intra * am2[i] * (big_delta + small_delta))))
#
#        denom = (D_intra ** 2) * (am2[i] ** 3) * ((x[2]) ** 2 * am2[i] - 1)
#        summ[:, i] = num / denom

    summ = func_mul(x, am2, small_delta, big_delta)

#    summ_rows = np.zeros((len(summ)))
#    for i in range(len(summ)):
#        summ_rows[i] = sum(summ[i, :])

    summ_rows = np.sum(summ, axis=1)
#    g_per = np.zeros(bvals.shape)

    g_per = func_bvec(bvecs, n)

#    for i in range(len(bvecs)):
#        g_per[i] = np.dot(bvecs[i, :], bvecs[i, :]) - \
#                   np.dot(bvecs[i, :], n) ** 2

    L2 = 2 * (g_per * gamma ** 2) * summ_rows * G ** 2

    yhat_cylinder = L1 + L2

    # zeppelin
    yhat_zeppelin = bvals * ((D_intra - (D_intra * (1 - x[3]))) *
                             (np.dot(bvecs, n) ** 2) + (D_intra * (1 - x[3])))

    # ball
    yhat_ball = (D_iso * bvals)

    # dot
    yhat_dot = np.dot(bvecs, np.array([0, 0, 0]))

    if debug:
        return L1, summ, summ_rows, g_per, L2, yhat_cylinder, yhat_zeppelin, \
            yhat_ball, yhat_dot
    return yhat_cylinder, yhat_zeppelin, yhat_ball, yhat_dot

# @profile
def activax_exvivo_model(x, bvals, bvecs, G, small_delta, big_delta,
                         gamma=gamma,
                         D_intra=0.6 * 10 ** 3, D_iso=2 * 10 ** 3,
                         debug=False):

    """
    Aax_exvivo_nlin

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
    exp(-yhat_cylinder), exp(-yhat_zeppelin), exp(-yhat_ball), exp(-yhat_dot)

    Notes
    --------
    The estimated dMRI normalized signal SActiveAx is assumed to be
    coming from the following four compartments:

    .. math::

        S_hat_ActiveAx = {f1}{S_cylinder(R,theta,phi)}+
                    {f2}{S_zeppelin(d_perp,theta,phi)}+
                    {f3}{S_ball}+
                    {f4}{S_dot}

        where d_perp=D_intra*(1-v)
        S_cylinder = exp(-yhat_cylinder)
        S_zeppelin = exp(-yhat_zeppelin)
        S_ball = exp(-yhat_ball)
        S_dot = exp(-yhat_dot)

    """
    res = activax_exvivo_compartments(x, bvals, bvecs, G, small_delta,
                                      big_delta, gamma=gamma,
                                      D_intra=0.6 * 10 ** 3, D_iso=2 * 10 ** 3,
                                      debug=False)

    yhat_cylinder, yhat_zeppelin, yhat_ball, yhat_dot = res

    phi = np.vstack([yhat_cylinder, yhat_zeppelin, yhat_ball, yhat_dot]).T
    phi = np.ascontiguousarray(phi)

    return np.exp(-phi)


# @profile
def activax_exvivo_compartments2(x_fe, bvals, bvecs, G, small_delta, big_delta,
                                 gamma=gamma, D_intra=0.6 * 10 ** 3,
                                 D_iso=2 * 10 ** 3, debug=False):

    """
    Aax_exvivo_eval

    Parameters
    ----------
    x_fe : array
        x_fe.shape = 7x1
        x_fe(0) x_fe(1) x_fe(2)  are f1 f2 f3
        x_fe(3) theta
        x_fe(4) phi
        x_fe(5) R
        x_fe(6) as f4

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
    yhat_cylinder, yhat_zeppelin, yhat_ball, yhat_dot

    Notes
    --------
    The estimated dMRI normalized signal SActiveAx is assumed to be
    coming from the following four compartments:

    .. math::

        SActiveAx = {f1}{exp(-yhat_cylinder)}+
                    {f2}{exp(-yhat_zeppelin)}+
                    {f3}{exp(-yhat_ball)}+
                    {f4}{exp(-yhat_dot)}

        where d_perp=D_intra*(1-v)

    """

    sinT = np.sin(x_fe[3])
    cosT = np.cos(x_fe[3])
    sinP = np.sin(x_fe[4])
    cosP = np.cos(x_fe[4])
    n = np.array([cosP * sinT, sinP * sinT, cosT])

    # Cylinder
    L = bvals * D_intra
#    print(L)
#    print(bvecs.shape)
#    print(n.shape)
    L1 = L * np.dot(bvecs, n) ** 2
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

    am2 = (am / x_fe[5]) ** 2

    summ = np.zeros((len(bvals), len(am)))

    for i in range(len(am)):
        num = (2 * D_intra * am2[i] * small_delta) - 2 + \
              (2 * np.exp(-(D_intra * am2[i] * small_delta))) + \
              (2 * np.exp(-(D_intra * am2[i] * big_delta))) - \
              (np.exp(-(D_intra * am2[i] * (big_delta - small_delta)))) - \
              (np.exp(-(D_intra * am2[i] * (big_delta + small_delta))))

        denom = (D_intra ** 2) * (am2[i] ** 3) * ((x_fe[5]) ** 2 * am2[i] - 1)
        summ[:, i] = num / denom

    summ_rows = np.sum(summ, axis=1)
    g_per = np.zeros(bvals.shape)

    for i in range(len(bvecs)):
        g_per[i] = np.dot(bvecs[i, :], bvecs[i, :]) - \
                   np.dot(bvecs[i, :], n) ** 2

    L2 = 2 * (g_per * gamma ** 2) * summ_rows * G ** 2

    yhat_cylinder = L1 + L2

    # zeppelin
    v = x_fe[0]/(x_fe[0] + x_fe[1])
    yhat_zeppelin = bvals * ((D_intra - (D_intra * (1 - v))) *
                             (np.dot(bvecs, n) ** 2) + (D_intra * (1 - v)))

    # ball
    yhat_ball = (D_iso * bvals)

    # dot
    yhat_dot = np.dot(bvecs, np.array([0, 0, 0]))

    if debug:
        return L1, summ, summ_rows, g_per, L2, yhat_cylinder, yhat_zeppelin, \
            yhat_ball, yhat_dot
    return yhat_cylinder, yhat_zeppelin, yhat_ball, yhat_dot


# @profile
def activax_exvivo_model2(x_fe, bvals, bvecs, G, small_delta, big_delta,
                          gamma=gamma, D_intra=0.6 * 10 ** 3,
                          D_iso=2 * 10 ** 3, debug=False):

    """
    Aax_exvivo_eval

    Parameters
    ----------
    x_fe: array
        x_fe.shape = 7x1
        x_fe(0) x_fe(1) x_fe(2)  are f1 f2 f3
        x_fe(3) theta
        x_fe(4) phi
        x_fe(5) R
        x_fe(6) as f4

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
    exp(-yhat_cylinder), exp(-yhat_zeppelin), exp(-yhat_ball), exp(-yhat_dot)

    Notes
    --------
    The estimated dMRI normalized signal SActiveAx is assumed to be
    coming from the following four compartments:

    .. math::

        SActiveAx = {f1}{S_cylinder(R,theta,phi)}+
                    {f2}{S_zeppelin(d_perp,theta,phi)}+
                    {f3}{S_ball}+
                    {f4}{S_dot}

        where d_perp=D_intra*(1-v)
        S_cylinder = exp(-yhat_cylinder)
        S_zeppelin = exp(-yhat_zeppelin)
        S_ball = exp(-yhat_ball)
        S_dot = exp(-yhat_dot)

    """

    res = activax_exvivo_compartments2(x_fe, bvals, bvecs, G, small_delta,
                                       big_delta, gamma=gamma,
                                       D_intra=0.6 * 10 ** 3,
                                       D_iso=2 * 10 ** 3, debug=False)

    yhat_cylinder, yhat_zeppelin, yhat_ball, yhat_dot = res

    phi = np.vstack([yhat_cylinder, yhat_zeppelin, yhat_ball, yhat_dot]).T
    phi = np.ascontiguousarray(phi)

    return np.exp(-phi)

#@profile
def activeax_cost_one(phi, signal):  # sigma

    """
    Aax_exvivo_nlin

    to make cost function for genetic algorithm

    Parameters
    ----------
    phi:
        phi.shape = number of data points x 4
    signal:
        signal.shape = number of data points x 1

    Returns
    -------
    (signal -  S)^T(signal -  S)

    Notes
    --------
    to make cost function for genetic algorithm:

    .. math::

        (signal -  S)^T(signal -  S)

    """

#    phi_mp = np.dot(np.linalg.inv(np.dot(phi.T, phi)), phi.T)  # moore-penrose
    phi_mp = gemm(1, np.linalg.inv(gemm(1, phi.T, phi)), phi.T)
    f = np.dot(phi_mp, signal)
    yhat = np.dot(phi, f)  # - sigma
    return np.dot((signal - yhat).T, signal - yhat)

#@profile
def stoc_search_cost_func(x, signal, bvals, bvecs, G, small_delta, big_delta):

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

    phi = activax_exvivo_model(x, bvals, bvecs, G,
                               small_delta, big_delta,
                               gamma=gamma,
                               D_intra=0.6 * 10 ** 3, D_iso=2 * 10 ** 3,
                               debug=False)

    error_one = activeax_cost_one(phi, signal)
    return error_one


#@profile
def dif_evol(signal, bvals, bvecs, G, small_delta, big_delta):

    """
    differential evolution algorithm instead of genetic algorithm in MIX paper

    Parameters
    ----------
    signal:
        signal.shape = number of data points x 1

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
    res_one.x(0) theta (radian)
    res_one.x(1) phi (radian)
    res_one.x(2) R (micrometers)
    res_one.x(3) v=f1/(f1+f2) (0.1 - 0.8)

    """

    bounds = [(0.01, np.pi), (0.01, np.pi), (0.1, 11), (0.1, 0.8)]
    res_one = differential_evolution(stoc_search_cost_func, bounds, args=(signal, bvals,
                                                             bvecs, G,
                                                             small_delta,
                                                             big_delta))


#    res_one = scipy.optimize.minimize(stoc_search_cost_func, 2*x, args=(signal, bvals,
#                                                             bvecs, G,
#                                                             small_delta,
#                                                             big_delta), method='Powell')


    return res_one.x


#@profile
def estimate_f(signal, phi):

    """
    Linear parameters fit using cvx

    Parameters
    ----------
    phi : array
        phi.shape = number of data points x 4
    signal : array
        signal.shape = number of data points x 1

    Returns
    -------
    f1, f2, f3, f4 (volume fractions)
    f1 = fe[0]
    f2 = fe[1]
    f3 = fe[2]
    f4 = fe[3]

    Notes
    --------
    cost function for genetic algorithm:

    .. math::

        minimize(norm((signal)- (phi*fe)))
    """

    # Create four scalar optimization variables.
    fe = cvx.Variable(4)
    # Create four constraints.
    constraints = [cvx.sum_entries(fe) == 1,
                   fe[0] >= 0,
                   fe[1] >= 0,
                   fe[2] >= 0,
                   fe[3] >= 0]

    # Form objective.
    obj = cvx.Minimize(cvx.sum_entries(cvx.square(phi * fe - signal[:, None])))

    # Form and solve problem.
    prob = cvx.Problem(obj, constraints)
    prob.solve()  # Returns the optimal value.

    return np.array(fe.value)


#@profile
def nls_cost_func(x_fe, signal_param):

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

    fe = np.zeros((1, 4))
    fe = np.squeeze(fe)
    fe[0:3] = x_fe[0:3]
    fe[3] = x_fe[6]

    signal = signal_param[:, 0]
    bvals = signal_param[:, 1]
    bvecs = signal_param[:, 2:5]
    G = signal_param[:, 5]
    small_delta = signal_param[:, 6]
    big_delta = signal_param[:, 7]

    phi = activax_exvivo_model2(x_fe, bvals, bvecs, G, small_delta, big_delta,
                                gamma=gamma, D_intra=0.6 * 10 ** 3,
                                D_iso=2 * 10 ** 3, debug=False)

    return np.sum((np.squeeze(np.dot(phi, fe)) - signal) ** 2)


#@profile
def final(signal, x, fe):

    """ Nonlinear least squares fitting

    Parameters
    ----------
    fe : array
        fe(0) fe(1) fe(2) fe(3)  are f1 f2 f3 f4

    x : array
        x(0) theta, x(1) phi, x(2) R

    signal:
        signal.shape = number of data points x 1

    Returns
    -------
    volume fractions (f1, f2, f3, f4), theta, phi, R

    res.x(0) res.x(1) res.x(2)  are f1 f2 f3
    res.x(3) theta
    res.x(4) phi
    res.x(5) R
    res.x(6) as f4

    Notes
    --------
    in this step we take the estimated volume fractions from convex
    optimization and theta, phi and R from genetic algorithm to find all 7
    unknown parameters using bounded non-linear least square fitting

    """

    x_fe = np.zeros([7])
    x_fe[:3] = fe[:3]
    x_fe[3:6] = x[:3]
    x_fe[6] = fe[3]
    bounds = ([0.01, 0.01,  0.01, 0.01, 0.01, 0.1, 0.01], [0.9,  0.9,  0.9,
              np.pi, np.pi, 11, 0.9])
    res = least_squares(nls_cost_func, x_fe, bounds=(bounds),
                        args=(signal,))
    return res
