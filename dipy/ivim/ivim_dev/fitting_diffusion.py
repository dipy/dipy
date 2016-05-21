import nibabel as nb
import sys
import csv
import numpy as np
import scipy.optimize as op
import scipy.stats as st
import getopt as go
import fitting_diffusion as ic
import scipy.ndimage as ndim
import scipy.integrate as si


# Initial ivim script by etpeterson'


class IvimCurve(object):
    # This class wraps up some IVIM curves and the minimize solver for easier
    # IVIM curve fitting
    __author__ = 'eric'

    fun = None
    x0 = None
    args = None
    bounds = None
    method = None

    def __init__(self, fun_in=None, x0_in=None, args_in=(), method_in=None, bounds_in=None, jac_in=None):
        self.fun = fun_in
        self.x0 = x0_in
        self.args = args_in
        self.bounds = bounds_in
        self.method = method_in
        self.jac = jac_in

    def some_function(self):
        return 1

    def minimize(self, x0_in=None, args_in=None):
        if x0_in is not None:
            self.x0 = x0_in
        if args_in is not None:
            self.args = args_in
        return op.minimize(self.fun, self.x0, args=self.args, bounds=self.bounds, method=self.method, jac=self.jac)
        # res = op.minimize(IVIM_fun_lsqr_f_Dstar_sumsq, [f_guess,
        # Dstar_guess], args=(bvals_le_200, img[i, j, k, 0:b_le_200_cutoff],
        # res.x), bounds=bnds_f_Dstar, method=min_method)  # ,
        # jac=IVIM_grad_lsqr_f_Dstar_sumsq)


class DiffusionCurve(object):

    def __init__(self, S0_reg_in=0, f_reg_in=0.1, D_reg_in=0, Dstar_reg_in=0.01):
        self.S0_reg = S0_reg_in
        self.f_reg = f_reg_in  # defaults to 0.1 for images
        self.D_reg = D_reg_in
        self.Dstar_reg = Dstar_reg_in  # defaults to 0.01 for images

    def IVIM_fun_lsqr_sumsq(self, x, b, s, x_other):
        return np.sum(np.linalg.norm(self.IVIM_fun_lsqr(x, b, s))) + self.S0_reg * abs(x[0] - x_other[0]) / x_other[0] + self.f_reg * abs(x[1] - x_other[1]) / x_other[1] + self.D_reg * abs(x[2] - x_other[2]) / x_other[2] + self.Dstar_reg * abs(x[3] - x_other[3]) / x_other[3]

    def IVIM_fun_lsqr(self, x, b, s):
        return self.IVIM_fun_wrap(x, b) - s

    def IVIM_fun_lsqr_S0_D_sumsq(self, x, b, s, x_other):
        return np.sum(np.linalg.norm(self.IVIM_fun_lsqr_S0_D(x, b, s, x_other))) + self.S0_reg * abs(x[0] - 1.0) + self.D_reg * abs(x[1])
        # return np.sum(IVIM_fun_lsqr_S0_D(x, b, s, x_other)**2.0)

    def IVIM_fun_lsqr_f_Dstar_sumsq(self, x, b, s, x_other):
        return np.sum(np.linalg.norm(self.IVIM_fun_lsqr_f_Dstar(x, b, s, x_other))) + self.f_reg * abs(x[0]) + self.Dstar_reg * abs(x[1] - x_other[1])
        # return np.sum(np.linalg.norm(IVIM_fun_lsqr_f_Dstar(x, b, s, x_other))) + 0.1*np.sum(np.linalg.norm(np.array(x)-np.array([0,x_other[1]]))) #regularize, promote smoothness here
        # return np.sum(np.linalg.norm(IVIM_fun_lsqr_f_Dstar(x, b, s, x_other))) + 0.1*np.linalg.norm(np.prod(np.array(x)-np.array([0,x_other[1]])))  # regularize on (f*(Dstar-D))^2
        # return np.sum(IVIM_fun_lsqr_f_Dstar(x, b, s, x_other)**2.0) +
        # 0.01*np.sum(x**2.0) #regularize, promote smoothness here

    def IVIM_fun_lsqr_Dstar_sumsq(self, x, b, s, x_other):
        return np.sum(np.linalg.norm(self.IVIM_fun_lsqr_Dstar(x, b, s, x_other)))

    def IVIM_fun_lsqr_S0_D(self, x, b, s, x_other):
        return self.IVIM_fun_wrap(np.array([x[0], x_other[0], x[1], x_other[1]]), b) - s

    def IVIM_fun_lsqr_f_Dstar(self, x, b, s, x_other):
        return self.IVIM_fun_wrap(np.array([x_other[0], x[0], x_other[1], x[1]]), b) - s

    def IVIM_fun_lsqr_Dstar(self, x, b, s, x_other):
        return self.IVIM_fun_wrap(np.array([x_other[0], x_other[1], x_other[2], x[0]]), b) - s

    def IVIM_fun_wrap(self, x, b):
        return self.IVIM_fun(b, x[0], x[1], x[2], x[3])

    def IVIM_fun(self, b, S0, f, D, Dstar):
        # this function evaluates the IVIM equation relative to S0, f, D, D* at the values b on the diffusion curve
        # should I be fitting S0????
        S = S0 * (f * np.exp(-b * Dstar) + (1.0 - f) * np.exp(-b * D))
        return S


def ivim_array_fit(fit_class, x0, data, mask, other_params, bvals):
    # we assume the final dimension of that data is what we want to fit
    # crop the final dimension because that's the one that's fit
    itershape = data.shape[0:-1]
    if type(x0) is list:
        numparams = len(x0)
        x0_use = x0
    else:
        numparams = x0.shape[-1]
    # convert tuple to a list and append 4 to the end
    fit = np.zeros(list(itershape) + [numparams, ])
    nit = np.zeros(itershape)
    success = np.zeros(itershape)
    fun = np.zeros(itershape)
    if type(other_params) is list:
        other_params_use = other_params
    n = np.prod(itershape)
    #residual = np.zeros(data.shape)
    #curve = np.zeros(data.shape)
    #(bvals_gt_200, img[i, j, k, b_le_200_cutoff:], noise_std, [0, 0])
    for item in np.ndindex(itershape):  # here's how to run a flat iterator!
        # +1 would avoid printing 0 twice
        idx = np.ravel_multi_index(item, itershape)
        if (idx % (n / 20) == 0):  # change this to change the printing frequency
            print('%d%%' % (idx * 100 / n))
        if (not any(data[item] == 0)) and (mask[item] > 0):
            # normdata = data[item]/data[item][0]
            if not type(other_params) is list:
                other_params_use = other_params[item]
            if not type(x0) is list:
                x0_use = x0[item]
            res = fit_class.minimize(
                x0_use, (bvals, data[item], other_params_use))
            # print(res.x)
            fit[item] = res.x  # vector of [S0, f, D, D*]
            nit[item] = res.nit  # single value, number of iterations
            # single value: True, False. if it minimized well
            success[item] = res.success
            fun[item] = res.fun  # single value, fitness function
            #residual[item, :] = img[i, j, k, :] - IVIM_fun(bvals, fit[i, j, k, 0], fit[i, j, k, 1], fit[i, j, k, 2], fit[i, j, k, 3])
            #curve[item, :] = IVIM_fun(bvals, fit[i, j, k, 0], fit[i, j, k, 1], fit[i, j, k, 2], fit[i, j, k, 3])*im0
    return fit, nit, success, fun
