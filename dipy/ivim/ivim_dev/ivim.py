#!/usr/bin/python
""" Classes and functions for fitting ivim model """
from __future__ import division, print_function, absolute_import

import warnings

import functools

import numpy as np

import scipy.optimize as op
import scipy.stats as st
import scipy.ndimage as ndim
import scipy.integrate as si

from dipy.core.gradients import gradient_table
from base import ReconstModel
from fitting_diffusion import DiffusionCurve, IvimCurve, ivim_array_fit

import matplotlib.pylab as pl


def ivim_prediction(ivim_params, gtab, S0=1.0):
    """
    Predict a signal given the parameters of the IVIM model.

    Parameters
    ----------
    ivim_params : ndarray (N, 3)
        Ivim parameters f, D, D_star for all N voxels.
        The last dimension should have the 3 parameters
        f :
        D:
        D_star :

    gtab : a GradientTable class instance
        The gradient table for this prediction

    S0 : float or ndarray
        The non diffusion-weighted signal in every voxel, or across all
        voxels. Default: 1

    Notes
    -----
    The predicted signal is given by:
    $S(b) = S_0*{f*e^{-b * D_star} + (1-f)* e^{-b * D}}$,

    References
    ----------
    .. [1]

    """
    bvals = gtab.bvals
    S = np.vectorize(lambda f, D, D_star, b: S0 *
                     (f * np.exp(-b * D_star) + (1 - f) * np.exp(-b * D)))
    N = len(ivim_params)
    pred_signal = []
    for i in range(N):
        f, D, D_star = ivim_params[i]
        pred_signal.append(S(f, D, D_star, bvals))

    return pred_signal


def _min_positive_signal(data):
    """ Helper function to establish the minimum positive signal of a given
    data

    Parameters
    ----------
    data: array ([X, Y, Z, ...], g)
        Data or response variables holding the data. Note that the last
        dimension should contain the data.

    Returns
    -------
    min_signal : float
        Minimum positive signal of the given data
    """
    data = data.ravel()
    if np.all(data == 0):
        return 0.0001
    else:
        return data[data > 0].min()


class IvimModel(ReconstModel):
    """Ivim model
    """

    def __init__(self, gtab, fit_method="one_stage_fit", *args, **kwargs):
        """ A Ivim Model

        Parameters
        ----------
        gtab : GradientTable class instance

        fit_method : str or callable (have to work)
                str can be one of the following:
            'one_stage_fit'
                description of one stage fit

            'two_stage_fit'

        args, kwargs : arguments and key-word arguments passed to the

        min_signal : float
            The minimum signal value. Needs to be a strictly positive
            number. Default: minimal signal in the data provided to `fit`.

        References
        ----------
        .. [1]
        .. [2]
        .. [3]

        """
        ReconstModel.__init__(self, gtab)

        if not callable(fit_method):
            try:
                fit_method = common_fit_methods[fit_method]
            except KeyError:
                e_s = '"' + str(fit_method) + '" is not a known fit '
                e_s += 'method, the fit method should either be a '
                e_s += 'function or one of the common fit methods'
                raise ValueError(e_s)
        self.fit_method = fit_method
        self.args = args
        self.kwargs = kwargs
        self.min_signal = self.kwargs.pop('min_signal', None)
        if self.min_signal is not None and self.min_signal <= 0:
            e_s = "The `min_signal` key-word argument needs to be strictly"
            e_s += " positive."
            raise ValueError(e_s)

    def fit(self, data, mask=None):
        """ Fit method of the Ivim model class

        Parameters
        ----------
        data : array
            The measured signal from one voxel.

        mask : array
            A boolean array used to mark the coordinates in the data that
            should be analyzed that has the shape data.shape[:-1]

        """
        # Have to implement masking
        data = data.astype(np.dtype('d'))
        (bvals, img, fit, residual, curve, img0,
            S0_prime, D, S0, Dstar_prime, f) = self.fit_method(data, self.gtab)

        return IvimFit(bvals, img, fit, residual, curve, img0,
                       S0_prime, D, S0, Dstar_prime, f)

    def predict(self, ivim_params, S0=1):
        """
        Predict a signal for this IvimModel class instance given parameters.

        Parameters
        ----------
            ivim_params : ndarray
            Ivim parameters f, D, D_star. The last dimension should
            have the 3 parameters
            f :
            D_star :
            D :

        S0 : float or ndarray
            The non diffusion-weighted signal in every voxel, or across all
            voxels. Default: 1
        """
        return ivim_prediction(ivim_params, self.gtab, S0)


class IvimFit(object):
    # Should ideally be of this type

    # def __init__(self, model, model_params):
    #     """ Initialize a IvimFit class instance.
    #     """
    #     self.model = model
    #     self.model_params = model_params

    # def __getitem__(self, index):
    #     model_params = self.model_params
    #     N = model_params.ndim
    #     if type(index) is not tuple:
    #         index = (index,)
    #     elif len(index) >= model_params.ndim:
    #         raise IndexError("IndexError: invalid index")
    #     index = index + (slice(None),) * (N - len(index))
    #     return type(self)(self.model, model_params[index])

    # @property
    # def shape(self):
    #     return self.model_params.shape[:-1]

    # @property
    # def D(self):
    #     """
    #     Returns the D values
    #     """
    #     # return self.model_params[..., :3]

    # def predict(self, gtab, S0=1, step=None):
    #     r"""
    #     Given a model fit, predict the signal

    #     Parameters
    #     ----------
    #     gtab : a GradientTable class instance
    #         This encodes the directions for which a prediction is made

    #     S0 : float array
    #        The mean non-diffusion weighted signal in each voxel. Default: 1 in
    #        all voxels.

    #     Notes
    #     -----
    #     The predicted signal is given by:

    #     """
    #     # return predict.reshape(shape + (gtab.bvals.shape[0], ))
    #     return 0

    def __init__(self, bvals, img, fit, residual,
                 curve, img0, S0_prime, D, S0, Dstar_prime, f):
        self.bvals = bvals
        self.img = img
        self.fit = fit
        self.residual = residual
        self.curve = curve
        self.img0 = img0
        self.S0_prime = S0_prime
        self.D = D
        self.S0 = S0
        self.Dstar_prime = Dstar_prime
        self.f = f

    def plot(self, num_plots=1):
        bvals = self.bvals
        img = self.img
        curve = self.curve
        bvals = self.bvals
        fit = self.fit
        img0 = self.img0
        residual = self.residual
        S0 = self.S0
        S0_prime = self.S0_prime
        D = self.D
        Dstar_prime = self.Dstar_prime
        f = self.f

        bvals_le_200 = bvals[bvals <= 200]
        shape = img.shape
        shape3d = shape[0:-1]

        dc = DiffusionCurve()

        plot_count = 0
        for item in np.ndindex(shape3d):
            if plot_count < num_plots:
                curve[item] = dc.IVIM_fun(bvals, fit[item][0], fit[item][1], fit[
                    item][2], fit[item][3]) * img0[item]
                residual[item] = img[item] - curve[item]

                D_line = dc.IVIM_fun(bvals, S0_prime[item], 0., D[
                    item], 0.) * img0[item]
                Dstar_line = dc.IVIM_fun(
                    bvals_le_200, S0[item], 1., 0., Dstar_prime[item]) * img0[item]

                pl.plot(bvals, D_line, 'b--', label='D curve')
                pl.plot(bvals_le_200, Dstar_line, 'r--', label='D* curve')
                pl.plot(bvals, img[item] * img0[item],
                        '.', label='Image values')
                pl.plot(bvals, curve[item], label='Fitted curve')
                pl.yscale('symlog')  # to protect against 0 or negative values
                pl.xlabel(r'b-value $(s/mm^2)$')
                pl.ylabel(r'Signal intensity $(a.u.)$')
                pl.legend(loc='best')
                # pl.gca().text(0.25, 0.75, 'S0=%f f=%f D=%f D*=%f' %(fit[item][0], fit[item][1], fit[item][2], fit[item][3]))
                # ax = pl.axes()
                pl.ylim(0, 1.1)
                text_fit = 'S0={:.2e} f={:.2e}\nD={:.2e} D*={:.2e}'.format(
                    fit[item][0], fit[item][1], fit[item][2], fit[item][3])
                #'S0=%f f=%f D=%f D*=%f' %(fit[item][0], fit[item][1], fit[item][2], fit[item][3])
                pl.gca().text(0.65, 0.85, text_fit, horizontalalignment='center',
                              verticalalignment='center', transform=pl.gca().transAxes)

                # pl.cla()
                pl.show()
                plot_count += 1
            else:
                return 1


def one_stage_fit(data, gtab, jac=False, min_method=None):
    """
    Computes one_stage_ivim fit.

    Parameters
    ----------
    design_matrix : array (g, 7)
        Design matrix holding the guess and boundaries
    data : array ([X, Y, Z, ...], g)
        Data or response variables holding the data. Note that the last
        dimension should contain the data. It makes no copies of data.

    Returns
    -------
    f :
    D_star :
    D :
    """
    img = data
    bvals = gtab.bvals
    shape = img.shape
    shape3d = shape[0:-1]
    mask = np.ones(shape3d)

    bnds = ((0.5, 1.5), (0, 1), (0, 0.5), (0, 0.5))  # (S0, f, D, D*)
    bnds_S0_D = (bnds[0], bnds[2])
    bnds_f_Dstar = (bnds[1], bnds[3])
    bnds_Dstar = (bnds[3],)
    # initial guesses are from the typl values in "Quantitative Measurement
    # of Brin Perfusion with Intravoxel Incoherent Motion MR Imaging"
    S0_guess = 1.0  # img.ravel().max()
    f_guess = 6e-2  # 6% or 0.06
    D_guess = 0.9e-3  # note that D<D* because the slope is lower in the diffusion section
    Dstar_guess = 7.0e-3  # note that D*>D because the slope is higher in the perfusion section
    # noise_std /= img.ravel().max()

    bvals_le_200 = bvals[bvals <= 200]
    bvals_gt_200 = bvals[bvals > 200]
    b_le_200_cutoff = len(bvals_le_200)

    fit = np.zeros(list(shape3d) + [4, ])
    nit = np.zeros(shape3d)
    success = np.zeros(shape3d)
    fun = np.zeros(shape3d)
    residual = np.zeros(shape)
    curve = np.zeros(shape)
    S0_prime = 0
    D = 0
    S0 = 0
    Dstar_prime = 0
    f = 0

    # normalize img but store the first value
    img0 = np.zeros(shape3d)
    np.copyto(img0, img[..., 0])
    for item in np.ndindex(shape3d):
        if img[item][0] > 0:  # so we dont' divide by 0 or anything funky
            img[item] /= img[item][0]

    dc1 = DiffusionCurve(S0_reg_in=0.01, f_reg_in=0.01,
                         D_reg_in=0.01, Dstar_reg_in=0.01)
    fitfun_S0_f_D_Dstar = IvimCurve(fun_in=dc1.IVIM_fun_lsqr_sumsq,
                                    method_in=min_method, bounds_in=bnds)
    print('Start one stage fit\n')
    print('Fitting S0, f, D, and D* \n')

    fit, nit, success, fun = ivim_array_fit(fitfun_S0_f_D_Dstar, [
        S0_guess, f_guess, D_guess, Dstar_guess], img[..., :], mask, [0, 0, 0, 0], bvals)

    return bvals, img, fit, residual, curve, img0, S0_prime, D, S0, Dstar_prime, f


def two_stage_fit(data, gtab, jac=False):
    img = data
    bvals = gtab.bvals
    shape = img.shape
    shape3d = shape[0:-1]
    mask = np.ones(shape3d)

    bnds = ((0.5, 1.5), (0, 1), (0, 0.5), (0, 0.5))  # (S0, f, D, D*)
    bnds_S0_D = (bnds[0], bnds[2])
    bnds_f_Dstar = (bnds[1], bnds[3])
    bnds_Dstar = (bnds[3],)
    # initial guesses are from the typl values in "Quantitative Measurement
    # of Brin Perfusion with Intravoxel Incoherent Motion MR Imaging"
    S0_guess = 1.0  # img.ravel().max()
    f_guess = 6e-2  # 6% or 0.06
    D_guess = 0.9e-3  # note that D<D* because the slope is lower in the diffusion section
    Dstar_guess = 7.0e-3  # note that D*>D because the slope is higher in the perfusion section
    # noise_std /= img.ravel().max()

    bvals_le_200 = bvals[bvals <= 200]
    bvals_gt_200 = bvals[bvals > 200]
    b_le_200_cutoff = len(bvals_le_200)

    fit = np.zeros(list(shape3d) + [4, ])
    nit = np.zeros(shape3d)
    success = np.zeros(shape3d)
    fun = np.zeros(shape3d)
    residual = np.zeros(shape)
    curve = np.zeros(shape)
    S0_prime = 0
    D = 0
    S0 = 0
    Dstar_prime = 0
    f = 0
    min_method = None
    # normalize img but store the first value
    img0 = np.zeros(shape3d)
    np.copyto(img0, img[..., 0])
    for item in np.ndindex(shape3d):
        if img[item][0] > 0:  # so we dont' divide by 0 or anything funky
            img[item] /= img[item][0]

    dc2 = DiffusionCurve(S0_reg_in=0., f_reg_in=0.,
                         D_reg_in=0., Dstar_reg_in=0.)
    fitfun_S0_D = IvimCurve(
        fun_in=dc2.IVIM_fun_lsqr_S0_D_sumsq, method_in=min_method, bounds_in=bnds_S0_D)
    print('Start two stage fit\n')
    print('Fitting S0prime and D\n')
    fit[..., 0:3:2], nit, success, fun = ivim_array_fit(
        fitfun_S0_D, [S0_guess, D_guess], img[..., b_le_200_cutoff:], mask, [0, 0], bvals_gt_200)
    # save the intercept for the first values, let's call it S0'
    S0_prime = np.copy(fit[..., 0])
    D = np.copy(fit[..., 2])  # save the slope for plotting later
    print('Fitting S0 and D*prime\n')
    fit[..., 0:4:3], nit, success, fun = ivim_array_fit(fitfun_S0_D, [
        S0_guess, Dstar_guess], img[..., :b_le_200_cutoff], mask, [0, 0], bvals_le_200)
    S0 = np.copy(fit[..., 0])  # save the intercept for plotting later
    Dstar_prime = np.copy(fit[..., 3])  # save the linear D* only fit
    print('Estimating f\n')
    # arbitrary range, but we want to cap it to [0,1]
    fit[..., 1] = 1 - S0_prime / fit[..., 0]
    fit[fit[..., 1] < 0, 1] = 0  # make sure we don't have f<0
    fit[fit[..., 1] > 1, 1] = 1  # make sure we don't have f>1
    f = np.copy(fit[..., 1])  # save the fraction for plotting later
    print('Fitting D*')
    fitfun_Dstar = IvimCurve(
        fun_in=dc2.IVIM_fun_lsqr_Dstar_sumsq, method_in=min_method, bounds_in=bnds_Dstar)
    fit[..., 3:], nit, success, fun = ivim_array_fit(
        fitfun_Dstar, [Dstar_guess], img[..., :], mask, fit[..., 0:3], bvals)

    print('End two stage fit\n')

    return bvals, img, fit, residual, curve, img0, S0_prime, D, S0, Dstar_prime, f


common_fit_methods = {'one_stage_fit': one_stage_fit,
                      'two_stage_fit': two_stage_fit,
                      }
