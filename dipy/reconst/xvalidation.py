"""
Cross-validation analysis of diffusion models




"""
from __future__ import division, print_function, absolute_import
from dipy.utils.six.moves import range

import numpy as np
import dipy.core.gradients as gt

def coeff_of_determination(data, model, axis=-1):
    """
    Parameters
    ----------
    data : ndarray
        The real data

    model : ndarray
        The predictions of a model

    axis: int, optional
        The axis along which samples are organized (default: -1)

    Returns
    -------
    COD : ndarray
       The coefficient of determination. This has shape data.shape[:-1]


    Notes
    -----

    See: http://en.wikipedia.org/wiki/Coefficient_of_determination

    The coefficient of determination is calculated as:

    .. math::

        R^2 = 100 * (1 - \frac{SSE}{SSD})

    where SSE is the sum of the squared error between the model and the data
    (sum of the squared residuals) and SSD is the sum of the squares of the
    deviations of the (variance * N)
    """

    residuals = data - model
    ss_err = np.sum(residuals ** 2, axis=axis)

    demeaned_data = data - np.mean(data, axis=axis)[..., np.newaxis]
    ss_tot = np.sum(demeaned_data **2, axis=axis)

    # Don't divide by 0:
    if np.all(ss_tot==0.0):
        return np.nan

    return 100 * (1 - (ss_err/ss_tot))



def kfold_xval(model, data, folds, *model_args, **model_kwargs):
    """
    Perform k-fold cross-validation to generate out-of-sample predictions for
    each measurement.

    Parameters
    ----------
    model : class instance of a Model

    data : ndarray
        Diffusion MRI data acquired with the gtab of the model

    folds : int
        The number of divisions to apply to the data

    model_args :
        Additional arguments to the model initialization

    model_kwargs :
        Additional key-word arguments to the model initialization

    Notes
    -----
    This function assumes that a prediction API is implemented in the Model
    class for which prediction is conducted. That is, the Fit object that gets
    generated upon fitting the model needs to have a `predict` method, which
    receives a GradientTable class instance as input and produces a predicted
    signal as output.

    It also assumes that the model object has `bval` and `bvec` attributes
    holding b-values and corresponding unit vectors.

    References
    ----------
    .. [1] Rokem, A., Chan, K.L. Yeatman, J.D., Pestilli, F., Mezer, A.,
       Wandell, B.A., 2014. Evaluating the accuracy of diffusion models at
       multiple b-values with cross-validation. ISMRM 2014.
    """
    # This should always be there, if the model inherits from
    # dipy.reconst.base.ReconstModel:
    gtab = model.gtab
    data_d = data[..., ~gtab.b0s_mask]
    modder =  np.mod(data_d.shape[-1], folds)
    # Make sure that an equal number of samples get left out in each fold:
    if modder!= 0:
        msg = "The number of folds must divide the diffusion-weighted "
        msg += "data equally, but "
        msg = "np.mod(%s, %s) is %s"%(data_d.shape[-1], folds, modder)
        raise ValueError(msg)

    data_0 = data[..., gtab.b0s_mask]
    S0 = np.mean(data_0, -1)
    n_in_fold = data_d.shape[-1]/folds
    prediction = np.zeros(data.shape)
    # We are going to leave out some randomly chosen samples in each iteration:
    order = np.random.permutation(data_d.shape[-1])

    nz_bval = gtab.bvals[~gtab.b0s_mask]
    nz_bvec = gtab.bvecs[~gtab.b0s_mask]

    for k in range(folds):
        fold_mask = np.ones(data_d.shape[-1], dtype=bool)
        fold_idx = order[k*n_in_fold:(k+1)*n_in_fold]
        fold_mask[fold_idx] = False
        this_data = np.concatenate([data_0, data_d[..., fold_mask]], -1)

        this_gtab = gt.gradient_table(np.hstack([gtab.bvals[gtab.b0s_mask],
                                                 nz_bval[fold_mask]]),
                                      np.concatenate([gtab.bvecs[gtab.b0s_mask],
                                                 nz_bvec[fold_mask]]))
        left_out_gtab = gt.gradient_table(np.hstack([gtab.bvals[gtab.b0s_mask],
                                                 nz_bval[~fold_mask]]),
                                      np.concatenate([gtab.bvecs[gtab.b0s_mask],
                                                 nz_bvec[~fold_mask]]))

        this_model = model.__class__(this_gtab, *model_args, **model_kwargs)
        this_fit = this_model.fit(this_data)
        this_predict = this_fit.predict(left_out_gtab, S0=S0)
        idx_to_assign = np.where(~gtab.b0s_mask)[0][~fold_mask]
        prediction[..., idx_to_assign]=this_predict[..., np.sum(gtab.b0s_mask):]

    # For the b0 measurements
    prediction[..., gtab.b0s_mask] = S0[..., None]

    return prediction
