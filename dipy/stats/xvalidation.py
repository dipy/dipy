"""
Cross-validation analysis of diffusion models




"""
from __future__ import division, print_function, absolute_import
from dipy.utils.six.moves import range

import numpy as np
import dipy.core.gradients as gt

def kfold_xval(model, data, folds, *model_args, **model_kwargs):
    """
    Given a Model object perform iterative k-fold cross-validation of fitting
    that model

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

    # I'm going to leave this cruft here, for now:
    ## #### This is DKI-specific
    ## len1 = 1
    ## len2 = 2
    ## while len1 != len2:
    ##     # We are going to leave out some randomly chosen samples in each
    ##     # iteration:
    ##     order = np.random.permutation(data_d.shape[-1])
    ##     len1 = []
    ##     len2 = []
    ##     for k in range(folds):
    ##         fold_mask = np.ones(data_d.shape[-1], dtype=bool)
    ##         fold_idx = order[k*n_in_fold:(k+1)*n_in_fold]
    ##         fold_mask[fold_idx] = False
    ##         len1.append(len(np.unique(nz_bval[fold_mask])))
    ##         len2.append(len(np.unique(nz_bval[~fold_mask])))

    ## print("Apparently it's possible...")
    ## ##### Up until here


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
