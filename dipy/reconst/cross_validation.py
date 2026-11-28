"""
Cross-validation analysis of diffusion models.
"""

import numpy as np

import dipy.core.gradients as gt
from dipy.utils.deprecator import warning_for_keywords


@warning_for_keywords()
def coeff_of_determination(data, model, *, axis=-1):
    r"""Calculate the coefficient of determination for a model prediction,
    relative to data.

    Parameters
    ----------
    data : ndarray
        The data
    model : ndarray
        The predictions of a model for this data. Same shape as the data.
    axis: int, optional
        The axis along which different samples are laid out.

    Returns
    -------
    COD : ndarray
       The coefficient of determination. This has shape `data.shape[:-1]`

    Notes
    -----
    See: https://en.wikipedia.org/wiki/Coefficient_of_determination

    The coefficient of determination is calculated as:

    .. math::

        R^2 = 100 * (1 - \frac{SSE}{SSD})

    where SSE is the sum of the squared error between the model and the data
    (sum of the squared residuals) and SSD is the sum of the squares of the
    deviations of the data from the mean of the data (variance * N).

    """
    residuals = data - model
    ss_err = np.sum(residuals**2, axis=axis)

    demeaned_data = data - np.mean(data, axis=axis)[..., np.newaxis]
    ss_tot = np.sum(demeaned_data**2, axis=axis)

    # Don't divide by 0:
    if np.all(ss_tot == 0.0):
        return np.nan

    return 100 * (1 - (ss_err / ss_tot))


def _spherical_density_folds(gtab, folds, *, shell_tol, rng):
    """Split DWI indices relative to the diffusion-weighted subset."""
    dwi = ~gtab.b0s_mask
    bvals = gtab.bvals[dwi]
    if (
        isinstance(folds, (bool, np.bool_))
        or not isinstance(folds, (int, np.integer))
        or not 2 <= folds <= bvals.size
    ):
        raise ValueError("folds must be an integer between 2 and the DWI count.")
    if not np.any(gtab.b0s_mask):
        raise ValueError("Spherical-density splitting requires at least one b0.")
    if not np.isfinite(shell_tol) or shell_tol < 0:
        raise ValueError("shell_tol must be finite and nonnegative.")
    if not np.isfinite(bvals).all():
        raise ValueError("Diffusion-weighted b-values must be finite.")

    directions = gtab.bvecs[dwi]
    norms = np.linalg.norm(directions, axis=1)
    if np.any(~np.isfinite(norms) | (norms == 0)):
        raise ValueError("Diffusion-weighted directions must be finite and nonzero.")
    directions = directions / norms[:, None]
    if gtab.btens is not None:
        linear = bvals[:, None, None] * directions[:, :, None] * directions[:, None, :]
        if not np.allclose(gtab.btens[dwi], linear):
            raise ValueError("Spherical-density splitting requires linear encoding.")

    centers = gt.unique_bvals_tolerance(bvals, tol=shell_tol)
    right = np.searchsorted(centers, bvals)
    left = np.maximum(right - 1, 0)
    right = np.minimum(right, len(centers) - 1)
    left_distance = np.abs(bvals - centers[left])
    right_distance = np.abs(bvals - centers[right])
    tied = (left != right) & (left_distance == right_distance)
    if np.any(tied | (np.minimum(left_distance, right_distance) > shell_tol)):
        raise ValueError(
            "Could not uniquely assign DWI measurements to b-value shells."
        )
    shell_labels = np.where(left_distance < right_distance, left, right)

    fold_order = rng.permutation(folds)
    assignments = np.empty(bvals.size, dtype=int)
    cursor = 0
    for label in range(len(centers)):
        shell = rng.permutation(np.flatnonzero(shell_labels == label))
        for group in _direction_groups(directions[shell], folds):
            # Carry shell remainders across shells so global sizes stay balanced.
            slots = (cursor + np.arange(len(group))) % folds
            assignments[rng.permutation(shell[group])] = fold_order[slots]
            cursor = (cursor + len(group)) % folds

    return [np.flatnonzero(assignments == fold) for fold in range(folds)]


def _direction_groups(directions, size):
    """Group nearby axes, starting in the sparsest sampled neighborhoods."""
    if len(directions) <= size:
        yield np.arange(len(directions))
        return

    similarities = np.clip(np.abs(directions @ directions.T), 0, 1)
    np.fill_diagonal(similarities, -np.inf)
    # Smaller similarity to the last neighbor means a larger angular radius.
    column = len(directions) - (size - 1)
    density = np.partition(similarities, column, axis=1)[:, column]
    remaining = np.ones(len(directions), dtype=bool)
    for anchor in np.argsort(density, kind="stable"):
        if not remaining[anchor]:
            continue
        remaining[anchor] = False
        neighbors = np.flatnonzero(remaining)
        if len(neighbors) >= size:
            scores = similarities[anchor, neighbors]
            cutoff_index = len(scores) - (size - 1)
            cutoff = np.partition(scores, cutoff_index)[cutoff_index]
            closer = neighbors[scores > cutoff]
            tied = neighbors[scores == cutoff][: size - 1 - len(closer)]
            neighbors = np.concatenate((closer, tied))
        remaining[neighbors] = False
        yield np.concatenate(([anchor], neighbors))


def _fold_gradient_table(gtab, indices, *, preserve_metadata):
    metadata = {}
    if preserve_metadata:
        metadata = {
            "b0_threshold": gtab.b0_threshold,
            "big_delta": gtab.big_delta,
            "small_delta": gtab.small_delta,
            "btens": None if gtab.btens is None else gtab.btens[indices],
        }
    return gt.gradient_table(gtab.bvals[indices], bvecs=gtab.bvecs[indices], **metadata)


def kfold_xval(
    model,
    data,
    folds,
    *model_args,
    split_strategy="random",
    shell_tol=20,
    **model_kwargs,
):
    """Perform k-fold cross-validation.

    It generates out-of-sample predictions for each measurement.

    See :footcite:p:`Rokem2014` for further details about the method.

    Parameters
    ----------
    model : Model class instance
        The type of the model to use for prediction. The corresponding Fit
        object must have a `predict` function implemented One of the following:
        `reconst.dti.TensorModel` or
        `reconst.csdeconv.ConstrainedSphericalDeconvModel`.
    data : ndarray
        Diffusion MRI data acquired with the GradientTable of the model. Shape
        will typically be `(x, y, z, b)` where `xyz` are spatial dimensions and
        b is the number of bvals/bvecs in the GradientTable.
    folds : int
        The number of divisions to apply to the data
    model_args : list
        Additional arguments to the model initialization
    split_strategy : {'random', 'spherical_density'}, optional
        The default randomly splits the DWI measurements into equal-sized folds.
        Spherical-density splitting distributes local groups of directions within
        each b-value shell across folds. It supports uneven counts and balances
        both total and per-shell fold sizes to within one measurement.
    shell_tol : float, optional
        Tolerance in s/mm² for identifying shells with spherical-density splitting.
    model_kwargs : dict
        Additional key-word arguments to the model initialization. If contains
        the kwarg `mask`, this will be used as a key-word argument to the `fit`
        method of the model object, rather than being used in the
        initialization of the model object. Pass `rng` as a `numpy.random.Generator`
        to reproduce a split.

    Notes
    -----
    This function assumes that a prediction API is implemented in the Model
    class for which prediction is conducted. That is, the Fit object that gets
    generated upon fitting the model needs to have a `predict` method, which
    receives a GradientTable class instance as input and produces a predicted
    signal as output.

    The model's gradient table supplies b-values and gradient directions. All
    b0 measurements are used in each fit and their mean fills the b0 positions
    of the returned array. Spherical-density splitting assumes shelled linear
    encoding and treats opposite gradient directions as equivalent axes. Repeated
    measurements of an axis can be assigned to different folds.

    References
    ----------
    .. footbibliography::

    """
    rng = model_kwargs.pop("rng", np.random.default_rng())

    # This should always be there, if the model inherits from
    # dipy.reconst.base.ReconstModel:
    gtab = model.gtab
    b0_indices = np.flatnonzero(gtab.b0s_mask)
    dwi_indices = np.flatnonzero(~gtab.b0s_mask)
    use_density = split_strategy == "spherical_density"
    if use_density:
        if not data.ndim or data.shape[-1] != len(gtab.bvals):
            raise ValueError("The data's last axis must match the gradient table.")
        holdouts = _spherical_density_folds(gtab, folds, shell_tol=shell_tol, rng=rng)
    elif split_strategy == "random":
        # Preserve the existing seeded random permutation.
        div_by_folds = np.mod(dwi_indices.size, folds)
        if div_by_folds != 0:
            raise ValueError(f"np.mod({dwi_indices.size}, {folds}) is {div_by_folds}")
        holdouts = np.split(rng.permutation(dwi_indices.size), folds)
    else:
        raise ValueError("split_strategy must be 'random' or 'spherical_density'.")

    S0 = np.mean(data[..., gtab.b0s_mask], axis=-1)
    prediction = np.zeros(data.shape)

    # Pop the mask, if there is one, out here for use in every fold:
    mask = model_kwargs.pop("mask", None)
    for holdout in holdouts:
        train = np.ones(dwi_indices.size, dtype=bool)
        train[holdout] = False
        train_indices = np.concatenate((b0_indices, dwi_indices[train]))
        held_dwi_indices = dwi_indices[~train]
        test_indices = np.concatenate((b0_indices, held_dwi_indices))
        this_gtab = _fold_gradient_table(
            gtab, train_indices, preserve_metadata=use_density
        )
        left_out_gtab = _fold_gradient_table(
            gtab, test_indices, preserve_metadata=use_density
        )
        this_model = model.__class__(this_gtab, *model_args, **model_kwargs)
        this_fit = this_model.fit(data[..., train_indices], mask=mask)
        if not hasattr(this_fit, "predict"):
            err_str = f"Models of type: {this_model.__class__} "
            err_str += "do not have an implementation of model prediction"
            err_str += " and do not support cross-validation"
            raise ValueError(err_str)
        this_predict = S0[..., None] * this_fit.predict(gtab=left_out_gtab, S0=1)

        prediction[..., held_dwi_indices] = this_predict[..., b0_indices.size :]

    # For the b0 measurements
    prediction[..., gtab.b0s_mask] = S0[..., None]
    return prediction
