"""Error metrics for observed and predicted reconstruction signals."""

import numpy as np
from numpy.lib.array_utils import normalize_axis_index


def _validate_signals(observed, predicted, axis):
    observed = np.asarray(observed)
    predicted = np.asarray(predicted)
    if observed.shape != predicted.shape:
        raise ValueError("Observed and predicted signals must have identical shapes.")
    if isinstance(axis, (bool, np.bool_)):
        raise TypeError("axis must be an integer.")
    axis = normalize_axis_index(axis, observed.ndim)
    if observed.size == 0:
        raise ValueError("Observed and predicted signals must be nonempty.")
    if observed.dtype.kind not in "iuf" or predicted.dtype.kind not in "iuf":
        raise TypeError("Observed and predicted signals must contain real numbers.")
    if not (np.isfinite(observed).all() and np.isfinite(predicted).all()):
        raise ValueError("Observed and predicted signals must contain finite values.")

    dtype = np.result_type(observed, predicted, np.float64)
    return observed.astype(dtype, copy=False), predicted.astype(dtype, copy=False), axis


def _mean_magnitude(values, axis, *, squared):
    magnitude = np.abs(values)
    peak = magnitude.max(axis=axis, keepdims=True)
    magnitude /= np.where(peak == 0, 1, peak)
    if squared:
        np.square(magnitude, out=magnitude)
    mean = magnitude.mean(axis=axis)
    if squared:
        mean = np.sqrt(mean)
    return peak.squeeze(axis=axis) * mean


def rmse(observed, predicted, *, axis=-1):
    """Compute the root mean squared error in the signal units.

    Parameters
    ----------
    observed, predicted : array_like
        Equally shaped, nonempty, finite real signals in the same units.
    axis : int, optional
        Measurement axis, defaulting to the last.

    Returns
    -------
    error : floating or ndarray
        Root mean squared residual with the measurement axis removed.

    Notes
    -----
    Calculations use at least double precision.

    Examples
    --------
    >>> from dipy.reconst.metrics import rmse
    >>> float(rmse([1, 3], [2, 4]))
    1.0
    """
    observed, predicted, axis = _validate_signals(observed, predicted, axis)
    return _mean_magnitude(observed - predicted, axis, squared=True)


def nrmse(observed, predicted, *, normalization="rms", axis=-1):
    """Divide RMSE by a scale computed from the observed signal.

    Parameters
    ----------
    observed, predicted : array_like
        Equally shaped, nonempty, finite real signals in the same units.
    normalization : {'rms', 'mean_absolute'}, optional
        Observed root mean square (default) or mean absolute magnitude.
    axis : int, optional
        Measurement axis, defaulting to the last.

    Returns
    -------
    error : floating or ndarray
        Dimensionless error with the measurement axis removed. A zero scale
        gives NaN.

    Examples
    --------
    >>> from dipy.reconst.metrics import nrmse
    >>> float(nrmse([1, 3], [2, 4], normalization="mean_absolute"))
    0.5
    """
    if not isinstance(normalization, str) or normalization not in (
        "rms",
        "mean_absolute",
    ):
        raise ValueError("normalization must be 'rms' or 'mean_absolute'.")

    observed, predicted, axis = _validate_signals(observed, predicted, axis)
    error = _mean_magnitude(observed - predicted, axis, squared=True)
    scale = _mean_magnitude(observed, axis, squared=normalization == "rms")
    result = np.full_like(error, np.nan)
    np.divide(error, scale, out=result, where=scale != 0)
    return result[()]
