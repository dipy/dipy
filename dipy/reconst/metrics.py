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
    # Rescaling avoids overflow or underflow when squaring large or small values.
    magnitude /= np.where(peak == 0, 1, peak)
    if squared:
        np.square(magnitude, out=magnitude)
    mean = magnitude.mean(axis=axis)
    if squared:
        mean = np.sqrt(mean)
    return peak.squeeze(axis=axis) * mean


def rmse(observed, predicted, *, axis=-1):
    """Compute the root mean squared error between paired signals.

    Parameters
    ----------
    observed, predicted : array_like
        Nonempty, real numeric signals with identical shapes. Corresponding
        entries must represent the same measurements in the same units.
    axis : int, optional
        Measurement axis to reduce. The default is the last axis.

    Returns
    -------
    error : floating or ndarray
        Square root of the mean squared residual, in the input signal units.
        The output shape is the input shape with `axis` removed. A 1D input
        gives a scalar.

    Raises
    ------
    ValueError
        If the shapes differ, the arrays are empty, or either contains NaN or
        infinity.
    TypeError
        If either signal has a non-real-numeric dtype or `axis` is not an integer.
    numpy.exceptions.AxisError
        If `axis` is out of bounds.

    Notes
    -----
    Calculations use at least double precision, including for integer inputs.

    Examples
    --------
    >>> from dipy.reconst.metrics import rmse
    >>> float(rmse([1, 3], [2, 4]))
    1.0
    """
    observed, predicted, axis = _validate_signals(observed, predicted, axis)
    return _mean_magnitude(observed - predicted, axis, squared=True)


def nrmse(observed, predicted, *, normalization="rms", axis=-1):
    """Compute the normalized root mean squared error between paired signals.

    Parameters
    ----------
    observed, predicted : array_like
        Nonempty, real numeric signals with identical shapes. Corresponding
        entries must represent the same measurements in the same units.
    normalization : {'rms', 'mean_absolute'}, optional
        Scale computed from the observed signal along `axis`. ``'rms'`` is
        the root mean square (the default). ``'mean_absolute'`` is the mean
        absolute magnitude and equals the mean for nonnegative observations.
    axis : int, optional
        Measurement axis to reduce. The default is the last axis.

    Returns
    -------
    error : floating or ndarray
        Dimensionless ratio with the input shape minus `axis`. A 1D input
        gives a scalar. A zero normalization scale gives NaN, even for an
        exact prediction.

    Raises
    ------
    ValueError
        If the shapes differ, the arrays are empty, either contains NaN or
        infinity, or `normalization` is unsupported.
    TypeError
        If either signal has a non-real-numeric dtype or `axis` is not an integer.
    numpy.exceptions.AxisError
        If `axis` is out of bounds.

    See Also
    --------
    rmse : Root mean squared error in the original signal units.

    Notes
    -----
    RMSE is computed in the original signal units, then divided by the
    observed scale. Both terms use the same measurements. Calculations use
    at least double precision.

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
