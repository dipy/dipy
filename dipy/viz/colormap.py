"""FURY colormaps plus DIPY-specific colormap and lookup-table helpers."""

import numpy as np

from dipy.utils.optpkg import optional_package

fury, has_fury, _ = optional_package("fury", min_version="2.1.0")
if has_fury:
    from fury.colormap import *
    from fury.colormap import create_colormap as fury_create_colormap, hsv2rgb
else:
    fury_create_colormap = hsv2rgb = fury


# TODO(fury#1393): ``fury.colormap.create_colormap`` only accepts colormap
# names. Remove the ``hsv_range`` option, and this wrapper, once
# https://github.com/fury-gl/fury/issues/1393 adds custom colormap support.
def create_colormap(v, *, name="plasma", auto=True, hsv_range=None):
    """Create RGB colors from a named colormap or from HSV ramps.

    Parameters
    ----------
    v : array_like
        Values mapped to colors.
    name : str, optional
        Name of the colormap, as accepted by ``fury.colormap.create_colormap``.
        Ignored when ``hsv_range`` is given.
    auto : bool, optional
        If True, ``v`` is rescaled to [0, 1] from ``v.min()`` to ``v.max()``.
        Otherwise ``v`` is clipped to [0, 1].
    hsv_range : tuple of three (low, high) pairs, optional
        Hue, saturation and value ramps, each in [0, 1], from ``v = 0`` to
        ``v = 1``. Replaces ``name`` when given.

    Returns
    -------
    ndarray of shape (..., 3)
        RGB colors of ``v``.

    Raises
    ------
    ValueError
        If ``hsv_range`` does not have three (low, high) pairs.
    """
    if hsv_range is None:
        return fury_create_colormap(v, name=name, auto=auto)
    if len(hsv_range) != 3:
        raise ValueError(
            f"hsv_range must hold hue, saturation and value ramps, got {hsv_range}."
        )
    v = np.asarray(v, dtype=float)
    if auto:
        v = np.interp(v, [v.min(), v.max()], [0, 1])
    else:
        v = np.clip(v, 0, 1)
    hsv = np.stack([np.interp(v, [0, 1], ramp) for ramp in hsv_range], axis=-1)
    return hsv2rgb(hsv)


def scalars_to_lut_indices(values, clim, n_colors):
    """Map scalar values to the nearest index of an evenly spaced lookup table.

    Parameters
    ----------
    values : array_like
        Scalar values. Values outside ``clim``, including infinities, are
        clipped to the first or last index.
    clim : tuple of float
        Finite scalar values ``(low, high)`` mapped to the first and last index.
    n_colors : int
        Number of entries of the lookup table.

    Returns
    -------
    ndarray of int
        Lookup-table indices, with the shape of ``values``.

    Raises
    ------
    ValueError
        If ``clim`` is not finite or ``low > high``, if ``n_colors`` is lower
        than 1, or if ``values`` contains NaN.

    Notes
    -----
    When ``low == high`` every value is mapped to the first index.
    """
    low, high = clim
    if not (np.isfinite(low) and np.isfinite(high) and low <= high):
        raise ValueError(f"clim must be finite with low <= high, got {clim}.")
    if n_colors < 1:
        raise ValueError(f"n_colors must be at least 1, got {n_colors}.")
    values = np.asarray(values, dtype=float)
    if np.isnan(values).any():
        raise ValueError("values must not contain NaN.")
    if low == high:
        return np.zeros(values.shape, dtype=np.intp)
    fraction = np.clip((values - low) / (high - low), 0, 1)
    return np.rint(fraction * (n_colors - 1)).astype(np.intp)


def map_scalars_to_lut(values, clim, lut):
    """Map scalar values to the nearest color of an evenly spaced lookup table.

    Parameters
    ----------
    values : array_like
        Scalar values. Values outside ``clim`` are clipped.
    clim : tuple of float
        Finite scalar values ``(low, high)`` mapped to the first and last color.
    lut : ndarray of shape (M, 3)
        RGB colors ordered from low to high values.

    Returns
    -------
    ndarray of shape (..., 3)
        RGB colors of ``values``.

    Raises
    ------
    ValueError
        If ``clim`` is not finite or ``low > high``, if ``lut`` is empty, or if
        ``values`` contains NaN.

    See Also
    --------
    scalars_to_lut_indices
    """
    return lut[scalars_to_lut_indices(values, clim, len(lut))]
