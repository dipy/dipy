import multiprocessing as mp
import warnings

import numpy as np
from scipy.ndimage import affine_transform

from dipy.testing.decorators import warning_for_keywords
from dipy.utils.multiproc import determine_num_processes


def _affine_transform(kwargs):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*scipy.*18.*", category=UserWarning)
        return affine_transform(**kwargs)


def _lanczos_kernel(values, num_lobes):
    """Lanczos kernel: sinc(x) * sinc(x/a) for |x| < a, else 0.

    Parameters
    ----------
    values : array-like
        Input values.
    num_lobes : int
        Kernel support half-width. Values where ``|x| >= num_lobes`` are
        set to 0 regardless of position in the array. Typically 2 or 3.

    Returns
    -------
    result : ndarray
        Kernel values at values.
    """
    values = np.asarray(values, dtype=np.float64)
    mask = np.abs(values) < num_lobes
    return np.where(mask, np.sinc(values) * np.sinc(values / num_lobes), 0.0)


def _lanczos_resample_1d_axis0(
    data, scale, out_size, *, num_lobes=2, mode="constant", cval=0
):
    """Apply 1D Lanczos resampling along axis 0 of an N-dimensional array.

    Parameters
    ----------
    data : ndarray, shape (in_size, ...)
        Input array. Must be float64.
    scale : float
        Ratio new_zoom / old_zoom for this axis.
    out_size : int
        Number of output samples along axis 0.
    num_lobes : int, optional
        Lanczos kernel lobes (2 or 3).
    mode : str, optional
        Boundary mode: 'constant', 'nearest', 'reflect', or 'wrap'.
    cval : float, optional
        Fill value for mode='constant'.

    Returns
    -------
    out : ndarray, shape (out_size, ...), float64
        Resampled array.
    """
    valid_modes = ("constant", "nearest", "reflect", "wrap")
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode {mode!r}. Valid modes: {valid_modes}")

    in_size = data.shape[0]
    extra_dims = data.ndim - 1
    bc_shape = (-1,) + (1,) * extra_dims

    out_idx = np.arange(out_size, dtype=np.float64)
    in_coords = out_idx * scale
    floor_coords = np.floor(in_coords).astype(np.intp)

    result = np.zeros((out_size,) + data.shape[1:], dtype=np.float64)
    weight_sum = np.zeros(out_size, dtype=np.float64)

    for delta in range(-(num_lobes - 1), num_lobes + 1):
        neigh = floor_coords + delta
        w = _lanczos_kernel(in_coords - neigh, num_lobes)

        if mode == "wrap":
            sample = data[np.mod(neigh, in_size)]
        elif mode == "reflect":
            period = 2 * (in_size - 1) if in_size > 1 else 1
            idx = np.mod(neigh, period)
            idx = np.where(idx >= in_size, period - idx, idx)
            sample = data[idx]
        else:
            sample = data[np.clip(neigh, 0, in_size - 1)]
            if mode == "constant":
                invalid = (neigh < 0) | (neigh >= in_size)
                if invalid.any():
                    sample = sample.copy()
                    sample[invalid] = float(cval)

        result += w.reshape(bc_shape) * sample
        weight_sum += w

    nonzero = (np.abs(weight_sum) > 1e-10).reshape(bc_shape)
    weight_sum_bc = weight_sum.reshape(bc_shape)
    fill = float(cval) if mode == "constant" else 0.0
    return np.where(nonzero, result / np.where(nonzero, weight_sum_bc, 1.0), fill)


def _lanczos_resample_3d(
    data, scale, output_shape, *, num_lobes=2, mode="constant", cval=0
):
    """Resample a 3D volume using separable Lanczos interpolation.

    Applies 1D Lanczos resampling along each spatial axis in sequence.
    Separability of the Lanczos kernel makes this equivalent to the full
    3D tensor-product kernel at a fraction of the cost.

    Parameters
    ----------
    data : ndarray, shape (I, J, K)
        Input 3D volume.
    scale : ndarray, shape (3,)
        Scaling factors per axis (new_zooms / zooms).
    output_shape : tuple of int
        Shape of the output volume.
    num_lobes : int, optional
        Lanczos kernel lobes (2 or 3).
    mode : str, optional
        Boundary mode: 'constant', 'nearest', 'reflect', or 'wrap'.
    cval : float, optional
        Fill value for mode='constant'.

    Returns
    -------
    out : ndarray, shape output_shape, dtype matches data.dtype
        Resampled volume.
    """
    vol = data.astype(np.float64)
    for axis in range(3):
        vol = np.moveaxis(vol, axis, 0)
        vol = _lanczos_resample_1d_axis0(
            vol,
            scale[axis],
            output_shape[axis],
            num_lobes=num_lobes,
            mode=mode,
            cval=cval,
        )
        vol = np.moveaxis(vol, 0, axis)
    return vol.astype(data.dtype)


@warning_for_keywords()
def reslice(
    data,
    affine,
    zooms,
    new_zooms,
    *,
    order=1,
    mode="constant",
    cval=0,
    num_processes=1,
    new_shape=None,
):
    """Reslice data with new voxel resolution defined by ``new_zooms``.

    Parameters
    ----------
    data : array, shape (I,J,K) or (I,J,K,N)
        3d volume or 4d volume with datasets
    affine : array, shape (4,4)
        mapping from voxel coordinates to world coordinates
    zooms : tuple, shape (3,)
        voxel size for (i,j,k) dimensions
    new_zooms : tuple, shape (3,)
        new voxel size for (i,j,k) after resampling
    order : int or str
        Interpolation order. Integer 0–5 selects spline order via scipy
        (0 nearest, 1 trilinear, …). String values ``'lanczos'`` or
        ``'lanczos2'`` select a 2-lobe Lanczos kernel; ``'lanczos3'``
        selects a 3-lobe Lanczos kernel.
    mode : string ('constant', 'nearest', 'reflect' or 'wrap')
        Points outside the boundaries of the input are filled according
        to the given mode.
    cval : float
        Value used for points outside the boundaries of the input if
        mode='constant'.
    num_processes : int, optional
        Split the calculation to a pool of children processes. This only
        applies to 4D `data` arrays. Default is 1. If < 0 the maximal number
        of cores minus ``num_processes + 1`` is used (enter -1 to use as many
        cores as possible). 0 raises an error. Ignored when order is
        ``'lanczos'``, ``'lanczos2'``, or ``'lanczos3'``.
    new_shape : tuple, shape (3,), optional
        Sets the shape the image should take after affine transformation.
        If None, it is calculated through the affine matrix and current shape.


    Returns
    -------
    data2 : array, shape (I,J,K) or (I,J,K,N)
        datasets resampled into isotropic voxel size
    affine2 : array, shape (4,4)
        new affine for the resampled image

    Examples
    --------
    >>> from dipy.io.image import load_nifti
    >>> from dipy.align.reslice import reslice
    >>> from dipy.data import get_fnames
    >>> f_name = get_fnames(name="aniso_vox")
    >>> data, affine, zooms = load_nifti(f_name, return_voxsize=True)
    >>> data.shape == (58, 58, 24)
    True
    >>> zooms
    (4.0, 4.0, 5.0)
    >>> new_zooms = (3.,3.,3.)
    >>> new_zooms
    (3.0, 3.0, 3.0)
    >>> data2, affine2 = reslice(data, affine, zooms, new_zooms)
    >>> data2.shape == (77, 77, 40)
    True

    """
    num_processes = determine_num_processes(num_processes)

    lanczos_lobes = None
    if isinstance(order, str):
        if order in ("lanczos", "lanczos2"):
            lanczos_lobes = 2
        elif order == "lanczos3":
            lanczos_lobes = 3
        else:
            raise ValueError(
                f"Unknown interpolation order {order!r}. "
                "Valid string values: 'lanczos', 'lanczos2', 'lanczos3'."
            )
    elif not (0 <= order <= 5):
        raise ValueError(f"order must be 0-5, got {order!r}.")

    _valid_modes_lanczos = ("constant", "nearest", "reflect", "wrap")
    _valid_modes_scipy = ("constant", "nearest", "reflect", "mirror", "wrap")
    _valid_modes = (
        _valid_modes_lanczos if lanczos_lobes is not None else _valid_modes_scipy
    )
    if mode not in _valid_modes:
        raise ValueError(f"Invalid mode {mode!r}. Valid modes: {_valid_modes}")

    # We are suppressing warnings emitted by scipy >= 0.18,
    # described in https://github.com/dipy/dipy/issues/1107.
    # These warnings are not relevant to us, as long as our offset
    # input to scipy's affine_transform is [0, 0, 0]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*scipy.*18.*", category=UserWarning)
        new_zooms = np.array(new_zooms, dtype="f8")
        zooms = np.array(zooms, dtype="f8")
        scale = new_zooms / zooms
        if new_shape is None:
            new_shape = zooms / new_zooms * np.array(data.shape[:3])
            new_shape = tuple(np.round(new_shape).astype("i8"))

        if data.ndim not in (3, 4):
            raise ValueError(
                f"dimension of data should be 3 or 4 but you provided {data.ndim}"
            )

        if lanczos_lobes is not None:
            if data.ndim == 3:
                data2 = _lanczos_resample_3d(
                    data,
                    scale,
                    new_shape,
                    num_lobes=lanczos_lobes,
                    mode=mode,
                    cval=cval,
                )
            else:
                data2 = np.zeros(new_shape + (data.shape[-1],), data.dtype)
                for vol_idx in range(data.shape[-1]):
                    data2[..., vol_idx] = _lanczos_resample_3d(
                        data[..., vol_idx],
                        scale,
                        new_shape,
                        num_lobes=lanczos_lobes,
                        mode=mode,
                        cval=cval,
                    )
        else:
            kwargs = {
                "matrix": scale,
                "output_shape": new_shape,
                "order": order,
                "mode": mode,
                "cval": cval,
            }
            if data.ndim == 3:
                data2 = affine_transform(input=data, **kwargs)
            else:
                data2 = np.zeros(new_shape + (data.shape[-1],), data.dtype)
                if num_processes == 1:
                    for i in range(data.shape[-1]):
                        affine_transform(
                            input=data[..., i], output=data2[..., i], **kwargs
                        )
                else:
                    params = []
                    for i in range(data.shape[-1]):
                        _kwargs = {"input": data[..., i]}
                        _kwargs.update(kwargs)
                        params.append(_kwargs)
                    mp.set_start_method("spawn", force=True)
                    pool = mp.Pool(num_processes)
                    for i, res in enumerate(pool.imap(_affine_transform, params)):
                        data2[..., i] = res
                    pool.close()

        affine_scale = np.eye(4)
        affine_scale[:3, :3] = np.diag(scale)
        affine2 = np.dot(affine, affine_scale)
    return (data2, affine2)
