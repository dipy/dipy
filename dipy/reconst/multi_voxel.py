"""Tools to easily make multi voxel models"""
import numpy as np
from numpy.lib.stride_tricks import as_strided
from dipy.core.ndindex import ndindex


def multi_voxel_model(SingleVoxelModel):
    """Class decorator to turn a single voxel model into a multi voxel model

    See Also
    --------
    dipy/docs/examples/multiVoxelModel.py

    """
    class MultiVoxelModel(SingleVoxelModel):
        """A subclass of SingleVoxelModel that fits many voxels"""

        def fit(self, data, mask=None):
            """Fit model for every voxel in data"""
            # If only one voxel just return a normal fit
            if data.ndim == 1:
                return SingleVoxelModel.fit(self, data)

            # Make a mask if mask is None
            if mask is None:
                shape = data.shape[:-1]
                strides = (0,) * len(shape)
                mask = as_strided(np.array(True), shape=shape, strides=strides)
            # Check the shape of the mask if mask is not None
            elif mask.shape != data.shape[:-1]:
                raise ValueError("mask and data shape do not match")

            # Fit data where mask is True
            fit_array = np.empty(data.shape[:-1], dtype=object)
            for ijk in ndindex(data.shape[:-1]):
                if mask[ijk]:
                    fit_array[ijk] = SingleVoxelModel.fit(self, data[ijk])
            return MultiVoxelFit(self, fit_array, mask)

    return MultiVoxelModel


class MultiVoxelFit(object):
    """Holds an array of fits and allows access to their attributes and
    methods"""
    def __init__(self, model, fit_array, mask):
        self.model = model
        self.fit_array = fit_array
        self.mask = mask

    @property
    def shape(self):
        return self.fit_array.shape

    def __getattr__(self, attr):
        result = CallableArray(self.fit_array.shape, dtype=object)
        for ijk in ndindex(result.shape):
            if self.mask[ijk]:
                result[ijk] = getattr(self.fit_array[ijk], attr)
        return _squash(result, self.mask)

    def __getitem__(self, index):
        item = self.fit_array[index]
        if isinstance(item, np.ndarray):
            return MultiVoxelFit(self.model, item, self.mask[index])
        else:
            return item


class CallableArray(np.ndarray):
    """An array which can be called like a function"""
    def __call__(self, *args, **kwargs):
        result = np.empty(self.shape, dtype=object)
        for ijk in ndindex(self.shape):
            item = self[ijk]
            if item is not None:
                result[ijk] = item(*args, **kwargs)
        return _squash(result)


def _squash(arr, mask=None, fill=0):
    """Try and make a standard array from an object array

    This function takes an object array and attempts to convert it to a more
    useful dtype. If array can be converted to a better dtype, Nones are
    replaced by `fill`. To make the behaviour of this function more clear, here
    are the most common cases:

    1.  `arr` is an array of scalars of type `T`. Returns an array like
        `arr.astype(T)`
    2.  `arr` is an array of arrays. All items in `arr` have the same shape
        `S`. Returns an array with shape `arr.shape + S`.
    3.  `arr` is an array of arrays of different shapes. Returns `arr`.
    4.  Items in `arr` are not ndarrys or scalars. Returns `arr`.

    Parameters
    ----------
    arr : array, dtype=object
        The array to be converted.
    mask : array, dtype=bool, optional
        Where arr has Nones.
    fill : number, optional
        Nones are replaced by fill.

    Returns
    -------
    result : array

    Examples
    --------
    >>> arr = np.empty(3, dtype=object)
    >>> arr.fill(2)
    >>> _squash(arr)
    array([2, 2, 2])
    >>> arr[0] = None
    >>> _squash(arr)
    array([0, 2, 2])
    >>> arr.fill(np.ones(2))
    >>> r = _squash(arr)
    >>> r.shape
    (3, 2)
    >>> r.dtype
    dtype('float64')

    """
    if mask is None:
        mask = arr != np.array(None)
    not_none = arr[mask]
    # all None, just return arr
    if not_none.size == 0:
        return arr
    first = not_none[0]
    # If the first item is an ndarray
    if type(first) is np.ndarray:
        shape = first.shape
        try:
            # Check the shapes of all items
            all_same_shape = all(item.shape == shape for item in not_none)
        except AttributeError:
            return arr
        # If items have different shapes just return arr
        if not all_same_shape:
            return arr
        # Find common dtype.  np.result_type can do this more simply, but it is
        # only available for numpy 1.6.0
        dtypes = set(a.dtype for a in not_none)
        tiny_arrs = [np.zeros((1,), dtype=dt) for dt in dtypes]
        dtype = reduce(np.add, tiny_arrs).dtype
        # Create output array and fill
        result = np.empty(arr.shape + shape, dtype=dtype)
        result.fill(fill)
        for ijk in ndindex(arr.shape):
            if arr[ijk] is not None:
                result[ijk] = arr[ijk]
        return result

    # If the first item is a scalar
    elif np.isscalar(first):
        "first is not an ndarray"
        all_scalars = all(np.isscalar(item) for item in not_none)
        if not all_scalars:
            return
        # See comment about np.result_type above
        dtype = reduce(np.add, not_none).dtype
        temp = arr.copy()
        temp[~mask] = fill
        return temp.astype(dtype)
    else:
        return arr
