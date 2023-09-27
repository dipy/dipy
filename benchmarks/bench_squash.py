""" Benchmarks for fast squashing

Run all benchmarks with::

    import dipy.reconst as dire
    dire.bench()

With Pytest, Run this benchmark with:

    pytest -svv -c bench.ini /path/to/bench_squash.py
"""

from functools import reduce

import numpy as np

from dipy.core.ndindex import ndindex

from numpy.testing import measure


def old_squash(arr, mask=None, fill=0):
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
    >>> old_squash(arr)
    array([2, 2, 2])
    >>> arr[0] = None
    >>> old_squash(arr)
    array([0, 2, 2])
    >>> arr.fill(np.ones(2))
    >>> r = old_squash(arr)
    >>> r.shape == (3, 2)
    True
    >>> r.dtype
    dtype('float64')
    """
    if mask is None:
        mask = np.vectorize(lambda x : x is not None)(arr)
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
            if mask[ijk]:
                result[ijk] = arr[ijk]
        return result

    # If the first item is a scalar
    elif np.isscalar(first):
        "first is not an ndarray"
        all_scalars = all(np.isscalar(item) for item in not_none)
        if not all_scalars:
            return arr
        # See comment about np.result_type above. We sum against the smallest
        # possible type, bool, and let numpy type promotion find the best
        # common type. The values might all be Python scalars so we need to
        # cast to numpy type at the end to be sure of having a dtype.
        dtype = np.asarray(sum(not_none, False)).dtype
        temp = arr.copy()
        temp[~mask] = fill
        return temp.astype(dtype)
    else:
        return arr


def bench_quick_squash():
    repeat = 10
    shape = (300, 200)
    arrs = np.zeros(shape, dtype=object)
    scalars = np.zeros(shape, dtype=object)
    for ijk in ndindex(arrs.shape):
        arrs[ijk] = np.ones((3, 5))
        scalars[ijk] = np.float32(0)
    print('\nSquashing benchmarks')
    for name, objs in (
        ('floats', np.zeros(shape, float).astype(object)),
        ('ints', np.zeros(shape, int).astype(object)),
        ('arrays', arrs),
        ('scalars', scalars),
    ):
        print(name)
        timed0 = measure("quick_squash(objs)", repeat)
        timed1 = measure("old_squash(objs)", repeat)
        print("fast %4.2f; slow %4.2f" % (timed0, timed1))
        objs[50, 50] = None
        timed0 = measure("quick_squash(objs)", repeat)
        timed1 = measure("old_squash(objs)", repeat)
        print("With None: fast %4.2f; slow %4.2f" % (timed0, timed1))
        timed0 = measure("quick_squash(objs, msk)", repeat)
        timed1 = measure("old_squash(objs, msk)", repeat)
        print("With mask: fast %4.2f; slow %4.2f" % (timed0, timed1))
        objs[50, 50] = np.float32(0)
        timed0 = measure("quick_squash(objs, msk)", repeat)
        timed1 = measure("old_squash(objs, msk)", repeat)
        print("Other dtype: fast %4.2f; slow %4.2f" % (timed0, timed1))
