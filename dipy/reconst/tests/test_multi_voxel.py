from functools import reduce

import numpy as np
import numpy.testing as npt

from dipy.reconst.multi_voxel import _squash, multi_voxel_fit, CallableArray
from dipy.core.sphere import unit_icosahedron
from dipy.testing.decorators import set_random_number_generator


def test_squash():
    A = np.ones((3, 3), dtype=float)
    B = np.asarray(A, object)
    npt.assert_array_equal(A, _squash(B))
    npt.assert_equal(_squash(B).dtype, A.dtype)

    B[2, 2] = None
    A[2, 2] = 0
    npt.assert_array_equal(A, _squash(B))
    npt.assert_equal(_squash(B).dtype, A.dtype)

    for ijk in np.ndindex(*B.shape):
        B[ijk] = np.ones((2,))
    A = np.ones((3, 3, 2))
    npt.assert_array_equal(A, _squash(B))
    npt.assert_equal(_squash(B).dtype, A.dtype)

    B[2, 2] = None
    A[2, 2] = 0
    npt.assert_array_equal(A, _squash(B))
    npt.assert_equal(_squash(B).dtype, A.dtype)

    # sub-arrays have different shapes ( (3,) and (2,) )
    B[0, 0] = np.ones((3,))
    npt.assert_(_squash(B) is B)

    # Check dtypes for arrays and scalars
    arr_arr = np.zeros((2,), dtype=object)
    scalar_arr = np.zeros((2,), dtype=object)
    numeric_types = [getattr(np, dtype) for dtype in (
        'int8', 'byte', 'int16', 'short', 'int32', 'intc', 'int_', 'int64',
        'longlong', 'uint8', 'ubyte', 'uint16', 'ushort', 'uint32', 'uintc',
        'uint', 'uint64', 'ulonglong', 'float16', 'half', 'float32', 'single',
        'float64', 'double', 'float96', 'float128', 'longdouble', 'complex64',
        'csingle', 'complex128', 'cdouble', 'complex192', 'complex256',
        'clongdouble') if hasattr(np, dtype)] + [bool]
    for dt0 in numeric_types:
        arr_arr[0] = np.zeros((3,), dtype=dt0)
        scalar_arr[0] = dt0(0)
        for dt1 in numeric_types:
            arr_arr[1] = np.zeros((3,), dtype=dt1)
            npt.assert_equal(_squash(arr_arr).dtype,
                             reduce(np.add, arr_arr).dtype)
            scalar_arr[1] = dt0(1)
            npt.assert_equal(_squash(scalar_arr).dtype,
                             reduce(np.add, scalar_arr).dtype)

    # Check masks and Nones
    arr = np.ones((3, 4), dtype=float)
    obj_arr = arr.astype(object)
    arr[1, 1] = 99
    obj_arr[1, 1] = None
    npt.assert_array_equal(_squash(obj_arr, mask=None, fill=99), arr)
    msk = arr == 1
    npt.assert_array_equal(_squash(obj_arr, mask=msk, fill=99), arr)
    msk[1, 1] = 1  # unmask None - object array back
    npt.assert_array_equal(_squash(obj_arr, mask=msk, fill=99), obj_arr)
    msk[1, 1] = 0  # remask, back to fill again
    npt.assert_array_equal(_squash(obj_arr, mask=msk, fill=99), arr)
    obj_arr[2, 3] = None  # add another unmasked None, object again
    npt.assert_array_equal(_squash(obj_arr, mask=msk, fill=99), obj_arr)

    # Check array of arrays
    obj_arrs = np.zeros((3,), dtype=object)
    for i in range(3):
        obj_arrs[i] = np.ones((4, 5))
    arr_arrs = np.ones((3, 4, 5))
    # No Nones
    npt.assert_array_equal(_squash(obj_arrs, mask=None, fill=99), arr_arrs)
    # None, implicit masking
    obj_masked = obj_arrs.copy()
    obj_masked[1] = None
    arr_masked = arr_arrs.copy()
    arr_masked[1] = 99
    npt.assert_array_equal(_squash(obj_masked, mask=None, fill=99),
                           arr_masked)
    msk = np.array([1, 0, 1], dtype=bool)  # explicit mask
    npt.assert_array_equal(_squash(obj_masked, mask=msk, fill=99),
                           arr_masked)


def test_CallableArray():
    callarray = CallableArray((2, 3), dtype=object)

    # Test without Nones
    callarray[:] = np.arange
    expected = np.empty([2, 3, 4])
    expected[:] = range(4)
    npt.assert_array_equal(callarray(4), expected)

    # Test with Nones
    callarray[0, 0] = None
    expected[0, 0] = 0
    npt.assert_array_equal(callarray(4), expected)


@set_random_number_generator()
def test_multi_voxel_fit(rng):

    class SillyModel:

        @multi_voxel_fit
        def fit(self, data, mask=None):
            return SillyFit(model, data)

        def predict(self, S0):
            return np.ones(10) * S0

    class SillyFit:

        def __init__(self, model, data):
            self.model = model
            self.data = data

        model_attr = 2.

        def odf(self, sphere):
            return np.ones(len(sphere.phi))

        @property
        def directions(self):
            n = rng.integers(0, 10)
            return np.zeros((n, 3))

        def predict(self, S0):
            return np.ones(self.data.shape) * S0

    # Test the single voxel case
    model = SillyModel()
    single_voxel = np.zeros(64)
    fit = model.fit(single_voxel)
    npt.assert_equal(type(fit), SillyFit)

    # Test without a mask
    many_voxels = np.zeros((2, 3, 4, 64))
    fit = model.fit(many_voxels)
    expected = np.empty((2, 3, 4))
    expected[:] = 2.
    npt.assert_array_equal(fit.model_attr, expected)
    expected = np.ones((2, 3, 4, 12))
    npt.assert_array_equal(fit.odf(unit_icosahedron), expected)
    npt.assert_equal(fit.directions.shape, (2, 3, 4))
    S0 = 100.
    npt.assert_equal(fit.predict(S0=S0), np.ones(many_voxels.shape) * S0)

    # Test with a mask
    mask = np.zeros((3, 3, 3)).astype('bool')
    mask[0, 0] = 1
    mask[1, 1] = 1
    mask[2, 2] = 1
    data = np.zeros((3, 3, 3, 64))
    fit = model.fit(data, mask)
    expected = np.zeros((3, 3, 3))
    expected[0, 0] = 2
    expected[1, 1] = 2
    expected[2, 2] = 2
    npt.assert_array_equal(fit.model_attr, expected)
    odf = fit.odf(unit_icosahedron)
    npt.assert_equal(odf.shape, (3, 3, 3, 12))
    npt.assert_array_equal(odf[~mask], 0)
    npt.assert_array_equal(odf[mask], 1)
    predicted = np.zeros(data.shape)
    predicted[mask] = S0
    npt.assert_equal(fit.predict(S0=S0), predicted)

    # Test fit.shape
    npt.assert_equal(fit.shape, (3, 3, 3))

    # Test indexing into a fit
    npt.assert_equal(type(fit[0, 0, 0]), SillyFit)
    npt.assert_equal(fit[:2, :2, :2].shape, (2, 2, 2))
