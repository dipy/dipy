import numpy as np
import numpy.testing as npt
from dipy.reconst.multi_voxel import _squash, multi_voxel_model, CallableArray
from dipy.reconst.shm import QballOdfModel
from dipy.core.sphere import unit_icosahedron


def test_squash():
    A = np.ones((3, 3), dtype=float)
    B = np.asarray(A, object)
    npt.assert_array_equal(A, _squash(B))

    B[2, 2] = None
    A[2, 2] = 0
    npt.assert_array_equal(A, _squash(B))

    for ijk in np.ndindex(*B.shape):
        B[ijk] = np.ones((2,))
    A = np.ones((3, 3, 2))
    npt.assert_array_equal(A, _squash(B))

    B[2, 2] = None
    A[2, 2] = 0
    npt.assert_array_equal(A, _squash(B))


def test_CallableArray():
    callarray = CallableArray((2, 3), dtype=object)

    # Test without Nones
    callarray[:] = range
    expected = np.empty([2, 3, 4])
    expected[:] = range(4)
    npt.assert_array_equal(callarray(4), expected)

    # Test with Nones
    callarray[0, 0] = None
    expected[0, 0] = 0
    npt.assert_array_equal(callarray(4), expected)


def test_multi_voxel_model():

    class SillyModel(object):

        def fit(self, data, mask=None):
            return SillyFit(model)

    class SillyFit(object):

        def __init__(self, model):
            self.model = model

        model_attr = 2.

        def odf(self, sphere):
            return np.ones(len(sphere.phi))

        @property
        def directions(self):
            n = np.random.randint(0, 10)
            return np.zeros((n, 3))

    # Wrap the SillyModel
    MultiVoxelSillyModel = multi_voxel_model(SillyModel)

    # Test the single voxel case
    model = MultiVoxelSillyModel()
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

    # Test with a mask
    mask = np.eye(3).astype('bool')
    data = np.zeros((3, 3, 64))
    fit = model.fit(data, mask)
    npt.assert_array_equal(fit.model_attr, np.eye(3)*2)
    odf = fit.odf(unit_icosahedron)
    npt.assert_equal(odf.shape, (3, 3, 12))
    npt.assert_array_equal(odf[~mask], 0)
    npt.assert_array_equal(odf[mask], 1)

    # Test fit.shape
    npt.assert_equal(fit.shape, (3, 3))

    # Test indexing into a fit
    npt.assert_equal(type(fit[0, 0]), SillyFit)
    npt.assert_equal(fit[:2, :2].shape, (2, 2))


