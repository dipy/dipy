"""Testing volumes"""

import numpy as np
import numpy.testing as npt
import pytest

from dipy.utils.volume import adjacency_calc, select_3d_volume


def test_adjacency_calc():
    """
    Test adjacency_calc function, which calculates indices of adjacent voxels
    """

    cutoff = 1.99
    for img_shape in [(50, 50), (50, 50, 5)]:
        mask = None
        adj = adjacency_calc(img_shape, mask=mask, cutoff=cutoff)
        # check that adj in the first voxel is correct
        adj[0].sort()
        if len(img_shape) == 2:
            npt.assert_equal(adj[0], [0, 1, 50, 51])
        if len(img_shape) == 3:
            npt.assert_equal(adj[0], [0, 1, 5, 6, 250, 251, 255, 256])

        mask = np.zeros(img_shape, dtype=int)
        if len(img_shape) == 2:
            mask[10:40, 20:30] = 1
        if len(img_shape) == 3:
            mask[10:40, 20:30, :] = 1
        adj = adjacency_calc(img_shape, mask=mask, cutoff=cutoff)
        # check that adj in the first voxel is correct
        if len(img_shape) == 2:
            npt.assert_equal(adj[0], [0, 1, 10, 11])
        if len(img_shape) == 3:
            npt.assert_equal(adj[0], [0, 1, 5, 6, 50, 51, 55, 56])
        # test that adj only corresponds to flattened and masked data
        npt.assert_equal(len(adj), mask.sum())


def test_select_3d_volume():
    rng = np.random.default_rng(0)
    vol = rng.random((10, 10, 10))
    npt.assert_array_equal(select_3d_volume(vol), vol)
    npt.assert_array_equal(select_3d_volume(vol, index=7), vol)

    data = rng.random((10, 10, 10, 5))
    npt.assert_array_equal(select_3d_volume(data), data[..., 0])
    npt.assert_array_equal(select_3d_volume(data, index=4), data[..., 4])
    npt.assert_array_equal(select_3d_volume(data, index=np.int64(2)), data[..., 2])

    for bad in (5, -1):
        with pytest.raises(ValueError, match="must be in"):
            select_3d_volume(data, index=bad, name="dwi")
    for bad in (1.5, True):
        with pytest.raises(TypeError, match="must be an integer"):
            select_3d_volume(data, index=bad)
    for shape in ((5, 5), (5, 5, 5, 5, 5)):
        with pytest.raises(ValueError, match="T1 must be a 3D or 4D"):
            select_3d_volume(np.zeros(shape), name="T1")
