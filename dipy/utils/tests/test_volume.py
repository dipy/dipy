""" Testing volumes """

import numpy as np
import numpy.testing as npt

from dipy.utils.volume import (adjacency_calc)

def test_adjacency_calc():
    """
    Test adjacency_calc function, which calculates indices of adjacent voxels
    """

    cutoff = 1.99
    for img_shape in [(50, 50), (50, 50, 5)]:

        mask = None
        adj = adjacency_calc(img_shape, mask, cutoff=cutoff)
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
        adj = adjacency_calc(img_shape, mask, cutoff=cutoff)
        # check that adj in the first voxel is correct
        if len(img_shape) == 2:
            npt.assert_equal(adj[0], [0, 1, 10, 11])
        if len(img_shape) == 3:
            npt.assert_equal(adj[0], [0, 1, 5, 6, 50, 51, 55, 56])
        # test that adj only corresponds to flattened and masked data
        npt.assert_equal(len(adj), mask.sum())

