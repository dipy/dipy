import numpy as np

from numpy.testing import (run_module_suite,
                           assert_,
                           assert_equal,
                           assert_array_almost_equal)
from dipy.align.imaffine import AffineMap

from dipy.segment.mask import brain_extraction
from dipy.segment.fast_patch_averaging import fast_patch_averaging


def spherical_phantom():
    S0 = np.zeros([40, 40, 40], dtype=np.float64)
    for i in range(40):
        for j in range(40):
            for k in range(40):
                dist = np.sqrt((i - 20) * (i - 20) +
                               (j - 20) * (j - 20) +
                               (k - 20) * (k - 20))

                if(dist < 5):
                    S0[i, j, k] = 150
                if(dist < 10 and dist >= 5):
                    S0[i, j, k] = 100
                if(dist < 15 and dist >= 10):
                    S0[i, j, k] = 50

    return S0


def test_static():
    # concentric cube phantom
    S0 = np.zeros([35, 35, 35], dtype=np.float64)
    S0[5:30, 5:30, 5:30] = 50
    S0[10:25, 10:25, 10:25] = 100
    S0[15:20, 15:20, 15:20] = 150
    mask = np.zeros([35, 35, 35], dtype = np.float64)
    mask[5:30, 5:30, 5:30] = 1.0
    [S0avg, S0mask] = fast_patch_averaging(
        S0.astype(np.float64), S0.astype(np.float64), mask.astype(np.float64), 0, 1, 1.0, 0.0)

    assert_array_almost_equal(S0, S0avg)

    # Same test for spherical phantom
    S0s = spherical_phantom()
    masks = np.zeros(S0s.shape)
    masks[S0s > 0] = 1
    [S0savg, S0smask] = fast_patch_averaging(
        S0s.astype(np.float64), S0s.astype(np.float64), masks.astype(np.float64), 0, 1, 1.0, 0.0)

    assert_array_almost_equal(S0s, S0savg)


def test_with_transform():
    aftrsar = np.array([[1.0, 0, 0, (2 * np.random.rand(1) - 1) * 3],
                        [0, 1.0, 0, (2 * np.random.rand(1) - 1) * 3],
                        [0, 0, 1.0, (2 * np.random.rand(1) - 1) * 3],
                        [0, 0, 0, 1.0]])
    aftrs = AffineMap(aftrsar)
    S0 = np.zeros([35, 35, 35], dtype=np.float64)
    S0[5:30, 5:30, 5:30] = 50
    S0[10:25, 10:25, 10:25] = 100
    S0[15:20, 15:20, 15:20] = 150
    S0mask = np.zeros([35, 35, 35])
    S0mask[S0 > 0] = 1
    S0temp = np.zeros([38, 42, 40], dtype=np.float64)
    S0temp[5:30, 5:30, 5:30] = 50
    S0temp[10:25, 10:25, 10:25] = 100
    S0temp[15:20, 15:20, 15:20] = 150
    # some transform to the mask
    S0temp = aftrs._apply_transform(S0temp, sampling_grid_shape=S0temp.shape)
    print(S0temp.shape)
    S0tempmask = np.zeros(S0temp.shape)
    S0tempmask[S0temp > 0] = 1
    # now we apply the whole brain extraction scheme
    [S0out,
     S0outmask] = brain_extraction(S0,
                                   np.eye(4),
                                   S0temp,
                                   aftrsar,
                                   S0tempmask,
                                   patch_radius=1,
                                   block_radius=1,
                                   parameter=1,
                                   threshold=0)

    assert_array_almost_equal(S0out, S0)

    # same test for the speherical phantom
    S0s = spherical_phantom()
    S0smask = np.zeros(S0s.shape)

    S0smask[S0s > 0] = 1
    S0stemp = aftrs._apply_transform(S0s, sampling_grid_shape=S0s.shape)
    S0stempmask = np.zeros(S0stemp.shape)
    S0stempmask[S0stemp > 0] = 1

    [S0sout,
     S0soutmask] = brain_extraction(S0s,
                                    np.eye(4),
                                    S0stemp,
                                    aftrsar,
                                    S0stempmask,
                                    patch_radius=1,
                                    block_radius=1,
                                    parameter=1,
                                    threshold=0)

    assert_array_almost_equal(S0s, S0sout)

if __name__ == '__main__':
    run_module_suite()
