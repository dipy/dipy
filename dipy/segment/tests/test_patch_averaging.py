import numpy as np

from numpy.testing import (run_module_suite,
                           assert_,
                           assert_equal,
                           assert_array_almost_equal)

from dipy.segment.mask import (brain_extraction, patch_averaging)
# import matplotlib.pyplot as plt


def test_static():
    # concentric cube phantom
    S0 = np.zeros([35, 35, 35], dtype=np.float64)
    S0[5:30, 5:30, 5:30] = 50
    S0[10:25, 10:25, 10:25] = 100
    S0[15:20, 15:20, 15:20] = 150
    mask = np.zeros([35, 35, 35])
    mask[5:30, 5:30, 5:30] = 1
    [S0avg, S0mask] = patch_averaging(
        S0, S0, mask, patch_radius=0, block_radius=1, parameter=1, threshold=0)

    print(np.sum(mask - S0mask))
    assert_array_almost_equal(S0, S0avg)
    assert_array_almost_equal(S0mask, mask)

# def test_with_transform():
# 	S0 = np.zeros([70,70,70], dtype = np.float64)
# 	S0[10:60, 10:60, 10:60] = 50
# 	S0[20:50, 20:50, 20:50] = 100
# 	S0[30:40, 30:40, 30:40] = 150
# 	S0temp = np.zeros([73,77,75], dtype = np.float64)
# 	S0temp[10:60, 10:60, 10:60] = 50
# 	S0temp[20:50, 20:50, 20:50] = 100
# 	S0temp[30:40, 30:40, 30:40] = 150
# 	mask = np.zeros([73, 77, 75])
# 	mask[20:50, 20:50, 20:50] = 1
    # some transform to the mask

if __name__ == '__main__':
    run_module_suite()
    # test_static()
    # plt.show()
