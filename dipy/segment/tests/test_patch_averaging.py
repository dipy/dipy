import numpy as np

from numpy.testing import (run_module_suite,
                           assert_,
                           assert_equal,
                           assert_array_almost_equal)
from dipy.align.imaffine import AffineMap

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


def test_with_translation():
	aftrsar = np.array([[1.0, 0, 0, (2 * np.random.rand(1) - 1)*3], 
    	[0, 1.0, 0, (2 * np.random.rand(1) - 1)*3], [0, 0, 1.0, (2 * np.random.rand(1) - 1)*3], [0, 0, 0, 1.0]])
	aftrs = AffineMap(aftrsar)
	S0 = np.zeros([35, 35, 35], dtype=np.float64)
	S0[5:30, 5:30, 5:30] = 50
	S0[10:25, 10:25, 10:25] = 100
	S0[15:20, 15:20, 15:20] = 150
	S0mask = np.zeros([35,35,35])
	S0temp = np.zeros([38, 42, 40], dtype=np.float64)
	S0temp[5:30, 5:30, 5:30] = 50
	S0temp[10:25, 10:25, 10:25] = 100
	S0temp[15:20, 15:20, 15:20] = 150
	mask = np.zeros([38, 42, 40])
	mask[5:30, 5:30, 5:30] = 1
    # some transform to the mask
	S0temp = aftrs._apply_transform(S0temp, sampling_grid_shape = S0temp.shape)
	print(S0temp.shape)

	# now we apply the whole brain extraction scheme
	[S0out,S0outmask] = brain_extraction(S0, np.eye(4), S0temp,
                     aftrsar, S0temp,
                     patch_radius=1, block_radius=1, parameter=1,
                     threshold=0)

	assert_array_almost_equal(S0out, S0)

if __name__ == '__main__':
    run_module_suite()
    # test_with_translation()
    # test_static()
    # plt.show()
