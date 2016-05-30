import numpy as np
from numpy.testing import (run_module_suite,
                           assert_,
                           assert_equal,
                           assert_array_almost_equal)
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.ascm import ascm

def test_ascm_static():
	S0 = 100 * np.ones((20, 20, 20), dtype='f8')
	S0n1 = nlmeans(S0, sigma=np.ones((20, 20, 20)), rician=False, patch_radius = 1, block_radius = 1 , type='blockwise')
	S0n2 = nlmeans(S0, sigma=np.ones((20, 20, 20)), rician=False, patch_radius = 2, block_radius = 1 , type='blockwise')
	S0n = ascm(S0, S0n1, S0n2, 1)
	assert_array_almost_equal(S0,S0n)

def test_ascm_random_noise():
	S0 = 100 + 2 * np.random.standard_normal((22, 23, 30))
	S0n1 = nlmeans(S0, sigma=np.ones((22, 23, 30)), rician=False, patch_radius = 1, block_radius = 1 , type='blockwise')
	S0n2 = nlmeans(S0, sigma=np.ones((22, 23, 30)), rician=False, patch_radius = 2, block_radius = 1 , type='blockwise')
	S0n = ascm(S0, S0n1, S0n2, 1)

	print(S0.mean(), S0.min(), S0.max())
	print(S0n.mean(), S0n.min(), S0n.max())
    
	assert_(S0n.min() > S0.min())
	assert_(S0n.max() < S0.max())
	assert_equal(np.round(S0n.mean()), 100)
    
def test_ascm_rmse_with_nlmeans():
	# checks the smoothness
	S0 = np.ones((30,30,30)) * 100
	S0[10:20,10:20,10:20] = 50
	S0[20:30,20:30,20:30] = 0
	S0_noise = S0 + 20 * np.random.standard_normal((30,30,30))
	print("Original RMSE", np.sum(np.abs(S0 - S0_noise)) / np.sum(S0))

	S0n1 = nlmeans(S0_noise, sigma=400, rician=False, patch_radius = 1, block_radius = 1 , type='blockwise')
	print("Smaller patch RMSE", np.sum(np.abs(S0 - S0n1)) / np.sum(S0))
	S0n2 = nlmeans(S0_noise, sigma=400, rician=False, patch_radius = 2, block_radius = 2 , type='blockwise')
	print("Larger patch RMSE", np.sum(np.abs(S0 - S0n2)) / np.sum(S0))
	S0n = ascm(S0, S0n1, S0n2, 400)
	print("ASCM RMSE", np.sum(np.abs(S0 - S0n)) / np.sum(S0))

	assert_(np.sum(np.abs(S0 - S0n)) / np.sum(S0) < np.sum(np.abs(S0 - S0n1)) / np.sum(S0))
	assert_(np.sum(np.abs(S0 - S0n)) / np.sum(S0) < np.sum(np.abs(S0 - S0_noise)) / np.sum(S0))
	assert_(90 < np.mean(S0n) < 110)

def test_sharpness():
	# check the edge-preserving nature
	S0 = np.ones((30,30,30)) * 100
	S0[10:20,10:20,10:20] = 50
	S0[20:30,20:30,20:30] = 0
	S0_noise = S0 + 20 * np.random.standard_normal((30,30,30))
	S0n1 = nlmeans(S0_noise, sigma=400, rician=False, patch_radius = 1, block_radius = 1 , type='blockwise')
	edg1 = np.abs(np.mean(S0n1[8,10:20,10:20] - S0n1[12,10:20,10:20]) - 50)
	print("Edge gradient smaller patch", edg1)
	S0n2 = nlmeans(S0_noise, sigma=400, rician=False, patch_radius = 2, block_radius = 2 , type='blockwise')
	edg2 = np.abs(np.mean(S0n2[8,10:20,10:20] - S0n2[12,10:20,10:20]) - 50)
	print("Edge gradient larger patch", edg2)
	S0n = ascm(S0, S0n1, S0n2, 400)
	edg = np.abs(np.mean(S0n[8,10:20,10:20] - S0n[12,10:20,10:20]) - 50)
	print("Edge gradient ASCM", edg)

	assert_(edg2 > edg1)
	assert_(edg2 > edg)
	assert_(np.abs(edg1 - edg) < 1.5)

def test_ascm_dtype():
	S0 = 200 * np.ones((20, 20, 20), dtype='f4')
	S0n1 = nlmeans(S0, sigma=1, rician=True, patch_radius = 1, block_radius = 1 , type = 'blockwise')
	S0n2 = nlmeans(S0, sigma=1, rician=True, patch_radius = 2, block_radius = 1 , type = 'blockwise')
	S0n = ascm(S0,S0n1,S0n2,1)
	assert_equal(S0.dtype, S0n.dtype)

	S0 = 200 * np.ones((20, 20, 20), dtype=np.uint16)
	S0n1 = nlmeans(S0, sigma=1, rician=True, patch_radius = 1, block_radius = 1 , type = 'blockwise')
	S0n2 = nlmeans(S0, sigma=1, rician=True, patch_radius = 2, block_radius = 1 , type = 'blockwise')
	S0n = ascm(S0,S0n1,S0n2,1)
	assert_equal(S0.dtype, S0n.dtype)

if __name__ == '__main__':
    run_module_suite()
    