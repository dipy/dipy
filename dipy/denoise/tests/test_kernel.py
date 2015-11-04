from dipy.denoise.enhancement_kernel import EnhancementKernel
from dipy.denoise.convolution5d import convolve_5d_sf
from dipy.reconst.shm import sh_to_sf, sf_to_sh
from dipy.data import get_sphere

import numpy as np
import numpy.testing as npt


def test_enhancement_kernel():
    pass
    # D33 = 1.0
    # D44 = 0.04
    # t = 1
    # k = EnhancementKernel(D33, D44, t, force_recompute=True, test_mode=True)

    # x = np.array([0, 0, 0], dtype=np.float64)
    # y = np.array([0, 0, 0], dtype=np.float64)
    # r = np.array([0, 0, .8], dtype=np.float64)
    # v = np.array([0, 0, 1], dtype=np.float64)
    # npt.assert_almost_equal(k.evaluate_kernel(x, y, r, v), 0.00932114)

    # x = np.array([1, 0, 0], dtype=np.float64)
    # y = np.array([0, 0, 0], dtype=np.float64)
    # r = np.array([0, 0, 1], dtype=np.float64)
    # v = np.array([0, 0, 1], dtype=np.float64)
    # npt.assert_almost_equal(k.evaluate_kernel(x, y, r, v), 0.0355297)

    D33 = 1.0
    D44 = 0.04
    t = 1
    k = EnhancementKernel(D33, D44, t, test_mode=True)

    lut = k.get_lookup_table()
    print lut.shape
    tsum = 0.0
    for x in range(0,7):
        for y in range(0,7):
            for z in range(0,7):
                for orient in range(100):
                    tsum += lut[0, orient, x,y,z]
    print "kernel sum=" + str(tsum) 


def test_symmetry():

    D33 = 1.0
    D44 = 0.04
    t = 1
    k = EnhancementKernel(D33, D44, t, force_recompute=True, test_mode=True)

    kernel = k.get_lookup_table()
    kslice = np.array((kernel[0, 0, :, 3, 3],
                       kernel[0, 0, 3, :, 3],
                       kernel[0, 0, 3, 3, :]))
    ksliceR = np.fliplr(kslice)
    diff = np.sum(kslice - ksliceR)
    npt.assert_equal(diff, 0.0)
    
def test_spike():
    
    D33 = 1.0
    D44 = 0.04
    t = 1
    k = EnhancementKernel(D33, D44, t, force_recompute=True, test_mode=True)

    spike = np.zeros((1,7,7,100),dtype=np.float64)
    spike[0,3,3,0] = 1

    csd_enh = convolve_5d_sf(spike, k, test_mode=True)

    totalsum = 0.0
    for i in range(0,100):
        totalsum += np.sum(np.array(k.get_lookup_table())[i,0,3,:,:] - np.array(csd_enh)[0,:,:,i])    
    npt.assert_equal(totalsum, 0.0)
    
def test_leftinvariance():
    pass

if __name__ == '__main__':

    #npt.run_module_suite()
    test_enhancement_kernel()


