import numpy as np
from numpy.testing import (run_module_suite,
                           assert_,
                           assert_equal,
                           assert_raises,
                           assert_array_almost_equal)
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.denspeed import add_padding_reflection, remove_padding
from matplotlib.pyplot import *


def test_nlmeans_padding():
    S0 = 100 + 2 * np.random.standard_normal((50, 50, 50))
    S0 = S0.astype('f8')
    S0n = add_padding_reflection(S0, 5)
    S0n2 = remove_padding(S0n, 5)
    assert_equal(S0.shape, S0n2.shape)


def test_nlmeans_static():
    S0 = 100 * np.ones((20, 20, 20), dtype='f8')
    S0n = nlmeans(S0, sigma = 1, rician=False)
    assert_array_almost_equal(S0, S0n)


def test_nlmeans_random_noise():
    S0 = 100 + 2 * np.random.standard_normal((50, 50, 50))

    S0n = nlmeans(S0, sigma = np.std(S0), rician=False)

    print(S0.mean(), S0.min(), S0.max())
    print(S0n.mean(), S0n.min(), S0n.max())

    assert_(S0n.min() > S0.min)
    assert_(S0n.max() > S0.max)
    assert_(np.abs(S0n.mean() - 100) < np.abs(S0.mean() - 100))


# def test_nlmeans_random_noise():
#     S0 = 100 + 2 * np.random.standard_normal((20, 20, 20))

#     S0 = S0.astype('f8')

#     from time import time

#     t1 = time()
#     S0n = nlmeans(S0, sigma = np.std(S0), rician=False)
#     t2 = time()
#     print('Time was', t2 - t1)

#     print(S0.mean(), S0.min(), S0.max())
#     print(S0n.mean(), S0n.min(), S0n.max())

#     print(S0.shape)
#     print(S0n.shape)

#     figure(1)
#     imshow(S0[:,:,10], interpolation='nearest')
#     figure(2)
#     imshow(S0n[:,:,10], interpolation='nearest')

#     D = np.abs(S0n - S0)

#     figure(3)
#     imshow(D[:,:,10], interpolation='nearest')


# def test_nlmeans_boundary():
#     S0 = 100 + np.zeros((20, 20, 20))

#     noise = 2 * np.random.standard_normal((20, 20, 20))

#     S0 += noise

#     S0[:10, :10, :10] = 300 + noise[:10, :10, :10]

#     S0 = S0.astype('f8')

#     from time import time

#     print(np.std(noise))

#     t1 = time()
#     S0n = nlmeans(S0, sigma = np.std(noise), rician=False)
#     t2 = time()
#     print('Time was', t2 - t1)

#     figure(1)
#     imshow(S0[:, :, 5], interpolation='nearest')
#     colorbar()

#     figure(2)
#     imshow(S0n[:, :, 5], interpolation='nearest')
#     colorbar()

#     D = np.abs(S0n - S0)

#     figure(3)
#     imshow(D[:, :, 5], interpolation='nearest')
#     colorbar()


# def test_reflected_border():    

#     data = np.ones((10, 10, 10))
#     data2 = add_padding_reflection(data, 5)
#     data3 = remove_padding(data2, 5)

#     assert_equal(data.shape, data3.shape)



# def test_copy_sub_array():

#     source = np.ones((10, 10, 10))
#     source[2, 2, 2] = 2
#     source[6, 6, 7] = 3
#     dest = np.zeros((5, 5, 6))

#     source = np.ones((10, 10, 10))
#     source[2, 2, 2] = 2
#     source[6, 6, 7] = 3

#     cdef cnp.ndarray[double, ndim=3, mode ='c'] dest2 = np.zeros((5, 5, 6))
#     copy_sub_array(<double *>dest2.data, 5, 5, 6, source, 2, 2, 2)
#     print(dest2)



#test_nlmeans_borders()
#test_nlmeans_static()
#test_nlmeans_random_noise()
test_nlmeans()
#test_nlmeans_boundary()

