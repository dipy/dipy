from time import time

import numpy as np
from numpy.testing import (assert_,
                           assert_equal,
                           assert_array_almost_equal,
                           assert_raises)
import pytest

from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.denspeed import (add_padding_reflection, remove_padding)
from dipy.utils.omp import cpu_count, have_openmp
from dipy.testing import assert_greater
from dipy.testing.decorators import set_random_number_generator


@set_random_number_generator()
def test_nlmeans_padding(rng):
    S0 = 100 + 2 * rng.standard_normal((50, 50, 50))
    S0 = S0.astype('f8')
    S0n = add_padding_reflection(S0, 5)
    S0n2 = remove_padding(S0n, 5)
    assert_equal(S0.shape, S0n2.shape)


def test_nlmeans_static():
    S0 = 100 * np.ones((20, 20, 20), dtype='f8')
    S0n = nlmeans(S0, sigma=np.ones((20, 20, 20)), rician=False)
    assert_array_almost_equal(S0, S0n)


def test_nlmeans_wrong():
    S0 = np.ones((2, 2, 2, 2, 2))
    assert_raises(ValueError, nlmeans, S0, 1.0)

    # test invalid values of num_threads
    data = np.ones((10, 10, 10))
    sigma = 1
    assert_raises(ValueError, nlmeans, data, sigma, num_threads=0)


@set_random_number_generator()
def test_nlmeans_random_noise(rng):
    S0 = 100 + 2 * rng.standard_normal((22, 23, 30))

    S0n = nlmeans(S0, sigma=np.ones((22, 23, 30)) * np.std(S0), rician=False)

    print(S0.mean(), S0.min(), S0.max())
    print(S0n.mean(), S0n.min(), S0n.max())

    assert_(S0n.min() > S0.min())
    assert_(S0n.max() < S0.max())
    assert_equal(np.round(S0n.mean()), 100)


@set_random_number_generator()
def test_nlmeans_boundary(rng):
    # nlmeans preserves boundaries

    S0 = 100 + np.zeros((20, 20, 20))

    noise = 2 * rng.standard_normal((20, 20, 20))

    S0 += noise

    S0[:10, :10, :10] = 300 + noise[:10, :10, :10]

    nlmeans(S0, sigma=np.ones((20, 20, 20)) * np.std(noise),
            rician=False)

    print(S0[9, 9, 9])
    print(S0[10, 10, 10])

    assert_(S0[9, 9, 9] > 290)
    assert_(S0[10, 10, 10] < 110)


def test_nlmeans_4D_and_mask():
    S0 = 200 * np.ones((20, 20, 20, 3), dtype='f8')

    mask = np.zeros((20, 20, 20))
    mask[10, 10, 10] = 1

    S0n = nlmeans(S0, sigma=1, mask=mask, rician=True)
    assert_equal(S0.shape, S0n.shape)
    assert_equal(np.round(S0n[10, 10, 10]), 200)
    assert_equal(S0n[8, 8, 8], 0)


def test_nlmeans_dtype():

    S0 = 200 * np.ones((20, 20, 20, 3), dtype='f4')
    mask = np.zeros((20, 20, 20))
    mask[10:14, 10:14, 10:14] = 1
    S0n = nlmeans(S0, sigma=1, mask=mask, rician=True)
    assert_equal(S0.dtype, S0n.dtype)

    S0 = 200 * np.ones((20, 20, 20), dtype=np.uint16)
    mask = np.zeros((20, 20, 20))
    mask[10:14, 10:14, 10:14] = 1
    S0n = nlmeans(S0, sigma=np.ones((20, 20, 20)), mask=mask, rician=True)
    assert_equal(S0.dtype, S0n.dtype)

@pytest.mark.skipif(not have_openmp,
                    reason='OpenMP does not appear to be available')
def test_nlmeans_4d_3dsigma_and_threads():
    # Input is 4D data and 3D sigma
    data = np.ones((50, 50, 50, 5))
    sigma = np.ones(data.shape[:3])
    mask = np.zeros(data.shape[:3])

    # mask[25-10:25+10] = 1
    mask[:] = 1

    print('cpu count %d' % (cpu_count(),))

    print('1')
    t = time()
    new_data = nlmeans(data, sigma, mask, num_threads=1)
    duration_1core = time() - t
    print(duration_1core)

    print('All')
    t = time()
    new_data2 = nlmeans(data, sigma, mask, num_threads=None)
    duration_all_core = time() - t
    print(duration_all_core)

    print('2')
    t = time()
    new_data3 = nlmeans(data, sigma, mask, num_threads=2)
    duration_2core = time() - t
    print(duration_2core)

    assert_array_almost_equal(new_data, new_data2)
    assert_array_almost_equal(new_data2, new_data3)

    if cpu_count() > 2:

        assert_equal(duration_all_core < duration_2core, True)
        assert_equal(duration_2core < duration_1core, True)

    if cpu_count() == 2:

        assert_greater(duration_1core, duration_2core)
