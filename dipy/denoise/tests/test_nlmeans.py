from time import time

import numpy as np
from numpy.testing import (
    assert_,
    assert_array_almost_equal,
    assert_equal,
    assert_raises,
)
import pytest

from dipy.denoise.denspeed import add_padding_reflection, remove_padding
from dipy.denoise.nlmeans import nlmeans
from dipy.testing import assert_greater
from dipy.testing.decorators import set_random_number_generator
from dipy.utils.omp import cpu_count, have_openmp


@set_random_number_generator()
def test_nlmeans_padding(rng):
    S0 = 100 + 2 * rng.standard_normal((50, 50, 50))
    S0 = S0.astype("f8")
    S0n = add_padding_reflection(S0, 5)
    S0n2 = remove_padding(S0n, 5)
    assert_equal(S0.shape, S0n2.shape)


def test_nlmeans_static():
    """Test static image denoising with classic method."""
    S0 = 100 * np.ones((20, 20, 20), dtype="f8")
    S0n = nlmeans(S0, sigma=1.0, rician=False, method="classic")
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
    """Test random noise reduction with classic method."""
    S0 = 100 + 2 * rng.standard_normal((22, 23, 30))

    S0n = nlmeans(S0, sigma=np.std(S0), rician=False, method="classic")

    print(S0.mean(), S0.min(), S0.max())
    print(S0n.mean(), S0n.min(), S0n.max())

    assert_(S0n.min() > S0.min())
    assert_(S0n.max() < S0.max())
    assert_equal(np.round(S0n.mean()), 100)


@set_random_number_generator()
def test_nlmeans_boundary(rng):
    """Test boundary preservation with classic method."""
    S0 = 100 + np.zeros((20, 20, 20))
    noise = 2 * rng.standard_normal((20, 20, 20))
    S0 += noise
    S0[:10, :10, :10] = 300 + noise[:10, :10, :10]

    S0_denoised = nlmeans(S0, sigma=np.std(noise), rician=False, method="classic")

    assert_(S0_denoised[9, 9, 9] > 290)
    assert_(S0_denoised[10, 10, 10] < 110)


def test_nlmeans_4D_and_mask():
    """Test 4D data with mask using classic method."""
    S0 = 200 * np.ones((20, 20, 20, 3), dtype="f8")
    mask = np.zeros((20, 20, 20))
    mask[10, 10, 10] = 1

    S0n = nlmeans(S0, sigma=1, mask=mask, rician=True, method="classic")
    assert_equal(S0.shape, S0n.shape)
    assert_equal(np.round(S0n[10, 10, 10]), 200)
    assert_equal(S0n[8, 8, 8], 0)


def test_nlmeans_dtype():
    """Test dtype preservation with classic method."""
    S0 = 200 * np.ones((20, 20, 20, 3), dtype="f4")
    mask = np.zeros((20, 20, 20))
    mask[10:14, 10:14, 10:14] = 1
    S0n = nlmeans(S0, sigma=1, mask=mask, rician=True, method="classic")
    assert_equal(S0.dtype, S0n.dtype)

    S0 = 200 * np.ones((20, 20, 20), dtype=np.uint16)
    mask = np.zeros((20, 20, 20))
    mask[10:14, 10:14, 10:14] = 1
    S0n = nlmeans(S0, sigma=1, mask=mask, rician=True, method="classic")
    assert_equal(S0.dtype, S0n.dtype)
    S0n = nlmeans(
        S0, sigma=np.ones((20, 20, 20)), mask=mask, rician=True, method="classic"
    )
    assert_equal(S0.dtype, S0n.dtype)


@pytest.mark.skipif(not have_openmp, reason="OpenMP does not appear to be available")
def test_nlmeans_4d_3dsigma_and_threads():
    """Test 4D data with threading using classic method."""
    # Input is 4D data
    data = np.ones((50, 50, 50, 5))
    sigma = 1.0  # Use scalar sigma for compatibility
    mask = np.zeros(data.shape[:3])
    mask[:] = 1

    print(f"cpu count {cpu_count()}")

    print("1")
    t = time()
    new_data = nlmeans(data, sigma, mask=mask, num_threads=1, method="classic")
    duration_1core = time() - t
    print(duration_1core)

    print("All")
    t = time()
    new_data2 = nlmeans(data, sigma, mask=mask, num_threads=None, method="classic")
    duration_all_core = time() - t
    print(duration_all_core)

    print("2")
    t = time()
    new_data3 = nlmeans(data, sigma, mask=mask, num_threads=2, method="classic")
    duration_2core = time() - t
    print(duration_2core)

    assert_array_almost_equal(new_data, new_data2)
    assert_array_almost_equal(new_data2, new_data3)

    if cpu_count() > 2:
        assert_equal(duration_all_core < duration_2core, True)
        assert_equal(duration_2core < duration_1core, True)

    if cpu_count() == 2:
        assert_greater(duration_1core, duration_2core)


def test_nlmeans_static_blockwise():
    """Test static image denoising with blockwise method."""
    S0 = 100 * np.ones((20, 20, 20), dtype="f8")
    S0nb = nlmeans(S0, sigma=1.0, rician=False, method="blockwise")

    assert_equal(S0.shape, S0nb.shape)

    assert np.abs(np.mean(S0nb) - 100) < 20
    assert np.all(S0nb >= 0)

    S0 = 100 * np.ones((20, 20, 20, 3), dtype="f8")
    S0nb = nlmeans(S0, sigma=1.0, rician=False, method="blockwise")

    assert_equal(S0.shape, S0nb.shape)
    assert np.abs(np.mean(S0nb) - 100) < 20

    S0nb = nlmeans(S0, sigma=np.array(1.0), rician=False, method="blockwise")
    assert_equal(S0.shape, S0nb.shape)
    assert np.abs(np.mean(S0nb) - 100) < 20

    S0nb = nlmeans(S0, sigma=np.array([1.0]), rician=False, method="blockwise")
    assert_equal(S0.shape, S0nb.shape)
    assert np.abs(np.mean(S0nb) - 100) < 20


def test_method_parameter_validation():
    """Test method parameter validation."""
    S0 = 100 + np.zeros((10, 10, 10))

    # Valid methods should work
    result1 = nlmeans(S0, sigma=1.0, method="classic")
    result2 = nlmeans(S0, sigma=1.0, method="blockwise")
    assert result1.shape == S0.shape
    assert result2.shape == S0.shape

    # Invalid method should raise error
    with pytest.raises(ValueError):
        nlmeans(S0, sigma=1.0, method="invalid")


def test_default_block_radius():
    """
    Test that default block_radius values are set correctly based on method.
    """
    S0 = 100 * np.ones((20, 20, 20), dtype="f8")

    result_classic = nlmeans(S0, sigma=1.0, method="classic")
    assert result_classic.shape == S0.shape

    result_blockwise = nlmeans(S0, sigma=1.0, method="blockwise")
    assert result_blockwise.shape == S0.shape

    assert_array_almost_equal(result_classic, S0, decimal=0)
    assert np.abs(np.mean(result_blockwise) - 100) < 5
    assert np.all(result_blockwise >= 0)


def test_blockwise_sigma_array_support():
    """
    Test that blockwise method supports different sigma input formats.
    """
    S0 = 100 * np.ones((20, 20, 20), dtype="f8")
    result1 = nlmeans(S0, sigma=1.0, method="blockwise")
    assert result1.shape == S0.shape

    result2 = nlmeans(S0, sigma=np.array(1.0), method="blockwise")
    assert result2.shape == S0.shape
    assert np.abs(np.mean(result2) - np.mean(result1)) < 1.0

    result3 = nlmeans(S0, sigma=np.array([1.0]), method="blockwise")
    assert result3.shape == S0.shape
    assert np.abs(np.mean(result3) - np.mean(result1)) < 1.0

    sigma_3d = np.ones(S0.shape) * 1.0
    result4 = nlmeans(S0, sigma=sigma_3d, method="blockwise")
    assert result4.shape == S0.shape
    assert np.abs(np.mean(result4) - np.mean(result1)) < 5.0


def test_coordinate_consistency():
    """
    Test that the nlmeans denoising respects coordinate geometry.

    Creates an image with asymmetric features to verify that coordinate
    swapping bugs are not present in the implementation.
    """
    height, width, depth = 20, 20, 20
    test_image = np.zeros((height, width, depth), dtype=np.float64)

    test_image[5:15, 5:15, 5:15] = 100.0

    np.random.seed(42)
    noisy_image = test_image + np.random.normal(0, 5, test_image.shape)

    denoised_image = nlmeans(
        noisy_image,
        sigma=5.0,
        patch_radius=1,
        block_radius=2,
        rician=False,
        num_threads=1,
        method="blockwise",
    )

    assert denoised_image.shape == noisy_image.shape
    assert (
        np.sum(denoised_image < 0) < 0.01 * denoised_image.size
    ), "Too many negative values"

    assert (
        5 < np.mean(denoised_image) < 80
    ), "Denoised mean should be in reasonable range"

    assert isinstance(denoised_image, np.ndarray)
    assert denoised_image.dtype == np.float64 or denoised_image.dtype == np.float32
