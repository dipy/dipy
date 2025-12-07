import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_raises

from dipy.correct.disco import dummy_distortion_correction, estimate_distortion_field


def test_dummy_distortion_correction():
    """Test the dummy distortion correction function."""
    # Create a simple 4D test dataset
    data_shape = (10, 10, 10, 32)
    data = np.random.rand(*data_shape)

    # Test with default parameters
    corrected_data, corrected_affine = dummy_distortion_correction(data)

    # Check that output has the same shape as input
    assert_equal(corrected_data.shape, data_shape)

    # Check that affine is 4x4
    assert_equal(corrected_affine.shape, (4, 4))

    # Since it's a dummy function, data should be unchanged (just a copy)
    assert_array_equal(corrected_data, data)

    # Test with custom affine
    custom_affine = np.eye(4) * 2
    custom_affine[3, 3] = 1
    corrected_data, corrected_affine = dummy_distortion_correction(
        data, affine=custom_affine
    )

    # Check that affine is preserved
    assert_array_equal(corrected_affine, custom_affine)

    # Test with different b0_threshold
    corrected_data, corrected_affine = dummy_distortion_correction(
        data, b0_threshold=100
    )
    assert_equal(corrected_data.shape, data_shape)


def test_dummy_distortion_correction_2d():
    """Test the dummy distortion correction with 2D data."""
    # Create a 2D test dataset
    data = np.random.rand(10, 10)

    # This should work without errors
    corrected_data, corrected_affine = dummy_distortion_correction(data)

    # Check that output has the same shape as input
    assert_equal(corrected_data.shape, data.shape)

    # Since it's a dummy function, data should be unchanged (just a copy)
    assert_array_equal(corrected_data, data)


def test_estimate_distortion_field():
    """Test the distortion field estimation function."""
    # Create a simple 3D b0 image
    b0_image = np.random.rand(10, 10, 10)

    # Test with default parameters
    distortion_field = estimate_distortion_field(b0_image)

    # Check that output has the same shape as input
    assert_equal(distortion_field.shape, b0_image.shape)

    # Since it's a dummy function, field should be zeros
    assert_array_equal(distortion_field, np.zeros_like(b0_image))

    # Test with different phase encoding directions
    for direction in ['x', 'y', 'z']:
        distortion_field = estimate_distortion_field(
            b0_image, phase_encoding_direction=direction
        )
        assert_equal(distortion_field.shape, b0_image.shape)
        assert_array_equal(distortion_field, np.zeros_like(b0_image))


def test_estimate_distortion_field_2d():
    """Test the distortion field estimation with 2D data."""
    # Create a 2D b0 image
    b0_image = np.random.rand(10, 10)

    # Test with default parameters
    distortion_field = estimate_distortion_field(b0_image)

    # Check that output has the same shape as input
    assert_equal(distortion_field.shape, b0_image.shape)

    # Since it's a dummy function, field should be zeros
    assert_array_equal(distortion_field, np.zeros_like(b0_image))
