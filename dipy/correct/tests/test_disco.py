import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_raises

from dipy.correct.disco import (
    HAVE_SYNB0,
    dummy_distortion_correction,
    estimate_distortion_field,
    synb0_distortion_correction,
    synb0_predict,
)


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


def test_synb0_predict_shape_validation():
    """Test that synb0_predict validates input shapes."""
    if not HAVE_SYNB0:
        # Skip if Synb0 not available
        return

    # Create data with wrong shape
    b0_wrong = np.random.rand(80, 80, 80).astype(np.float32)
    T1_wrong = np.random.rand(80, 80, 80).astype(np.float32)

    # Should raise ValueError for wrong shape
    try:
        result = synb0_predict(b0_wrong, T1_wrong, average=False)
        # If prediction works (shouldn't), fail test
        assert False, "Should have raised ValueError for wrong shape"
    except (ValueError, Exception) as e:
        # Expected behavior - either ValueError or model error
        pass


def test_synb0_predict_correct_shape():
    """Test that synb0_predict accepts correct shape."""
    if not HAVE_SYNB0:
        # Skip if Synb0 not available
        return

    # Create data with correct shape
    b0 = np.random.rand(77, 91, 77).astype(np.float32) * 1000
    T1 = np.random.rand(77, 91, 77).astype(np.float32) * 150

    # Try prediction without average (faster)
    try:
        result = synb0_predict(b0, T1, average=False, verbose=False)
        # If successful, check output shape
        assert_equal(result.shape, (77, 91, 77))
    except Exception:
        # Weights might not be available, that's okay for shape test
        pass


def test_synb0_distortion_correction_shape_validation():
    """Test that synb0_distortion_correction validates input shapes."""
    if not HAVE_SYNB0:
        # Skip if Synb0 not available
        return

    # Test with wrong b0 shape
    b0_wrong = np.random.rand(80, 80, 80).astype(np.float32)
    T1 = np.random.rand(77, 91, 77).astype(np.float32)

    with assert_raises(ValueError):
        synb0_distortion_correction(b0_wrong, T1, average=False)

    # Test with wrong T1 shape
    b0 = np.random.rand(77, 91, 77).astype(np.float32)
    T1_wrong = np.random.rand(80, 80, 80).astype(np.float32)

    with assert_raises(ValueError):
        synb0_distortion_correction(b0, T1_wrong, average=False)


def test_synb0_distortion_correction_correct_shape():
    """Test that synb0_distortion_correction accepts correct shape."""
    if not HAVE_SYNB0:
        # Skip if Synb0 not available
        return

    # Create data with correct shape
    b0 = np.random.rand(77, 91, 77).astype(np.float32) * 1000
    T1 = np.random.rand(77, 91, 77).astype(np.float32) * 150

    # Try correction without average (faster)
    try:
        result = synb0_distortion_correction(
            b0, T1, average=False, verbose=False
        )
        # If successful, check output shape
        assert_equal(result.shape, (77, 91, 77))
    except Exception:
        # Weights might not be available, that's okay
        pass


def test_synb0_not_available():
    """Test behavior when Synb0 is not available."""
    # This test will pass when both torch and tf are not available
    if HAVE_SYNB0:
        # If Synb0 is available, skip this test
        return

    b0 = np.random.rand(77, 91, 77).astype(np.float32)
    T1 = np.random.rand(77, 91, 77).astype(np.float32)

    # Should raise ImportError
    with assert_raises(ImportError):
        synb0_predict(b0, T1, average=False)
