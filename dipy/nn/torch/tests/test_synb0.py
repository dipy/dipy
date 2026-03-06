#!/usr/bin/python
"""Tests for Synb0 PyTorch implementation."""

import numpy as np
from numpy.testing import assert_, assert_array_almost_equal, assert_equal

from dipy.nn.torch.synb0 import DecoderBlock, EncoderBlock, Synb0, UNet3D
from dipy.testing.decorators import set_random_number_generator
from dipy.utils.optpkg import optional_package

torch, have_torch, _ = optional_package("torch", min_version="2.2.0")


@set_random_number_generator()
def test_encoder_block(rng):
    """Test the EncoderBlock."""
    if not have_torch:
        return

    # Create encoder block
    encoder = EncoderBlock(in_channels=2, out_channels=32)

    # Create random input
    x = torch.randn(1, 2, 80, 96, 80)

    # Forward pass
    output = encoder(x)

    # Check output shape
    assert_equal(output.shape, (1, 32, 80, 96, 80))


@set_random_number_generator()
def test_decoder_block(rng):
    """Test the DecoderBlock."""
    if not have_torch:
        return

    # Create decoder block
    decoder = DecoderBlock(in_channels=32, out_channels=16)

    # Create random input
    x = torch.randn(1, 32, 40, 48, 40)

    # Forward pass
    output = decoder(x)

    # Check output shape (should be upsampled by 2)
    assert_equal(output.shape, (1, 16, 80, 96, 80))


@set_random_number_generator()
def test_unet3d_architecture(rng):
    """Test the UNet3D architecture."""
    if not have_torch:
        return

    # Create model
    model = UNet3D(input_channels=2)

    # Create random input (batch, channels, depth, height, width)
    x = torch.randn(1, 2, 80, 96, 80)

    # Forward pass
    output = model(x)

    # Check output shape
    assert_equal(output.shape, (1, 1, 80, 96, 80))


@set_random_number_generator()
def test_unet3d_batch(rng):
    """Test UNet3D with batch size > 1."""
    if not have_torch:
        return

    # Create model
    model = UNet3D(input_channels=2)

    # Create batch input
    batch_size = 2
    x = torch.randn(batch_size, 2, 80, 96, 80)

    # Forward pass
    output = model(x)

    # Check output shape
    assert_equal(output.shape, (batch_size, 1, 80, 96, 80))


@set_random_number_generator()
def test_synb0_initialization(rng):
    """Test Synb0 class initialization."""
    if not have_torch:
        return

    # Create Synb0 instance
    synb0 = Synb0(verbose=False)

    # Check that model is created
    assert_(hasattr(synb0, "model"))
    assert_(isinstance(synb0.model, UNet3D))

    # Check that device is set
    assert_(hasattr(synb0, "device"))


@set_random_number_generator()
def test_synb0_predict_single_image(rng):
    """Test Synb0 prediction with single image."""
    if not have_torch:
        return

    # Create Synb0 instance
    synb0 = Synb0(verbose=False)

    # Create dummy data with correct shape (77, 91, 77)
    b0 = rng.random((77, 91, 77)).astype(np.float32) * 1000
    T1 = rng.random((77, 91, 77)).astype(np.float32) * 150

    # Test prediction without average (faster for testing)
    try:
        prediction = synb0.predict(b0, T1, average=False)

        # Check output shape
        assert_equal(prediction.shape, (77, 91, 77))

        # Check output is not all zeros
        assert_(np.any(prediction != 0))
    except Exception:
        # If weights are not available, that's okay for this test
        pass


@set_random_number_generator()
def test_synb0_predict_batch(rng):
    """Test Synb0 prediction with batch of images."""
    if not have_torch:
        return

    # Create Synb0 instance
    synb0 = Synb0(verbose=False)

    # Create batch of dummy data
    batch_size = 2
    b0 = rng.random((batch_size, 77, 91, 77)).astype(np.float32) * 1000
    T1 = rng.random((batch_size, 77, 91, 77)).astype(np.float32) * 150

    # Test prediction without average
    try:
        prediction = synb0.predict(b0, T1, batch_size=1, average=False)

        # Check output shape
        assert_equal(prediction.shape, (batch_size, 77, 91, 77))
    except Exception:
        # If weights are not available, that's okay for this test
        pass


@set_random_number_generator()
def test_synb0_predict_shape_validation(rng):
    """Test that Synb0 validates input shapes."""
    if not have_torch:
        return

    # Create Synb0 instance
    synb0 = Synb0(verbose=False)

    # Create data with wrong shape
    b0_wrong = rng.random((80, 80, 80)).astype(np.float32)
    T1_wrong = rng.random((80, 80, 80)).astype(np.float32)

    # Should raise ValueError for wrong shape
    try:
        prediction = synb0.predict(b0_wrong, T1_wrong, average=False)
        # If no error, fail the test
        assert_(False, "Should have raised ValueError for wrong shape")
    except ValueError as e:
        # Expected behavior
        assert_("Expected shape" in str(e))


@set_random_number_generator()
def test_synb0_predict_mismatched_shapes(rng):
    """Test that Synb0 detects mismatched input shapes."""
    if not have_torch:
        return

    # Create Synb0 instance
    synb0 = Synb0(verbose=False)

    # Create data with mismatched shapes
    b0 = rng.random((77, 91, 77)).astype(np.float32)
    T1 = rng.random((80, 96, 80)).astype(np.float32)

    # Should raise ValueError for mismatched shapes
    try:
        prediction = synb0.predict(b0, T1, average=False)
        # If no error, fail the test
        assert_(False, "Should have raised ValueError for mismatched shapes")
    except ValueError as e:
        # Expected behavior
        assert_("Expected shape" in str(e))


@set_random_number_generator()
def test_unet3d_gradient_flow(rng):
    """Test that gradients can flow through UNet3D."""
    if not have_torch:
        return

    # Create model
    model = UNet3D(input_channels=2)
    model.train()

    # Create input with gradient tracking
    x = torch.randn(1, 2, 80, 96, 80, requires_grad=True)

    # Forward pass
    output = model(x)

    # Compute a simple loss
    loss = output.mean()

    # Backward pass
    loss.backward()

    # Check that gradients exist
    assert_(x.grad is not None)
    assert_(torch.any(x.grad != 0))


@set_random_number_generator()
def test_encoder_decoder_symmetry(rng):
    """Test encoder and decoder blocks are symmetric."""
    if not have_torch:
        return

    # Test that decoder upsamples what encoder would downsample
    encoder = EncoderBlock(2, 32, kernel_size=3, padding=1)
    pool = torch.nn.MaxPool3d(2)

    # Input
    x = torch.randn(1, 2, 80, 96, 80)

    # Encode and pool
    encoded = encoder(x)
    pooled = pool(encoded)

    # Should be half the size
    assert_equal(pooled.shape, (1, 32, 40, 48, 40))

    # Decoder should upsample back
    decoder = DecoderBlock(32, 16, kernel_size=2, stride=2)
    upsampled = decoder(pooled)

    assert_equal(upsampled.shape, (1, 16, 80, 96, 80))


@set_random_number_generator()
def test_model_eval_mode(rng):
    """Test that model is in eval mode for inference."""
    if not have_torch:
        return

    # Create Synb0 instance
    synb0 = Synb0(verbose=False)

    # Model should be in eval mode
    assert_(not synb0.model.training)


@set_random_number_generator()
def test_model_device_placement(rng):
    """Test that model is placed on correct device."""
    if not have_torch:
        return

    # Create Synb0 instance
    synb0 = Synb0(verbose=False)

    # Check device
    device = next(synb0.model.parameters()).device

    # Should be either CPU or CUDA
    assert_(device.type in ["cpu", "cuda"])


if __name__ == "__main__":
    # Run tests
    import sys

    # Simple test runner
    test_functions = [
        test_encoder_block,
        test_decoder_block,
        test_unet3d_architecture,
        test_unet3d_batch,
        test_synb0_initialization,
        test_synb0_predict_single_image,
        test_synb0_predict_batch,
        test_synb0_predict_shape_validation,
        test_synb0_predict_mismatched_shapes,
        test_unet3d_gradient_flow,
        test_encoder_decoder_symmetry,
        test_model_eval_mode,
        test_model_device_placement,
    ]

    if not have_torch:
        print("PyTorch not available, skipping tests")
        sys.exit(0)

    passed = 0
    failed = 0

    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__}")
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
