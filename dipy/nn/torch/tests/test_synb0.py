#!/usr/bin/python
"""Tests for Synb0 PyTorch implementation."""

import numpy as np
from numpy.testing import assert_, assert_equal
import pytest

from dipy.nn.torch.synb0 import Synb0, UNet3D
from dipy.testing.decorators import set_random_number_generator
from dipy.utils.optpkg import optional_package

torch, have_torch, _ = optional_package("torch", min_version="2.2.0")

skip_no_torch = pytest.mark.skipif(not have_torch, reason="Requires PyTorch >= 2.2.0")


@skip_no_torch
@set_random_number_generator()
def test_unet3d_architecture(rng):
    """Test the UNet3D architecture."""
    # Create model
    model = UNet3D(n_in=2, n_out=1)

    # Create random input (batch, channels, depth, height, width)
    x = torch.randn(1, 2, 80, 96, 80)

    # Forward pass
    output = model(x)

    # Check output shape
    assert_equal(output.shape, (1, 1, 80, 96, 80))


@skip_no_torch
@set_random_number_generator()
def test_unet3d_batch(rng):
    """Test UNet3D with batch size > 1."""
    # Create model
    model = UNet3D(n_in=2, n_out=1)

    # Create batch input
    batch_size = 2
    x = torch.randn(batch_size, 2, 80, 96, 80)

    # Forward pass
    output = model(x)

    # Check output shape
    assert_equal(output.shape, (batch_size, 1, 80, 96, 80))


@skip_no_torch
@set_random_number_generator()
def test_synb0_initialization(rng):
    """Test Synb0 class initialization."""
    # Create Synb0 instance
    synb0 = Synb0(verbose=False)

    # Check that model is created
    assert_(hasattr(synb0, "model"))
    assert_(isinstance(synb0.model, UNet3D))

    # Check that device is set
    assert_(hasattr(synb0, "device"))


@skip_no_torch
@set_random_number_generator()
def test_synb0_predict_single_image(rng):
    """Test Synb0 prediction with single image."""
    # Create Synb0 instance
    synb0 = Synb0(verbose=False)

    # Create dummy data with correct shape (77, 91, 77)
    b0 = rng.random((77, 91, 77)).astype(np.float32) * 1000
    T1 = rng.random((77, 91, 77)).astype(np.float32) * 150

    # Test prediction without average (faster for testing)
    # Skip if default weights are not available (network fetch may fail)
    pytest.importorskip("requests")
    prediction = synb0.predict(b0, T1, average=False)

    # Check output shape
    assert_equal(prediction.shape, (77, 91, 77))

    # Check output is not all zeros
    assert_(np.any(prediction != 0))


@skip_no_torch
@set_random_number_generator()
def test_synb0_predict_batch(rng):
    """Test Synb0 prediction with batch of images."""
    # Create Synb0 instance
    synb0 = Synb0(verbose=False)

    # Create batch of dummy data
    batch_size = 2
    b0 = rng.random((batch_size, 77, 91, 77)).astype(np.float32) * 1000
    T1 = rng.random((batch_size, 77, 91, 77)).astype(np.float32) * 150

    # Skip if default weights are not available
    pytest.importorskip("requests")
    prediction = synb0.predict(b0, T1, batch_size=1, average=False)

    # Check output shape
    assert_equal(prediction.shape, (batch_size, 77, 91, 77))


@skip_no_torch
@set_random_number_generator()
def test_synb0_predict_shape_validation(rng):
    """Test that Synb0 validates input shapes."""
    # Create Synb0 instance
    synb0 = Synb0(verbose=False)

    # Create data with wrong shape
    b0_wrong = rng.random((80, 80, 80)).astype(np.float32)
    T1_wrong = rng.random((80, 80, 80)).astype(np.float32)

    with pytest.raises(ValueError, match="Expected shape"):
        synb0.predict(b0_wrong, T1_wrong, average=False)


@skip_no_torch
@set_random_number_generator()
def test_synb0_predict_mismatched_shapes(rng):
    """Test that Synb0 detects mismatched input shapes."""
    # Create Synb0 instance
    synb0 = Synb0(verbose=False)

    # Create data with mismatched shapes
    b0 = rng.random((77, 91, 77)).astype(np.float32)
    T1 = rng.random((80, 96, 80)).astype(np.float32)

    with pytest.raises(ValueError, match="Expected shape"):
        synb0.predict(b0, T1, average=False)


@skip_no_torch
@set_random_number_generator()
def test_unet3d_gradient_flow(rng):
    """Test that gradients can flow through UNet3D."""
    # Create model
    model = UNet3D(n_in=2, n_out=1)
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


@skip_no_torch
@set_random_number_generator()
def test_model_eval_mode(rng):
    """Test that model is in eval mode for inference."""
    # Create Synb0 instance
    synb0 = Synb0(verbose=False)

    # Model should be in eval mode
    assert_(not synb0.model.training)


@skip_no_torch
@set_random_number_generator()
def test_model_device_placement(rng):
    """Test that model is placed on correct device."""
    # Create Synb0 instance
    synb0 = Synb0(verbose=False)

    # Check device
    device = next(synb0.model.parameters()).device

    # Should be either CPU or CUDA
    assert_(device.type in ["cpu", "cuda"])
