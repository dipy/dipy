"""Tests for dipy.viz.skyline.render.image."""

import numpy as np
import pytest

try:
    from dipy.viz.skyline.render.image import Image3D

    HAS_SKYLINE = True
except Exception:
    HAS_SKYLINE = False

pytestmark = pytest.mark.skipif(not HAS_SKYLINE, reason="Requires Fury >= 2.0.0a6")


@pytest.fixture
def image_3d_4d():
    """Fixture to create an Image3D with 4D data."""
    rng = np.random.default_rng(42)
    data = rng.random((20, 20, 20, 5)).astype(np.float32)
    affine = np.eye(4)
    return Image3D("test_4d", data, affine=affine, render_callback=None)


def test_slice_state_preserved_on_direction_change(image_3d_4d):
    """Slice positions must not reset when changing gradient direction.

    Regression test for https://github.com/dipy/dipy/issues/3890
    """
    img = image_3d_4d

    # Move to a non-default slice position
    img.state = np.array([3.0, 7.0, 12.0])
    saved_state = img.state.copy()

    # Simulate direction change — this calls _create_slicer_actor internally
    img._volume_idx = 1
    img._create_slicer_actor()

    np.testing.assert_array_equal(
        img.state,
        saved_state,
        err_msg="Slice state reset after direction change (issue #3890)",
    )
