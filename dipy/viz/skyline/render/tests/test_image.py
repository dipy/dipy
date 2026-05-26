import numpy as np
import pytest

pytest.importorskip("fury")

from dipy.viz.skyline.render.image import Image3D  # noqa: E402
from dipy.viz.skyline.render.renderer import (  # noqa: E402
    slice_slider_bounds,
    slice_slider_values_from_state,
    slice_state_from_slider_values,
)


def _image_stub():
    """Build ``Image3D``-like object for lightweight method tests.

    Returns
    -------
    Image3D
        Instance with the minimum attributes needed by tested methods.
    """
    obj = Image3D.__new__(Image3D)
    obj._synchronize = True
    obj.state = np.array([0.0, 0.0, 0.0], dtype=float)
    obj._has_directions = False
    obj._volume_idx = 0
    obj.dwi = np.zeros((2, 2, 2), dtype=np.float32)
    obj.colormap = "Gray"
    obj.interpolation = "linear"
    obj._scene_calls = []
    obj.apply_scene_op = lambda func, *args: obj._scene_calls.append(
        (getattr(func, "__name__", ""), args)
    )
    return obj


def test_set_interpolation_forces_nearest_for_divergent_colormap():
    """``_set_interpolation`` enforces nearest when colormap is divergent."""

    class _Material:
        def __init__(self):
            self.interpolation = "Linear"

    class _Actor:
        def __init__(self):
            self.material = _Material()

    image = _image_stub()
    image._slicer = type("Slicer", (), {"children": [_Actor(), _Actor()]})()
    image._has_directions = False
    image._value_percentiles = (0, 100)
    image._apply_colormap("Divergent")

    image.interpolation = "nearest"
    assert all(
        actor.material.interpolation == "nearest" for actor in image._slicer.children
    )


def test_update_state_updates_slice_state_without_direction_switch():
    """``update_state`` updates slices and skips volume switch for 3-value state."""
    image = _image_stub()
    image._has_directions = True
    image.dwi = np.zeros((2, 2, 2, 4), dtype=np.float32)
    image._slicer = object()

    image.update_state(np.array([1.0, 2.0, 3.0]))

    assert np.array_equal(image.state, np.array([1.0, 2.0, 3.0]))
    assert image._volume_idx == 0
    assert len(image._scene_calls) == 1


def test_update_state_switches_direction_when_index_is_valid():
    """``update_state`` changes direction and recreates slicer for valid index."""
    image = _image_stub()
    image._has_directions = True
    image.dwi = np.zeros((2, 2, 2, 5), dtype=np.float32)
    image._slicer = object()

    image.update_state(np.array([1.0, 1.0, 1.0, 3.0]))

    assert image._volume_idx == 3
    assert len(image._scene_calls) == 2


def test_image_slice_slider_bounds_use_affine_scaled_shape():
    """Image slice sliders use affine-scaled bounds for large voxels."""
    image = _image_stub()
    image.dwi = np.zeros((10, 20, 30), dtype=np.float32)
    image.affine = np.diag([2.0, 1.0, 3.0, 1.0])

    bounds = slice_slider_bounds(image.dwi.shape[:3], affine=image.affine)

    assert bounds == ((0, 20), (0, 20), (0, 90))


def test_image_slice_slider_value_maps_to_world_state():
    """Image slice slider values map back to world slice coordinates."""
    image = _image_stub()
    image.affine = np.array(
        [
            [2.0, 0.0, 0.0, 10.0],
            [0.0, 0.5, 0.0, 20.0],
            [0.0, 0.0, 3.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    slider_state = np.array([8.0, 4.0, 15.0])

    image.state = slice_state_from_slider_values(slider_state, affine=image.affine)
    displayed_state = slice_slider_values_from_state(image.state, affine=image.affine)

    assert np.allclose(image.state, (18.0, 22.0, 45.0))
    assert np.allclose(displayed_state, slider_state)
