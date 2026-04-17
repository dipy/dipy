import numpy as np
import pytest

pytest.importorskip("fury")

from dipy.viz.skyline.render.image import Image3D  # noqa: E402


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
