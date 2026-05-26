import pytest

pytest.importorskip("fury", minversion="2.0.0a6")

from dipy.viz.skyline.render.sh_billboard import SlicedSphGlyphMaterial  # noqa: E402


class _UniformBuffer:
    """Small uniform buffer stub for material property tests."""

    def __init__(self):
        """Small uniform buffer stub for material property tests."""
        self.data = {
            "active_slice_x": 0.0,
            "active_slice_y": 0.0,
            "active_slice_z": 0.0,
            "vis_x": 0,
        }
        self.update_count = 0

    def update_full(self):
        """Record a full uniform buffer update."""
        self.update_count += 1


class _SlicedSphGlyphMaterial(SlicedSphGlyphMaterial):
    """Test material exposing a lightweight uniform buffer."""

    @property
    def uniform_buffer(self):
        """Return the lightweight uniform buffer."""
        return self._uniform_buffer


def test_active_slice_uniforms_preserve_float_values():
    """Sliced SH material stores active slices as float uniforms."""
    material = _SlicedSphGlyphMaterial.__new__(_SlicedSphGlyphMaterial)
    material._uniform_buffer = _UniformBuffer()

    material.active_slice_x = 1.25
    material.active_slice_y = 2.5
    material.active_slice_z = 3.75

    assert material.active_slice_x == 1.25
    assert material.active_slice_y == 2.5
    assert material.active_slice_z == 3.75
    assert material.uniform_buffer.update_count == 3


def test_visibility_uniforms_still_store_integer_values():
    """Sliced SH material visibility flags remain integer uniforms."""
    material = _SlicedSphGlyphMaterial.__new__(_SlicedSphGlyphMaterial)
    material._uniform_buffer = _UniformBuffer()

    material.vis_x = 1.8

    assert material.vis_x == 1
