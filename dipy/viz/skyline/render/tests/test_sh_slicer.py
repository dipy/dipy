import numpy as np
import pytest

pytest.importorskip("fury", minversion="2.0.0a6")

from dipy.viz.skyline.render.sh_slicer import SHGlyph3D  # noqa: E402


class _SHSlicer:
    """Small SH slicer stub recording slice updates."""

    def __init__(self):
        """Small SH slicer stub recording slice updates."""
        self.slices = []

    def set_slice(self, axis, idx):
        """Record the slice update.

        Parameters
        ----------
        axis : str
            Axis name.
        idx : float
            Slice coordinate.
        """
        self.slices.append((axis, idx))


def test_set_slices_passes_world_coordinates_to_shader_slicer():
    """SH slices are set with world coordinates for affine-scaled data."""
    glyph = SHGlyph3D.__new__(SHGlyph3D)
    glyph.affine = np.diag([2.0, 2.0, 2.0, 1.0])
    glyph.state = np.array([20.0, 10.0, 4.0], dtype=float)
    glyph._last_state = [-1, -1, -1]
    glyph._slicer = _SHSlicer()

    glyph.set_slices()

    assert glyph._slicer.slices == [("x", 20.0), ("y", 10.0), ("z", 4.0)]
    assert glyph._last_state == [20.0, 10.0, 4.0]
