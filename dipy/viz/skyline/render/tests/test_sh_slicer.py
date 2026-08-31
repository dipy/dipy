import numpy as np
import pytest

from dipy.utils.optpkg import optional_package

_, has_fury, _ = optional_package("fury", min_version="2.0.0")
if not has_fury:
    pytest.skip("Requires fury>=2.0.0", allow_module_level=True)
else:
    from dipy.viz.skyline.render.sh_slicer import SHGlyph3D


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


def _make_glyph(shape, state, affine):
    """Build a bare ``SHGlyph3D`` wired to a recording slicer stub.

    Parameters
    ----------
    shape : tuple of int
        Spatial shape of the SH volume.
    state : array-like
        Current slicing state in world coordinates.
    affine : ndarray
        Voxel-to-world affine.

    Returns
    -------
    SHGlyph3D
        Instance with ``_slicer`` replaced by a recording stub.
    """
    glyph = SHGlyph3D.__new__(SHGlyph3D)
    glyph.affine = affine
    glyph.shape = shape
    glyph.state = np.asarray(state, dtype=float)
    glyph._last_state = [-1, -1, -1]
    glyph._slicer = _SHSlicer()
    return glyph


def test_set_slices_passes_voxel_coordinates_to_shader_slicer():
    """SH slices are set with voxel coordinates for affine-scaled data."""
    glyph = _make_glyph((30, 20, 10), (20.0, 10.0, 4.0), np.diag([2.0, 2.0, 2.0, 1.0]))

    glyph.set_slices()

    assert glyph._slicer.slices == [("x", 10.0), ("y", 5.0), ("z", 2.0)]
    assert glyph._last_state == [20.0, 10.0, 4.0]


def test_set_slices_clips_voxel_coordinates_to_volume_shape():
    """Out-of-bounds world states are clipped to valid voxel indices."""
    glyph = _make_glyph(
        (30, 20, 10), (-4.0, 10.0, 100.0), np.diag([2.0, 2.0, 2.0, 1.0])
    )

    glyph.set_slices()

    assert glyph._slicer.slices == [("x", 0.0), ("y", 5.0), ("z", 9.0)]
    assert glyph._last_state == [-4.0, 10.0, 100.0]
