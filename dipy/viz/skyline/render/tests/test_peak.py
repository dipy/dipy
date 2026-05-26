import numpy as np
import pytest

pytest.importorskip("fury", minversion="2.0.0a6")

from dipy.viz.skyline.render.peak import Peak3D  # noqa: E402


class _PeakSlicer:
    """Small peak slicer stub storing cross-section assignments."""

    def __init__(self):
        """Small peak slicer stub storing cross-section assignments."""
        self.cross_section = np.zeros(3, dtype=np.float32)


def test_voxel_from_world_state_uses_inverse_affine_and_clips():
    """Peak world state is mapped to clipped voxel coordinates."""
    peak = Peak3D.__new__(Peak3D)
    peak.affine = np.diag([2.0, 0.5, 3.0, 1.0])
    peak.peaks = np.zeros((5, 7, 9, 3), dtype=np.float32)

    voxel_state = peak._voxel_from_world_state((20.0, 2.0, 12.0))

    assert np.array_equal(voxel_state, np.array([4, 4, 4], dtype=np.int16))


def test_apply_cross_section_from_world_state_sets_world_cross_section():
    """Peak cross-section keeps world coordinates when the slicer uses world space."""
    peak = Peak3D.__new__(Peak3D)
    peak.affine = np.diag([2.0, 2.0, 2.0, 1.0])
    peak.peaks = np.zeros((11, 11, 11, 3), dtype=np.float32)
    peak.state = np.array([20.0, 10.0, 4.0], dtype=np.float32)
    peak._cross_section_space = "world"
    peak._slicer = _PeakSlicer()

    peak._apply_cross_section_from_state()

    assert np.allclose(peak._slicer.cross_section, (20.0, 10.0, 4.0))
    assert np.allclose(peak._cross_section_state, (20.0, 10.0, 4.0))
