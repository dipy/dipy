import numpy as np
import pytest

pytest.importorskip("fury", minversion="2.0.0")

from dipy.viz.skyline.render.peak import (  # noqa: E402
    Peak3D,
    create_peak_visualization,
)


class _PeakChunk:
    """Single ``VectorField`` chunk stub storing cross-section assignments."""

    def __init__(self):
        """Single ``VectorField`` chunk stub storing cross-section assignments."""
        self.cross_section = np.zeros(3, dtype=np.float32)


class _PeakSlicer:
    """Peak slicer stub mirroring the ``Group`` of chunks ``peaks_slicer`` returns.

    Only the chunk actors carry ``cross_section``; the group itself does not.
    """

    def __init__(self, *, n_chunks=2):
        """Peak slicer stub mirroring the ``Group`` of chunks returned by Fury."""
        self.children = [_PeakChunk() for _ in range(n_chunks)]


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

    # every chunk of the slicer group must be moved, not just the first
    for chunk in peak._slicer.children:
        assert np.allclose(chunk.cross_section, (20.0, 10.0, 4.0))
    assert np.allclose(peak._cross_section_state, (20.0, 10.0, 4.0))


def _pam(shape=(6, 7, 8, 3), *, affine=None):
    """Minimal ``PeaksAndMetrics``-like object for ``create_peak_visualization``."""

    class _PAM:
        pass

    pam = _PAM()
    rng = np.random.default_rng(42)
    pam.peak_dirs = rng.random((*shape, 3)).astype(np.float32)
    pam.peak_values = rng.random(shape).astype(np.float32)
    pam.affine = affine
    return pam


@pytest.mark.parametrize("affine", [None, np.diag([2.0, 2.0, 2.0, 1.0])])
def test_create_peak_visualization_builds_real_actor(affine):
    """``create_peak_visualization`` constructs a Fury-backed ``Peak3D``."""
    peak = create_peak_visualization((_pam(affine=affine), "peaks.pam5"), 0)

    assert peak.actor is not None
    # peaks_slicer returns a Group of chunks; cross_section lives on the chunks
    assert peak.actor.children
    assert np.asarray(peak.state).shape == (3,)
    assert np.allclose(peak._get_cross_section(), peak.actor.children[0].cross_section)


def test_peak_cross_section_update_reaches_every_chunk():
    """Updating the state propagates the cross section to all chunk actors."""
    peak = create_peak_visualization((_pam(),), 0)

    peak.state = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    peak._apply_cross_section_from_state()

    for chunk in peak.actor.children:
        assert np.allclose(chunk.cross_section, (1.0, 2.0, 3.0))


def test_peak_set_opacity_reaches_every_chunk():
    """Opacity is applied to each chunk material, not the material-less group."""
    peak = create_peak_visualization((_pam(),), 0)

    peak._set_opacity(0.4)

    assert peak.actor.material is None  # the group itself carries no material
    for chunk in peak.actor.children:
        assert np.isclose(chunk.material.opacity, 0.4)


def test_peak_set_slice_visibility_reaches_every_chunk():
    """Per-axis slice visibility is applied to each chunk material."""
    peak = create_peak_visualization((_pam(),), 0)

    peak._set_slice_visibility((False, True, False))

    for chunk in peak.actor.children:
        assert list(chunk.material.visibility) == [False, True, False]
