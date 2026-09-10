import numpy as np
import numpy.testing as npt
import pytest

from dipy.data import default_sphere
from dipy.direction.peaks import PeaksAndMetrics
from dipy.utils.optpkg import optional_package

_, has_fury, _ = optional_package("fury", min_version="2.0.0")
if not has_fury:
    pytest.skip("Requires fury>=2.0.0", allow_module_level=True)
else:
    from dipy.viz.skyline.render.peak import Peak3D, create_peak_visualization

SHAPE = (5, 7, 9)


def _peak_dirs(shape=SHAPE, n_peaks=3):
    """One unit peak along +x in every voxel, plus zeroed extra peaks."""
    dirs = np.zeros((*shape, n_peaks, 3), dtype=np.float32)
    dirs[..., 0, 0] = 1.0
    return dirs


def _pam(shape=SHAPE, affine=None, n_peaks=3):
    pam = PeaksAndMetrics()
    pam.affine = np.eye(4) if affine is None else affine
    pam.peak_dirs = _peak_dirs(shape, n_peaks)
    pam.peak_values = np.ones((*shape, n_peaks), dtype=np.float32)
    pam.peak_indices = np.zeros((*shape, n_peaks), dtype=np.int32)
    pam.sphere = default_sphere
    return pam


def _peak(affine=None, shape=SHAPE):
    return Peak3D(
        "peaks.pam5",
        _peak_dirs(shape),
        affine=np.eye(4) if affine is None else affine,
    )


@pytest.mark.parametrize("bad_input", ["not a tuple", (), (1, 2, 3)])
def test_create_peak_visualization_rejects_invalid_input(bad_input):
    with pytest.raises(ValueError, match="Input must be a tuple"):
        create_peak_visualization(bad_input, 0)


def test_create_peak_visualization_names_by_index():
    viz = create_peak_visualization((_pam(),), 2)

    assert viz.path == "Peaks_2"
    assert isinstance(viz, Peak3D)
    assert viz.viz_type == "peak"


def test_create_peak_visualization_uses_the_given_filename():
    viz = create_peak_visualization((_pam(), "peaks.pam5"), 0)

    assert viz.path == "peaks.pam5"


def test_create_peak_visualization_carries_the_pam_geometry():
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    viz = create_peak_visualization((_pam(affine=affine), "peaks.pam5"), 0)

    assert viz.peaks.shape == (*SHAPE, 3, 3)
    npt.assert_allclose(viz.affine, affine)


def test_create_peak_visualization_forwards_the_opacity():
    viz = create_peak_visualization((_pam(), "peaks.pam5"), 0, opacity=40)

    assert viz.opacity == 40


def test_peak3d_defaults():
    peak = _peak()

    assert peak._scale == 1.0
    assert peak._synchronize is True
    assert peak._slice_visibility == [True, True, True]
    assert peak.actor is peak._slicer
    assert len(peak.actor.children) >= 1


def test_peak3d_info_lists_shape_dtype_and_affine():
    peak = _peak()

    info = peak._populate_info()

    assert f"Peaks shape: {(*SHAPE, 3, 3)}" in info
    assert "Peaks dtype: float32" in info
    assert "Affine:" in info


def test_peak3d_info_without_an_affine():
    peak = Peak3D("peaks.pam5", _peak_dirs(), affine=None)

    assert "Affine:" not in peak._populate_info()


def test_peak3d_bounds_follow_the_identity_affine():
    peak = _peak()

    npt.assert_allclose(peak.bounds[0], (0, 0, 0))
    npt.assert_allclose(peak.bounds[1], np.array(SHAPE) - 1)


def test_peak3d_bounds_follow_a_scaled_affine():
    peak = _peak(affine=np.diag([2.0, 2.0, 2.0, 1.0]))

    npt.assert_allclose(peak.bounds[0], (0, 0, 0))
    npt.assert_allclose(peak.bounds[1], (np.array(SHAPE) - 1) * 2.0)


def test_peak3d_bounds_without_an_affine():
    peak = Peak3D("peaks.pam5", _peak_dirs(), affine=None)

    npt.assert_allclose(peak.bounds[1], np.array(SHAPE) - 1)


def test_peak3d_cross_section_is_shared_by_every_chunk():
    peak = _peak()

    peak._set_cross_section(np.array([1, 2, 3], dtype=np.int16))

    for chunk in peak.actor.children:
        npt.assert_array_equal(chunk.cross_section, (1, 2, 3))
    npt.assert_array_equal(peak._get_cross_section(), (1, 2, 3))


def test_peak3d_voxel_from_world_state_inverts_the_affine():
    peak = _peak(affine=np.diag([2.0, 0.5, 3.0, 1.0]))

    voxel_state = peak._voxel_from_world_state((4.0, 2.0, 6.0))

    npt.assert_array_equal(voxel_state, np.array([2, 4, 2], dtype=np.int16))


def test_peak3d_voxel_from_world_state_clips_to_the_volume():
    peak = _peak(affine=np.diag([2.0, 0.5, 3.0, 1.0]))

    voxel_state = peak._voxel_from_world_state((1000.0, 1000.0, 1000.0))

    npt.assert_array_equal(voxel_state, np.array(SHAPE, dtype=np.int16) - 1)


def test_peak3d_voxel_from_world_state_clips_negatives_to_zero():
    peak = _peak(affine=np.diag([2.0, 0.5, 3.0, 1.0]))

    voxel_state = peak._voxel_from_world_state((-50.0, -50.0, -50.0))

    npt.assert_array_equal(voxel_state, np.zeros(3, dtype=np.int16))


def test_peak3d_without_an_affine_uses_voxel_cross_sections():
    peak = Peak3D("peaks.pam5", _peak_dirs(), affine=None)

    peak.state = np.array([1.4, 2.6, 3.5], dtype=np.float32)
    peak._apply_cross_section_from_state()

    npt.assert_array_equal(peak._cross_section_state, np.round((1.4, 2.6, 3.5)))
    npt.assert_array_equal(
        peak.actor.children[0].cross_section, np.round((1.4, 2.6, 3.5))
    )


def test_peak3d_cross_section_space_is_voxel_without_an_affine():
    peak = Peak3D("peaks.pam5", _peak_dirs(), affine=None)

    assert peak._cross_section_space == "voxel"


def test_peak3d_cross_section_space_is_inferred_from_the_affine():
    peak = _peak(affine=np.diag([4.0, 4.0, 4.0, 1.0]))

    assert peak._cross_section_space in {"world", "voxel"}
    assert peak._infer_cross_section_space() == peak._cross_section_space


def test_peak3d_update_state_moves_the_cross_section():
    peak = _peak(affine=np.diag([2.0, 2.0, 2.0, 1.0]))
    target = np.array([4.0, 6.0, 8.0], dtype=np.float32)

    peak.update_state(target)

    npt.assert_allclose(peak.state, target)
    expected_voxel = peak._voxel_from_world_state(target)
    if peak._cross_section_space == "world":
        npt.assert_allclose(peak._cross_section_state, expected_voxel * 2.0)
    else:
        npt.assert_allclose(peak._cross_section_state, expected_voxel)


def test_peak3d_update_state_ignores_extra_components():
    peak = _peak()

    peak.update_state(np.array([1.0, 2.0, 3.0, 9.0]))

    npt.assert_allclose(peak.state, (1.0, 2.0, 3.0))


def test_peak3d_update_state_is_ignored_when_sync_is_off():
    peak = _peak()
    before = np.array(peak.state, dtype=np.float32).copy()
    peak._synchronize = False

    peak.update_state(np.array([4.0, 4.0, 4.0]))

    npt.assert_allclose(peak.state, before)


def test_peak3d_set_slice_visibility_reaches_every_chunk():
    peak = _peak()

    peak._set_slice_visibility((True, False, True))

    for chunk in peak.actor.children:
        assert tuple(chunk.material.visibility) == (True, False, True)


def test_peak3d_set_opacity_reaches_every_chunk():
    peak = _peak()

    peak._set_opacity(0.4)

    for chunk in peak.actor.children:
        assert chunk.material.opacity == pytest.approx(0.4)


def test_peak3d_scale_is_applied_when_the_actor_is_rebuilt():
    peak = _peak()

    peak._scale = 2.0
    peak._create_peak_actor()

    assert peak._scale == 2.0
    assert peak.actor is peak._slicer
