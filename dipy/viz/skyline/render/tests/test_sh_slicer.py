import numpy as np
import numpy.testing as npt
import pytest

from dipy.utils.optpkg import optional_package

_, has_fury, _ = optional_package("fury", min_version="2.0.0")
if not has_fury:
    pytest.skip("Requires fury>=2.0.0", allow_module_level=True)
else:
    from fury import window

    from dipy.viz.skyline.render.renderer import affine_voxel_sizes
    from dipy.viz.skyline.render.sh_slicer import (
        SHGlyph3D,
        SHSlicer,
        _descoteaux_to_fury_standard,
        create_shm_visualization,
    )

SH_ORDER = 8
N_DESCOTEAUX = sum(2 * ell + 1 for ell in range(0, SH_ORDER + 1, 2))
SHAPE = (6, 5, 4)


def _coeffs(shape=SHAPE, n_coeffs=N_DESCOTEAUX):
    coeffs = np.zeros((*shape, n_coeffs), dtype=np.float32)
    coeffs[..., 0] = 1.0
    return coeffs


def _glyph(affine=None, shape=SHAPE, **kwargs):
    return create_shm_visualization(
        (_coeffs(shape), np.eye(4) if affine is None else affine, "odf.pam5"),
        0,
        **kwargs,
    )


def _material(glyph):
    return glyph._slicer._glyph_actor.material


@pytest.mark.parametrize("bad_input", ["not a tuple", (), (1,), (1, 2, 3, 4, 5)])
def test_create_shm_visualization_rejects_invalid_input(bad_input):
    with pytest.raises(ValueError, match="Input must be a tuple"):
        create_shm_visualization(bad_input, 0)


def test_create_shm_visualization_names_by_index():
    viz = create_shm_visualization((_coeffs(), np.eye(4)), 5)

    assert viz.path == "SH_Glyphs_5"
    assert viz.name == "ODFs (SH_Glyphs_5)"
    assert isinstance(viz, SHGlyph3D)
    assert viz.viz_type == "sh_glyph"


def test_create_shm_visualization_uses_the_given_filename():
    viz = create_shm_visualization((_coeffs(), np.eye(4), "odf.pam5"), 0)

    assert viz.path == "odf.pam5"


def test_create_shm_visualization_takes_the_basis_from_a_four_tuple():
    viz = create_shm_visualization(
        (_coeffs(), np.eye(4), "odf.pam5", "descoteaux07"), 0
    )

    assert viz.shape == SHAPE


def test_descoteaux_to_fury_standard_expands_to_the_full_basis():
    coeffs = np.arange(N_DESCOTEAUX, dtype=np.float32).reshape(1, 1, 1, -1)

    converted = _descoteaux_to_fury_standard(coeffs, SH_ORDER)

    assert converted.shape == (1, 1, 1, (SH_ORDER + 1) ** 2)
    assert converted.dtype == coeffs.dtype


def test_descoteaux_to_fury_standard_mirrors_the_order_of_m():
    coeffs = np.zeros((1, 1, 1, N_DESCOTEAUX), dtype=np.float32)
    coeffs[0, 0, 0, 0] = 1.0
    coeffs[0, 0, 0, 1] = 2.0

    converted = _descoteaux_to_fury_standard(coeffs, SH_ORDER)

    assert converted[0, 0, 0, 0] == 1.0
    assert converted[0, 0, 0, 2 * 2 + 2 + 2] == 2.0


def test_descoteaux_to_fury_standard_leaves_odd_orders_empty():
    coeffs = np.ones((1, 1, 1, N_DESCOTEAUX), dtype=np.float32)

    converted = _descoteaux_to_fury_standard(coeffs, SH_ORDER)

    npt.assert_array_equal(converted[0, 0, 0, 1:4], np.zeros(3))
    assert np.count_nonzero(converted) == N_DESCOTEAUX


def test_sh_slicer_caps_l_max_to_a_lower_order_present_in_descoteaux_coeffs():
    """A default/too-large l_max must never crash on a lower-order file."""
    order4_ncoeffs = sum(2 * ell + 1 for ell in range(0, 4 + 1, 2))  # 15
    coeffs = np.zeros((2, 2, 2, order4_ncoeffs), dtype=np.float32)
    coeffs[..., 0] = 1.0

    slicer = SHSlicer(coeffs, basis_type="descoteaux07")  # default l_max=8

    assert slicer.l_max == 4
    assert slicer.n_coeffs == (4 + 1) ** 2


def test_sh_slicer_truncates_higher_order_descoteaux_coeffs_to_l_max():
    """A file with more detail than requested is still capped at l_max."""
    order10_ncoeffs = sum(2 * ell + 1 for ell in range(0, 10 + 1, 2))  # 66
    coeffs = np.zeros((2, 2, 2, order10_ncoeffs), dtype=np.float32)
    coeffs[..., 0] = 1.0

    slicer = SHSlicer(coeffs, l_max=8, basis_type="descoteaux07")

    assert slicer.l_max == 8
    assert slicer.n_coeffs == (8 + 1) ** 2


def test_sh_slicer_rejects_invalid_descoteaux_coefficient_count():
    """A coefficient count that isn't a valid even-order SH basis size
    must raise, not silently misinterpret or IndexError deep inside the
    conversion loop."""
    coeffs = np.zeros((2, 2, 2, 20), dtype=np.float32)

    with pytest.raises(ValueError):
        SHSlicer(coeffs, basis_type="descoteaux07")


def test_sh_glyph_starts_at_the_volume_center():
    glyph = _glyph()

    assert glyph.shape == SHAPE
    npt.assert_allclose(glyph.bounds[0], (0, 0, 0))
    npt.assert_allclose(glyph.bounds[1], np.array(SHAPE) - 1)
    npt.assert_array_equal(glyph.state, np.mean(glyph.bounds, axis=0).astype(int))


def test_sh_glyph_info_lists_dimensions_and_order():
    glyph = _glyph()

    info = glyph._populate_info()

    assert f"Dimensions: {SHAPE}" in info
    assert f"SH Order: {glyph._slicer.l_max}" in info
    assert f"SH Coefficients: {glyph._slicer.n_coeffs}" in info


def test_sh_glyph_info_without_an_affine():
    glyph = SHGlyph3D("odf.pam5", _coeffs(), affine=None, basis_type="descoteaux07")

    info = glyph._populate_info()
    assert f"Dimensions: {SHAPE}" in info
    assert "Voxel Order:" not in info
    assert "Affine:" not in info


def test_sh_glyph_info_reports_voxel_order_and_affine():
    affine = np.diag([-1.0, 1.0, 1.0, 1.0])
    glyph = SHGlyph3D("odf.pam5", _coeffs(), affine=affine, basis_type="descoteaux07")

    info = glyph._populate_info()
    assert "Voxel Order: LAS" in info
    assert "Affine:" in info


def test_sh_glyph_actor_is_the_slicer_group():
    glyph = _glyph()

    assert glyph.actor is glyph._slicer.actor
    assert glyph._slicer._glyph_actor is not None


def test_sh_glyph_set_slices_passes_world_positions_with_diagonal_affine():
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    glyph = _glyph(affine=affine)

    glyph.state = np.array([4.0, 6.0, 2.0])
    glyph.set_slices()

    material = _material(glyph)
    assert material.active_slice_x == 4.0
    assert material.active_slice_y == 6.0
    assert material.active_slice_z == 2.0
    npt.assert_allclose(glyph._last_state, (4.0, 6.0, 2.0))


def test_sh_glyph_set_slices_snaps_to_the_nearest_voxel_with_rotated_affine():
    """Slicing snaps the continuous state to the nearest voxel's exact world
    center (matching how ``Peak3D`` derives its cross section), so glyphs on
    the same voxel-grid plane line up under a rotated affine.
    """
    affine = np.array(
        [
            [-2.5, 0.08, 0.07, 113.64],
            [0.07, 2.45, -0.49, -104.41],
            [0.08, 0.49, 2.45, -31.5],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    glyph = _glyph(affine=affine)

    glyph.state = np.array([50.0, -20.0, 10.0])
    glyph.set_slices()

    nearest_voxel = glyph._voxel_from_world_state(glyph.state)
    npt.assert_array_equal(nearest_voxel, [5, 4, 3])
    expected_world = (affine @ np.r_[nearest_voxel, 1.0])[:3]

    material = _material(glyph)
    assert material.active_slice_x == pytest.approx(expected_world[0])
    assert material.active_slice_y == pytest.approx(expected_world[1])
    assert material.active_slice_z == pytest.approx(expected_world[2])


def test_sh_glyph_set_slices_clips_the_voxel_index_with_affine():
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    glyph = _glyph(affine=affine)

    glyph.state = np.array([-40.0, 6.0, 1000.0])
    glyph.set_slices()

    material = _material(glyph)
    assert material.active_slice_x == 0.0
    assert material.active_slice_y == 6.0
    assert material.active_slice_z == 6.0
    npt.assert_allclose(glyph._last_state, (-40.0, 6.0, 1000.0))


def test_sh_glyph_set_slices_clips_to_volume_without_affine():
    glyph = SHGlyph3D("test", _coeffs(), affine=None, basis_type="descoteaux07")

    glyph.state = np.array([-5.0, 2.0, 100.0])
    glyph.set_slices()

    material = _material(glyph)
    assert material.active_slice_x == 0.0
    assert material.active_slice_y == 2.0
    assert material.active_slice_z == float(SHAPE[2] - 1)


def test_sh_glyph_update_state_moves_the_slices():
    glyph = _glyph()

    glyph.update_state(np.array([1.0, 2.0, 3.0, 9.0]))

    npt.assert_allclose(glyph.state, (1.0, 2.0, 3.0))
    assert _material(glyph).active_slice_y == 2.0


def test_sh_glyph_update_state_is_ignored_when_sync_is_off():
    glyph = _glyph()
    before = np.array(glyph.state, dtype=float).copy()
    glyph._synchronize = False

    glyph.update_state(np.array([0.0, 0.0, 0.0]))

    npt.assert_allclose(glyph.state, before)


def test_sh_glyph_hiding_an_axis_clears_its_visibility_uniform():
    glyph = _glyph()

    glyph._slice_visibility = [True, False, True]
    glyph.set_slice_visibility()

    material = _material(glyph)
    assert material.vis_x == 1
    assert material.vis_y == 0
    assert material.vis_z == 1
    assert glyph._last_state[1] == -1


def test_sh_glyph_showing_an_axis_restores_its_visibility_uniform():
    glyph = _glyph()
    glyph._slice_visibility = [False, False, False]
    glyph.set_slice_visibility()

    glyph._slice_visibility = [True, True, True]
    glyph.set_slice_visibility()

    material = _material(glyph)
    assert (material.vis_x, material.vis_y, material.vis_z) == (1, 1, 1)
    npt.assert_allclose(glyph._last_state, glyph.state)


def test_sh_slicer_set_slice_is_a_noop_for_an_unchanged_index():
    glyph = _glyph()
    glyph._slicer.set_slice("x", 2.0)

    glyph._slicer.set_slice("x", 2.0)

    assert glyph._slicer._cur["x"] == 2.0
    assert _material(glyph).active_slice_x == 2.0


def test_sh_slicer_set_scale_rescales_the_glyphs():
    glyph = _glyph()
    slicer = glyph._slicer
    original_scale = slicer.scale

    slicer.set_scale(original_scale * 2.0)

    assert slicer.scale == pytest.approx(original_scale * 2.0)
    assert _material(glyph).scale == pytest.approx(original_scale * 2.0)


def test_sh_slicer_set_scale_ignores_an_unchanged_value():
    glyph = _glyph()
    slicer = glyph._slicer

    slicer.set_scale(slicer.scale)

    assert slicer.scale == pytest.approx(slicer.scale)


def test_sh_slicer_set_opacity_switches_the_alpha_mode():
    glyph = _glyph()
    slicer = glyph._slicer

    slicer.set_opacity(0.4)

    material = _material(glyph)
    assert material.opacity == pytest.approx(0.4)
    assert material.alpha_mode == "blend"

    slicer.set_opacity(1.0)

    assert material.alpha_mode == "solid"


def test_sh_slicer_skips_an_all_zero_volume():
    coeffs = np.zeros((*SHAPE, N_DESCOTEAUX), dtype=np.float32)

    glyph = create_shm_visualization((coeffs, np.eye(4), "empty.pam5"), 0)

    assert glyph._slicer._glyph_actor is None
    assert len(glyph.actor.children) == 0


def test_sh_slicer_honours_a_mask():
    mask = np.zeros(SHAPE, dtype=bool)
    mask[0, 0, 0] = True

    glyph = _glyph(mask=mask)

    assert glyph._slicer._glyph_actor is not None
    assert glyph._slicer.mask is mask


def test_sh_slicer_masking_everything_out_leaves_no_actor():
    glyph = _glyph(mask=np.zeros(SHAPE, dtype=bool))

    assert glyph._slicer._glyph_actor is None


def test_sh_glyph_offscreen_slice_visibility():
    """Sparse nonzero voxels under a rotated affine: only on-slice glyphs render."""
    affine = np.array(
        [
            [0, 2, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )
    coeffs = np.zeros((4, 3, 5, N_DESCOTEAUX), dtype=np.float32)
    # Place nonzero SH at exactly two voxels: (1, 1, 1) and (3, 2, 4).
    coeffs[1, 1, 1, 0] = 1.0
    coeffs[3, 2, 4, 0] = 1.0
    glyph = create_shm_visualization((coeffs, affine, "odf.pam5"), 0)
    actor = glyph.actor

    scene = window.Scene()
    scene.add(actor)

    # Baseline: hide all axes -> nothing visible.
    glyph._slicer.hide_axis("x")
    glyph._slicer.hide_axis("y")
    glyph._slicer.hide_axis("z")
    baseline = window.snapshot(scene=scene, return_array=True, fname=None)
    baseline_fg = int(np.sum(baseline > 0))

    # Set slice to world x=2 (the transformed center of voxel (1, 1, 1)).
    glyph._slicer.show_axis("x")
    glyph._slicer.set_slice("x", 2.0)
    arr_x1 = window.snapshot(scene=scene, return_array=True, fname=None)
    fg_x1 = int(np.sum(arr_x1 > 0))
    # Voxel (1, 1, 1) transforms to world x=2 -> visible; strictly more foreground.
    assert fg_x1 > baseline_fg, (
        f"Expected visible glyph at world x=2: fg={fg_x1} vs baseline={baseline_fg}"
    )

    # Move slice to world x=4 (the transformed center of voxel (3, 2, 4)).
    glyph._slicer.set_slice("x", 4.0)
    arr_x3 = window.snapshot(scene=scene, return_array=True, fname=None)
    fg_x3 = int(np.sum(arr_x3 > 0))
    # Voxel (3, 2, 4) transforms to world x=4 -> visible.
    assert fg_x3 > baseline_fg, (
        f"Expected visible glyph at world x=4: fg={fg_x3} vs baseline={baseline_fg}"
    )

    # Move to world x=0, far outside both voxels' tolerance -- matches baseline.
    glyph._slicer.set_slice("x", 0.0)
    arr_x0 = window.snapshot(scene=scene, return_array=True, fname=None)
    fg_x0 = int(np.sum(arr_x0 > 0))
    assert fg_x0 <= baseline_fg + 10, (
        f"Expected no glyphs at world x=0: fg={fg_x0} vs baseline={baseline_fg}"
    )


def test_sh_glyph_slice_plane_excludes_one_voxel_step_away_under_rotation():
    """A genuine (non-permutation) rotation must not blur adjacent layers.

    The prior permutation-only affine coverage cannot detect an
    over-generous slice-plane tolerance, since a pure axis swap makes the
    per-row L1 sum equal the per-row max. Here the affine mixes two axes
    through a real rotation angle, so the sum and max genuinely differ:
    the voxel one grid step away along ``i`` sits closer to the active
    slice than the (too generous) L1-sum tolerance, but farther than the
    correct max tolerance.
    """
    theta = np.radians(50.0)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    spacing = 2.0
    affine = np.array(
        [
            [spacing * cos_t, -spacing * sin_t, 0, 0],
            [spacing * sin_t, spacing * cos_t, 0, 0],
            [0, 0, spacing, 0],
            [0, 0, 0, 1],
        ]
    )
    shape = (6, 6, 6)
    coeffs = np.zeros((*shape, N_DESCOTEAUX), dtype=np.float32)
    coeffs[2, 2, 2, 0] = 1.0

    glyph = create_shm_visualization((coeffs, affine, "odf.pam5"), 0)
    actor = glyph.actor

    scene = window.Scene()
    scene.add(actor)

    glyph._slicer.hide_axis("x")
    glyph._slicer.hide_axis("y")
    glyph._slicer.hide_axis("z")
    baseline = window.snapshot(scene=scene, return_array=True, fname=None)
    baseline_fg = int(np.sum(baseline > 0))

    world_own = (affine @ np.array([2, 2, 2, 1.0]))[0]
    world_neighbor = (affine @ np.array([3, 2, 2, 1.0]))[0]

    glyph._slicer.show_axis("x")
    glyph._slicer.set_slice("x", float(world_own))
    arr_own = window.snapshot(scene=scene, return_array=True, fname=None)
    fg_own = int(np.sum(arr_own > 0))
    assert fg_own > baseline_fg, (
        f"Expected the voxel to render on its own slice: fg={fg_own} vs baseline={baseline_fg}"
    )

    glyph._slicer.set_slice("x", float(world_neighbor))
    arr_neighbor = window.snapshot(scene=scene, return_array=True, fname=None)
    fg_neighbor = int(np.sum(arr_neighbor > 0))
    assert fg_neighbor <= baseline_fg + 10, (
        "Expected no glyph one grid step away along the mixed axis: "
        f"fg={fg_neighbor} vs baseline={baseline_fg}"
    )


def test_sh_glyph_default_scale_uses_affine_voxel_sizes():
    """Default glyph scale must derive from the affine's voxel sizes, not
    a single diagonal entry, so it stays correct for rotated/axis-swapped
    affines where the diagonal alone can be zero or misleading.
    """
    affine = np.array(
        [
            [0, 2, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )
    glyph = _glyph(affine=affine)

    assert glyph._scale == pytest.approx(np.mean(affine_voxel_sizes(affine)))
    assert glyph._scale == pytest.approx(2.0)


def test_sh_glyph_default_scale_matches_diagonal_affine():
    """For a uniform diagonal affine, the new voxel-size-based default
    scale matches the old (diagonal-only) behavior exactly.
    """
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    glyph = _glyph(affine=affine)

    assert glyph._scale == pytest.approx(abs(affine[0, 0]))


def test_sh_slicer_model_space_centers_are_raw_voxel_coords():
    """SHSlicer places glyph centers at integer voxel coordinates, not
    pre-scaled by any voxel size -- the affine is applied once, as a
    group transform, by SHGlyph3D on top of these raw centers.
    """
    slicer = SHSlicer(
        _coeffs(shape=(3, 3, 3)), l_max=SH_ORDER, basis_type="descoteaux07"
    )
    actor = slicer.build().children[0]

    centers = actor.billboard_centers
    npt.assert_array_equal(centers, np.round(centers))
    assert centers.min() == 0
    assert centers.max() == 2
