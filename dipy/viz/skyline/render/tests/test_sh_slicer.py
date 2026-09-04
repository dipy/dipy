import numpy as np
import numpy.testing as npt
import pytest

from dipy.utils.optpkg import optional_package

_, has_fury, _ = optional_package("fury", min_version="2.0.0")
if not has_fury:
    pytest.skip("Requires fury>=2.0.0", allow_module_level=True)
else:
    from dipy.viz.skyline.render.sh_slicer import (
        SHGlyph3D,
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
    assert "Voxel Sizes:" in info


def test_sh_glyph_info_without_an_affine():
    glyph = SHGlyph3D("odf.pam5", _coeffs(), affine=None, basis_type="descoteaux07")

    assert "Voxel Sizes:" not in glyph._populate_info()


def test_sh_glyph_actor_is_the_slicer_group():
    glyph = _glyph()

    assert glyph.actor is glyph._slicer.actor
    assert glyph._slicer._glyph_actor is not None


def test_sh_glyph_set_slices_maps_world_state_to_voxel_uniforms():
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    glyph = _glyph(affine=affine)

    glyph.state = np.array([4.0, 6.0, 2.0])
    glyph.set_slices()

    material = _material(glyph)
    assert material.active_slice_x == 2.0
    assert material.active_slice_y == 3.0
    assert material.active_slice_z == 1.0
    npt.assert_allclose(glyph._last_state, (4.0, 6.0, 2.0))


def test_sh_glyph_set_slices_clips_to_the_volume():
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    glyph = _glyph(affine=affine)

    glyph.state = np.array([-40.0, 6.0, 1000.0])
    glyph.set_slices()

    material = _material(glyph)
    assert material.active_slice_x == 0.0
    assert material.active_slice_z == float(SHAPE[2] - 1)
    npt.assert_allclose(glyph._last_state, (-40.0, 6.0, 1000.0))


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
