import numpy as np
import numpy.testing as npt
import pytest

from dipy.utils.optpkg import optional_package

_, has_fury, _ = optional_package("fury", min_version="2.0.0")
if not has_fury:
    pytest.skip("Requires fury>=2.0.0", allow_module_level=True)
else:
    from dipy.viz.skyline.render import sh_billboard
    from dipy.viz.skyline.render.sh_billboard import (
        SlicedSphGlyphMaterial,
        SphGlyphBillboard,
        _calculate_lut_chunking,
        _get_gpu_max_buffer_size,
        sph_glyph_billboard_sliced,
    )

# ``sph_glyph_billboard_sliced`` works in the full (standard) SH basis, where the
# coefficient count is ``(l_max + 1) ** 2``.
L_MAX = 8
N_COEFFS = (L_MAX + 1) ** 2


def _glyph_inputs(n_glyphs=4):
    coeffs = np.zeros((n_glyphs, N_COEFFS), dtype=np.float32)
    coeffs[:, 0] = 1.0
    centers = np.arange(n_glyphs * 3, dtype=np.float32).reshape(n_glyphs, 3)
    voxel_coords = np.arange(n_glyphs * 3, dtype=np.int32).reshape(n_glyphs, 3)
    return coeffs, centers, voxel_coords


def test_gpu_max_buffer_size_is_positive_and_cached():
    first = _get_gpu_max_buffer_size()
    second = _get_gpu_max_buffer_size()

    assert first > 0
    assert first == second
    assert (
        sh_billboard._GPU_DEVICE_LIMITS_CACHE["max_storage_buffer_binding_size"]
        == first
    )


def test_lut_chunking_fits_a_small_layout_in_one_chunk():
    plan = _calculate_lut_chunking(10, 64)

    assert plan["n_chunks"] == 1
    assert plan["glyphs_per_chunk"] == 10
    assert plan["chunk_sizes"] == [10]
    assert plan["total_samples"] == 640
    assert plan["samples_per_chunk"] == 640
    assert plan["feasible"] is True


def test_lut_chunking_splits_a_layout_that_exceeds_the_buffer_limit():
    usable = int(_get_gpu_max_buffer_size() * 0.90)
    samples_per_glyph = 4096
    glyphs_per_chunk = (usable // 4) // samples_per_glyph
    glyph_count = glyphs_per_chunk * 3

    plan = _calculate_lut_chunking(glyph_count, samples_per_glyph)

    assert plan["n_chunks"] == 3
    assert plan["glyphs_per_chunk"] == glyphs_per_chunk
    assert plan["chunk_sizes"] == [glyphs_per_chunk] * 3
    assert sum(plan["chunk_sizes"]) == glyph_count
    assert plan["total_samples"] == glyph_count * samples_per_glyph
    assert plan["feasible"] is True


def test_lut_chunking_keeps_a_short_final_chunk():
    usable = int(_get_gpu_max_buffer_size() * 0.90)
    samples_per_glyph = 4096
    glyphs_per_chunk = (usable // 4) // samples_per_glyph
    glyph_count = glyphs_per_chunk + 7

    plan = _calculate_lut_chunking(glyph_count, samples_per_glyph)

    assert plan["n_chunks"] == 2
    assert plan["chunk_sizes"] == [glyphs_per_chunk, 7]
    assert sum(plan["chunk_sizes"]) == glyph_count


def test_lut_chunking_is_infeasible_when_one_glyph_cannot_fit():
    plan = _calculate_lut_chunking(2, 10**9)

    assert plan["feasible"] is False
    assert plan["n_chunks"] == 0
    assert plan["glyphs_per_chunk"] == 0
    assert plan["chunk_sizes"] == []
    assert plan["total_samples"] == 2 * 10**9


def test_lut_chunking_is_infeasible_beyond_the_chunk_ceiling():
    usable = int(_get_gpu_max_buffer_size() * 0.90)
    samples_per_glyph = 4096
    glyphs_per_chunk = (usable // 4) // samples_per_glyph
    glyph_count = glyphs_per_chunk * (sh_billboard._MAX_LUT_CHUNKS + 1)

    plan = _calculate_lut_chunking(glyph_count, samples_per_glyph)

    assert plan["n_chunks"] == sh_billboard._MAX_LUT_CHUNKS + 1
    assert plan["feasible"] is False


def test_lut_chunking_accounts_for_the_sample_width():
    narrow = _calculate_lut_chunking(10**6, 4096, bytes_per_sample=2)
    wide = _calculate_lut_chunking(10**6, 4096, bytes_per_sample=8)

    assert narrow["n_chunks"] <= wide["n_chunks"]
    assert narrow["total_samples"] == wide["total_samples"]


def test_sliced_material_defaults_hide_no_axis():
    material = SlicedSphGlyphMaterial()

    assert material.active_slice_x == -1.0
    assert material.active_slice_y == -1.0
    assert material.active_slice_z == -1.0
    assert (material.vis_x, material.vis_y, material.vis_z) == (1, 1, 1)


def test_sliced_material_accepts_initial_values():
    material = SlicedSphGlyphMaterial(
        active_slice_x=2.0, active_slice_y=3.0, active_slice_z=4.0, vis_y=0
    )

    assert material.active_slice_x == 2.0
    assert material.active_slice_y == 3.0
    assert material.active_slice_z == 4.0
    assert (material.vis_x, material.vis_y, material.vis_z) == (1, 0, 1)


def test_sliced_material_keeps_fractional_slice_positions():
    material = SlicedSphGlyphMaterial()

    material.active_slice_x = 1.25
    material.active_slice_y = 2.5
    material.active_slice_z = 3.75

    assert material.active_slice_x == 1.25
    assert material.active_slice_y == 2.5
    assert material.active_slice_z == 3.75


def test_sliced_material_writes_slices_into_the_uniform_buffer():
    material = SlicedSphGlyphMaterial()

    material.active_slice_x = 6.5

    assert float(material.uniform_buffer.data["active_slice_x"]) == 6.5


@pytest.mark.parametrize("axis", ["vis_x", "vis_y", "vis_z"])
def test_sliced_material_visibility_flags_are_integers(axis):
    material = SlicedSphGlyphMaterial()

    setattr(material, axis, 1.8)

    assert getattr(material, axis) == 1
    assert isinstance(getattr(material, axis), int)

    setattr(material, axis, 0)

    assert getattr(material, axis) == 0


def test_sph_glyph_billboard_sliced_builds_one_actor_for_every_glyph():
    coeffs, centers, voxel_coords = _glyph_inputs()

    actor = sph_glyph_billboard_sliced(coeffs, centers, voxel_coords)

    assert isinstance(actor, SphGlyphBillboard)
    assert isinstance(actor.material, SlicedSphGlyphMaterial)
    assert actor.material.active_slice_x == -1.0


def test_sph_glyph_billboard_sliced_infers_l_max_from_the_coefficients():
    coeffs, centers, voxel_coords = _glyph_inputs()

    actor = sph_glyph_billboard_sliced(coeffs, centers, voxel_coords)

    assert actor.l_max == L_MAX


def test_sph_glyph_billboard_sliced_truncates_the_shaded_coefficients():
    """``l_max`` limits what the material shades, not the actor's own order."""
    coeffs, centers, voxel_coords = _glyph_inputs()

    actor = sph_glyph_billboard_sliced(coeffs, centers, voxel_coords, l_max=4)

    assert actor.material.n_coeffs == (4 + 1) ** 2
    assert actor.n_coeff == N_COEFFS
    assert actor.l_max == L_MAX


def test_sph_glyph_billboard_sliced_shades_every_coefficient_by_default():
    coeffs, centers, voxel_coords = _glyph_inputs()

    actor = sph_glyph_billboard_sliced(coeffs, centers, voxel_coords)

    assert actor.material.n_coeffs == -1


def test_sph_glyph_billboard_sliced_rejects_an_l_max_above_the_coefficients():
    coeffs, centers, voxel_coords = _glyph_inputs()

    with pytest.raises(ValueError, match="exceeds degree supported by coeffs"):
        sph_glyph_billboard_sliced(coeffs, centers, voxel_coords, l_max=L_MAX + 2)


def test_sph_glyph_billboard_l_max_rejects_a_negative_order():
    coeffs, centers, voxel_coords = _glyph_inputs()
    actor = sph_glyph_billboard_sliced(coeffs, centers, voxel_coords)

    with pytest.raises(ValueError, match="non-negative integer"):
        actor.l_max = -1


def test_sph_glyph_billboard_l_max_rejects_a_non_integer_order():
    coeffs, centers, voxel_coords = _glyph_inputs()
    actor = sph_glyph_billboard_sliced(coeffs, centers, voxel_coords)

    with pytest.raises(ValueError, match="non-negative integer"):
        actor.l_max = 2.5


def test_sph_glyph_billboard_l_max_rejects_an_order_beyond_the_coefficients():
    coeffs, centers, voxel_coords = _glyph_inputs()
    actor = sph_glyph_billboard_sliced(coeffs, centers, voxel_coords)

    with pytest.raises(ValueError, match="exceeds the number of"):
        actor.l_max = L_MAX + 2


@pytest.mark.parametrize("color_type", ["orientation", "sign"])
def test_sph_glyph_billboard_sliced_color_types(color_type):
    coeffs, centers, voxel_coords = _glyph_inputs()

    actor = sph_glyph_billboard_sliced(
        coeffs, centers, voxel_coords, color_type=color_type
    )

    assert actor is not None


def test_sph_glyph_billboard_sliced_scale_and_opacity():
    coeffs, centers, voxel_coords = _glyph_inputs()

    actor = sph_glyph_billboard_sliced(
        coeffs, centers, voxel_coords, scale=2.0, opacity=0.5, shininess=10
    )

    assert actor.material.opacity == pytest.approx(0.5)


def test_sph_glyph_billboard_sliced_without_hermite_interpolation():
    coeffs, centers, voxel_coords = _glyph_inputs()

    actor = sph_glyph_billboard_sliced(
        coeffs, centers, voxel_coords, use_hermite=False, lut_res=4
    )

    assert actor is not None


def test_sph_glyph_billboard_sliced_single_glyph():
    coeffs, centers, voxel_coords = _glyph_inputs(n_glyphs=1)

    actor = sph_glyph_billboard_sliced(coeffs, centers, voxel_coords)

    assert actor is not None
    npt.assert_array_equal(voxel_coords.shape, (1, 3))
