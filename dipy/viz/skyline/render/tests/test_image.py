import numpy as np
import numpy.testing as npt
import pytest

from dipy.utils.optpkg import optional_package

_, has_fury, _ = optional_package("fury", min_version="2.0.0")
if not has_fury:
    pytest.skip("Requires fury>=2.0.0", allow_module_level=True)
else:
    from dipy.viz.skyline.render.image import Image3D, create_image_visualization
    from dipy.viz.skyline.render.renderer import (
        slice_slider_bounds,
        slice_slider_values_from_state,
        slice_state_from_slider_values,
    )

AFFINE = np.eye(4)
SHAPE = (10, 11, 12)


def _volume(shape=SHAPE, seed=0):
    return np.random.default_rng(seed).random(shape).astype(np.float32)


def _image(**kwargs):
    kwargs.setdefault("affine", AFFINE)
    return Image3D("vol.nii.gz", _volume(), **kwargs)


def _materials(image):
    return [actor.material for actor in image.actor.children]


def _info_line(info, prefix):
    """The remainder of the single ``info`` line starting with ``prefix``."""
    return next(
        line.split(prefix, 1)[1].strip()
        for line in info.splitlines()
        if line.startswith(prefix)
    )


def _voxel_sizes(info):
    return np.fromstring(_info_line(info, "Voxel Sizes:").strip("[]"), sep=" ")


def test_create_image_visualization_rejects_invalid_input():
    with pytest.raises(ValueError, match="Input must be a tuple"):
        create_image_visualization("not a tuple", 0)
    with pytest.raises(ValueError, match="Input must be a tuple"):
        create_image_visualization((_volume(),), 0)


def test_create_image_visualization_names_by_index():
    viz = create_image_visualization((_volume(), AFFINE), 4)

    assert viz.path == "Image_4"
    assert isinstance(viz, Image3D)


def test_create_image_visualization_uses_the_given_filename():
    viz = create_image_visualization((_volume(), AFFINE, "t1.nii.gz"), 0)

    assert viz.path == "t1.nii.gz"
    assert viz.viz_type == "image"


def test_create_image_visualization_forwards_its_options():
    viz = create_image_visualization(
        (_volume(), AFFINE, "t1.nii.gz"),
        0,
        interpolation="nearest",
        opacity=50,
        value_percentiles=(5, 95),
        colormap="Viridis",
    )

    assert viz.interpolation == "nearest"
    assert viz.opacity == 50
    assert viz._value_percentiles == (5, 95)
    assert viz.colormap == "Viridis"


def test_image3d_defaults():
    image = _image()

    assert image.rgb is False
    assert image._has_directions is False
    assert image._volume_idx == 0
    assert image.interpolation == "linear"
    assert image._slice_visibility == [True, True, True]
    assert image._synchronize is True
    assert image.actor is image._slicer


def test_image3d_empty_interpolation_falls_back_to_linear():
    assert _image(interpolation=None).interpolation == "linear"


def test_image3d_state_starts_at_the_volume_center():
    image = _image()

    npt.assert_allclose(image.state, np.mean(image.bounds, axis=0))
    assert len(image.actor.children) == 3


def test_image3d_active_volume_is_the_whole_volume_for_3d_data():
    image = _image()

    assert image.active_volume is image.dwi


def test_image3d_active_volume_selects_a_direction_for_4d_data():
    data = _volume((6, 7, 8, 5))
    image = Image3D("dwi.nii.gz", data, affine=AFFINE)

    assert image._has_directions is True
    npt.assert_array_equal(image.active_volume, data[..., 0])


def test_image3d_rgb_volume_is_not_directional():
    data = _volume((6, 7, 8, 3))
    image = Image3D("rgb.nii.gz", data, affine=AFFINE, rgb=True)

    assert image.rgb is True
    assert image._has_directions is False
    assert image.active_volume is data


@pytest.mark.parametrize("n_channels", [2, 5])
def test_image3d_rgb_requires_three_or_four_channels(n_channels):
    data = _volume((4, 4, 4, n_channels))

    with pytest.raises(ValueError, match="must be 3 \\(RGB\\) or 4 \\(RGBA\\)"):
        Image3D("rgb.nii.gz", data, affine=AFFINE, rgb=True)


def test_image3d_value_range_follows_the_percentiles():
    data = _volume()
    image = Image3D("vol.nii.gz", data, affine=AFFINE, value_percentiles=(10, 90))

    npt.assert_allclose(
        image._value_range_from_percentile(data), np.percentile(data, (10, 90))
    )
    npt.assert_allclose(image.value_range, np.percentile(data, (10, 90)))


def test_image3d_gray_colormap_clears_the_material_map():
    image = _image(colormap="Gray")

    assert image._is_divergent_colormap() is False
    assert image._is_distinct_colormap() is False
    for material in _materials(image):
        assert material.map is None
        npt.assert_allclose(material.clim, image.value_range)


def test_image3d_named_colormap_sets_a_material_map():
    image = _image(colormap="Viridis")

    for material in _materials(image):
        assert material.map is not None
        npt.assert_allclose(material.clim, image.value_range)


def test_image3d_divergent_colormap_is_symmetric_and_nearest():
    image = _image(colormap="Divergent")

    assert image._is_divergent_colormap() is True
    assert image.interpolation == "nearest"
    max_abs = float(np.max(np.abs(image.active_volume)))
    for material in _materials(image):
        assert material.map is not None
        assert material.interpolation == "nearest"
        npt.assert_allclose(material.clim, (-max_abs, max_abs))


def test_image3d_divergent_colormap_handles_an_all_zero_volume():
    image = Image3D(
        "zeros.nii.gz",
        np.zeros(SHAPE, dtype=np.float32),
        affine=AFFINE,
        colormap="Divergent",
    )

    for material in _materials(image):
        npt.assert_allclose(material.clim, (-1.0, 1.0))


def test_image3d_distinct_colormap_is_nearest():
    image = _image(colormap="Distinct")

    assert image._is_distinct_colormap() is True
    assert image.interpolation == "nearest"
    for material in _materials(image):
        assert material.map is not None
        assert material.interpolation == "nearest"


def test_image3d_switching_colormaps_updates_every_slice():
    image = _image(colormap="Gray")

    image._apply_colormap("Plasma")

    assert image.colormap == "Plasma"
    assert all(material.map is not None for material in _materials(image))

    image._apply_colormap("Gray")

    assert all(material.map is None for material in _materials(image))


def test_image3d_colormap_options_cover_the_special_cases():
    image = _image()

    assert "Gray" in image._colormap_options
    assert "Divergent" in image._colormap_options
    assert "Distinct" in image._colormap_options


def test_image3d_info_lists_dimensions_dtype_and_affine():
    image = _image()

    info = image._populate_info()

    assert f"Dimensions: {SHAPE}" in info
    assert "Data Type: float32" in info
    npt.assert_allclose(_voxel_sizes(info), (1.0, 1.0, 1.0))
    assert "Voxel Order: RAS" in info
    assert "Affine:" in info
    assert "Directions:" not in info


def test_image3d_info_reports_directions_for_4d_data():
    image = Image3D("dwi.nii.gz", _volume((4, 5, 6, 7)), affine=AFFINE)

    assert "Directions: 7" in image._populate_info()


def test_image3d_info_reports_the_las_voxel_order():
    affine = np.diag([-2.0, 2.0, 2.0, 1.0])
    image = Image3D("vol.nii.gz", _volume(), affine=affine)

    info = image._populate_info()

    assert "Voxel Order: LAS" in info
    npt.assert_allclose(_voxel_sizes(info), (2.0, 2.0, 2.0))


def test_image3d_info_without_an_affine():
    image = Image3D("vol.nii.gz", _volume(), affine=None)

    info = image._populate_info()

    assert "Affine:" not in info
    assert "Voxel Order:" not in info


def test_image3d_update_state_moves_the_slices():
    image = _image()
    target = image.state + 1.0

    image.update_state(target)

    npt.assert_allclose(image.state, target[:3])


def test_image3d_update_state_is_ignored_when_sync_is_off():
    image = _image()
    before = np.array(image.state, dtype=float).copy()
    image._synchronize = False

    image.update_state(before + 5.0)

    npt.assert_allclose(image.state, before)


def test_image3d_update_state_switches_the_active_direction():
    image = Image3D("dwi.nii.gz", _volume((6, 7, 8, 5)), affine=AFFINE)
    state = np.append(np.array(image.state, dtype=float), 3.0)

    image.update_state(state)

    assert image._volume_idx == 3
    npt.assert_array_equal(image.active_volume, image.dwi[..., 3])


def test_image3d_update_state_keeps_the_direction_for_a_three_value_state():
    image = Image3D("dwi.nii.gz", _volume((6, 7, 8, 5)), affine=AFFINE)

    image.update_state(np.array(image.state, dtype=float))

    assert image._volume_idx == 0


def test_image3d_update_state_ignores_an_out_of_range_direction():
    image = Image3D("dwi.nii.gz", _volume((6, 7, 8, 4)), affine=AFFINE)
    state = np.append(np.array(image.state, dtype=float), 9.0)

    image.update_state(state)

    assert image._volume_idx == 0


def test_image3d_opacity_below_full_enables_blending():
    image = _image()

    image._set_opacity(40)

    for material in _materials(image):
        assert material.depth_write is False
        assert material.alpha_mode == "blend"


def test_image3d_full_opacity_restores_bayer_dithering():
    image = _image()
    image._set_opacity(40)

    image._set_opacity(100)

    for material in _materials(image):
        assert material.depth_write is True
        assert material.alpha_mode == "bayer"


def test_image3d_set_slice_state_applies_visibility():
    image = _image()

    image._set_slice_state((True, False, True), image.state)

    visibilities = [actor.visible for actor in image.actor.children]
    assert visibilities == [True, False, True]


def test_image3d_set_clim_updates_every_slice():
    image = _image()

    image._set_clim((0.25, 0.75))

    for material in _materials(image):
        npt.assert_allclose(material.clim, (0.25, 0.75))


def test_image3d_set_interpolation_updates_every_slice():
    image = _image()

    image._set_interpolation("nearest")

    assert image.interpolation == "nearest"
    for material in _materials(image):
        assert material.interpolation == "nearest"


def test_image3d_slice_slider_bounds_use_affine_scaled_shape():
    image = Image3D("vol.nii.gz", _volume((10, 20, 30)), affine=np.diag([2, 1, 3, 1]))

    bounds = slice_slider_bounds(image.dwi.shape[:3], affine=image.affine)

    assert bounds == ((0, 20), (0, 20), (0, 90))


def test_image3d_slice_slider_value_maps_to_world_state():
    affine = np.array(
        [
            [2.0, 0.0, 0.0, 10.0],
            [0.0, 0.5, 0.0, 20.0],
            [0.0, 0.0, 3.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    image = Image3D("vol.nii.gz", _volume(), affine=affine)
    slider_state = np.array([8.0, 4.0, 15.0])

    image.state = slice_state_from_slider_values(slider_state, affine=image.affine)
    displayed_state = slice_slider_values_from_state(image.state, affine=image.affine)

    npt.assert_allclose(image.state, (18.0, 22.0, 45.0))
    npt.assert_allclose(displayed_state, slider_state)
