import numpy as np
import pytest

from dipy.utils.optpkg import optional_package

_, has_fury, _ = optional_package("fury", min_version="2.0.0")
if not has_fury:
    pytest.skip("Requires fury>=2.0.0", allow_module_level=True)
else:
    from dipy.viz.skyline.render.image import Image3D
    from dipy.viz.skyline.render.renderer import (
        Visualization,
        affine_voxel_sizes,
        create_window,
        slice_slider_bounds,
        slice_slider_values_from_state,
        slice_state_from_slider_values,
        voxel_values_from_slice_state,
    )
    from dipy.viz.skyline.render.roi import ROI3D
    from dipy.viz.skyline.render.sh_slicer import create_shm_visualization


def test_affine_voxel_sizes_use_affine_columns():
    """Affine voxel sizes are computed from matrix columns."""
    affine = np.array(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    voxel_sizes = affine_voxel_sizes(affine)

    assert np.allclose(voxel_sizes, (2.0, 3.0, 4.0))


def test_slice_slider_bounds_use_original_shape_without_affine():
    """Slice slider bounds use original sizes when no affine is provided."""
    bounds = slice_slider_bounds((10, 20, 30))

    assert bounds == ((0, 10), (0, 20), (0, 30))


def test_slice_slider_bounds_use_affine_scaled_shape_for_large_voxels():
    """Slice slider bounds use affine-scaled sizes for voxels at least one."""
    affine = np.diag([2.0, 1.0, 3.0, 1.0])

    bounds = slice_slider_bounds((10, 20, 30), affine=affine)

    assert bounds == ((0, 20), (0, 20), (0, 90))


def test_slice_slider_bounds_use_original_shape_for_small_voxels():
    """Slice slider bounds keep original sizes for voxels smaller than one."""
    affine = np.diag([0.5, 0.25, 0.9, 1.0])

    bounds = slice_slider_bounds((10, 20, 30), affine=affine)

    assert bounds == ((0, 10), (0, 20), (0, 30))


def test_slice_slider_bounds_apply_rule_per_axis():
    """Slice slider bounds apply the affine voxel-size rule independently."""
    affine = np.diag([2.0, 0.5, 1.25, 1.0])

    bounds = slice_slider_bounds((10, 20, 30), affine=affine)

    assert bounds == ((0, 20), (0, 20), (0, 38))


def test_slice_slider_values_convert_to_world_state():
    """Slice slider values convert back to affine world coordinates."""
    affine = np.array(
        [
            [2.0, 0.0, 0.0, 10.0],
            [0.0, 0.5, 0.0, 20.0],
            [0.0, 0.0, 3.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    state = slice_state_from_slider_values((8.0, 4.0, 15.0), affine=affine)

    assert np.allclose(state, (18.0, 22.0, 45.0))


def test_slice_slider_values_round_trip_from_world_state():
    """Slice slider values round-trip with affine world coordinates."""
    affine = np.array(
        [
            [2.0, 0.0, 0.0, 10.0],
            [0.0, 0.5, 0.0, 20.0],
            [0.0, 0.0, 3.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    world_state = np.array([18.0, 22.0, 45.0])

    slider_values = slice_slider_values_from_state(world_state, affine=affine)
    round_trip = slice_state_from_slider_values(slider_values, affine=affine)

    assert np.allclose(slider_values, (8.0, 4.0, 15.0))
    assert np.allclose(round_trip, world_state)


def test_slice_slider_max_value_maps_to_affine_scaled_world_extent():
    """Slider max maps to world extent for large affine voxels."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])

    state = slice_state_from_slider_values((20.0, 20.0, 20.0), affine=affine)

    assert np.allclose(state, (20.0, 20.0, 20.0))


def test_voxel_values_from_slice_state_inverts_affine():
    """World slice states can be converted back to voxel coordinates."""
    affine = np.array(
        [
            [2.0, 0.0, 0.0, 10.0],
            [0.0, 0.5, 0.0, 20.0],
            [0.0, 0.0, 3.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    voxel_values = voxel_values_from_slice_state((18.0, 22.0, 45.0), affine=affine)

    assert np.allclose(voxel_values, (4.0, 4.0, 5.0))


def _volume(shape=(6, 7, 8), seed=0):
    return np.random.default_rng(seed).random(shape).astype(np.float32)


def test_create_window_stealth_is_offscreen(tmp_path):
    """Stealth mode builds an offscreen ShowManager without an ImGui overlay."""
    show_manager = create_window(
        visualizer_type="stealth", size=(64, 48), title=str(tmp_path / "scene")
    )

    assert len(show_manager.screens) == 1
    assert show_manager._imgui is None
    assert tuple(show_manager.size) == (64, 48)


def test_create_window_exits_on_an_unknown_visualizer_type(caplog):
    """An unrecognized visualizer type is reported and aborts the process."""
    import logging

    with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as excinfo:
        create_window(visualizer_type="hologram")

    assert excinfo.value.code == 1
    assert "'hologram' is not recognized" in caplog.text


def test_affine_voxel_sizes_ignores_translation():
    """Voxel sizes come from the rotation/scale block, not the offset."""
    affine = np.array(
        [
            [0.0, -2.0, 0.0, 100.0],
            [3.0, 0.0, 0.0, -50.0],
            [0.0, 0.0, 4.0, 25.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    assert np.allclose(affine_voxel_sizes(affine), (3.0, 2.0, 4.0))


def test_slice_slider_bounds_without_an_affine_keep_integer_sizes():
    assert slice_slider_bounds((1, 1, 1)) == ((0, 1), (0, 1), (0, 1))


def test_slice_state_round_trip_without_an_affine():
    values = (3.0, 4.0, 5.0)

    state = slice_state_from_slider_values(values)

    assert np.allclose(state, values)
    assert np.allclose(slice_slider_values_from_state(state), values)
    assert np.allclose(voxel_values_from_slice_state(state), values)


def test_visualization_base_requires_an_actor():
    """The base class leaves ``actor`` to its subclasses."""

    class Bare(Visualization):
        def _populate_info(self):
            return ""

    bare = Bare("nowhere.nii.gz", None)

    with pytest.raises(NotImplementedError, match="must implement the actor"):
        _ = bare.actor


def test_visualization_names_itself_from_the_path():
    image = Image3D("/data/subject/t1.nii.gz", _volume(), affine=np.eye(4))

    assert image.path == "/data/subject/t1.nii.gz"
    assert image.name == "t1.nii.gz"
    assert image.active is False
    assert image._visible is True


def test_visualization_falls_back_to_a_placeholder_name():
    image = Image3D(None, _volume(), affine=np.eye(4))

    assert image.path == "Unnamed Visualization"
    assert image.name == "Unnamed Visualization"


def test_visualization_prefixes_roi_and_odf_names():
    roi = np.zeros((16, 16, 16), dtype=np.uint8)
    roi[4:12, 4:12, 4:12] = 1
    roi_viz = ROI3D("mask.nii.gz", roi, affine=np.eye(4))

    n_coeffs = sum(2 * ell + 1 for ell in range(0, 9, 2))
    coeffs = np.zeros((2, 2, 2, n_coeffs), dtype=np.float32)
    coeffs[..., 0] = 1.0
    odf_viz = create_shm_visualization((coeffs, np.eye(4), "odf.pam5"), 0)

    assert roi_viz.name == "ROI (mask.nii.gz)"
    assert odf_viz.name == "ODFs (odf.pam5)"


@pytest.mark.parametrize("visible", [True, False])
def test_visualization_actor_visibility_is_forwarded(visible):
    image = Image3D("t1.nii.gz", _volume(), affine=np.eye(4))

    image._set_actor_visible(visible)

    assert image.actor.visible is visible


def test_visualization_render_calls_the_callback():
    calls = []
    image = Image3D(
        "t1.nii.gz",
        _volume(),
        affine=np.eye(4),
        render_callback=lambda: calls.append(1),
    )
    calls.clear()

    image.render()

    assert calls == [1]


def test_visualization_render_without_a_callback_is_a_noop():
    image = Image3D("t1.nii.gz", _volume(), affine=np.eye(4))

    image.render()

    assert image._render_callback is None


def test_visualization_scene_ops_run_immediately_without_a_deferrer():
    image = Image3D("t1.nii.gz", _volume(), affine=np.eye(4))
    calls = []

    image.apply_scene_op(calls.append, "now")

    assert calls == ["now"]


def test_visualization_scene_ops_are_deferred_when_a_callback_is_set():
    image = Image3D("t1.nii.gz", _volume(), affine=np.eye(4))
    deferred = []
    image._scene_op_callback = lambda func, *args, **kwargs: deferred.append(
        (func, args, kwargs)
    )
    calls = []

    image.apply_scene_op(calls.append, "later")

    assert calls == []
    assert len(deferred) == 1
    assert deferred[0][1] == ("later",)


def test_visualization_type_is_none_for_an_unknown_subclass():
    class Mystery(Visualization):
        def _populate_info(self):
            return ""

    assert Mystery("x", None).viz_type is None
