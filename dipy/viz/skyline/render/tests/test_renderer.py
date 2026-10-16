import logging
import os
import re

import numpy as np
import pytest

from dipy.utils.optpkg import optional_package

_, has_fury, _ = optional_package("fury", min_version="2.0.0")
if not has_fury:
    pytest.skip("Requires fury>=2.0.0", allow_module_level=True)
else:
    from fury import window
    from fury.lib import OrthographicCamera, PointerEvent

    from dipy.viz.skyline.render.image import Image3D
    from dipy.viz.skyline.render.renderer import (
        Visualization,
        _show_anatomical_gizmo,
        affine_voxel_sizes,
        create_window,
        format_affine_info,
        slice_slider_bounds,
        slice_slider_values_from_state,
        slice_state_from_slider_values,
        voxel_values_from_slice_state,
    )
    from dipy.viz.skyline.render.roi import ROI3D
    from dipy.viz.skyline.render.sh_slicer import create_shm_visualization


@pytest.fixture(autouse=True)
def restore_fury_offscreen():
    previous_fury_offscreen = os.environ.get("FURY_OFFSCREEN")
    try:
        yield
    finally:
        if previous_fury_offscreen is None:
            os.environ.pop("FURY_OFFSCREEN", None)
        else:
            os.environ["FURY_OFFSCREEN"] = previous_fury_offscreen


ANATOMICAL_VIEWS = {
    "L": ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
    "R": ((1, 0, 0), (0, 0, 1), (0, -1, 0)),
    "P": ((0, -1, 0), (0, 0, 1), (-1, 0, 0)),
    "A": ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
    "S": ((0, 0, 1), (0, 1, 0), (-1, 0, 0)),
    "I": ((0, 0, -1), (0, 1, 0), (1, 0, 0)),
}
PERMUTED_AFFINE = np.array(
    [[0, -2, 0, 20], [-3, 0, 0, -15], [0, 0, 4, 7], [0, 0, 0, 1]],
    dtype=float,
)


@pytest.fixture
def gizmo_manager():
    manager = window.ShowManager(size=(160, 120), window_type="offscreen", imgui=False)
    try:
        yield manager
    finally:
        manager.close()


def _add_gizmo_volume(manager, affine):
    """Add asymmetric slices and the anatomical gizmo.

    Parameters
    ----------
    manager : ShowManager
        Offscreen manager hosting the scene.
    affine : ndarray
        Voxel-to-world transform for the synthetic volume.
    """
    volume = np.arange(9 * 11 * 13, dtype=np.float32).reshape(9, 11, 13)
    image = Image3D("asymmetric.nii.gz", volume, affine=affine)
    manager.screens[0].scene.add(image.actor)
    manager.render()
    _show_anatomical_gizmo(manager)
    return image


def _click_gizmo(manager, label, target_kind="disk"):
    """Dispatch a pointer event to a real gizmo disk or label.

    Parameters
    ----------
    manager : ShowManager
        Manager hosting the gizmo.
    label : str
        Anatomical endpoint label.
    target_kind : str, optional
        Disk or label event target.
    """
    endpoint = ("L", "R", "P", "A", "I", "S").index(label)
    child_index = 1 + 3 * endpoint + (target_kind == "label")
    group = _gizmo_group(manager)
    manager.renderer.dispatch_event(
        PointerEvent(
            type="pointer_down",
            x=-1,
            y=-1,
            target=group.children[child_index],
        )
    )


def _assert_anatomical_basis(camera, label):
    """Assert the camera's actual viewing direction and upright basis.

    Parameters
    ----------
    camera : Camera
        Camera after the pointer event.
    label : str
        Clicked anatomical endpoint.
    """
    side, up, _ = ANATOMICAL_VIEWS[label]
    np.testing.assert_allclose(camera.world.forward, -np.asarray(side), atol=1e-6)
    np.testing.assert_allclose(camera.world.matrix[:3, 1], up, atol=1e-6)


def _gizmo_group(manager):
    """Find the anatomical overlay in the real UI scene.

    Parameters
    ----------
    manager : ShowManager
        Manager hosting the main-screen gizmo.
    """
    return next(
        actor.children[0]
        for actor in manager.screens[0].scene.ui_scene.children
        if actor.name == "Skyline anatomical axes"
    )


def _assert_gizmo_alignment(manager):
    """Compare displayed endpoint directions with camera-space world landmarks.

    Parameters
    ----------
    manager : ShowManager
        Manager hosting the camera and overlay.
    """
    camera = manager.screens[0].camera
    group = _gizmo_group(manager)
    for endpoint, label in enumerate(("L", "R", "P", "A", "I", "S")):
        direction, _, _ = ANATOMICAL_VIEWS[label]
        expected = (camera.world.inverse_matrix @ np.r_[direction, 0])[:3]
        expected[2] *= -1
        for child in group.children[1 + 3 * endpoint : 3 + 3 * endpoint]:
            displayed = (child.world.position - group.world.position) / (30, 30, 0.45)
            np.testing.assert_allclose(displayed, expected, atol=1e-6)
            np.testing.assert_allclose(
                child.world.matrix[:2, :2] / 30, np.eye(2), atol=1e-6
            )


def _process_gizmo_callbacks(manager):
    """Process the manager's registered repeating callbacks.

    Parameters
    ----------
    manager : ShowManager
        Manager whose real timer callbacks should run.
    """
    for callback, _, _, args in manager.callbacks.values():
        callback(*args)


@pytest.mark.parametrize("affine", [np.eye(4), PERMUTED_AFFINE], ids=["RAS", "PLS"])
@pytest.mark.parametrize("target_kind", ["disk", "label"])
@pytest.mark.parametrize("label", ANATOMICAL_VIEWS)
def test_axes_gizmo_anatomical_views(gizmo_manager, affine, target_kind, label):
    _add_gizmo_volume(gizmo_manager, affine)
    screen = gizmo_manager.screens[0]
    target = screen.scene.main_scene.get_world_bounding_box().mean(axis=0)
    screen.camera.world.position = target + (30, 40, 50)
    screen.camera.look_at(target)

    _click_gizmo(gizmo_manager, label, target_kind)

    camera = screen.camera
    _assert_anatomical_basis(camera, label)
    _, up, screen_left = ANATOMICAL_VIEWS[label]
    landmarks = np.array([target, target + up, target + screen_left])
    camera_landmarks = (camera.world.inverse_matrix @ np.c_[landmarks, np.ones(3)].T).T
    np.testing.assert_allclose(camera_landmarks[0, :2], 0, atol=1e-6)
    assert camera_landmarks[0, 2] < 0
    assert camera_landmarks[1, 1] > camera_landmarks[0, 1]
    assert camera_landmarks[2, 0] < camera_landmarks[0, 0]
    _assert_gizmo_alignment(gizmo_manager)
    _process_gizmo_callbacks(gizmo_manager)
    _assert_gizmo_alignment(gizmo_manager)

    opposite_label = {"L": "R", "R": "L", "P": "A", "A": "P", "S": "I", "I": "S"}[label]
    for expected_label in (opposite_label, label):
        _click_gizmo(gizmo_manager, label, target_kind)
        _assert_anatomical_basis(camera, expected_label)
        _process_gizmo_callbacks(gizmo_manager)
        _assert_gizmo_alignment(gizmo_manager)


def test_axes_gizmo_view_transitions(gizmo_manager):
    image = _add_gizmo_volume(gizmo_manager, PERMUTED_AFFINE)
    state = image.state.copy()
    slice_states = [
        (child.visible, child.world.matrix.copy()) for child in image.actor.children
    ]
    camera = gizmo_manager.screens[0].camera
    target = image.actor.get_world_bounding_box().mean(axis=0)
    camera.world.position = target + (30, 40, 50)
    camera.look_at(target)

    for clicked_label, expected_label in (
        ("S", "S"),
        ("S", "I"),
        ("S", "S"),
        ("L", "L"),
        ("L", "R"),
        ("L", "L"),
        ("R", "R"),
        ("A", "A"),
        ("A", "P"),
        ("P", "A"),
        ("I", "I"),
        ("R", "R"),
    ):
        _click_gizmo(gizmo_manager, clicked_label)
        _assert_anatomical_basis(camera, expected_label)
        _process_gizmo_callbacks(gizmo_manager)
        _assert_gizmo_alignment(gizmo_manager)

    np.testing.assert_array_equal(image.state, state)
    for child, (visible, transform) in zip(image.actor.children, slice_states):
        assert child.visible == visible
        np.testing.assert_array_equal(child.world.matrix, transform)


def test_axes_gizmo_empty_scene(gizmo_manager):
    gizmo_manager.render()
    _show_anatomical_gizmo(gizmo_manager)
    camera = gizmo_manager.screens[0].camera

    for label in ("L", "A"):
        _click_gizmo(gizmo_manager, label)
        assert np.isfinite(camera.world.position).all()
        assert np.isfinite(camera.world.matrix).all()
        _assert_anatomical_basis(camera, label)
        _assert_gizmo_alignment(gizmo_manager)


def test_axes_gizmo_only_changes_main_view():
    manager = window.ShowManager(
        size=(160, 120), screen_config=[2], window_type="offscreen", imgui=False
    )
    try:
        _add_gizmo_volume(manager, PERMUTED_AFFINE)
        other_camera = manager.screens[1].camera
        other_state = other_camera.get_state()

        _click_gizmo(manager, "A")

        _assert_anatomical_basis(manager.screens[0].camera, "A")
        for key, value in other_state.items():
            np.testing.assert_array_equal(other_camera.get_state()[key], value)
    finally:
        manager.close()


def test_axes_gizmo_orthographic_view():
    camera = OrthographicCamera(30, 30)
    manager = window.ShowManager(
        size=(160, 120), camera=camera, window_type="offscreen", imgui=False
    )
    try:
        _add_gizmo_volume(manager, PERMUTED_AFFINE)

        for expected_label in ("R", "L", "R"):
            _click_gizmo(manager, "R")

            assert isinstance(manager.screens[0].camera, OrthographicCamera)
            assert np.isfinite(camera.world.position).all()
            assert np.isfinite(camera.world.matrix).all()
            _assert_anatomical_basis(camera, expected_label)
            _process_gizmo_callbacks(manager)
            _assert_gizmo_alignment(manager)
            target = (
                manager.screens[0]
                .scene.main_scene.get_world_bounding_box()
                .mean(axis=0)
            )
            assert np.linalg.norm(camera.world.position - target) >= 1.0
    finally:
        manager.close()


def test_axes_gizmo_follows_controller_rotation(gizmo_manager):
    _add_gizmo_volume(gizmo_manager, PERMUTED_AFFINE)
    screen = gizmo_manager.screens[0]
    _click_gizmo(gizmo_manager, "L")
    screen.controller.rotate((0.15, 0.1), screen.viewport.rect, animate=False)
    _process_gizmo_callbacks(gizmo_manager)

    _assert_gizmo_alignment(gizmo_manager)

    _click_gizmo(gizmo_manager, "A", "label")
    _process_gizmo_callbacks(gizmo_manager)
    _assert_anatomical_basis(screen.camera, "A")
    _assert_gizmo_alignment(gizmo_manager)


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


def test_format_affine_info_reports_voxel_sizes_order_and_matrix():
    affine = np.diag([-2.0, 3.0, 4.0, 1.0])
    affine[:3, 3] = [5.0, -6.0, 7.0]

    info = format_affine_info(affine)

    voxel_sizes_line = next(
        line for line in info.splitlines() if line.startswith("Voxel Sizes:")
    )
    voxel_sizes = np.fromstring(
        voxel_sizes_line.split(":", 1)[1].strip().strip("[]"), sep=" "
    )
    assert np.allclose(voxel_sizes, (2.0, 3.0, 4.0))
    assert "Voxel Order: LAS" in info
    assert "Affine:" in info
    affine_block = info.split("Affine:\n", 1)[1]
    parsed_affine = np.array(
        [float(value) for value in re.findall(r"-?\d+\.?\d*", affine_block)]
    ).reshape(4, 4)
    assert np.allclose(parsed_affine, affine)


def test_format_affine_info_resolves_a_permuted_affine():
    """A permuted affine must resolve through ``nib.aff2axcodes`` rather than
    a sign-only heuristic on a single matrix entry.
    """
    affine = np.eye(4)
    affine[:3, :3] = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]

    assert "Voxel Order: SRA" in format_affine_info(affine)


def test_format_affine_info_accepts_a_plain_list_affine():
    """Callers may still pass a nested-list affine, as ``ROI3D``/``Peak3D``
    accept unvalidated affines through their public tuple-based factories.
    """
    affine = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]

    assert "Voxel Order: RAS" in format_affine_info(affine)


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

    try:
        assert os.environ["FURY_OFFSCREEN"] == "1"
        assert len(show_manager.screens) == 1
        assert show_manager._imgui is None
        assert tuple(show_manager.size) == (64, 48)
    finally:
        show_manager.close()


def test_create_window_uses_native_snapshot_dimensions():
    direct = window.ShowManager(
        size=(96, 72), title="direct", window_type="offscreen", imgui=False
    )
    skyline = create_window(visualizer_type="stealth", size=(96, 72), title="skyline")
    try:
        direct.render()
        skyline.render()
        direct.window.draw()
        skyline.window.draw()
        assert skyline.snapshot().shape == direct.snapshot().shape
    finally:
        direct.close()
        skyline.close()


def test_create_window_exits_on_an_unknown_visualizer_type(caplog):
    """An unrecognized visualizer type is reported and aborts the process."""
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
