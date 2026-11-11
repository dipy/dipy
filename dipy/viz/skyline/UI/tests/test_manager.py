"""Tests for the Skyline sidebar, driven through real ImGui frames.

``UIWindow.render`` is executed against a live ImGui context, with the sections
supplied by a real offscreen ``Skyline`` viewer, so the sidebar draws the same
widgets it draws in the application.
"""

import logging
import time

import nibabel as nib
import numpy as np
import numpy.testing as npt
import pytest

from dipy.io.image import save_nifti
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.reconst.shm import real_sh_descoteaux, real_sh_tournier
from dipy.utils.optpkg import optional_package
from dipy.viz.skyline.compute import process_async_callbacks

_, has_fury, _ = optional_package("fury", min_version="2.1.0")
_, has_imgui, _ = optional_package(
    "imgui_bundle", min_version="1.92.600", max_version="1.92.801"
)
if not (has_fury and has_imgui):
    pytest.skip(
        "Requires fury>=2.1.0 and imgui_bundle>=1.92.600,<=1.92.801",
        allow_module_level=True,
    )
else:
    from fury.actor import show_slices
    from fury.utils import create_sh_basis_matrix
    from imgui_bundle import icons_fontawesome_6, imgui

    from dipy.viz.skyline.UI.manager import (
        _GROUP_LABELS,
        _GROUP_ORDER,
        UIManager,
        UIWindow,
    )
    from dipy.viz.skyline.app import skyline

AFFINE = np.eye(4)
SHAPE = (8, 9, 10)


@pytest.fixture(scope="module")
def viewer(tmp_path_factory):
    """One real offscreen viewer holding every visualization type."""
    out_dir = tmp_path_factory.mktemp("skyline_ui")
    rng = np.random.default_rng(0)
    data = rng.random(SHAPE).astype(np.float32)
    mask = (data > 0.5).astype(np.uint8)
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.uint32)
    lines = [
        np.array([[index * 0.1, 0.5 * t, 0.25 * t] for t in range(8)])
        for index in range(8)
    ]
    reference = nib.Nifti1Image(np.zeros(SHAPE, dtype=np.float32), AFFINE)
    sft = StatefulTractogram(lines, reference, Space.RASMM)

    peak_dirs = np.zeros((3, 3, 3, 5, 3), dtype=np.float32)
    peak_dirs[..., 0, 0] = 1.0
    peak_values = np.ones((3, 3, 3, 5), dtype=np.float32)

    n_coeffs = sum(2 * ell + 1 for ell in range(0, 9, 2))
    coeffs = np.zeros((2, 2, 2, n_coeffs), dtype=np.float32)
    coeffs[..., 0] = 1.0

    return skyline(
        visualizer_type="stealth",
        images=[(data, AFFINE, "vol.nii.gz")],
        peaks=[(peak_dirs, AFFINE, "peaks.pam5", peak_values)],
        rois=[(mask, AFFINE, "roi.nii.gz")],
        surfaces=[(vertices, faces, "surf.gii")],
        tractograms=[(sft, "tracts.trk")],
        sh_coeffs=[(coeffs, AFFINE, "odf")],
        out_dir=str(out_dir),
        out_stealth_png="sidebar.png",
    )


@pytest.fixture
def import_viewer(tmp_path):
    coeffs = np.tile(np.array([4, 0, 0.2, 0, -0.1, 0], dtype=np.float32), (2, 2, 2, 1))
    return skyline(
        visualizer_type="stealth",
        sh_coeffs=[(coeffs, AFFINE, "existing_odf")],
        out_dir=str(tmp_path),
        out_stealth_png="import.png",
    )


@pytest.fixture
def overlapping_images_viewer(tmp_path):
    data = np.arange(np.prod(SHAPE), dtype=np.float32).reshape(SHAPE)
    return skyline(
        visualizer_type="stealth",
        images=[
            (data, AFFINE, "first.nii.gz"),
            (data + 1, AFFINE, "second.nii.gz"),
        ],
        out_dir=str(tmp_path),
        out_stealth_png="overlapping.png",
    )


def _window(**kwargs):
    kwargs.setdefault("logo_tex_ref", imgui.ImTextureRef())
    kwargs.setdefault("size", (400, 900))
    return UIWindow(kwargs.pop("title", "Image Controls"), **kwargs)


def _populate(window, viewer):
    for viz in viewer.visualizations:
        window.add(f"{viz.path}:{viz.name}", viz.renderer, viz_type=viz.viz_type)
    return window


def _finish_loading(viewer):
    deadline = time.monotonic() + 10
    while viewer._loading_total and time.monotonic() < deadline:
        process_async_callbacks()
        viewer._drain_pending_visualizations()
        time.sleep(0.01)


def test_ui_manager_registers_windows_by_name(ui):
    manager = UIManager()
    window = _window()

    manager.add_window("sidebar", window)

    assert manager.windows == {"sidebar": window}


def test_ui_manager_starts_empty():
    assert UIManager().windows == {}


def test_uiwindow_defaults(ui):
    window = _window()

    assert window.title == "Image Controls"
    assert window.is_open is True
    assert window.pos == (0, 0)
    assert window.size == (400, 900)
    assert window.sections == {}
    assert window.section_open_states == {}
    assert window.request_file_dialog is False
    assert window._bg_color == (0.1, 0.1, 0.1)
    assert window._color_picker_popup_id == "bg_color_picker_popup##Image Controls"
    assert window._is_dialog_open is False


def test_uiwindow_replaces_missing_callbacks_with_no_ops(ui):
    window = _window()

    assert window.render_callback() is None
    assert window.file_dialog_callback() is None
    assert window.bg_color_callback() is None
    assert window.snapshot_callback() is None


def test_uiwindow_replaces_non_callable_callbacks_with_no_ops(ui):
    window = _window(
        render_callback="not callable",
        file_dialog_callback=3,
        bg_color_callback=None,
        snapshot_callback=[],
    )

    assert window.render_callback() is None
    assert window.file_dialog_callback() is None
    assert window.bg_color_callback() is None
    assert window.snapshot_callback() is None


def test_uiwindow_keeps_supplied_callbacks(ui):
    calls = []
    window = _window(render_callback=lambda: calls.append("render"))

    window.render_callback()

    assert calls == ["render"]


def test_add_registers_a_section_closed(ui):
    window = _window()

    window.add(
        "vol.nii.gz:Image", lambda *a, **k: (False, False, False), viz_type="image"
    )

    assert list(window.sections) == ["vol.nii.gz:Image"]
    assert window.section_open_states == {"vol.nii.gz:Image": False}


def test_add_keeps_the_open_state_of_an_existing_section(ui):
    window = _window()
    window.add("a:b", lambda *a, **k: (False, False, False), viz_type="image")
    window.section_open_states["a:b"] = True

    window.add("a:b", lambda *a, **k: (False, False, False), viz_type="image")

    assert window.section_open_states["a:b"] is True


def test_remove_drops_the_section_and_its_state(ui):
    window = _window()
    window.add("a:b", lambda *a, **k: (False, False, False), viz_type="image")

    window.remove("a:b")

    assert window.sections == {}
    assert window.section_open_states == {}


def test_remove_is_a_noop_for_an_unknown_section(ui):
    window = _window()

    window.remove("never added")

    assert window.sections == {}


@pytest.mark.parametrize(
    "sh_basis, vector, evaluate_basis, legacy",
    [
        ("descoteaux07", [4, 0, 0.2, 0, -0.1, 0], real_sh_descoteaux, False),
        (
            "descoteaux07_legacy",
            [4, 0, -0.2, 0, -0.1, 0],
            real_sh_descoteaux,
            True,
        ),
        (
            "tournier07",
            [4, 0, -0.1 * np.sqrt(2), 0, -0.2 * np.sqrt(2), 0],
            real_sh_tournier,
            True,
        ),
        ("tournier19", [4, 0, -0.1, 0, -0.2, 0], real_sh_tournier, False),
    ],
)
def test_file_dialog_closed_imports_odfs_with_declared_basis(
    ui, tmp_path, import_viewer, sh_basis, vector, evaluate_basis, legacy
):
    data = np.tile(np.array(vector, dtype=np.float32), (2, 2, 2, 1))
    path = str(tmp_path / "selected_odf.nii.gz")
    save_nifti(path, data, AFFINE)
    existing_coeffs = import_viewer._sh_glyph_visualizations[0]._slicer.coeffs_4d.copy()
    existing_info = import_viewer._sh_glyph_visualizations[0]._info
    assert "Input SH basis: descoteaux07" in existing_info
    window = _populate(
        _window(
            file_dialog_callback=import_viewer._append_visualization, sh_basis=sh_basis
        ),
        import_viewer,
    )
    window._is_dialog_open = True
    ui.frame(window.render)
    window._sh_basis = (
        "descoteaux07_legacy" if sh_basis == "descoteaux07" else "descoteaux07"
    )

    window._file_dialog_closed(shm_coeffs=[path], sh_basis=sh_basis)
    _finish_loading(import_viewer)
    ui.frame(_populate(window, import_viewer).render)

    assert window._is_dialog_open is False
    glyphs = {glyph.path: glyph for glyph in import_viewer._sh_glyph_visualizations}
    assert set(glyphs) == {"existing_odf", path}
    assert f"Input SH basis: {sh_basis}" in glyphs[path]._info
    theta = np.array([0.41, 0.9, 1.37])
    phi = np.array([0.23, 1.11, 2.29])
    vertices = np.column_stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))
    )
    if legacy:
        with pytest.warns(PendingDeprecationWarning, match="The legacy .* basis"):
            source_basis, _, _ = evaluate_basis(2, theta, phi, legacy=True)
    else:
        source_basis, _, _ = evaluate_basis(2, theta, phi, legacy=False)
    standard_basis = create_sh_basis_matrix(vertices, 2)
    npt.assert_allclose(
        glyphs[path]._slicer.coeffs_4d @ standard_basis.T,
        data @ source_basis.T,
        atol=1e-6,
        rtol=1e-5,
    )
    npt.assert_array_equal(glyphs["existing_odf"]._slicer.coeffs_4d, existing_coeffs)
    assert glyphs["existing_odf"]._info == existing_info


def test_file_dialog_closed_imports_non_odf_selections(
    ui, tmp_path, import_viewer, caplog
):
    image = np.arange(np.prod(SHAPE), dtype=np.float32).reshape(SHAPE)
    roi = np.zeros(SHAPE, dtype=np.uint8)
    roi[2:5, 3:6, 4:7] = 1
    peak_dirs = np.zeros((2, 2, 2, 1, 3), dtype=np.float32)
    peak_dirs[..., 0] = 1
    image_path = str(tmp_path / "image.nii.gz")
    roi_path = str(tmp_path / "roi.nii.gz")
    peak_path = str(tmp_path / "peaks.nii.gz")
    for path, data in ((image_path, image), (roi_path, roi), (peak_path, peak_dirs)):
        save_nifti(path, data, AFFINE)
    existing_coeffs = import_viewer._sh_glyph_visualizations[0]._slicer.coeffs_4d.copy()
    existing_info = import_viewer._sh_glyph_visualizations[0]._info
    window = _populate(
        _window(file_dialog_callback=import_viewer._append_visualization), import_viewer
    )
    window._is_dialog_open = True
    ui.frame(window.render)

    with caplog.at_level(logging.ERROR):
        window._file_dialog_closed(
            filenames=[image_path],
            rois=[roi_path],
            peaks=[peak_path],
            sh_basis="unknown",
        )
        _finish_loading(import_viewer)
    ui.frame(_populate(window, import_viewer).render)

    assert window._is_dialog_open is False
    assert {viz.path for viz in import_viewer.visualizations} == {
        "existing_odf",
        image_path,
        roi_path,
        peak_path,
    }
    npt.assert_array_equal(import_viewer._image_visualizations[0].dwi, image)
    npt.assert_array_equal(import_viewer._roi_visualizations[0].roi, roi)
    npt.assert_array_equal(import_viewer._peak_visualizations[0].peaks, peak_dirs)
    npt.assert_array_equal(
        import_viewer._sh_glyph_visualizations[0]._slicer.coeffs_4d, existing_coeffs
    )
    assert import_viewer._sh_glyph_visualizations[0]._info == existing_info
    assert not any(record.levelno >= logging.ERROR for record in caplog.records)


@pytest.mark.parametrize("selection", [None, []])
def test_file_dialog_closed_cancellation_preserves_visualizations(
    ui, import_viewer, selection, caplog
):
    existing_visualizations = tuple(import_viewer.visualizations)
    existing_coeffs = import_viewer._sh_glyph_visualizations[0]._slicer.coeffs_4d.copy()
    existing_info = import_viewer._sh_glyph_visualizations[0]._info
    window = _populate(
        _window(file_dialog_callback=import_viewer._append_visualization), import_viewer
    )
    window._is_dialog_open = True
    ui.frame(window.render)

    with caplog.at_level(logging.ERROR):
        window._file_dialog_closed(
            filenames=selection,
            rois=selection,
            peaks=selection,
            shm_coeffs=selection,
            sh_basis="unknown",
        )
        _finish_loading(import_viewer)
    ui.frame(window.render)

    assert window._is_dialog_open is False
    assert tuple(import_viewer.visualizations) == existing_visualizations
    npt.assert_array_equal(
        import_viewer._sh_glyph_visualizations[0]._slicer.coeffs_4d, existing_coeffs
    )
    assert import_viewer._sh_glyph_visualizations[0]._info == existing_info
    assert not any(record.levelno >= logging.ERROR for record in caplog.records)


def test_update_bg_color_stores_and_notifies(ui):
    seen = []
    window = _window(bg_color_callback=seen.append)

    window._update_bg_color((0.2, 0.4, 0.6))

    assert window._bg_color == (0.2, 0.4, 0.6)
    assert seen == [(0.2, 0.4, 0.6)]


@pytest.mark.parametrize(
    "filenames,expected",
    [
        (["/tmp/skyline_snapshot.png"], "/tmp/skyline_snapshot.png"),
        ("/tmp/skyline_snapshot.png", "/tmp/skyline_snapshot.png"),
        (["/tmp/first.png", "/tmp/second.png"], "/tmp/first.png"),
    ],
)
def test_snapshot_dialog_closed_forwards_the_selected_path(ui, filenames, expected):
    captured = []
    window = _window(snapshot_callback=captured.append)

    window._snapshot_dialog_closed(filenames=filenames, rois=None, shm_coeffs=None)

    assert captured == [expected]


@pytest.mark.parametrize("filenames", [None, []])
def test_snapshot_dialog_closed_ignores_a_cancelled_dialog(ui, filenames):
    captured = []
    window = _window(snapshot_callback=captured.append)

    window._snapshot_dialog_closed(filenames=filenames)

    assert captured == []


def test_update_loader_toggles_the_overlay_and_message(ui):
    window = _window()

    window.update_loader(show=True, message="Loading Files...")

    assert window._show_loader is True
    assert window._loading_message == "Loading Files..."

    window.update_loader(show=False)

    assert window._show_loader is False
    assert window._loading_message == "Loading Files..."


def test_render_draws_an_empty_sidebar(ui):
    window = _window()

    ui.frame(window.render)
    ui.frame(window.render)

    assert window.sections == {}


def test_render_emits_geometry(ui):
    window = _window()

    ui.frame(window.render)
    ui.frame(window.render)
    draw_data = imgui.get_draw_data()

    assert draw_data.total_vtx_count > 0


def test_render_draws_every_visualization_section(ui, viewer):
    window = _populate(_window(), viewer)

    for _ in range(3):
        ui.frame(window.render)

    assert len(window.sections) == len(viewer.visualizations)
    assert set(window.section_open_states) == set(window.sections)


def test_render_draws_open_sections_of_every_visualization_type(ui, viewer):
    window = _populate(_window(), viewer)
    for name in window.section_open_states:
        window.section_open_states[name] = True

    for _ in range(3):
        ui.frame(window.render)

    assert all(window.section_open_states.values())
    assert {viz_type for _, viz_type in window.sections.values()} == {
        "image",
        "peak",
        "roi",
        "surface",
        "tractography",
        "sh_glyph",
    }


@pytest.mark.parametrize("first_is_open", [False, True])
def test_image_titles_select_front_layer_without_expanding(
    ui, overlapping_images_viewer, first_is_open
):
    viewer = overlapping_images_viewer
    first, second = viewer._image_visualizations
    window = _window()
    header_geometry = {}
    section_names = {}
    base_poses = {}

    def slice_planes(image):
        return np.asarray(
            [
                child.plane if hasattr(child, "plane") else child.material.plane
                for child in image.actor.children
            ]
        )

    def register_image(image):
        section_name = f"{image.path}:{image.name}"
        section_names[image] = section_name

        def render_image(is_open, *, group_visible=True):
            start = imgui.get_cursor_screen_pos()
            width = imgui.get_content_region_avail().x
            header_geometry[image] = (start.x, start.y, width)
            return image.renderer(is_open, group_visible=group_visible)

        window.add(section_name, render_image, viz_type="image")
        show_slices(image.actor, image.state)
        base_poses[image] = slice_planes(image).copy()

    register_image(first)
    register_image(second)
    window.section_open_states[section_names[first]] = first_is_open
    initial_expansion = window.section_open_states.copy()
    viewer._arrange_image_actors()

    def render():
        window.render()
        viewer._arrange_image_actors()

    def click_header(image, *, arrow=False):
        for _ in range(3):
            ui.frame(render)
        x, y, width = header_geometry[image]
        position = (x + width - 26 if arrow else x + 70, y + 20)
        for _ in range(3):
            ui.frame(render, mouse=position)
        ui.frame(render, mouse=position, press=True)
        ui.frame(render, mouse=position)

    def assert_front_and_base(front, back):
        assert viewer.active_image is front
        np.testing.assert_allclose(slice_planes(back), base_poses[back])
        np.testing.assert_allclose(slice_planes(front)[:, :3], base_poses[front][:, :3])
        np.testing.assert_allclose(
            slice_planes(front)[:, 3], base_poses[front][:, 3] + 0.01
        )

    click_header(first)
    assert_front_and_base(first, second)
    assert window.section_open_states == initial_expansion

    click_header(second)
    assert_front_and_base(second, first)
    assert window.section_open_states == initial_expansion

    click_header(first, arrow=True)
    assert window.section_open_states[section_names[first]] is not first_is_open
    assert window.section_open_states[section_names[second]] is False
    assert_front_and_base(second, first)

    click_header(first, arrow=True)
    assert window.section_open_states == initial_expansion
    assert_front_and_base(second, first)


def test_render_groups_sections_under_the_documented_labels(ui, viewer):
    window = _populate(_window(), viewer)

    for _ in range(3):
        ui.frame(window.render)

    assert set(window._group_open) == {
        viz_type for _, viz_type in window.sections.values()
    }
    assert all(window._group_open.values())
    assert all(window._group_visible.values())
    assert set(_GROUP_LABELS) == set(_GROUP_ORDER)


def test_render_draws_sections_with_an_unrecognised_type(ui):
    window = _window()
    calls = []

    def section(is_open, **kwargs):
        calls.append(kwargs)
        return is_open, False, False

    window.add("misc:thing", section, viz_type="not-a-known-group")

    for _ in range(2):
        ui.frame(window.render)

    assert calls == [{}, {}]
    assert window._group_open == {}


def _closing_section(is_open, **kwargs):
    return is_open, True, False


def test_render_removes_sections_that_ask_to_close(ui):
    refreshes = []
    window = _window(render_callback=lambda: refreshes.append("refresh"))
    window.add("gone:Image", _closing_section, viz_type="image")

    ui.frame(window.render)
    ui.frame(window.render)

    assert window.sections == {}
    assert window.section_open_states == {}
    assert refreshes == ["refresh"]


def test_render_removes_sections_without_a_render_callback(ui):
    window = _window()
    window.add("gone:Image", _closing_section, viz_type="image")

    ui.frame(window.render)
    ui.frame(window.render)

    assert window.sections == {}


def test_render_shows_the_loading_overlay(ui):
    window = _window()
    window.update_loader(show=True, message="Loading Files...")

    for _ in range(3):
        ui.frame(window.render)

    assert window._show_loader is True


def test_render_opens_the_file_dialog_popup_on_request(ui):
    window = _window()
    window.request_file_dialog = True

    for _ in range(2):
        ui.frame(window.render)

    assert window.request_file_dialog is True


def test_render_tracks_the_background_color_picker_state(ui):
    window = _window()

    ui.frame(window.render)
    ui.frame(window.render)

    assert window._color_picker_open is False
    assert window._draft_color == window._bg_color


@pytest.fixture
def slice_viewer(tmp_path):
    data = np.arange(np.prod(SHAPE), dtype=np.float32).reshape(SHAPE)
    peaks = np.zeros((*SHAPE, 1, 3), dtype=np.float32)
    peaks[..., 0, 0] = 1
    coeffs = np.zeros((*SHAPE, 45), dtype=np.float32)
    coeffs[..., 0] = 1
    return skyline(
        visualizer_type="stealth",
        images=[(data, AFFINE, "first"), (data + 1, AFFINE, "second")],
        peaks=[(peaks, AFFINE, "peaks", np.ones((*SHAPE, 1), dtype=np.float32))],
        sh_coeffs=[(coeffs, AFFINE, "odf")],
        out_dir=str(tmp_path),
    )


@pytest.mark.parametrize("synchronized", [True, False])
def test_global_slices_remain_interactive_with_collapsed_groups(
    ui, slice_viewer, synchronized
):
    viewer = slice_viewer
    first, second = viewer._image_visualizations
    peak = viewer._peak_visualizations[0]
    glyph = viewer._sh_glyph_visualizations[0]
    for viz in viewer.visualizations:
        viz._synchronize = synchronized
    peak._synchronize = False
    before = [np.array(viz.state).copy() for viz in viewer.visualizations]
    geometry = {}

    def controls():
        result = viewer._get_global_slice_controls()
        start = imgui.get_cursor_screen_pos()
        spacing = imgui.get_style().item_spacing.y
        message = (
            f"{icons_fontawesome_6.ICON_FA_CROSSHAIRS} Reference: {result[0]}"
            if result[0] is not None
            else "Enable Synchronize Slices to adjust positions."
        )
        message_height = imgui.calc_text_size(
            message, wrap_width=imgui.get_content_region_avail().x
        ).y
        geometry["row_y"] = (
            start.y
            + imgui.get_text_line_height_with_spacing()
            + message_height
            + 5 * spacing
        )
        icon_width = imgui.calc_text_size(icons_fontawesome_6.ICON_FA_TOGGLE_ON).x
        geometry["toggle_x"] = start.x + icon_width / 2
        geometry["slider_x"] = (
            start.x + icon_width + 8 + imgui.calc_text_size("Y").x + 16 + 7
        )
        geometry["slider_end"] = start.x + imgui.get_content_region_avail().x - 57
        geometry["row_step"] = 20 + spacing
        return result

    window = _populate(
        _window(
            slice_controls_callback=controls,
            slice_position_callback=viewer._set_global_slice_position,
            slice_visibility_callback=viewer._set_global_slice_visibility,
        ),
        viewer,
    )
    window._group_open.update(image=False, peak=False, sh_glyph=False)
    expansion = window.section_open_states.copy()
    groups = window._group_open.copy()

    def render():
        viewer._is_drawing_ui = True
        try:
            window.render()
        finally:
            viewer._is_drawing_ui = False
        viewer._flush_pending_sync_requests()
        viewer._flush_pending_scene_ops()
        viewer._arrange_image_actors()

    def click(position):
        for _ in range(3):
            ui.frame(render, mouse=position)
        ui.frame(render, mouse=position, press=True)
        ui.frame(render, mouse=position)

    def assert_visibility(expected):
        for image in (first, second):
            assert [child.visible for child in image.actor.children] == expected
        for chunk in peak.actor.children:
            assert list(chunk.material.visibility) == expected
        material = glyph._slicer._glyph_actor.material
        assert [material.vis_x, material.vis_y, material.vis_z] == expected

    for _ in range(3):
        ui.frame(render)
    assert_visibility([True, True, True])
    toggle_position = (geometry["toggle_x"], geometry["row_y"] + 8)
    click(toggle_position)
    assert_visibility([False, True, True])
    first.set_slice_visibility([True, True, True])
    ui.frame(render)
    assert viewer._get_global_slice_controls()[3] == [True, True, True]
    assert [child.visible for child in second.actor.children] == [False, True, True]
    assert glyph._slicer._glyph_actor.material.vis_x == 0
    click(toggle_position)
    assert_visibility([False, True, True])
    click(toggle_position)
    assert_visibility([True, True, True])

    slider_position = (
        geometry["slider_x"]
        + (geometry["slider_end"] - geometry["slider_x"]) / SHAPE[1],
        geometry["row_y"] + geometry["row_step"] + 10,
    )
    click(slider_position)
    if synchronized:
        for viz in (first, second, glyph):
            assert viz.state[1] == 1
        np.testing.assert_allclose(first.actor.children[1].material.plane[3], 1.001)
        np.testing.assert_allclose(second.actor.children[1].material.plane[3], 1.011)
        assert glyph._slicer._glyph_actor.material.active_slice_y == 1
    else:
        for viz, state in zip(viewer.visualizations, before):
            np.testing.assert_array_equal(viz.state, state)
    np.testing.assert_array_equal(peak.state, before[2])
    for chunk in peak.actor.children:
        np.testing.assert_array_equal(chunk.cross_section, np.round(before[2]))
    assert window.section_open_states == expansion
    assert window._group_open == groups
