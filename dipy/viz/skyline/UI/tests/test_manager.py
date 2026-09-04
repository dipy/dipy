"""Tests for the Skyline sidebar, driven through real ImGui frames.

``UIWindow.render`` is executed against a live ImGui context, with the sections
supplied by a real offscreen ``Skyline`` viewer, so the sidebar draws the same
widgets it draws in the application.
"""

import nibabel as nib
import numpy as np
import pytest

from dipy.data import default_sphere
from dipy.direction.peaks import PeaksAndMetrics
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.utils.optpkg import optional_package

_, has_fury, _ = optional_package("fury", min_version="2.0.0")
_, has_imgui, _ = optional_package(
    "imgui_bundle", min_version="1.92.600", max_version="1.92.801"
)
if not (has_fury and has_imgui):
    pytest.skip(
        "Requires fury>=2.0.0 and imgui_bundle>=1.92.600,<=1.92.801",
        allow_module_level=True,
    )
else:
    from imgui_bundle import imgui

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

    pam = PeaksAndMetrics()
    pam.affine = AFFINE
    pam.peak_dirs = np.zeros((3, 3, 3, 5, 3), dtype=np.float32)
    pam.peak_dirs[..., 0, 0] = 1.0
    pam.peak_values = np.ones((3, 3, 3, 5), dtype=np.float32)
    pam.peak_indices = np.zeros((3, 3, 3, 5), dtype=np.int32)
    pam.sphere = default_sphere

    n_coeffs = sum(2 * ell + 1 for ell in range(0, 9, 2))
    coeffs = np.zeros((2, 2, 2, n_coeffs), dtype=np.float32)
    coeffs[..., 0] = 1.0

    return skyline(
        visualizer_type="stealth",
        images=[(data, AFFINE, "vol.nii.gz")],
        peaks=[(pam, "peaks.pam5")],
        rois=[(mask, AFFINE, "roi.nii.gz")],
        surfaces=[(vertices, faces, "surf.gii")],
        tractograms=[(sft, "tracts.trk")],
        sh_coeffs=[(coeffs, AFFINE, "odf")],
        out_dir=str(out_dir),
        out_stealth_png="sidebar.png",
    )


def _window(**kwargs):
    kwargs.setdefault("logo_tex_ref", imgui.ImTextureRef())
    kwargs.setdefault("size", (400, 900))
    return UIWindow(kwargs.pop("title", "Image Controls"), **kwargs)


def _populate(window, viewer):
    for viz in viewer.visualizations:
        window.add(f"{viz.path}:{viz.name}", viz.renderer, viz.viz_type)
    return window


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

    window.add("vol.nii.gz:Image", lambda *a, **k: (False, False, False), "image")

    assert list(window.sections) == ["vol.nii.gz:Image"]
    assert window.section_open_states == {"vol.nii.gz:Image": False}


def test_add_keeps_the_open_state_of_an_existing_section(ui):
    window = _window()
    window.add("a:b", lambda *a, **k: (False, False, False), "image")
    window.section_open_states["a:b"] = True

    window.add("a:b", lambda *a, **k: (False, False, False), "image")

    assert window.section_open_states["a:b"] is True


def test_remove_drops_the_section_and_its_state(ui):
    window = _window()
    window.add("a:b", lambda *a, **k: (False, False, False), "image")

    window.remove("a:b")

    assert window.sections == {}
    assert window.section_open_states == {}


def test_remove_is_a_noop_for_an_unknown_section(ui):
    window = _window()

    window.remove("never added")

    assert window.sections == {}


def test_file_dialog_closed_forwards_every_selection_kind(ui):
    captured = {}
    window = _window(file_dialog_callback=lambda **kwargs: captured.update(kwargs))
    window._is_dialog_open = True

    window._file_dialog_closed(
        filenames=["a.nii.gz"], rois=["r.nii.gz"], shm_coeffs=["s.pam5"]
    )

    assert captured == {
        "filenames": ["a.nii.gz"],
        "rois": ["r.nii.gz"],
        "shm_coeffs": ["s.pam5"],
    }
    assert window._is_dialog_open is False


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

    window.add("misc:thing", section, "not-a-known-group")

    for _ in range(2):
        ui.frame(window.render)

    assert calls == [{}, {}]
    assert window._group_open == {}


def _closing_section(is_open, **kwargs):
    return is_open, True, False


def test_render_removes_sections_that_ask_to_close(ui):
    refreshes = []
    window = _window(render_callback=lambda: refreshes.append("refresh"))
    window.add("gone:Image", _closing_section, "image")

    ui.frame(window.render)
    ui.frame(window.render)

    assert window.sections == {}
    assert window.section_open_states == {}
    assert refreshes == ["refresh"]


def test_render_removes_sections_without_a_render_callback(ui):
    window = _window()
    window.add("gone:Image", _closing_section, "image")

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
