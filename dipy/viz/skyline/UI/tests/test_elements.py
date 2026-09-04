"""Tests for the Skyline ImGui widgets.

Widgets are drawn in real ImGui frames and driven with real mouse events fed
through ``ImGuiIO``; the assertions are on what the widgets actually return for
that input.
"""

import numpy as np
import pytest

from dipy.utils.optpkg import optional_package

_, has_fury, _ = optional_package("fury", min_version="2.0.0")
if not has_fury:
    pytest.skip("Requires fury>=2.0.0", allow_module_level=True)
else:
    from dipy.viz.skyline.UI import elements
    from dipy.viz.skyline.UI.elements import (
        colors_equal,
        normalize_picker_color,
    )

_, has_imgui, _ = optional_package(
    "imgui_bundle", min_version="1.92.600", max_version="1.92.801"
)

if has_imgui:
    from dipy.viz.skyline.UI.elements import (
        color_picker,
        create_numeric_input,
        downloader,
        dropdown,
        loading,
        open_confirmation_dialog,
        render_group,
        render_section_header,
        segmented_switch,
        thin_slider,
        toggle_button,
        two_disk_slider,
        uploader,
        warning_message,
    )

requires_imgui = pytest.mark.skipif(
    not has_imgui, reason="Requires imgui_bundle>=1.92.600,<=1.92.801"
)

# Hit points measured from a real frame drawn at the harness origin (50, 50).
# The section header is laid out right-to-left as show / close / info / arrow.
HEADER_WIDTH = 300
HEADER_BODY = (120, 70)
HEADER_SHOW_ICON = (266, 70)
HEADER_CLOSE_ICON = (286, 70)
HEADER_ARROW_ICON = (324, 70)
# ``create_numeric_input`` with a hidden label draws its spinner column here.
SPINNER_UP = (145, 58)
SPINNER_DOWN = (145, 74)
# ``thin_slider(label="", width=200)`` track extent; a visible label shifts the
# track right by its text width, so the interactive tests draw an empty label.
SLIDER_TRACK_LEFT = (70, 60)
SLIDER_TRACK_MID = (140, 60)
SLIDER_TRACK_RIGHT = (205, 60)


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), True),
        ((1, 0, 0), (1.0, 0.0, 0.0), True),
        (np.array([0.5, 0.25, 1.0]), (0.5, 0.25, 1.0), True),
        ((1.0, 0.0, 0.0, 0.5), (1.0, 0.0, 0.0, 1.0), True),
        (np.array([1, 2, 3, 4]), (1, 2, 3), True),
    ],
)
def test_colors_equal_numeric_sequences(a, b, expected):
    """``colors_equal`` matches RGB regardless of alpha or container type."""
    assert colors_equal(a, b) is expected


def test_colors_equal_plain_strings():
    """``colors_equal`` compares two string operands with lexicographic equality."""
    assert colors_equal("direction", "direction") is True
    assert colors_equal("direction", "random") is False


def test_colors_equal_mixed_string_and_sequence_false():
    """``colors_equal`` is False when only one operand is a string color name."""
    assert colors_equal("direction", (1.0, 0.0, 0.0)) is False
    assert colors_equal((1.0, 0.0, 0.0), "direction") is False


def test_colors_equal_rejects_non_vector_shapes():
    """``colors_equal`` returns False for non-1D array-like inputs."""
    assert colors_equal(np.zeros((2, 3)), np.zeros((2, 3))) is False


@pytest.mark.parametrize(
    "color,fallback,expected",
    [
        ((0.25, 0.5, 0.75), None, (0.25, 0.5, 0.75)),
        (np.array([0.0, 1.0, 0.0, 0.9]), None, (0.0, 1.0, 0.0)),
        ("direction", (0.1, 0.2, 0.3), (0.1, 0.2, 0.3)),
        ((), (0.0, 0.5, 1.0), (0.0, 0.5, 1.0)),
        (np.array([0.1, 0.2]), (1.0, 1.0, 0.0), (1.0, 1.0, 0.0)),
    ],
)
def test_normalize_picker_color(color, fallback, expected):
    """``normalize_picker_color`` returns RGB floats or the supplied fallback."""
    if fallback is None:
        assert normalize_picker_color(color) == expected
    else:
        assert normalize_picker_color(color, fallback=fallback) == expected


def test_normalize_picker_color_default_fallback():
    """String inputs use the default red fallback when none is passed."""
    assert normalize_picker_color("direction") == (1.0, 0.0, 0.0)


@requires_imgui
def test_ensure_last_dir_creates_missing_directory(tmp_path):
    missing_dir = tmp_path / "new" / ".dipy"
    original_last_dir = elements._LAST_DIR
    elements._LAST_DIR = missing_dir

    try:
        resolved_dir = elements._ensure_last_dir()

        assert resolved_dir == missing_dir
        assert missing_dir.exists()
        assert missing_dir.is_dir()
    finally:
        elements._LAST_DIR = original_last_dir


@requires_imgui
def test_ensure_last_dir_uses_parent_when_last_dir_is_file(tmp_path):
    file_path = tmp_path / "last_location.txt"
    file_path.write_text("placeholder")
    original_last_dir = elements._LAST_DIR
    elements._LAST_DIR = file_path

    try:
        resolved_dir = elements._ensure_last_dir()

        assert resolved_dir == file_path.parent
        assert resolved_dir.exists()
        assert resolved_dir.is_dir()
    finally:
        elements._LAST_DIR = original_last_dir


@requires_imgui
def test_set_last_dir_remembers_the_containing_directory(tmp_path):
    original_last_dir = elements._LAST_DIR
    selected = tmp_path / "scan.nii.gz"
    selected.write_bytes(b"")

    try:
        elements._set_last_dir(str(selected))

        assert elements._LAST_DIR == tmp_path
        assert elements._ensure_last_dir() == tmp_path
    finally:
        elements._LAST_DIR = original_last_dir


@requires_imgui
def test_imgui_add_rect_exposes_thickness_keyword():
    """The installed bindings accept the ``thickness`` keyword we rely on."""
    signature = elements.imgui.ImDrawList.add_rect.__doc__ or ""

    assert "thickness" in signature
    assert "flags" in signature


@requires_imgui
@pytest.mark.parametrize(
    "pos,size,padding,expected",
    [
        ((10, 20), (30, 40), 4, ((6, 16), (44, 64))),
        ((0, 0), (10, 10), 0, ((0, 0), (10, 10))),
    ],
)
def test_calculate_hit_box_expands_by_the_padding(pos, size, padding, expected):
    hit_min, hit_max = elements._calculate_hit_box(pos, size, padding)

    assert (hit_min.x, hit_min.y) == expected[0]
    assert (hit_max.x, hit_max.y) == expected[1]


@requires_imgui
def test_toggle_button_is_unchanged_without_input(ui):
    assert ui.draw(lambda: toggle_button(False, label="Show")) == (False, False)
    assert ui.draw(lambda: toggle_button(True)) == (False, True)


@requires_imgui
def test_toggle_button_turns_on_when_clicked(ui):
    assert ui.click(lambda: toggle_button(False, label="Show")) == (True, True)


@requires_imgui
def test_toggle_button_turns_off_when_clicked(ui):
    assert ui.click(lambda: toggle_button(True)) == (True, False)


@requires_imgui
def test_warning_message_draws_without_returning_a_value(ui):
    assert ui.draw(lambda: warning_message("Large tractogram")) is None


@requires_imgui
def test_downloader_and_uploader_draw_their_labels(ui):
    calls = []

    assert ui.draw(lambda: downloader("Save", calls.append)) is None
    assert ui.draw(lambda: uploader("BUAN", calls.append)) is None
    assert ui.draw(lambda: uploader("BUAN", calls.append, selected="pvals.npy")) is None
    assert calls == []


@requires_imgui
def test_color_picker_is_closed_until_clicked(ui):
    changed, color, is_open = ui.draw(
        lambda: color_picker(selected_color=(1.0, 0.0, 0.0), popup_id="p")
    )

    assert (changed, is_open) == (False, False)
    assert color == (1.0, 0.0, 0.0)


@requires_imgui
def test_color_picker_opens_its_popup_on_click(ui):
    changed, color, is_open = ui.click(
        lambda: color_picker(
            label="Color", selected_color=(0.2, 0.4, 0.6), popup_id="p"
        )
    )

    assert is_open is True
    assert changed is False
    np.testing.assert_allclose(color, (0.2, 0.4, 0.6), atol=1e-3)


@requires_imgui
def test_section_header_is_inert_without_input(ui):
    result = ui.draw(
        lambda: render_section_header(
            "Images", is_open=False, type="image", width=HEADER_WIDTH
        )
    )

    assert result == (False, True, False, False)


@requires_imgui
def test_section_header_body_click_opens_the_section(ui):
    result = ui.click(
        lambda: render_section_header(
            "Images", is_open=False, type="image", width=HEADER_WIDTH
        ),
        at=HEADER_BODY,
    )

    assert result == (True, True, False, True)


@requires_imgui
def test_section_header_arrow_toggles_open_without_reporting_a_change(ui):
    result = ui.click(
        lambda: render_section_header(
            "Images", is_open=False, type="image", width=HEADER_WIDTH
        ),
        at=HEADER_ARROW_ICON,
    )

    assert result == (True, True, False, False)


@requires_imgui
def test_section_header_arrow_closes_an_open_section(ui):
    result = ui.click(
        lambda: render_section_header(
            "Images", is_open=True, type="image", width=HEADER_WIDTH
        ),
        at=HEADER_ARROW_ICON,
    )

    assert result == (False, True, False, False)


@requires_imgui
def test_section_header_show_icon_toggles_visibility(ui):
    result = ui.click(
        lambda: render_section_header(
            "Images", is_open=False, is_visible=True, type="image", width=HEADER_WIDTH
        ),
        at=HEADER_SHOW_ICON,
    )

    assert result == (False, False, False, False)


@requires_imgui
def test_section_header_close_icon_requests_removal(ui):
    result = ui.click(
        lambda: render_section_header(
            "Images", is_open=False, type="image", width=HEADER_WIDTH
        ),
        at=HEADER_CLOSE_ICON,
    )

    assert result == (False, True, True, False)


@requires_imgui
def test_section_header_without_close_and_info_icons_moves_the_arrow(ui):
    with_icons = ui.click(
        lambda: render_section_header(
            "Images", is_open=False, type="image", width=HEADER_WIDTH
        ),
        at=HEADER_CLOSE_ICON,
    )
    without_icons = ui.click(
        lambda: render_section_header(
            "Images",
            is_open=False,
            type="image",
            width=HEADER_WIDTH,
            show_close=False,
            show_info=False,
        ),
        at=HEADER_CLOSE_ICON,
    )

    assert with_icons == (False, True, True, False)
    assert without_icons[2] is False


@requires_imgui
@pytest.mark.parametrize(
    "viz_type",
    ["image", "roi", "surface", "peak", "tractography", "sh_glyph", None],
)
def test_section_header_draws_every_visualization_icon(ui, viz_type):
    result = ui.draw(
        lambda: render_section_header(
            "Section", is_open=True, type=viz_type, width=HEADER_WIDTH
        )
    )

    assert result == (True, True, False, False)


@requires_imgui
def test_section_header_truncates_a_long_label(ui):
    result = ui.draw(
        lambda: render_section_header(
            "an extremely long visualization name that will not fit at all",
            is_open=False,
            type="image",
            width=160,
            info="tooltip body",
        )
    )

    assert result == (False, True, False, False)


@requires_imgui
def test_section_header_uses_the_available_width_by_default(ui):
    result = ui.draw(lambda: render_section_header("Images", type="image"))

    assert result == (True, True, False, False)


@requires_imgui
def test_numeric_input_is_unchanged_without_input(ui):
    assert ui.draw(lambda: create_numeric_input("##a", 5)) == (False, 5)


@requires_imgui
def test_numeric_input_up_arrow_increments(ui):
    assert ui.click(
        lambda: create_numeric_input("##b", 5, width=106), at=SPINNER_UP
    ) == (True, 6)


@requires_imgui
def test_numeric_input_down_arrow_decrements(ui):
    assert ui.click(
        lambda: create_numeric_input("##c", 5, width=106), at=SPINNER_DOWN
    ) == (True, 4)


@requires_imgui
def test_numeric_input_honours_the_step(ui):
    assert ui.click(
        lambda: create_numeric_input("##d", 10, step=5, width=106), at=SPINNER_UP
    ) == (True, 15)


@requires_imgui
def test_numeric_input_float_step_produces_a_float(ui):
    changed, value = ui.click(
        lambda: create_numeric_input(
            "##e", 1.5, value_type="float", step=0.25, width=106
        ),
        at=SPINNER_UP,
    )

    assert changed is True
    assert value == pytest.approx(1.75)


@requires_imgui
def test_numeric_input_clamps_a_non_positive_float_step_to_one(ui):
    changed, value = ui.click(
        lambda: create_numeric_input("##f", 1.0, value_type="float", step=0, width=106),
        at=SPINNER_UP,
    )

    assert (changed, value) == (True, 2.0)


@requires_imgui
def test_numeric_input_warns_when_a_float_is_given_to_an_int_field(ui, caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        changed, value = ui.draw(lambda: create_numeric_input("##g", 2.6))

    assert (changed, value) == (False, 3)
    assert "converted to int for integer input" in caplog.text


@requires_imgui
def test_numeric_input_rejects_an_unknown_value_type(ui):
    with pytest.raises(ValueError, match="value_type must be either"):
        ui.draw(lambda: create_numeric_input("##h", 1, value_type="complex"))


@requires_imgui
def test_numeric_input_draws_a_visible_label_and_a_fixed_label_column(ui):
    assert ui.draw(lambda: create_numeric_input("Slices", 3)) == (False, 3)
    assert ui.draw(lambda: create_numeric_input("Slices", 3, label_width=80)) == (
        False,
        3,
    )
    assert ui.draw(lambda: create_numeric_input("##i", 3, label_width=80)) == (False, 3)


@requires_imgui
def test_segmented_switch_is_unchanged_without_input(ui):
    assert ui.draw(lambda: segmented_switch("Type", ["Line", "Tube"], "Line")) == (
        False,
        "Line",
    )


@requires_imgui
def test_segmented_switch_selects_another_segment(ui):
    def render():
        return segmented_switch("Type", ["Line", "Tube"], "Line")

    ui.draw(render)
    rect_min, rect_max = ui.last_rect
    last_segment = ((rect_min.x + rect_max.x) * 0.5, (rect_min.y + rect_max.y) * 0.5)

    assert ui.press_release(render, at=last_segment) == (True, "Tube")


@requires_imgui
def test_segmented_switch_keeps_the_current_segment(ui):
    def render():
        return segmented_switch("Type", ["Line", "Tube"], "Tube")

    ui.draw(render)
    rect_min, rect_max = ui.last_rect
    last_segment = ((rect_min.x + rect_max.x) * 0.5, (rect_min.y + rect_max.y) * 0.5)

    assert ui.press_release(render, at=last_segment) == (False, "Tube")


@requires_imgui
def test_segmented_switch_returns_early_without_options(ui):
    assert ui.draw(lambda: segmented_switch("Type", [], "Line")) == (False, "Line")


@requires_imgui
def test_segmented_switch_falls_back_to_the_first_option(ui):
    assert ui.draw(lambda: segmented_switch("Type", ["Line", "Tube"], "Ribbon")) == (
        False,
        "Line",
    )


@requires_imgui
def test_segmented_switch_accepts_a_fixed_width(ui):
    assert ui.draw(
        lambda: segmented_switch("Type", ["Line", "Tube"], "Line", width=180)
    ) == (False, "Line")


@requires_imgui
def test_dropdown_is_unchanged_without_input(ui):
    assert ui.draw(lambda: dropdown("Colormap", ["Gray", "Jet"], "Gray")) == (
        False,
        "Gray",
    )


@requires_imgui
def test_dropdown_returns_early_without_options(ui):
    assert ui.draw(lambda: dropdown("Colormap", [], "Gray")) == (False, "Gray")


@requires_imgui
def test_dropdown_falls_back_to_the_first_option(ui):
    assert ui.draw(lambda: dropdown("Colormap", ["Gray", "Jet"], "Plasma")) == (
        False,
        "Gray",
    )


@requires_imgui
def test_dropdown_accepts_explicit_width_and_height(ui):
    assert ui.draw(
        lambda: dropdown("Colormap", ["Gray", "Jet"], "Jet", width=200, height=36)
    ) == (False, "Jet")


@requires_imgui
def test_dropdown_opens_its_list_on_click(ui):
    assert ui.click(
        lambda: dropdown("Colormap", ["Gray", "Jet", "Viridis"], "Gray")
    ) == (False, "Gray")


@requires_imgui
def test_thin_slider_is_unchanged_without_input(ui):
    assert ui.draw(lambda: thin_slider("Opacity", 50.0, 0.0, 100.0, width=200)) == (
        False,
        50.0,
    )


@requires_imgui
def test_thin_slider_snaps_to_the_minimum_at_the_track_start(ui):
    changed, value = ui.click(
        lambda: thin_slider("", 50.0, 0.0, 100.0, width=200),
        at=SLIDER_TRACK_LEFT,
    )

    assert changed is True
    assert value == 0.0


@requires_imgui
def test_thin_slider_value_grows_from_left_to_right(ui):
    def render():
        return thin_slider("", 50.0, 0.0, 100.0, width=200)

    left = ui.click(render, at=SLIDER_TRACK_LEFT)[1]
    middle = ui.click(render, at=SLIDER_TRACK_MID)[1]
    right = ui.click(render, at=SLIDER_TRACK_RIGHT)[1]

    assert left < middle < right
    assert 0.0 <= left and right <= 100.0


@requires_imgui
def test_thin_slider_respects_a_shifted_range(ui):
    changed, value = ui.click(
        lambda: thin_slider("", 0.0, -90.0, 90.0, width=200),
        at=SLIDER_TRACK_LEFT,
    )

    assert (changed, value) == (True, -90.0)


@requires_imgui
def test_thin_slider_returns_integers_for_an_int_slider(ui):
    changed, value = ui.click(
        lambda: thin_slider("", 5, 0, 10, width=200, value_type="int"),
        at=SLIDER_TRACK_LEFT,
    )

    assert changed is True
    assert isinstance(value, int)
    assert value == 0


@requires_imgui
def test_thin_slider_warns_when_a_float_is_given_to_an_int_slider(ui, caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        ui.draw(lambda: thin_slider("Slice", 5.5, 0, 10, width=200, value_type="int"))

    assert "converted to int for integer slider" in caplog.text


@requires_imgui
def test_thin_slider_rejects_an_unknown_value_type(ui):
    with pytest.raises(ValueError, match="value_type must be either"):
        ui.draw(lambda: thin_slider("X", 1.0, 0.0, 2.0, value_type="decimal"))


@requires_imgui
def test_thin_slider_shows_a_unit_and_a_visibility_toggle(ui):
    result = ui.draw(
        lambda: thin_slider(
            "Length",
            5.0,
            0.0,
            10.0,
            width=200,
            value_unit="mm",
            show_toggle=True,
            toggle=True,
        )
    )

    assert result == (False, 5.0, True)


@requires_imgui
def test_thin_slider_toggle_can_be_switched_off(ui):
    def render():
        return thin_slider(
            "Length", 5.0, 0.0, 10.0, width=200, show_toggle=True, toggle=True
        )

    ui.draw(render)

    changed, value, toggle = ui.click(render, at=(52, 60))

    assert (changed, value) == (False, 5.0)
    assert toggle is False


@requires_imgui
def test_thin_slider_uses_the_available_width_by_default(ui):
    assert ui.draw(lambda: thin_slider("Opacity", 25.0, 0.0, 100.0)) == (False, 25.0)


@requires_imgui
def test_two_disk_slider_is_unchanged_without_input(ui):
    assert ui.draw(
        lambda: two_disk_slider("Range", (20.0, 80.0), 0.0, 100.0, width=200)
    ) == (False, (20.0, 80.0))


@requires_imgui
def test_two_disk_slider_moves_the_nearest_thumb(ui):
    changed, (low, high) = ui.click(
        lambda: two_disk_slider("Range", (20.0, 80.0), 0.0, 100.0, width=200)
    )

    assert changed is True
    assert low > 20.0
    assert high == 80.0
    assert low < high


@requires_imgui
def test_two_disk_slider_keeps_the_thumbs_apart(ui):
    changed, (low, high) = ui.click(
        lambda: two_disk_slider(
            "Range", (20.0, 80.0), 0.0, 100.0, width=200, min_gap=40.0
        )
    )

    assert changed is True
    assert high - low >= 40.0


@requires_imgui
def test_two_disk_slider_returns_integers_for_an_int_range(ui):
    changed, (low, high) = ui.click(
        lambda: two_disk_slider("Size", (2, 8), 0, 10, width=200, value_type="int")
    )

    assert changed is True
    assert isinstance(low, int) and isinstance(high, int)


@requires_imgui
def test_two_disk_slider_warns_when_floats_are_given_to_an_int_range(ui, caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        ui.draw(
            lambda: two_disk_slider(
                "Size", (2.5, 8.5), 0, 10, width=200, value_type="int"
            )
        )

    assert "converted to int for integer slider" in caplog.text


@requires_imgui
def test_two_disk_slider_rejects_an_unknown_value_type(ui):
    with pytest.raises(ValueError, match="value_type must be either"):
        ui.draw(lambda: two_disk_slider("R", (0.0, 1.0), 0.0, 1.0, value_type="ratio"))


@requires_imgui
def test_two_disk_slider_rejects_a_malformed_range(ui):
    with pytest.raises(ValueError, match="values must be a 2-item iterable"):
        ui.draw(lambda: two_disk_slider("R", (0.0, 0.5, 1.0), 0.0, 1.0))


@requires_imgui
def test_two_disk_slider_shows_display_values_and_a_unit(ui):
    assert ui.draw(
        lambda: two_disk_slider(
            "Length",
            (25.0, 75.0),
            0.0,
            100.0,
            width=200,
            value_unit="mm",
            display_values=(12.5, 87.5),
        )
    ) == (False, (25.0, 75.0))


@requires_imgui
def test_two_disk_slider_uses_the_available_width_by_default(ui):
    assert ui.draw(lambda: two_disk_slider("Range", (10.0, 90.0), 0.0, 100.0)) == (
        False,
        (10.0, 90.0),
    )


@requires_imgui
def test_render_group_returns_each_row_result(ui):
    results = ui.draw(
        lambda: render_group(
            "Sliders",
            [
                (thin_slider, ("X", 1.0, 0.0, 10.0), {"width": 120}),
                (thin_slider, ("Y", 2.0, 0.0, 10.0), {"width": 120}),
            ],
        )
    )

    assert results == [(False, 1.0), (False, 2.0)]


@requires_imgui
def test_render_group_accepts_rows_without_arguments(ui):
    results = ui.draw(lambda: render_group("Toggles", [(lambda: toggle_button(True),)]))

    assert results == [(False, True)]


@requires_imgui
def test_render_group_accepts_rows_with_positional_arguments_only(ui):
    results = ui.draw(lambda: render_group("Toggles", [(toggle_button, (False,))]))

    assert results == [(False, False)]


@requires_imgui
def test_loading_overlay_can_be_shown_and_hidden(ui):
    from imgui_bundle import imgui

    def show():
        imgui.open_popup("LoadingOverlay")
        return loading("LoadingOverlay", "Loading Files...", True)

    assert ui.draw(show) is None
    assert ui.draw(lambda: loading("LoadingOverlay", "Loading Files...", True)) is None
    assert ui.draw(lambda: loading("LoadingOverlay", "Loading Files...", False)) is None


@requires_imgui
def test_confirmation_dialog_reports_the_popup_state(ui):
    from imgui_bundle import imgui

    def opener():
        imgui.open_popup("Confirm##switch")

    assert ui.draw(lambda: open_confirmation_dialog("Confirm##switch", "Sure?")) == (
        "open"
    )
    ui.draw(opener)
    assert ui.draw(lambda: open_confirmation_dialog("Confirm##switch", "Sure?")) == (
        "already_open"
    )


@requires_imgui
def test_confirmation_dialog_accepts_custom_button_labels(ui):
    from imgui_bundle import imgui

    def opener():
        imgui.open_popup("Confirm##labels")

    ui.draw(opener)
    state = ui.draw(
        lambda: open_confirmation_dialog(
            "Confirm##labels", "Switch to tubes?", okay_text="Switch", cancel_text="No"
        )
    )

    assert state == "already_open"
