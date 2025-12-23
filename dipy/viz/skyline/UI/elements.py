import math

from imgui_bundle import icons_fontawesome_6, imgui

from dipy.utils.logging import logger
from dipy.viz.skyline.UI.theme import SLIDER_THEME, THEME


def render_section_header(
    label,
    *,
    is_open=True,
    icon=None,
    width=0,
    height=28,
    padding_x=12,
):
    """Draw a custom section header with a toggle arrow.

    Parameters
    ----------
    label : str
        Text to render in the header.
    is_open : bool, optional
        Current open state for the section. The returned value reflects toggling.
    icon : str, optional
        Optional icon text to prefix the label.
    width : int, optional
        Header width in pixels. If 0 or negative, uses available width.
    height : int, optional
        Header height in pixels.
    padding_x : int, optional
        Horizontal padding between the icon and label.

    Returns
    -------
    bool
        Updated open state after handling input.
    """
    imgui.push_id(label)

    total_width = width if width and width > 0 else imgui.get_content_region_avail().x
    bg = imgui.get_color_u32(THEME["background"])
    text_color = imgui.get_color_u32(THEME["text"])
    accent_color = imgui.get_color_u32(THEME["text_highlight"])

    start = imgui.get_cursor_screen_pos()
    end = (start.x + total_width, start.y + height)
    draw_list = imgui.get_window_draw_list()

    draw_list.add_rect_filled(start, end, bg)

    hovered = imgui.is_mouse_hovering_rect(start, end)
    active = hovered and imgui.is_mouse_down(imgui.MouseButton_.left)

    color = accent_color if (hovered or active or is_open) else text_color

    brain_icon = icons_fontawesome_6.ICON_FA_BRAIN
    brain_size = imgui.calc_text_size(brain_icon)
    brain_pos = (start.x, start.y + (height - brain_size.y) * 0.5)
    draw_list.add_text(brain_pos, color, brain_icon)

    label_text = f"{icon}  {label}" if icon else label
    label_size = imgui.calc_text_size(label_text)
    label_pos = (
        start.x + brain_size.x + padding_x,
        start.y + (height - label_size.y) * 0.5,
    )
    draw_list.add_text(label_pos, color, label_text)

    arrow_icon = (
        icons_fontawesome_6.ICON_FA_ANGLE_UP
        if is_open
        else icons_fontawesome_6.ICON_FA_ANGLE_DOWN
    )
    arrow_size = imgui.calc_text_size(arrow_icon)
    arrow_pos = (
        start.x + brain_size.x + label_size.x + padding_x * 3,
        start.y + (height - arrow_size.y) * 0.6,
    )
    draw_list.add_text(arrow_pos, color, arrow_icon)

    imgui.set_cursor_screen_pos(start)
    imgui.invisible_button("section_header_button", (total_width, height))
    if imgui.is_item_clicked():
        is_open = not is_open

    imgui.pop_id()
    return is_open


def render_group(label, items, *, row_height=26, label_width=36, line_indent=8):
    """Render a grouped list with a tree-like label column and custom rows.

    Parameters
    ----------
    label : str
        Group heading shown above the table.
    items : list of tuple
        Sequence of items where each entry is
        ``(render_fn)`` or ``(render_fn, args, kwargs)``.
        ``render_fn`` is called with the provided args/kwargs in the content column.
    row_height : int, optional
        Height of each row in pixels.
    label_width : int, optional
        Width of the label column in pixels.
    line_indent : int, optional
        Horizontal indent for the guide line from the row start.

    Returns
    -------
    list of tuple
        A list of ``(changed, new_value)`` returned from each ``render_fn`` call.
        Returns ``None`` if no items are supplied.
    """
    if not items:
        return

    label_color = THEME["text"]
    line_color = imgui.get_color_u32(label_color)

    imgui.text_colored(label_color, label)
    imgui.spacing()

    flags = imgui.TableFlags_.sizing_fixed_fit | imgui.TableFlags_.pad_outer_x
    if imgui.begin_table(f"group_{label}", 2, flags):
        imgui.table_setup_column(
            "labels", imgui.TableColumnFlags_.width_fixed, label_width
        )
        imgui.table_setup_column("content", imgui.TableColumnFlags_.width_stretch)

        draw_list = imgui.get_window_draw_list()
        text_height = imgui.get_text_line_height()

        render_data = []
        total_items = len(items)
        for idx, item in enumerate(items):
            render_fn, *rest = item
            args, kwargs = (), {}
            if rest:
                args = rest[0] if len(rest) >= 1 else ()
                kwargs = rest[1] if len(rest) >= 2 else {}

            imgui.table_next_row(imgui.TableRowFlags_.none, row_height)

            imgui.table_set_column_index(0)
            row_pos = imgui.get_cursor_screen_pos()
            line_x = row_pos.x + line_indent
            text_x = line_x + 16
            center_y = row_pos.y + row_height * 0.4
            is_last = idx == total_items - 1
            vertical_end_y = center_y if is_last else row_pos.y + row_height

            draw_list.add_line(
                imgui.ImVec2(line_x, row_pos.y),
                imgui.ImVec2(line_x, vertical_end_y),
                line_color,
                1.0,
            )
            draw_list.add_line(
                imgui.ImVec2(line_x, center_y),
                imgui.ImVec2(text_x, center_y),
                line_color,
                1.0,
            )

            text_y = center_y - text_height * 0.5
            imgui.set_cursor_screen_pos((text_x, text_y))
            imgui.dummy((1, text_height))

            imgui.table_set_column_index(1)
            changed, new = render_fn(*args, **kwargs)
            render_data.append((changed, new))

        imgui.end_table()
        return render_data


def create_numeric_input(label, value, *, value_type="int", step=1, format="%.3f"):
    if value_type == "int" and isinstance(value, float):
        value = int(value)
        logger.warning(
            "Value converted to int for integer input."
            " Please provide value_type as 'float' if float is intended."
        )

    if value_type == "int":
        changed, new_val = imgui.input_int(
            label,
            value,
            step=int(step),
            step_fast=int(step * 10),
        )
    else:
        changed, new_val = imgui.input_float(
            label, float(value), step=step, step_fast=step * 10, format=format
        )

    return changed, new_val


def thin_slider(
    label,
    value,
    min_value,
    max_value,
    *,
    width=300,
    step=1.0,
    track_height=2.0,
    thumb_radius=7.0,
    hitbox_height=20.0,
    text_format=".3f",
    value_type="float",
    value_unit=None,
):
    """Render a compact slider with a thin track and circular thumb.

    Parameters
    ----------
    label : str
        Text rendered next to the slider.
    value : float
        Current slider value.
    min_value : float
        Lower bound for the slider.
    max_value : float
        Upper bound for the slider.
    width : int, optional
        Widget width in pixels. Negative values use the available width.
    step : float, optional
        Increment applied when using keyboard arrows.
    track_height : float, optional
        Thickness of the slider track in pixels.
    thumb_radius : float, optional
        Radius of the circular thumb in pixels.
    hitbox_height : float, optional
        Height of the invisible button capturing pointer interactions.
    text_format : str, optional
        Format specification passed when displaying float values.
    value_type : {"float", "int"}, optional
        Numeric type enforced for the slider value.
    value_unit : str or None, optional
        Optional unit suffix appended to the value display.

    Returns
    -------
    tuple(bool, float or int)
        Whether the slider value changed and the resulting numeric value.
    """
    if value_type not in {"float", "int"}:
        raise ValueError("value_type must be either 'float' or 'int'")
    if value_type == "int" and isinstance(value, float):
        logger.warning(
            "Value converted to int for integer slider."
            " Please provide value_type as 'float' if float is intended."
        )

    imgui.push_id(label)
    if width > 0:
        imgui.push_item_width(width)

    label_color = SLIDER_THEME["label_color"]
    imgui.text_colored(label_color, label)

    imgui.same_line(0, 16)

    total_h = max(hitbox_height, thumb_radius * 2 + 4)
    available_size = (
        (imgui.get_content_region_avail().x, total_h) if width < 0 else (width, total_h)
    )
    imgui.invisible_button(f"#thin_slider_btn_{label}", available_size, 0)
    bb_min = imgui.get_item_rect_min()
    bb_max = imgui.get_item_rect_max()
    draw_list = imgui.get_window_draw_list()

    x0 = bb_min.x + thumb_radius
    x1 = bb_max.x - thumb_radius
    y_center = (bb_min.y + bb_max.y) / 2.0

    cur_val = float(value)
    min_numeric = float(min_value)
    max_numeric = float(max_value)
    cur_val = max(min_numeric, min(max_numeric, cur_val))
    original_val = cur_val

    hovered = imgui.is_item_hovered()
    active = imgui.is_item_active()
    focused = imgui.is_item_focused()

    track_y = y_center
    if active:
        mx, _my = imgui.get_mouse_pos()
        new_ratio = (mx - x0) / max(1.0, (x1 - x0))
        new_ratio = min(max(new_ratio, 0.0), 1.0)
        cur_val = min_numeric + new_ratio * (max_numeric - min_numeric)

    step_amount = float(step)
    if value_type == "int":
        step_amount = max(1.0, round(step_amount))
    if focused and imgui.is_key_pressed(imgui.Key.left_arrow):
        cur_val = max(min_numeric, cur_val - step_amount)
    if focused and imgui.is_key_pressed(imgui.Key.right_arrow):
        cur_val = min(max_numeric, cur_val + step_amount)

    def convert_value(val):
        if value_type == "int":
            rounded = int(round(val))
            lower = math.ceil(min_numeric)
            upper = math.floor(max_numeric)
            return max(int(lower), min(int(upper), rounded))
        return float(val)

    typed_original = convert_value(original_val)
    typed_value = convert_value(cur_val)
    if value_type == "int":
        cur_val = float(typed_value)

    ratio = (
        (cur_val - min_numeric) / (max_numeric - min_numeric)
        if max_numeric != min_numeric
        else 0.0
    )
    ratio = min(max(ratio, 0.0), 1.0)
    thumb_x = x0 + (x1 - x0) * ratio

    radius = thumb_radius
    if hovered:
        radius = thumb_radius * 1.08
    if active:
        radius = thumb_radius * 1.18

    track_color = imgui.get_color_u32(SLIDER_THEME["track_color"])
    track_covered_color = imgui.get_color_u32(SLIDER_THEME["track_covered_color"])

    draw_list.add_rect_filled(
        (x0, track_y - track_height / 2.0),
        (x1, track_y + track_height / 2.0),
        track_color,
        min(track_height / 2.0, 4.0),
    )
    draw_list.add_rect_filled(
        (x0, track_y - track_height / 2.0),
        (thumb_x, track_y + track_height / 2.0),
        track_covered_color,
        min(track_height / 2.0, 4.0),
    )

    thumb_color = imgui.get_color_u32(SLIDER_THEME["thumb_color"])
    shadow_color = imgui.get_color_u32(SLIDER_THEME["shadow_color"])
    draw_list.add_circle_filled((thumb_x, track_y), radius + 4.0, shadow_color)
    draw_list.add_circle_filled((thumb_x, track_y), radius, thumb_color)

    value_color = SLIDER_THEME["value_color"]

    if value_unit is not None:
        display_text = f"{typed_value:{text_format}} {value_unit}"
    else:
        display_text = f"{typed_value:{text_format}}"

    max_text_size = 48
    text_size = imgui.calc_text_size(display_text)
    imgui.same_line(
        0, max_text_size - text_size.x if text_size.x < max_text_size else 8
    )
    imgui.text_colored(value_color, display_text)

    if width > 0:
        imgui.pop_item_width()
    imgui.pop_id()

    value_changed = typed_value != typed_original
    return value_changed, typed_value
