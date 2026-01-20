import math

from imgui_bundle import icons_fontawesome_6, imgui

from dipy.utils.logging import logger
from dipy.viz.skyline.UI.theme import (
    DROPDOWN_THEME,
    SLIDER_THEME,
    SWITCH_THEME,
    THEME,
    WINDOW_THEME,
)


def render_section_header(
    label,
    *,
    is_open=True,
    is_visible=True,
    type=None,
    width=0,
    height=40,
    padding_x=12,
):
    """Draw a custom section header with a toggle arrow.

    Parameters
    ----------
    label : str
        Text to render in the header.
    is_open : bool, optional
        Current open state for the section. The returned value reflects toggling.
    is_visible : bool, optional
        Current visibility state for the section.
    type : str, optional
        Type of section. Used to determine the icon shown.
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
    bg = imgui.get_color_u32(WINDOW_THEME["background_color"])
    text_color = imgui.get_color_u32(WINDOW_THEME["title_color"])
    accent_color = imgui.get_color_u32(WINDOW_THEME["title_active_color"])
    collapse_color = imgui.get_color_u32(WINDOW_THEME["collapse_color"])

    start = imgui.get_cursor_screen_pos()
    end = (start.x + total_width - 10, start.y + height)
    draw_list = imgui.get_window_draw_list()

    draw_list.add_rect_filled(start, end, bg)

    hovered = imgui.is_mouse_hovering_rect(start, end)
    active = hovered and imgui.is_mouse_down(imgui.MouseButton_.left)

    color = accent_color if (hovered or active or is_open) else text_color

    brain_icon = icons_fontawesome_6.ICON_FA_BRAIN
    brain_size = imgui.calc_text_size(brain_icon)
    brain_pos = (start.x, start.y + (height - brain_size.y) * 0.5)
    draw_list.add_text(brain_pos, color, brain_icon)

    label_size = imgui.calc_text_size(label)
    icon_size = imgui.ImVec2(0, 0)
    arrow_icon = (
        icons_fontawesome_6.ICON_FA_ANGLE_UP
        if is_open
        else icons_fontawesome_6.ICON_FA_ANGLE_DOWN
    )
    info_icon = icons_fontawesome_6.ICON_FA_CIRCLE_INFO
    close_icon = icons_fontawesome_6.ICON_FA_XMARK
    show_icon = (
        icons_fontawesome_6.ICON_FA_EYE
        if is_visible
        else icons_fontawesome_6.ICON_FA_EYE_SLASH
    )
    info_icon_size = imgui.calc_text_size(info_icon)
    close_icon_size = imgui.calc_text_size(close_icon)
    show_icon_size = imgui.calc_text_size(show_icon)
    arrow_icon_size = imgui.calc_text_size(arrow_icon)
    icon_size = imgui.ImVec2(
        info_icon_size.x
        + close_icon_size.x
        + show_icon_size.x
        + arrow_icon_size.x
        + (padding_x * 4),
        arrow_icon_size.y,
    )
    table_icon_size = imgui.ImVec2(0, 0)
    if type == "image":
        table_icon = icons_fontawesome_6.ICON_FA_TABLE
        table_icon_size = imgui.calc_text_size(table_icon)
        icon_size = imgui.ImVec2(
            icon_size.x + table_icon_size.x + padding_x,
            icon_size.y,
        )

    available_label_width = total_width - brain_size.x - icon_size.x - padding_x * 2
    ellipsis_text = "..."
    show_ellipsis = label_size.x > available_label_width > 0
    display_text = label
    label_pos = (
        start.x + brain_size.x + padding_x,
        start.y + (height - label_size.y) * 0.5,
    )

    if show_ellipsis:
        ellipsis_size = imgui.calc_text_size(ellipsis_text)
        max_label_width = max(0, available_label_width - ellipsis_size.x)
        trimmed_label = label
        while trimmed_label and imgui.calc_text_size(trimmed_label).x > max_label_width:
            trimmed_label = trimmed_label[:-1]
        display_text = trimmed_label.rstrip()
        display_text += ellipsis_text

    draw_list.add_text(label_pos, color, display_text)

    pos_x = end[0] - icon_size.x
    pos_y = start.y + (height - icon_size.y) * 0.6

    draw_list.add_text((pos_x, pos_y), text_color, show_icon)
    pos_x += show_icon_size.x + padding_x

    draw_list.add_text((pos_x, pos_y), text_color, close_icon)
    pos_x += close_icon_size.x + padding_x

    if type == "image":
        draw_list.add_text((pos_x, pos_y), text_color, table_icon)
        pos_x += table_icon_size.x + padding_x

    draw_list.add_text((pos_x, pos_y), text_color, info_icon)
    pos_x += info_icon_size.x + padding_x

    draw_list.add_text((pos_x, pos_y), collapse_color, arrow_icon)
    draw_list.add_rect_filled(
        (label_pos[0], height + start.y - 2),
        end,
        text_color,
        0,
        0,
    )

    imgui.set_cursor_screen_pos(start)
    imgui.invisible_button("section_header_button", (total_width, height))

    if imgui.is_item_clicked():
        is_open = not is_open

    imgui.pop_id()
    return is_open, is_visible


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


def segmented_switch(label, options, value, *, width=0, height=28):
    """Render a segmented switch control.

    Parameters
    ----------
    label : str
        Text rendered next to the switch.
    options : list of str
        Labels for each segment in the switch.
    value : str
        Currently selected option. If not found in ``options``, the first option
        is used.
    width : int, optional
        Total width for the switch. If 0 or negative, uses the available width.
    height : int, optional
        Height for each segment in pixels.

    Returns
    -------
    tuple(bool, str)
        Whether the selection changed and the resulting option value.
    """
    if not options:
        return False, value

    imgui.push_id(label)

    value_options = [option.lower() for option in options]
    current_value = value if value in value_options else options[0]
    label_color = THEME["text"]
    imgui.push_style_var(imgui.StyleVar_.frame_padding, (12.0, 6.0))
    imgui.align_text_to_frame_padding()
    imgui.text_colored(label_color, label)
    imgui.same_line(0, 28)

    available_width = imgui.get_content_region_avail().x
    total_width = width if width and width > 0 else available_width
    count = len(options)
    button_width = total_width / count if total_width > 0 else 80.0

    selected_bg = imgui.get_color_u32(SWITCH_THEME["active_color"])
    selected_text = SWITCH_THEME["active_text_color"]
    inactive_text = SWITCH_THEME["inactive_text_color"]
    container_bg = SWITCH_THEME["background_color"]
    border_color = imgui.get_color_u32(SWITCH_THEME["border_color"])
    container_rounding = 6.0

    imgui.push_style_var(imgui.StyleVar_.item_spacing, (0.0, 0.0))
    imgui.push_style_var(imgui.StyleVar_.frame_border_size, 0.0)
    imgui.push_style_var(imgui.StyleVar_.frame_rounding, container_rounding)

    imgui.push_style_color(imgui.Col_.button, (0, 0, 0, 0))
    imgui.push_style_color(imgui.Col_.button_hovered, (0, 0, 0, 0))
    imgui.push_style_color(imgui.Col_.button_active, (0, 0, 0, 0))

    changed = False
    new_value = current_value

    button_height = max(height, imgui.get_frame_height())
    start = imgui.get_cursor_screen_pos()
    end = (start.x + button_width * count, start.y + button_height)
    draw_list = imgui.get_window_draw_list()
    draw_list.add_rect_filled(
        start, end, imgui.get_color_u32(container_bg), container_rounding
    )

    for idx, option in enumerate(value_options):
        if idx > 0:
            imgui.same_line(0, 0)

        segment_start = (start.x + idx * button_width, start.y)
        segment_end = (segment_start[0] + button_width, start.y + button_height)

        is_selected = option == current_value
        if is_selected:
            if count == 1:
                corner_flags = imgui.ImDrawFlags_.round_corners_all
            elif idx == 0:
                corner_flags = imgui.ImDrawFlags_.round_corners_left
            elif idx == count - 1:
                corner_flags = imgui.ImDrawFlags_.round_corners_right
            else:
                corner_flags = imgui.ImDrawFlags_.round_corners_none
            draw_list.add_rect_filled(
                segment_start,
                segment_end,
                selected_bg,
                container_rounding,
                corner_flags,
            )

        imgui.push_style_color(
            imgui.Col_.text, selected_text if is_selected else inactive_text
        )

        if imgui.button(options[idx], (button_width, button_height)):
            if option != current_value:
                changed = True
                new_value = option

        imgui.pop_style_color(1)

    draw_list.add_rect(start, end, border_color, container_rounding, 0, 1.5)

    imgui.pop_style_color(3)
    imgui.pop_style_var(4)
    imgui.pop_id()

    return changed, new_value


def dropdown(label, options, value, *, width=0):
    """Render a themed dropdown/combobox control.

    Parameters
    ----------
    label : str
        Text rendered next to the dropdown.
    options : list[str]
        Available options displayed in the dropdown.
    value : str
        Currently selected option. If not present in ``options``, the first
        option is used.
    width : int, optional
        Width of the dropdown in pixels. If 0 or negative, uses available space.

    Returns
    -------
    tuple(bool, str)
        Whether the selection changed and the resulting option value.
    """
    if not options:
        return False, value

    imgui.push_id(label)
    label_color = THEME["text"]
    imgui.align_text_to_frame_padding()
    imgui.text_colored(label_color, label)
    imgui.same_line(0, 16)

    current_value = value if value in options else options[0]
    available_width = imgui.get_content_region_avail().x
    combo_width = width if width and width > 0 else max(140.0, available_width)
    padding_x = 10.0
    arrow_icon = icons_fontawesome_6.ICON_FA_ANGLE_DOWN

    frame_bg = DROPDOWN_THEME["background_color"]
    border_color = DROPDOWN_THEME["border_color"]
    text_color = DROPDOWN_THEME["selected_color"]
    highlight = DROPDOWN_THEME["hover_color"]
    arrow_color = DROPDOWN_THEME["arrow_color"]

    imgui.set_next_item_width(combo_width)
    imgui.push_style_var(imgui.StyleVar_.frame_rounding, 6.0)
    imgui.push_style_var(imgui.StyleVar_.frame_border_size, 1.0)
    imgui.push_style_var(imgui.StyleVar_.item_spacing, (0.0, 10.0))
    imgui.push_style_color(imgui.Col_.frame_bg, frame_bg)
    imgui.push_style_color(imgui.Col_.frame_bg_hovered, frame_bg)
    imgui.push_style_color(imgui.Col_.frame_bg_active, frame_bg)
    imgui.push_style_color(imgui.Col_.border, border_color)
    imgui.push_style_color(imgui.Col_.text, text_color)
    imgui.push_style_color(imgui.Col_.header, highlight)
    imgui.push_style_color(imgui.Col_.header_hovered, highlight)
    imgui.push_style_color(imgui.Col_.header_active, highlight)

    combo_flags = imgui.ComboFlags_.height_regular | imgui.ComboFlags_.no_arrow_button
    changed = False
    new_value = current_value
    opened = imgui.begin_combo(f"##{label}_dropdown", current_value, combo_flags)

    frame_min = imgui.get_item_rect_min()
    frame_max = imgui.get_item_rect_max()
    draw_list = imgui.get_window_draw_list()
    arrow_size = imgui.calc_text_size(arrow_icon)
    arrow_pos = (
        frame_max.x - padding_x - arrow_size.x,
        frame_min.y + (frame_max.y - frame_min.y - arrow_size.y) * 0.52,
    )
    draw_list.add_text(arrow_pos, imgui.get_color_u32(arrow_color), arrow_icon)

    if opened:
        for option in options:
            is_selected = option == current_value
            selectable_result = imgui.selectable(option, is_selected)
            pressed = (
                selectable_result[0]
                if isinstance(selectable_result, tuple)
                else selectable_result
            )
            if pressed:
                new_value = option
            if is_selected:
                imgui.set_item_default_focus()
        changed = new_value != current_value
        imgui.end_combo()
        current_value = new_value

    imgui.pop_style_color(8)
    imgui.pop_style_var(3)

    imgui.pop_id()
    return changed, new_value


def thin_slider(
    label,
    value,
    min_value,
    max_value,
    *,
    width=0,
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
        (imgui.get_content_region_avail().x, total_h)
        if width <= 0
        else (width, total_h)
    )
    imgui.invisible_button(
        f"#thin_slider_btn_{label}", (available_size[0] - 50, available_size[1]), 0
    )
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
        display_text = f"{typed_value:{text_format}}{value_unit}"
    else:
        display_text = f"{typed_value:{text_format}}"

    max_text_size = 50
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


def two_disk_slider(
    label,
    values,
    min_value,
    max_value,
    *,
    width=0,
    step=1.0,
    track_height=2.0,
    thumb_radius=7.0,
    hitbox_height=20.0,
    text_format=".3f",
    value_type="float",
    value_unit=None,
    min_gap=0.0,
    display_values=None,
):
    """Render a range slider with two circular thumbs on a thin track.

    Parameters
    ----------
    label : str
        Text rendered next to the slider.
    values : tuple[float, float]
        Current lower and upper values for the range.
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
        Radius of each circular thumb in pixels.
    hitbox_height : float, optional
        Height of the invisible button capturing pointer interactions.
    text_format : str, optional
        Format specification passed when displaying float values.
    value_type : {"float", "int"}, optional
        Numeric type enforced for the slider values.
    value_unit : str or None, optional
        Optional unit suffix appended to the value display.
    min_gap : float, optional
        Minimum allowed gap between the two thumbs.
    display_values : tuple[float, float] or None, optional
        Optional values shown in the text readout instead of the slider values.
        Useful for displaying absolute values while the thumbs operate on
        percentiles.

    Returns
    -------
    tuple(bool, tuple[float or int, float or int])
        Whether the slider values changed and the resulting numeric range.
    """
    if value_type not in {"float", "int"}:
        raise ValueError("value_type must be either 'float' or 'int'")
    if len(values) != 2:
        raise ValueError("values must be a 2-item iterable (low, high)")

    low_val, high_val = values
    if value_type == "int" and any(isinstance(v, float) for v in values):
        logger.warning(
            "Values converted to int for integer slider."
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
        (imgui.get_content_region_avail().x, total_h)
        if width <= 0
        else (width, total_h)
    )
    slider_width = max(40, available_size[0])
    imgui.invisible_button(
        f"#two_disk_slider_btn_{label}", (slider_width, available_size[1]), 0
    )
    bb_min = imgui.get_item_rect_min()
    bb_max = imgui.get_item_rect_max()
    draw_list = imgui.get_window_draw_list()

    min_numeric = float(min_value)
    max_numeric = float(max_value)
    low_val = max(min_numeric, min(max_numeric, float(low_val)))
    high_val = max(min_numeric, min(max_numeric, float(high_val)))
    if low_val > high_val:
        low_val, high_val = high_val, low_val

    original_low = low_val
    original_high = high_val

    hovered = imgui.is_item_hovered()
    active = imgui.is_item_active()

    def convert_value(val):
        if value_type == "int":
            rounded = int(round(val))
            lower = math.ceil(min_numeric)
            upper = math.floor(max_numeric)
            return max(int(lower), min(int(upper), rounded))
        return float(val)

    step_amount = float(step)
    if value_type == "int":
        step_amount = max(1.0, round(step_amount))

    state = imgui.get_state_storage()
    active_key = imgui.get_id(f"{label}_two_disk_active")
    active_thumb = state.get_int(active_key, -1)

    value_display_width = 50
    text_padding = 6.0
    track_left = bb_min.x + value_display_width + text_padding + thumb_radius
    track_right = bb_max.x - value_display_width - text_padding - thumb_radius
    if track_right <= track_left:
        track_right = track_left + 1.0

    def ratio_from_value(val):
        if max_numeric == min_numeric:
            return 0.0
        return min(max((val - min_numeric) / (max_numeric - min_numeric), 0.0), 1.0)

    left_ratio = ratio_from_value(low_val)
    right_ratio = ratio_from_value(high_val)
    left_x = track_left + (track_right - track_left) * left_ratio
    right_x = track_left + (track_right - track_left) * right_ratio

    if not imgui.is_mouse_down(imgui.MouseButton_.left):
        active_thumb = -1
        state.set_int(active_key, -1)

    if imgui.is_item_clicked(imgui.MouseButton_.left):
        mouse_x, _mouse_y = imgui.get_mouse_pos()
        dist_left = abs(mouse_x - left_x)
        dist_right = abs(mouse_x - right_x)
        active_thumb = 0 if dist_left <= dist_right else 1
        state.set_int(active_key, active_thumb)

    if active and active_thumb != -1:
        mouse_x, _mouse_y = imgui.get_mouse_pos()
        new_ratio = (mouse_x - track_left) / max(1.0, (track_right - track_left))
        new_ratio = min(max(new_ratio, 0.0), 1.0)
        new_val = min_numeric + new_ratio * (max_numeric - min_numeric)
        if active_thumb == 0:
            low_val = min(new_val, high_val - min_gap)
        else:
            high_val = max(new_val, low_val + min_gap)

    focused = imgui.is_item_focused()
    if focused and imgui.is_key_pressed(imgui.Key.left_arrow):
        low_val = max(min_numeric, low_val - step_amount)
    if focused and imgui.is_key_pressed(imgui.Key.right_arrow):
        low_val = min(high_val - min_gap, low_val + step_amount)
    if focused and imgui.is_key_pressed(imgui.Key.down_arrow):
        high_val = max(low_val + min_gap, high_val - step_amount)
    if focused and imgui.is_key_pressed(imgui.Key.up_arrow):
        high_val = min(max_numeric, high_val + step_amount)

    typed_low = convert_value(low_val)
    typed_high = convert_value(max(high_val, typed_low + min_gap))
    if value_type == "int":
        low_val = float(typed_low)
        high_val = float(typed_high)

    left_ratio = ratio_from_value(low_val)
    right_ratio = ratio_from_value(high_val)
    left_x = track_left + (track_right - track_left) * left_ratio
    right_x = track_left + (track_right - track_left) * right_ratio

    y_center = (bb_min.y + bb_max.y) / 2.0
    track_color = imgui.get_color_u32(SLIDER_THEME["track_color"])
    track_covered_color = imgui.get_color_u32(SLIDER_THEME["track_covered_color"])
    draw_list.add_rect_filled(
        (track_left, y_center - track_height / 2.0),
        (track_right, y_center + track_height / 2.0),
        track_color,
        min(track_height / 2.0, 4.0),
    )
    draw_list.add_rect_filled(
        (left_x, y_center - track_height / 2.0),
        (right_x, y_center + track_height / 2.0),
        track_covered_color,
        min(track_height / 2.0, 4.0),
    )

    thumb_color = imgui.get_color_u32(SLIDER_THEME["thumb_color"])
    shadow_color = imgui.get_color_u32(SLIDER_THEME["shadow_color"])
    mouse_x, mouse_y = imgui.get_mouse_pos()
    left_hovered = hovered and (abs(mouse_x - left_x) <= thumb_radius * 1.4)
    right_hovered = hovered and (abs(mouse_x - right_x) <= thumb_radius * 1.4)

    def draw_thumb(x_pos, is_hovered, is_active):
        radius = thumb_radius
        if is_hovered:
            radius = thumb_radius * 1.08
        if is_active:
            radius = thumb_radius * 1.18
        draw_list.add_circle_filled((x_pos, y_center), radius + 4.0, shadow_color)
        draw_list.add_circle_filled((x_pos, y_center), radius, thumb_color)

    draw_thumb(left_x, left_hovered, active_thumb == 0 and active)
    draw_thumb(right_x, right_hovered, active_thumb == 1 and active)

    value_color = imgui.get_color_u32(SLIDER_THEME["value_color"])
    display_low, display_high = (
        display_values if display_values is not None else (typed_low, typed_high)
    )
    if value_unit is not None and display_values is None:
        left_text = f"{display_low:{text_format}}{value_unit}"
        right_text = f"{display_high:{text_format}}{value_unit}"
    else:
        left_text = f"{display_low:{text_format}}"
        right_text = f"{display_high:{text_format}}"

    left_size = imgui.calc_text_size(left_text)
    right_size = imgui.calc_text_size(right_text)
    text_y = y_center - left_size.y * 0.5
    draw_list.add_text(imgui.ImVec2(bb_min.x, text_y), value_color, left_text)
    draw_list.add_text(
        imgui.ImVec2(bb_max.x - right_size.x, y_center - right_size.y * 0.5),
        value_color,
        right_text,
    )

    if width > 0:
        imgui.pop_item_width()
    imgui.pop_id()

    value_changed = typed_low != convert_value(
        original_low
    ) or typed_high != convert_value(original_high)
    return value_changed, (typed_low, typed_high)
