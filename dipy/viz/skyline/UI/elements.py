from imgui_bundle import imgui

from dipy.utils.logging import logger


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


def create_slider(
    label, value, *, min_value, max_value, value_type="int", format="%.3f"
):
    if value_type == "int" and isinstance(value, float):
        value = int(value)
        logger.warning(
            "Value converted to int for integer slider."
            " Please provide value_type as 'float' if float is intended."
        )

    if value_type == "int":
        changed, new_val = imgui.slider_int(
            label, value, int(min_value), int(max_value)
        )
    else:
        changed, new_val = imgui.slider_float(
            label, float(value), float(min_value), float(max_value), format=format
        )

    return changed, new_val
