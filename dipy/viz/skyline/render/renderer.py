import sys

from fury import window
from imgui_bundle import imgui

from dipy.utils.logging import logger
from dipy.viz.skyline.UI.elements import render_section_header


class Visualization:
    def __init__(self, name, render_callback):
        self._render_callback = render_callback
        self.name = name

    def render(self):
        if self._render_callback is not None:
            self._render_callback()

    @property
    def actor(self):
        raise NotImplementedError("Subclasses must implement the actor property.")

    def renderer(self, name, is_open):
        """Provides a callback from UIManager to handle visualization.

        Parameters
        ----------
        name : str
            Name of the visualization section
        is_open : bool
            If the UI section is open
        """
        is_open, is_visible, is_removed = render_section_header(
            name, is_open=is_open, is_visible=self.actor.visible
        )
        self.actor.visible = is_visible

        if is_open and is_visible:
            padding = 20
            imgui.begin_group()
            imgui.push_style_var(imgui.StyleVar_.window_padding, (padding, padding / 2))
            child_flags = (
                imgui.ChildFlags_.always_use_window_padding
                | imgui.ChildFlags_.auto_resize_y
            )
            if imgui.begin_child(f"{name}_content_child", (0, 0), child_flags):
                self.render_widgets()
            imgui.end_child()
            imgui.pop_style_var()
            imgui.dummy((0, padding / 2))
            imgui.end_group()

        return is_open, is_removed

    def render_widgets(self):
        """Render control widgets for visualization.

        Raises
        ------
        NotImplementedError
            if the method is not implemented in the subclass.
        """
        raise NotImplementedError(
            "Subclasses must implement the render_widgets method."
        )


def create_window(
    *, visualizer_type="standalone", size=(1200, 1000), screen_config=None
):
    """Create a FURY ShowManager based on the visualizer type.

    This function enables the creation of different types of windows for visualizations.
    This will be used to create multi-window structure in the skyline visualizer,
    where multiple instances can co-exist.

    Parameters
    ----------
    visualizer_type : str, optional
        Type of visualizer to create. The options are:
        - "standalone": A standalone window with full interactivity.
        - "gui": A Qt-based GUI window.
        - "jupyter": An inline Jupyter notebook visualizer.
        - "stealth": An offscreen visualizer without GUI.
    size : tuple, optional
        Size of the window
    screen_config : list, optional
        Defines the screen layout. Can be a list of integers (vertical/horizontal
        sections) or a list of explicit bounding box tuples (x, y, w, h).

    Returns
    -------
    ShowManager
        An instance of FURY's ShowManager configured according to the
        specified visualizer type.
    """
    if visualizer_type == "standalone":
        window_type = "default"
    elif visualizer_type == "gui":
        window_type = "qt"
    elif visualizer_type == "jupyter":
        window_type = "jupyter"
    elif visualizer_type == "stealth":
        window_type = "offscreen"
    else:
        logger.error(
            f"Visualizer type '{visualizer_type}' is not recognized. "
            "Please provide one of the following: "
            "'standalone', 'gui', 'jupyter', 'stealth'."
        )
        sys.exit(1)

    if visualizer_type != "stealth":
        show_m = window.ShowManager(
            title="DIPY Skyline",
            size=size,
            window_type=window_type,
            screen_config=screen_config,
            imgui=True,
            imgui_draw_function=lambda: None,
            pixel_ratio=2,
        )
    else:
        show_m = window.ShowManager(
            title="DIPY Skyline Visualizer",
            size=size,
            window_type=window_type,
            screen_config=screen_config,
            imgui=False,
        )

    return show_m
