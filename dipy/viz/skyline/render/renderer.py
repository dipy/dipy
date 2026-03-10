import sys

from fury import window
from imgui_bundle import imgui

from dipy.utils.logging import logger
from dipy.viz.skyline.UI.elements import render_section_header


class Visualization:
    def __init__(self, name, render_callback):
        self._render_callback = render_callback
        self.name = name
        self.active = False
        self._info = self._populate_info()

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
        type = None
        if self.__class__.__name__ == "Image3D":
            type = "image"
        elif self.__class__.__name__ == "Surface":
            type = "surface"
        elif self.__class__.__name__ == "Peak3D":
            type = "peak"
        elif self.__class__.__name__ == "ROI3D":
            type = "roi"
        elif (
            self.__class__.__name__ == "Streamline3D"
            or self.__class__.__name__ == "ClusterStreamline3D"
        ):
            type = "tractography"
        elif self.__class__.__name__ == "SHGlyph3D":
            type = "sh_glyph"
        else:
            logger.warning(
                f"Visualization type '{self.__class__.__name__}' is not recognized. "
                "UI rendering may not be fully functional for this visualization."
            )
        is_open, is_visible, is_removed, is_selected = render_section_header(
            name,
            is_open=is_open,
            is_visible=self.actor.visible,
            info=self._info,
            type=type,
        )
        self.actor.visible = is_visible
        self.active = is_selected
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

    def _populate_info(self):
        return self.name


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
            title="DIPY SKYLINE",
            size=size,
            window_type=window_type,
            screen_config=screen_config,
            imgui=True,
            imgui_draw_function=lambda: None,
            pixel_ratio=1.25,
        )
    else:
        show_m = window.ShowManager(
            title="DIPY SKYLINE",
            size=size,
            window_type=window_type,
            screen_config=screen_config,
            pixel_ratio=1.25,
            imgui=False,
        )
    if hasattr(show_m, "show_axes_gizmo"):
        show_m.show_axes_gizmo(labels=["L", "R", "P", "A", "S", "I"])
    logger.info(
        "Created visualizer currently assumes Neurological convention for axes."
    )
    return show_m
