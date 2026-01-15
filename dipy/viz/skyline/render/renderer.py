import sys

from fury import window

from dipy.utils.logging import logger


class Visualization:
    def __init__(self, render_callback):
        self._render_callback = render_callback

    def render(self):
        if self._render_callback is not None:
            self._render_callback()

    @property
    def actor(self):
        raise NotImplementedError("Subclasses must implement the actor property.")

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
            title="DIPY Skyline Visualizer",
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
