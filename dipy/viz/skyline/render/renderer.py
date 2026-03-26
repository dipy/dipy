"""Shared FURY window factory and base class for Skyline visualizations."""

from pathlib import Path
import sys

from PIL import Image

from dipy.utils.logging import logger
from dipy.utils.optpkg import optional_package
from dipy.viz.skyline.UI.elements import render_section_header
from dipy.viz.skyline.UI.theme import LOGO_SMALL

fury_trip_msg = (
    "Skyline requires Fury version 2.0.0a6 or higher."
    " Please upgrade Fury by `pip install -U fury --pre` to use Skyline."
)
fury, has_fury_v2, _ = optional_package(
    "fury",
    min_version="2.0.0a6",
    trip_msg=fury_trip_msg,
)
if has_fury_v2:
    from fury import window
    import glfw
else:
    window = fury.window

imgui_bundle, has_imgui, _ = optional_package("imgui_bundle", min_version="1.92.600")
if has_imgui:
    imgui = imgui_bundle.imgui


class Visualization:
    """Bridge between a data layer, a FURY actor, and Skyline sidebar widgets.

    Subclasses implement :attr:`actor`, :meth:`render_widgets`, and optionally
    :meth:`_populate_info`.

    Parameters
    ----------
    path : str or None
        Source path or label used for sidebar naming.
    render_callback : callable or None
        Invoked to request a full window redraw after widget edits.
    """

    def __init__(self, path, render_callback):
        self._render_callback = render_callback
        self._scene_op_callback = None
        self.path = path if path is not None else "Unnamed Visualization"
        if self.__class__.__name__ == "ROI3D":
            self.name = f"ROI ({Path(self.path).name})"
        elif self.__class__.__name__ == "SHGlyph3D":
            self.name = f"ODFs ({Path(self.path).name})"
        else:
            self.name = Path(self.path).name
        self.active = False
        self._visible = True
        self._info = self._populate_info()

    def render(self):
        """Request a window redraw through :attr:`_render_callback` when set."""
        if self._render_callback is not None:
            self._render_callback()

    def apply_scene_op(self, func, *args, **kwargs):
        """Run ``func`` immediately or defer it via :attr:`_scene_op_callback`."""
        if self._scene_op_callback is not None:
            self._scene_op_callback(func, *args, **kwargs)
            return
        func(*args, **kwargs)

    def _set_actor_visible(self, visible):
        self.actor.visible = visible

    @property
    def actor(self):
        raise NotImplementedError("Subclasses must implement the actor property.")

    @property
    def viz_type(self):
        """Return the visualization type identifier string."""
        name = self.__class__.__name__
        if name == "Image3D":
            return "image"
        elif name == "Surface":
            return "surface"
        elif name == "Peak3D":
            return "peak"
        elif name == "ROI3D":
            return "roi"
        elif name in ("Streamline3D", "ClusterStreamline3D"):
            return "tractography"
        elif name == "SHGlyph3D":
            return "sh_glyph"
        return None

    def renderer(self, is_open, group_visible=True):
        """Draw the sidebar header and optional widget body for this layer.

        Parameters
        ----------
        is_open : bool
            Whether the collapsible section should start expanded this frame.
        group_visible : bool, optional
            Whether the parent group is visible. When False, the actor is
            hidden regardless of the individual visibility toggle.

        Returns
        -------
        is_open : bool
            Updated expanded state after handling input.
        is_removed : bool
            True if the user requested removal.
        should_enable_group : bool
            True if a hidden group must be re-enabled because visibility was toggled.
        """
        viz_type = self.viz_type
        if viz_type is None:
            logger.warning(
                f"Visualization type '{self.__class__.__name__}' is not recognized. "
                "UI rendering may not be fully functional for this visualization."
            )
        effective_visible = self._visible and group_visible
        is_open, new_visible, is_removed, is_selected = render_section_header(
            self.name,
            is_open=is_open,
            is_visible=effective_visible,
            info=self._info,
            type=viz_type,
        )

        should_enable_group = False
        if group_visible:
            self._visible = new_visible
        elif new_visible:
            # Group is hidden but user clicked visibility ON
            # Enable the group and show this item
            self._visible = True
            should_enable_group = True
        # If group hidden and user kept it off (new_visible=False), do nothing

        self.apply_scene_op(self._set_actor_visible, self._visible and group_visible)
        self.active = is_selected
        if is_open:
            padding = 20
            imgui.begin_group()
            imgui.push_style_var(imgui.StyleVar_.window_padding, (padding, padding / 2))
            child_flags = (
                imgui.ChildFlags_.always_use_window_padding
                | imgui.ChildFlags_.auto_resize_y
            )
            if imgui.begin_child(f"{self.name}_content_child", (0, 0), child_flags):
                self.render_widgets()
            imgui.end_child()
            imgui.pop_style_var()
            imgui.dummy((0, padding / 2))
            imgui.end_group()

        return is_open, is_removed, should_enable_group

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
    *,
    visualizer_type="standalone",
    size=(1200, 1000),
    screen_config=None,
    title="DIPY SKYLINE",
):
    """Create a FURY ShowManager based on the visualizer type.

    Used to host the main scene, optional ImGui overlay, and multi-viewport
    layouts in Skyline.

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
    title : str, optional
        Window title; in stealth mode may be combined with ``out_dir`` upstream.

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
            title=title,
            size=size,
            window_type=window_type,
            screen_config=screen_config,
            imgui=True,
            imgui_draw_function=lambda: None,
            pixel_ratio=1.5,
        )
        if window_type == "default":
            with Image.open(LOGO_SMALL) as img:
                img = img.convert("RGBA")
                glfw.set_window_icon(show_m.window._window, 1, [(img)])
                glfw.poll_events()
                img.close()

    else:
        show_m = window.ShowManager(
            title=title,
            size=size,
            window_type=window_type,
            screen_config=screen_config,
            pixel_ratio=1.5,
            imgui=False,
        )
    if hasattr(show_m, "show_axes_gizmo"):
        show_m.show_axes_gizmo(labels=["L", "R", "P", "A", "S", "I"])
    logger.info(
        "Created visualizer currently assumes Neurological convention for axes."
    )
    return show_m
