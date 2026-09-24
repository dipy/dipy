"""Shared FURY window factory and base class for Skyline visualizations."""

from pathlib import Path
import sys

import numpy as np

from dipy.core.gradients import get_orientation_from_affine
from dipy.utils.logging import logger
from dipy.utils.optpkg import optional_package
from dipy.viz.skyline.UI.elements import render_section_header
from dipy.viz.skyline.UI.theme import LOGO_SMALL

fury_trip_msg = (
    "Skyline requires Fury version 2.0.0 or higher."
    " Please upgrade Fury by `pip install -U fury --pre` to use Skyline."
)
fury, has_fury_v2, _ = optional_package(
    "fury",
    min_version="2.0.0",
    trip_msg=fury_trip_msg,
)

if has_fury_v2:
    from PIL import Image
    from fury import window
    import glfw
else:
    window = fury
    Image = fury
    glfw = fury

imgui_bundle, has_imgui, _ = optional_package(
    "imgui_bundle", min_version="1.92.600", max_version="1.92.801"
)
if has_imgui:
    imgui = imgui_bundle.imgui


def affine_voxel_sizes(affine):
    """Return voxel sizes from an affine matrix.

    Parameters
    ----------
    affine : ndarray
        Voxel-to-world affine.

    Returns
    -------
    ndarray
        Per-axis voxel sizes from the affine columns.
    """
    return np.linalg.norm(np.asarray(affine)[:3, :3], axis=0)


def format_affine_info(affine):
    """Build the shared "voxel sizes + order + affine" block for info panels.

    Parameters
    ----------
    affine : array_like
        Voxel-to-world affine.

    Returns
    -------
    str
        "Voxel Sizes: <sizes>\\nVoxel Order: <code>\\nAffine:\\n<matrix>".
    """
    affine = np.asarray(affine)
    voxel_sizes = affine_voxel_sizes(affine)
    voxel_order = get_orientation_from_affine(affine)
    affine_str = np.array2string(np.round(affine, 2), separator=" ", prefix="")
    return (
        f"Voxel Sizes: {np.round(voxel_sizes, 1)}\n"
        f"Voxel Order: {voxel_order}\n"
        f"Affine:\n{affine_str}"
    )


def slice_slider_bounds(shape, *, affine=None):
    """Return affine-aware integer bounds for slice sliders.

    Parameters
    ----------
    shape : tuple(int, int, int)
        Original spatial data shape.
    affine : ndarray, optional
        Voxel-to-world affine used to position slices in world coordinates.

    Returns
    -------
    tuple(tuple(int, int), tuple(int, int), tuple(int, int))
        Per-axis inclusive slider bounds.
    """
    spatial_shape = np.asarray(shape[:3], dtype=float)
    if affine is None:
        max_bounds = spatial_shape
    else:
        voxel_sizes = affine_voxel_sizes(affine)
        max_bounds = np.where(
            voxel_sizes >= 1.0, spatial_shape * voxel_sizes, spatial_shape
        )

    max_bounds = np.maximum(np.ceil(max_bounds - 1e-12).astype(int), 0)
    return tuple((0, int(max_bound)) for max_bound in max_bounds)


def slice_state_from_slider_values(slider_values, *, affine=None):
    """Convert slice slider values to slicing state coordinates.

    Parameters
    ----------
    slider_values : array-like
        Per-axis values displayed by the slice sliders.
    affine : ndarray, optional
        Voxel-to-world affine used by the visualization.

    Returns
    -------
    ndarray
        Slicing state in world coordinates when affine is provided, otherwise
        voxel coordinates.
    """
    slider_values = np.asarray(slider_values[:3], dtype=float)
    if affine is None:
        return slider_values

    voxel_sizes = affine_voxel_sizes(affine)
    scaled_axes = voxel_sizes >= 1.0
    voxel_values = slider_values.copy()
    np.divide(
        slider_values,
        voxel_sizes,
        out=voxel_values,
        where=scaled_axes,
    )
    return (np.asarray(affine) @ np.r_[voxel_values, 1.0])[:3]


def slice_slider_values_from_state(state, *, affine=None):
    """Convert slicing state coordinates to slice slider values.

    Parameters
    ----------
    state : array-like
        Current slicing state in world coordinates when affine is provided,
        otherwise voxel coordinates.
    affine : ndarray, optional
        Voxel-to-world affine used by the visualization.

    Returns
    -------
    ndarray
        Per-axis values to display in slice sliders.
    """
    state = np.asarray(state[:3], dtype=float)
    if affine is None:
        return state

    voxel_values = voxel_values_from_slice_state(state, affine=affine)
    voxel_sizes = affine_voxel_sizes(affine)
    return np.where(voxel_sizes >= 1.0, voxel_values * voxel_sizes, voxel_values)


def voxel_values_from_slice_state(state, *, affine=None):
    """Convert slicing state coordinates to voxel coordinates.

    Parameters
    ----------
    state : array-like
        Current slicing state in world coordinates when affine is provided,
        otherwise voxel coordinates.
    affine : ndarray, optional
        Voxel-to-world affine used by the visualization.

    Returns
    -------
    ndarray
        Per-axis voxel coordinates.
    """
    state = np.asarray(state[:3], dtype=float)
    if affine is None:
        return state

    return (np.linalg.inv(np.asarray(affine)) @ np.r_[state, 1.0])[:3]


class Visualization:
    """Base class for a single visualization layer in the Skyline sidebar.

    Parameters
    ----------
    path : str or Path
        Path to the resource on disk.
    render_callback : callable, optional
        Callback used to request a render/update.
    """

    def __init__(self, path, render_callback):
        """Initialize the visualization layer.

        Parameters
        ----------
        path : str or Path
            Path to the resource on disk.
        render_callback : callable, optional
            Callback used to request a render/update.
        """
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
        """Run ``func`` immediately or defer it via :attr:`_scene_op_callback`.

        Parameters
        ----------
        func : callable
            Scene-mutating callable to run, either directly or through the
            deferral callback.
        *args
            Positional arguments forwarded to ``func``.
        **kwargs
            Keyword arguments forwarded to ``func``.
        """
        if self._scene_op_callback is not None:
            self._scene_op_callback(func, *args, **kwargs)
            return
        func(*args, **kwargs)

    def _set_actor_visible(self, visible):
        """Show or hide the actor by setting its ``visible`` attribute.

        Parameters
        ----------
        visible : bool
            Whether the actor should be visible.
        """
        self.actor.visible = visible

    @property
    def actor(self):
        """The FURY actor rendered for this visualization.

        Returns
        -------
        object
            The visualization's underlying FURY actor object.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the subclass.
        """
        raise NotImplementedError("Subclasses must implement the actor property.")

    @property
    def viz_type(self):
        """The visualization type identifier derived from the subclass name.

        Returns
        -------
        str or None
            One of ``"image"``, ``"surface"``, ``"peak"``, ``"roi"``,
            ``"tractography"``, or ``"sh_glyph"`` depending on the concrete
            subclass, or None if the subclass name is not recognized.
        """
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

    def renderer(self, is_open, *, group_visible=True):
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
            If the method is not implemented in the subclass.
        """
        raise NotImplementedError(
            "Subclasses must implement the render_widgets method."
        )

    def _populate_info(self):
        """Build the info string shown in the sidebar for this visualization.

        Returns
        -------
        str
            The visualization's display name; subclasses override this to
            include additional details such as affine and voxel information.
        """
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
    visualizer_type : {"standalone", "gui", "jupyter", "stealth"}, optional
        Type of visualizer to create:

        - "standalone": a standalone window with full interactivity.
        - "gui": a Qt-based GUI window.
        - "jupyter": an inline Jupyter notebook visualizer.
        - "stealth": an offscreen visualizer without GUI.
    size : tuple of int, optional
        Window size in pixels as ``(width, height)``.
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

    Notes
    -----
    If ``visualizer_type`` is not one of the recognized values, the error is
    logged and the process exits via ``sys.exit(1)`` instead of raising a
    Python exception.
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
