"""Mesh surfaces (FreeSurfer/GIFTI) with Phong or basic materials."""

import numpy as np

from dipy.utils.optpkg import optional_package
from dipy.viz.skyline.UI.elements import color_picker, colors_equal, thin_slider
from dipy.viz.skyline.render.renderer import Visualization

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
    from fury.actor import surface
imgui_bundle, has_imgui, _ = optional_package("imgui_bundle", min_version="1.92.600")
if has_imgui:
    imgui = imgui_bundle.imgui


def create_surface_visualization(
    input,
    idx,
    *,
    color=(1, 0, 0),
    opacity=100,
    texture=None,
    material="phong",
    render_callback=None,
):
    """Create surface visualization from input

    Parameters
    ----------
    input : tuple
        Tuple of the (vertices, faces, filename) or (vertices, faces)
    idx : int
        Index of the surface for naming purposes if filename is not provided.
    color : tuple, optional
        Color of the surface rendering.
    opacity : int, optional
        Opacity of the surface rendering.
    texture : ndarray, optional
        Texture to use for surface.
    material : str, optional
        Material type for surface.
    render_callback : callable, optional
        Callback function to be called after rendering.

    Returns
    -------
    Surface
        The created Surface object.
    """
    if not isinstance(input, tuple) or len(input) not in (2, 3):
        raise ValueError(
            "Input must be a tuple containing (vertices, faces, filename) or "
            "(vertices, faces) for surface visualization."
        )

    if len(input) == 2:
        vertices, faces = input
        filename = f"Surface_{idx}"
    else:
        vertices, faces, filename = input

    return Surface(
        filename,
        vertices,
        faces,
        color=color,
        opacity=opacity,
        texture=texture,
        material=material,
        render_callback=render_callback,
    )


class Surface(Visualization):
    """Represent ``Surface`` in Skyline.

    Parameters
    ----------
    name : str
        Display name used in the Skyline UI.
    vertices : ndarray
        Value for ``vertices``.
    faces : ndarray
        Value for ``faces``.
    affine : ndarray, optional
        Voxel-to-world affine used to position slices in world coordinates.
    color : tuple(float, float, float), optional
        Value for ``color``.
    opacity : int, optional
        Slice opacity in percent, expected in ``[0, 100]``.
    texture : ndarray, optional
        Value for ``texture``.
    material : str, optional
        Value for ``material``.
    render_callback : callable, optional
        Callback used to request a render/update.
    """

    def __init__(
        self,
        name,
        vertices,
        faces,
        *,
        affine=None,
        color=(1, 0, 0),
        opacity=100,
        texture=None,
        material="phong",
        render_callback=None,
    ):
        """Represent ``Surface`` in Skyline.

        Parameters
        ----------
        name : str
            Display name used in the Skyline UI.
        vertices : ndarray
            Value for ``vertices``.
        faces : ndarray
            Value for ``faces``.
        affine : ndarray, optional
            Voxel-to-world affine used to position slices in world coordinates.
        color : tuple(float, float, float), optional
            Value for ``color``.
        opacity : int, optional
            Slice opacity in percent, expected in ``[0, 100]``.
        texture : ndarray, optional
            Value for ``texture``.
        material : str, optional
            Value for ``material``.
        render_callback : callable, optional
            Callback used to request a render/update.
        """
        self.vertices = vertices
        self.faces = faces
        self.affine = affine
        self.color = color
        self._draft_color = color
        self._color_picker_open = False
        self._color_picker_popup_id = f"surface_color_picker_popup##{name}"
        self.opacity = opacity
        self.texture = texture
        self.material = material
        self._create_surface_actor()
        super().__init__(name, render_callback)

    def _create_surface_actor(self):
        """Handle  create surface actor for ``Surface``."""
        self._surface_actor = surface(
            self.vertices,
            self.faces,
            material=self.material,
            colors=self.color,
            opacity=self.opacity / 100.0,
        )
        self._surface_actor.material.alpha_mode = "blend"
        if self.opacity < 100:
            self._surface_actor.material.depth_write = False

    def _set_opacity(self, opacity):
        """Handle  set opacity for ``Surface``.

        Parameters
        ----------
        opacity : int, optional
            Slice opacity in percent, expected in ``[0, 100]``.
        """
        self._surface_actor.material.opacity = opacity / 100.0
        self._surface_actor.material.depth_write = opacity >= 100

    def _populate_info(self):
        """Handle  populate info for ``Surface``.

        Returns
        -------
        object
            Returned value.
        """
        info = f"No. of vertices: {len(self.vertices)}\nNo. of faces: {len(self.faces)}"
        return info

    @property
    def actor(self):
        """Handle actor for ``Surface``.

        Returns
        -------
        object
            Returned value.
        """
        return self._surface_actor

    def render_widgets(self):
        """Handle render widgets for ``Surface``."""
        changed, new = thin_slider(
            "Opacity",
            self.opacity,
            0,
            100,
            value_type="int",
            text_format=".0f",
            value_unit="%",
            step=1,
        )
        if changed:
            self.opacity = new
            self.apply_scene_op(self._set_opacity, self.opacity)

        imgui.spacing()
        color = np.asarray(self.color) * 255
        color = color.astype(np.uint8)
        selected_color = self._draft_color if self._color_picker_open else self.color
        changed, new_color, is_open = color_picker(
            selected_color=selected_color,
            tooltip="Pick Surface color",
            label=color,
            popup_id=self._color_picker_popup_id,
        )
        if is_open and not self._color_picker_open:
            self._draft_color = self.color
        if changed:
            self._draft_color = new_color
        if self._color_picker_open and not is_open:
            if not colors_equal(self._draft_color, self.color):
                self.color = self._draft_color
                self.apply_scene_op(self._create_surface_actor)
                self.render()
            self._draft_color = self.color
        self._color_picker_open = is_open
