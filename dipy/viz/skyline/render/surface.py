import numpy as np

from dipy.utils.optpkg import optional_package
from dipy.viz.skyline.UI.elements import color_picker, thin_slider
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
else:
    actor = fury.actor
imgui_bundle, has_imgui, _ = optional_package("imgui_bundle", min_version="1.92.600")
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
        super().__init__(name, render_callback)
        self.vertices = vertices
        self.faces = faces
        self.affine = affine
        self.color = color
        self.opacity = opacity
        self.texture = texture
        self.material = material
        self._create_surface_actor()

    def _create_surface_actor(self):
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

    @property
    def actor(self):
        return self._surface_actor

    def render_widgets(self):
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
            self._surface_actor.material.opacity = self.opacity / 100.0
            if self.opacity < 100:
                self._surface_actor.material.depth_write = False
            else:
                self._surface_actor.material.depth_write = True

        imgui.spacing()
        color = np.asarray(self.color) * 255
        color = color.astype(np.uint8)
        changed, new_color = color_picker(
            selected_color=self.color,
            tooltip="Pick surface color",
            label=color,
        )
        if changed:
            self.color = (new_color[0], new_color[1], new_color[2])
            self._create_surface_actor()
            self.render()
