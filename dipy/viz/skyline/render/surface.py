"""Mesh surfaces (FreeSurfer/GIFTI) with Phong or basic materials."""

import numpy as np

from dipy.utils.optpkg import optional_package
from dipy.viz.skyline.UI.elements import color_picker, colors_equal, thin_slider
from dipy.viz.skyline.render.renderer import Visualization

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
    from fury.actor import surface

imgui_bundle, has_imgui, _ = optional_package(
    "imgui_bundle", min_version="1.92.600", max_version="1.92.801"
)
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
    """Create a Surface visualization from already-loaded mesh data.

    Parameters
    ----------
    input : tuple
        Tuple of ``(vertices, faces, filename)`` or ``(vertices, faces)``
        holding already-loaded mesh vertices and faces, with an optional
        filename.
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

    Raises
    ------
    ValueError
        If the input is not a tuple of length 2 or 3.
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
    """A triangular mesh surface rendered with a Phong or basic material.

    Parameters
    ----------
    name : str
        Display name used in the Skyline UI.
    vertices : ndarray
        Vertex positions of the surface mesh, shape ``(N, 3)``.
    faces : ndarray
        Triangle face indices into ``vertices``, shape ``(M, 3)``.
    affine : ndarray, optional
        Voxel-to-world affine; accepted but not currently used by this class.
    color : tuple(float, float, float), optional
        RGB color applied to the surface mesh, in ``[0, 1]``.
    opacity : int, optional
        Surface opacity in percent, expected in ``[0, 100]``.
    texture : ndarray, optional
        Texture image; accepted but not currently applied to the rendered
        mesh.
    material : str, optional
        Material type for the mesh (``"phong"`` or ``"basic"``).
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
        """Initialize the mesh surface visualization.

        Parameters
        ----------
        name : str
            Display name used in the Skyline UI.
        vertices : ndarray
            Vertex positions of the surface mesh, shape ``(N, 3)``.
        faces : ndarray
            Triangle face indices into ``vertices``, shape ``(M, 3)``.
        affine : ndarray, optional
            Voxel-to-world affine; accepted but not currently used by this
            class.
        color : tuple(float, float, float), optional
            RGB color applied to the surface mesh, in ``[0, 1]``.
        opacity : int, optional
            Surface opacity in percent, expected in ``[0, 100]``.
        texture : ndarray, optional
            Texture image; accepted but not currently applied to the
            rendered mesh.
        material : str, optional
            Material type for the mesh (``"phong"`` or ``"basic"``).
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
        """Create the mesh actor for the surface geometry.

        Builds the actor with ``fury.actor.surface`` using the vertices,
        faces, color, material, and opacity state, sets the alpha blend
        mode, and disables depth writing when opacity is below 100%. The
        ``texture`` attribute is not passed to the underlying actor.
        """
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
        """Set the surface opacity and toggle depth writing.

        Sets the mesh material's opacity to ``opacity / 100`` and disables
        depth writing below 100% opacity. Unlike :class:`Image3D` and
        :class:`ROI3D`, the alpha blend mode stays ``"blend"`` regardless
        of opacity.

        Parameters
        ----------
        opacity : int
            Surface opacity in percent, expected in ``[0, 100]``.
        """
        self._surface_actor.material.opacity = opacity / 100.0
        self._surface_actor.material.depth_write = opacity >= 100

    def _populate_info(self):
        """Build the informational text describing the surface mesh.

        Returns
        -------
        str
            Text with the vertex and face counts.
        """
        info = f"No. of vertices: {len(self.vertices)}\nNo. of faces: {len(self.faces)}"
        return info

    @property
    def actor(self):
        """The mesh actor rendering the surface.

        Returns
        -------
        Mesh
            The actor of the surface visualization.
        """
        return self._surface_actor

    def render_widgets(self):
        """Draw the ImGui controls for surface opacity and color.

        Renders an opacity slider and a color picker; committing a new
        color rebuilds the mesh actor via :meth:`_create_surface_actor`.
        """
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


if not has_fury_v2:
    create_surface_visualization = Surface = fury
