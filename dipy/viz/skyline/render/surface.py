from fury.actor import surface

from dipy.viz.skyline.UI.elements import thin_slider
from dipy.viz.skyline.render.renderer import Visualization


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
        render_callback=None,
    ):
        super().__init__(name, render_callback)
        self.vertices = vertices
        self.faces = faces
        self.affine = affine
        self.color = color
        self.opacity = opacity
        self.texture = texture
        self._create_surface_actor()

    def _create_surface_actor(self):
        self._surface_actor = surface(
            self.vertices,
            self.faces,
            colors=self.color,
            opacity=self.opacity / 100.0,
        )

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
