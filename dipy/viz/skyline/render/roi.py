from fury.actor import contour_from_roi, set_group_opacity

from dipy.viz.skyline.UI.elements import thin_slider
from dipy.viz.skyline.render.renderer import Visualization


class ROI3D(Visualization):
    def __init__(
        self,
        name,
        roi,
        *,
        affine=None,
        opacity=100,
        color=(1, 0, 0),
        render_callback=None,
    ):
        super().__init__(name, render_callback)
        self.roi = roi
        self.affine = affine
        self.opacity = opacity
        self.color = color
        self._create_roi_actor()

    def _create_roi_actor(self):
        self._roi_surface = contour_from_roi(
            self.roi, affine=self.affine, color=self.color, opacity=self.opacity / 100.0
        )

    @property
    def actor(self):
        return self._roi_surface

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
            set_group_opacity(self._roi_surface, self.opacity / 100.0)
