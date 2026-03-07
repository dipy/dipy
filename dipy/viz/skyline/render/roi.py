from pathlib import Path

from fury.actor import contour_from_roi, set_group_opacity

from dipy.viz.skyline.UI.elements import thin_slider
from dipy.viz.skyline.render.renderer import Visualization


def create_roi_visualization(
    input,
    idx,
    *,
    opacity=100,
    color=(1, 0, 0),
    render_callback=None,
):
    """Create ROI visualization from input

    Parameters
    ----------
    input : tuple
        Tuple of the (roi, affine, filename) or (roi, affine)
    idx : int
        Index of the ROI for naming purposes if filename is not provided.
    opacity : int, optional
        Opacity of the ROI rendering.
    color : tuple, optional
        Color of the ROI rendering.
    render_callback : callable, optional
        Callback function to be called after rendering.

    Returns
    -------
    ROI3D
        The created ROI3D object.
    """
    if not isinstance(input, tuple) or len(input) not in (2, 3):
        raise ValueError(
            "Input must be a tuple containing (roi, affine, filename) or "
            "(roi, affine) for ROI visualization."
        )

    if len(input) == 2:
        roi, affine = input
        filename = f"ROI_{idx}"
    else:
        roi, affine, filename = input
        filename = Path(filename).name if filename is not None else f"ROI_{idx}"

    return ROI3D(
        filename,
        roi,
        affine=affine,
        color=color,
        opacity=opacity,
        render_callback=render_callback,
    )


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
        for actor in self._roi_surface.children:
            actor.material.alpha_mode = "blend"
            if self.opacity < 100:
                actor.material.depth_write = False

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
            for actor in self._roi_surface.children:
                if self.opacity < 100:
                    actor.material.depth_write = False
                else:
                    actor.material.depth_write = True
