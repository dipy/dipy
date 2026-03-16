import numpy as np

from dipy.utils.logging import logger
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
    from fury.actor import contour_from_roi, set_group_opacity
else:
    actor = fury.actor
imgui_bundle, has_imgui, _ = optional_package("imgui_bundle", min_version="1.92.600")
imgui = imgui_bundle.imgui


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
        self.roi = roi
        if self.roi is None:
            raise ValueError("ROI data cannot be None for ROI visualization.")
        elif not isinstance(self.roi, np.ndarray):
            raise ValueError("ROI data must be a numpy array for ROI visualization.")

        if self.roi.ndim == 4:
            logger.info(
                "Input has 4 dims, taking the first volume for ROI visualization."
            )
            self.roi = self.roi[..., 0]
        self.affine = affine
        self.opacity = opacity
        self.color = color
        self._create_roi_actor()
        super().__init__(name, render_callback)

    def _create_roi_actor(self):
        self._roi_surface = contour_from_roi(
            self.roi, affine=self.affine, color=self.color, opacity=self.opacity / 100.0
        )
        for actor in self._roi_surface.children:
            actor.material.alpha_mode = "blend"
            if self.opacity < 100:
                actor.material.depth_write = False

    def _populate_info(self):
        info = f"ROI shape: {self.roi.shape}\nROI dtype: {self.roi.dtype}\n"
        info += f"Total voxels in ROI: {np.sum(self.roi > 0)}\n"
        if self.affine is not None:
            affine_str = np.array2string(
                np.round(self.affine, 2), separator=" ", prefix=""
            )
            info += f"Affine:\n{affine_str}\n"
        return info

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
            self._create_roi_actor()
            self.render()
