from fury.actor import set_group_opacity, show_slices, volume_slicer
import numpy as np

from dipy.viz.skyline.UI.elements import render_group, thin_slider
from dipy.viz.skyline.render.renderer import Visualization


class Slicer(Visualization):
    def __init__(
        self,
        volume,
        *,
        affine=None,
        interpolation="linear",
        render_callback=None,
        opacity=100,
        value_percentiles=(2, 98),
    ):
        super().__init__(render_callback=render_callback)
        self.volume = volume
        self.affine = affine
        self.slicer = volume_slicer(
            volume,
            affine=affine,
            interpolation=interpolation,
        )
        self.bounds = self.slicer.get_bounding_box()
        self.state = np.mean(self.bounds, axis=0).astype(int)
        show_slices(self.slicer, self.state)
        self.opacity = opacity

        self.slicer.add_event_handler(self._pick_voxel, "pointer_down")

    def _pick_voxel(self, event):
        info = event.pick_info
        intensity_interpolated = np.asarray(info["rgba"].rgb).mean()
        intensity_raw = self.volume[info["index"]]
        print(
            f"Voxel {info['index']}:"
            f"\nInterpolated intensity: {intensity_interpolated:.2f}"
            f"\nRaw intensity: {intensity_raw}"
        )

    @property
    def actor(self):
        return self.slicer

    def render_widgets(self):
        changed, new = thin_slider(
            "Opacity",
            self.opacity,
            0,
            100,
            value_type="int",
            text_format=".0f",
            value_unit="%",
            width=250,
            step=1,
        )
        if changed:
            self.opacity = new
            set_group_opacity(self.slicer, self.opacity / 100.0)

        axis_labels = ("X", "Y", "Z")
        slider_bounds = (
            (int(self.bounds[0][0] + 1), int(self.bounds[1][0] - 1)),
            (int(self.bounds[0][1] + 1), int(self.bounds[1][1] - 1)),
            (int(self.bounds[0][2] + 1), int(self.bounds[1][2] - 1)),
        )
        slicers = []
        for axis, label in enumerate(axis_labels):
            min_bound, max_bound = slider_bounds[axis]
            slicers.append(
                (
                    thin_slider,
                    (label, self.state[axis], min_bound, max_bound),
                    {
                        "value_type": "int",
                        "text_format": ".0f",
                        "width": 250,
                        "step": 1,
                    },
                )
            )
        render_data = render_group("Slice", slicers)
        for idx, (changed, new) in enumerate(render_data):
            if changed:
                self.state[idx] = new
        show_slices(self.slicer, self.state)

        self.render()
