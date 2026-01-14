import sys

from fury.actor import set_group_opacity, show_slices, volume_slicer
from imgui_bundle import imgui
import numpy as np

from dipy.utils.logging import logger
from dipy.viz.skyline.UI.elements import (
    render_group,
    segmented_switch,
    thin_slider,
    two_disk_slider,
)
from dipy.viz.skyline.render.renderer import Visualization


class Image3D(Visualization):
    def __init__(
        self,
        volume,
        *,
        affine=None,
        interpolation="linear",
        render_callback=None,
        opacity=100,
        rgb=False,
        value_percentiles=(2, 98),
    ):
        super().__init__(render_callback=render_callback)
        self.dwi = volume
        self.affine = affine

        if (
            rgb
            and self.dwi.ndim == 4
            and (self.dwi.shape[3] != 3 and self.dwi.shape[3] != 4)
        ):
            logger.error(
                "When specifying rgb=True, the last dimension of the volume "
                "must be 3 (RGB) or 4 (RGBA)."
            )
            sys.exit(1)
        self.rgb = rgb

        self._volume_idx = 0
        self.interpolation = interpolation or "linear"
        self._value_percentiles = value_percentiles

        self._create_slicer_actor()
        self.opacity = opacity

        self.slicer.add_event_handler(self._pick_voxel, "pointer_down")

    def _pick_voxel(self, event):
        info = event.pick_info
        intensity_interpolated = np.asarray(info["rgba"].rgb).mean()
        intensity_raw = self.dwi[info["index"]]
        print(
            f"Voxel {info['index']}:"
            f"\nInterpolated intensity: {intensity_interpolated:.2f}"
            f"\nRaw intensity: {intensity_raw}"
        )

    def _create_slicer_actor(self):
        if self.dwi.ndim == 4 and not self.rgb:
            volume = self.dwi[..., self._volume_idx]
            self.value_range = self._value_range_from_percentile(volume)
            self.slicer = volume_slicer(
                volume,
                affine=self.affine,
                interpolation=self.interpolation,
                value_range=self.value_range,
                alpha_mode="bayer",
                depth_write=True,
            )
        else:
            self.value_range = self._value_range_from_percentile(self.dwi)
            self.slicer = volume_slicer(
                self.dwi,
                affine=self.affine,
                interpolation=self.interpolation,
                value_range=self.value_range,
                alpha_mode="bayer",
                depth_write=True,
            )
        self.bounds = self.slicer.get_bounding_box()
        self.state = np.mean(self.bounds, axis=0).astype(int)
        show_slices(self.slicer, self.state)
        self.render()

    def _value_range_from_percentile(self, volume):
        p_low, p_high = self._value_percentiles
        vmin, vmax = np.percentile(volume, (p_low, p_high))
        return vmin, vmax

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
            step=1,
        )
        if changed:
            self.opacity = new
            set_group_opacity(self.slicer, self.opacity / 100.0)

        imgui.spacing()
        volume_for_range = (
            self.dwi[..., self._volume_idx]
            if self.dwi.ndim == 4 and not self.rgb
            else self.dwi
        )
        intensity_changed, new_percentiles = two_disk_slider(
            "Intensities",
            self._value_percentiles,
            0,
            100,
            text_format=".1f",
            step=1,
            min_gap=0.1,
            display_values=self.value_range,
        )
        if intensity_changed:
            self._value_percentiles = new_percentiles
            self.value_range = self._value_range_from_percentile(volume_for_range)
            for actor in self.slicer.children:
                actor.material.clim = self.value_range

        imgui.spacing()
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
                        "step": 1,
                    },
                )
            )
        render_data = render_group("Slice", slicers)
        for idx, (changed, new) in enumerate(render_data):
            if changed:
                self.state[idx] = new
            show_slices(self.slicer, self.state)

        imgui.spacing()
        if self.dwi.ndim == 4 and not self.rgb:
            volume_changed, new_idx = thin_slider(
                "Directions",
                self._volume_idx,
                0,
                self.dwi.shape[3] - 1,
                value_type="int",
                text_format=".0f",
                step=1,
            )
            if volume_changed:
                self._volume_idx = new_idx
                self._create_slicer_actor()

        imgui.spacing()
        changed, new = segmented_switch(
            "Interpolation", ["Linear", "Nearest"], self.interpolation
        )
        if changed:
            self.interpolation = new
            for actor in self.slicer.children:
                actor.material.interpolation = self.interpolation

        imgui.spacing()
