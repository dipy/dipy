import sys

from fury.actor import set_group_opacity, show_slices, volume_slicer
from fury.lib import gfx
from imgui_bundle import imgui
import numpy as np

from dipy.utils.logging import logger
from dipy.viz.skyline.UI.elements import (
    dropdown,
    render_group,
    segmented_switch,
    thin_slider,
    two_disk_slider,
)
from dipy.viz.skyline.render.renderer import Visualization


class Image3D(Visualization):
    def __init__(
        self,
        name,
        volume,
        *,
        affine=None,
        interpolation="linear",
        render_callback=None,
        opacity=100,
        rgb=False,
        value_percentiles=(2, 98),
        colormap="Gray",
    ):
        super().__init__(name, render_callback)
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

        self._has_directions = self.dwi.ndim == 4 and not rgb

        self._volume_idx = 0
        self.interpolation = interpolation or "linear"
        self._value_percentiles = value_percentiles
        self._colormap_options = ("Gray", "Inferno", "Magma", "Plasma", "Viridis")
        self.colormap = colormap

        self._create_slicer_actor()
        self.opacity = opacity

        self._slicer.add_event_handler(self._pick_voxel, "pointer_down")

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
        if self._has_directions:
            volume = self.dwi[..., self._volume_idx]
            self.value_range = self._value_range_from_percentile(volume)
            self._slicer = volume_slicer(
                volume,
                affine=self.affine,
                interpolation=self.interpolation,
                value_range=self.value_range,
                alpha_mode="solid",
                depth_write=True,
            )
        else:
            self.value_range = self._value_range_from_percentile(self.dwi)
            self._slicer = volume_slicer(
                self.dwi,
                affine=self.affine,
                interpolation=self.interpolation,
                value_range=self.value_range,
                alpha_mode="solid",
                depth_write=True,
            )
        self._apply_colormap(self.colormap)
        self.bounds = self._slicer.get_bounding_box()
        self.state = np.mean(self.bounds, axis=0).astype(int)
        show_slices(self._slicer, self.state)
        self.render()

    def _value_range_from_percentile(self, volume):
        p_low, p_high = self._value_percentiles
        vmin, vmax = np.percentile(volume, (p_low, p_high))
        return vmin, vmax

    def _apply_colormap(self, colormap):
        self.colormap = colormap
        for actor in self._slicer.children:
            actor.material.map = getattr(gfx.cm, self.colormap.lower())

    @property
    def actor(self):
        return self._slicer

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
            if self.opacity == 100:
                for actor in self.actor.children:
                    actor.material.alpha_mode = "solid"
            else:
                for actor in self.actor.children:
                    actor.material.alpha_mode = "bayer"
            set_group_opacity(self._slicer, self.opacity / 100.0)

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
            show_slices(self._slicer, self.state)

        imgui.spacing()
        volume_for_range = (
            self.dwi[..., self._volume_idx] if self._has_directions else self.dwi
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
            for actor in self._slicer.children:
                actor.material.clim = self.value_range

        if self._has_directions:
            imgui.spacing()
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
        colormap_changed, new_cmap = dropdown(
            "Colormap", self._colormap_options, self.colormap
        )
        if colormap_changed:
            self._apply_colormap(new_cmap)

        imgui.spacing()
        imgui.spacing()
        changed, new = segmented_switch(
            "Interpolation", ["Linear", "Nearest"], self.interpolation
        )
        if changed:
            self.interpolation = new
            for actor in self._slicer.children:
                actor.material.interpolation = self.interpolation

        imgui.spacing()
