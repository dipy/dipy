from pathlib import Path

from fury.actor import (
    set_group_opacity,
    set_group_visibility,
    show_slices,
    volume_slicer,
)
from fury.lib import gfx
from imgui_bundle import imgui
import numpy as np

from dipy.viz.skyline.UI.elements import (
    dropdown,
    render_group,
    segmented_switch,
    thin_slider,
    toggle_button,
    two_disk_slider,
)
from dipy.viz.skyline.UI.theme import THEME
from dipy.viz.skyline.render.renderer import Visualization


def create_image_visualization(
    input,
    idx,
    *,
    interpolation="linear",
    render_callback=None,
    opacity=100,
    rgb=False,
    value_percentiles=(2, 98),
    colormap="Gray",
    sync_callabck=None,
):
    """Create image visualization from input

    Parameters
    ----------
    input : tuple
        Tuple of the (data, affine, filename) or (data, affine)
    idx : int
        Index of the image for naming purposes when filename is not provided.
    interpolation : str, optional
        Interpolation method for volume rendering. Options are "linear" or "nearest".
    render_callback : callable, optional
        Callback function to be called after rendering.
    opacity : int, optional
        Opacity of the volume rendering.
    rgb : bool, optional
        Whether the image is RGB
    value_percentiles : tuple, optional
        Percentiles for intensity value range. For example, (2, 98) will set the
        intensity range to be between the 2nd and 98th percentiles of the image
        intensities.
    colormap : str, optional
        The colormap to use for rendering. Options include "Gray", "Inferno", "Magma",
        "Plasma", and "Viridis". This parameter is ignored if rgb=True.
    sync_callabck : callable, optional
        Callback function to synchronize slice positions across visualizations.

    Returns
    -------
    Image3D
        The created Image3D object.

    Raises
    ------
    ValueError
        If the input is not a tuple of length 2 or 3, or if rgb=True and the last
        dimension of the volume is not 3 or 4.
    """
    if not isinstance(input, tuple) or len(input) not in (2, 3):
        raise ValueError(
            "Input must be a tuple containing (data, affine, filename) or "
            "(data, affine) for image visualization."
        )

    if len(input) == 2:
        data, affine = input
        filename = f"Image_{idx}"
    else:
        data, affine, filename = input
        filename = Path(filename).name

    return Image3D(
        filename,
        data,
        affine=affine,
        interpolation=interpolation,
        render_callback=render_callback,
        opacity=opacity,
        rgb=rgb,
        value_percentiles=value_percentiles,
        colormap=colormap,
        sync_callabck=sync_callabck,
    )


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
        sync_callabck=None,
    ):
        self.dwi = volume
        self.affine = affine

        if (
            rgb
            and self.dwi.ndim == 4
            and (self.dwi.shape[3] != 3 and self.dwi.shape[3] != 4)
        ):
            raise ValueError(
                "When specifying rgb=True, the last dimension of the volume "
                "must be 3 (RGB) or 4 (RGBA)."
            )
        self.rgb = rgb

        self._has_directions = self.dwi.ndim == 4 and not rgb

        self._volume_idx = 0
        self.interpolation = interpolation or "linear"
        self._value_percentiles = value_percentiles
        self._colormap_options = ("Gray", "Inferno", "Magma", "Plasma", "Viridis")
        self.colormap = colormap
        self._picked_voxel = None
        self._picked_intensity = None
        self._slice_visibility = [True, True, True]
        self._synchronize = True
        self._sync_callabck = sync_callabck
        super().__init__(name, render_callback)

        self._create_slicer_actor()
        self.opacity = opacity

    def _pick_voxel(self, event):
        info = event.pick_info
        voxel = info["index"]
        self._picked_voxel = voxel
        self._picked_intensity = self._active_volume()[voxel]

    def _active_volume(self):
        return self.dwi[..., self._volume_idx] if self._has_directions else self.dwi

    def _create_slicer_actor(self):
        if self._has_directions:
            volume = self.dwi[..., self._volume_idx]
            self.value_range = self._value_range_from_percentile(volume)
            self._slicer = volume_slicer(
                volume,
                affine=self.affine,
                interpolation=self.interpolation,
                value_range=self.value_range,
                alpha_mode="blend",
                depth_write=True,
            )
        else:
            self.value_range = self._value_range_from_percentile(self.dwi)
            self._slicer = volume_slicer(
                self.dwi,
                affine=self.affine,
                interpolation=self.interpolation,
                value_range=self.value_range,
                alpha_mode="blend",
                depth_write=True,
            )
        self._apply_colormap(self.colormap)
        self.bounds = self._slicer.get_bounding_box()
        self.state = np.mean(self.bounds, axis=0)
        self._slicer.add_event_handler(self._pick_voxel, "pointer_down")
        show_slices(self._slicer, self.state)
        self.render()

    def _value_range_from_percentile(self, volume):
        p_low, p_high = self._value_percentiles
        vmin, vmax = np.percentile(volume, (p_low, p_high))
        return vmin, vmax

    def _apply_colormap(self, colormap):
        self.colormap = colormap
        if self.colormap.lower() == "gray":
            for actor in self._slicer.children:
                actor.material.map = None
        else:
            for actor in self._slicer.children:
                actor.material.map = getattr(gfx.cm, self.colormap.lower())

    @property
    def actor(self):
        return self._slicer

    def _populate_info(self):
        info = f"Dimensions: {self.dwi.shape[:3]}"
        if self._has_directions:
            info += f"\nDirections: {self.dwi.shape[3]}"
        info += f"\nData Type: {self.dwi.dtype}"
        if self.affine is not None:
            voxel_sizes = np.sqrt(np.sum(self.affine[:3, :3] ** 2, axis=0))
            info += f"\nVoxel Sizes: {voxel_sizes}"
            voxel_order = "LAS" if self.affine[0, 0] < 0 else "RAS"
            info += f"\nVoxel Order: {voxel_order}"
            info += f"\nAffine:\n{self.affine}"
        return info

    def update_state(self, new_state):
        if self._synchronize:
            self.state = new_state
            show_slices(self._slicer, self.state)

    def render_widgets(self):
        changed, new = toggle_button(self._synchronize, label="Synchronize Slices")
        if changed:
            print(f"Synchronize set to {new}")
            self._synchronize = new

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
                        "value_type": "float",
                        "text_format": ".0f",
                        "step": 1,
                        "show_toggle": True,
                        "toggle": self._slice_visibility[axis],
                    },
                )
            )
        render_data = render_group("Slice", slicers)
        for idx, (changed, new, toggle) in enumerate(render_data):
            if changed:
                self.state[idx] = new
                self._synchronize and self._sync_callabck(self, self.state)
            self._slice_visibility[idx] = toggle
        set_group_visibility(self._slicer, self._slice_visibility)
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
            "Colormap", self._colormap_options, self.colormap, height=26
        )
        if colormap_changed:
            self._apply_colormap(new_cmap)

        imgui.spacing()

        voxel = str(self._picked_voxel) if self._picked_voxel is not None else ""
        intensity = (
            str(self._picked_intensity) if self._picked_intensity is not None else ""
        )
        value_color = THEME["primary"]
        label_color = THEME["text"]
        intesity_pos_x = imgui.get_content_region_avail().x * 0.5
        imgui.push_id("voxel_info")
        imgui.text_colored(label_color, "Voxel ")
        imgui.same_line()
        imgui.text_colored(value_color, voxel)
        imgui.same_line()
        imgui.set_cursor_pos_x(intesity_pos_x)
        imgui.text_colored(label_color, "Intensity ")
        imgui.same_line()
        imgui.text_colored(value_color, intensity)
        imgui.pop_id()

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
