import numpy as np

from dipy.utils.optpkg import optional_package
from dipy.viz.skyline.UI.elements import (
    create_numeric_input,
    render_group,
    thin_slider,
    toggle_button,
)
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
    from fury.actor import peaks_slicer
    from fury.transform import apply_transformation
else:
    actor = fury.actor

imgui_bundle, has_imgui, _ = optional_package("imgui_bundle", min_version="1.92.600")
if has_imgui:
    imgui = imgui_bundle.imgui


def create_peak_visualization(
    input,
    idx,
    *,
    opacity=100,
    render_callback=None,
    sync_callabck=None,
):
    """Create peak visualization from input

    Parameters
    ----------
    input : tuple
        Tuple of the (pam, filename) or (pam,)
    idx : int
        Index of the peak for naming purposes if filename is not provided.
    opacity : int, optional
        Opacity of the peak rendering.
    render_callback : callable, optional
        Callback function to be called after rendering.
    sync_callabck : callable, optional
        Callback function to synchronize slice positions across visualizations.

    Returns
    -------
    Peak3D
        The created Peak3D object.
    """
    if not isinstance(input, tuple) or len(input) not in (1, 2):
        raise ValueError(
            "Input must be a tuple containing (pam, filename) or (pam,) "
            "for peak visualization."
        )

    if len(input) == 1:
        pam = input[0]
        filename = f"Peaks_{idx}"
    else:
        pam, filename = input

    peak_values = 1.0
    if pam.peak_values is not None:
        max = np.percentile(pam.peak_values, 99)
        peak_values = np.clip(pam.peak_values, 0, max)

    return Peak3D(
        filename,
        pam.peak_dirs,
        affine=pam.affine,
        peak_values=peak_values,
        opacity=opacity,
        render_callback=render_callback,
        sync_callabck=sync_callabck,
    )


class Peak3D(Visualization):
    def __init__(
        self,
        name,
        peaks,
        *,
        affine=None,
        peak_values=1.0,
        opacity=100,
        render_callback=None,
        sync_callabck=None,
    ):
        self.peaks = peaks
        self.affine = affine
        self.peak_values = peak_values
        self._scale = 1.0
        self.opacity = opacity
        self._synchronize = True
        self._sync_callabck = sync_callabck
        self._slice_visibility = [True, True, True]
        self._create_peak_actor()
        super().__init__(name, render_callback)

    def _create_peak_actor(self):
        self._slicer = peaks_slicer(
            self.peaks,
            affine=self.affine,
            peak_values=self.peak_values * self._scale,
            visibility=self._slice_visibility,
        )
        self.state = np.asarray(self._slicer.cross_section, dtype=np.float32)
        self._cross_section_state = np.asarray(self.state, dtype=np.float32)
        lower_bounds = np.zeros(3)
        upper_bounds = np.array(self.peaks.shape[:3]) - 1
        if self.affine is not None:
            self.bounds = apply_transformation(
                np.array([lower_bounds, upper_bounds]), self.affine
            )
        else:
            self.bounds = np.asarray([lower_bounds, upper_bounds])
        self._cross_section_space = self._infer_cross_section_space()
        self._apply_cross_section_from_state()

    def _populate_info(self):
        info = f"Peaks shape: {self.peaks.shape}\n"
        info += f"Peaks dtype: {self.peaks.dtype}\n"
        if self.affine is not None:
            affine_str = np.array2string(
                np.round(self.affine, 2), separator=" ", prefix=""
            )
            info += f"Affine:\n{affine_str}\n"
        return info

    @property
    def actor(self):
        return self._slicer

    def _infer_cross_section_space(self):
        if self.affine is None:
            return "voxel"

        cross_section = np.asarray(self._slicer.cross_section, dtype=np.float32)
        voxel_center = (np.array(self.peaks.shape[:3], dtype=np.float32) - 1.0) * 0.5
        world_center = apply_transformation(
            np.array([voxel_center], dtype=np.float32), self.affine
        )[0]

        world_dist = np.linalg.norm(cross_section - world_center)
        voxel_dist = np.linalg.norm(cross_section - voxel_center)
        return "world" if world_dist <= voxel_dist else "voxel"

    def _voxel_from_world_state(self, world_state):
        voxel_state = apply_transformation(
            np.array([world_state], dtype=np.float32), np.linalg.inv(self.affine)
        )[0]
        voxel_state = np.round(voxel_state).astype(np.int16)
        max_idx = np.array(self.peaks.shape[:3], dtype=np.int16) - 1
        return np.clip(voxel_state, 0, max_idx)

    def _apply_cross_section_from_state(self):
        if self.affine is None:
            voxel_state = np.round(self.state).astype(np.int16)
            self._cross_section_state = voxel_state.astype(np.float32)
            self._slicer.cross_section = voxel_state
            return

        voxel_state = self._voxel_from_world_state(self.state)
        if self._cross_section_space == "world":
            world_state = apply_transformation(
                np.array([voxel_state], dtype=np.float32), self.affine
            )[0]
            self._cross_section_state = np.asarray(world_state, dtype=np.float32)
            self._slicer.cross_section = self._cross_section_state
        else:
            self._cross_section_state = voxel_state.astype(np.float32)
            self._slicer.cross_section = voxel_state

    def update_state(self, new_state):
        if self._synchronize:
            self.state = np.asarray(new_state[:3], dtype=np.float32)
            self._apply_cross_section_from_state()

    def render_widgets(self):
        changed, new = toggle_button(self._synchronize, label="Synchronize Slices")
        if changed:
            self._synchronize = new

        imgui.spacing()

        changed, new_scale = create_numeric_input(
            "Scale", self._scale, value_type="float", format="%.1f", step=0.1, height=24
        )

        if changed:
            new_scale = float(new_scale)
            if abs(new_scale - self._scale) > 1e-4:
                self._scale = new_scale
                self._create_peak_actor()
                self.render()

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
            self._slicer.material.opacity = self.opacity / 100.0

        imgui.spacing()
        axis_labels = ("X", "Y", "Z")

        def _axis_slider_bounds(axis):
            lower = float(min(self.bounds[0][axis], self.bounds[1][axis]))
            upper = float(max(self.bounds[0][axis], self.bounds[1][axis]))
            min_bound = int(np.ceil(lower))
            max_bound = int(np.floor(upper))
            if min_bound > max_bound:
                mid = int(round((lower + upper) * 0.5))
                return mid, mid
            return min_bound, max_bound

        slider_bounds = tuple(_axis_slider_bounds(axis) for axis in range(3))
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
                self.state[idx] = float(new)
                self._apply_cross_section_from_state()
                if self._synchronize and self._sync_callabck is not None:
                    self._sync_callabck(self, self.state)
            self._slice_visibility[idx] = toggle
        self._slicer.material.visibility = self._slice_visibility
        imgui.spacing()
