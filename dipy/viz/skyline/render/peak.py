from pathlib import Path

from fury.actor import peaks_slicer
from fury.transform import apply_transformation
from imgui_bundle import imgui
import numpy as np

from dipy.viz.skyline.UI.elements import (
    create_numeric_input,
    render_group,
    thin_slider,
    toggle_button,
)
from dipy.viz.skyline.render.renderer import Visualization


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
        filename = Path(filename).name if filename is not None else f"Peaks_{idx}"

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
        super().__init__(name, render_callback)
        self.peaks = peaks
        self.affine = affine
        self.peak_values = peak_values
        self._scale = 1.0
        self.opacity = opacity
        self._synchronize = True
        self._sync_callabck = sync_callabck
        self._slice_visibility = [True, True, True]
        self._create_peak_actor()

    def _create_peak_actor(self):
        self._slicer = peaks_slicer(
            self.peaks,
            affine=self.affine,
            peak_values=self.peak_values * self._scale,
            visibility=self._slice_visibility,
        )
        self.state = self._slicer.cross_section
        lower_bounds = np.zeros(3)
        upper_bounds = np.array(self.peaks.shape[:3]) - 1
        if self.affine is not None:
            self.bounds = apply_transformation(
                np.array([lower_bounds, upper_bounds]), self.affine
            )
        else:
            self.bounds = np.asarray([lower_bounds, upper_bounds])

    @property
    def actor(self):
        return self._slicer

    def update_state(self, new_state):
        if self._synchronize:
            self.state = new_state
            self._slicer.cross_section = self.state

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
                self._slicer.cross_section = self.state
                self._synchronize and self._sync_callabck(self, self.state)
            self._slice_visibility[idx] = toggle
        self._slicer.material.visibility = self._slice_visibility
        imgui.spacing()
