from fury.actor import peaks_slicer
from imgui_bundle import imgui

from dipy.viz.skyline.UI.elements import render_group, thin_slider
from dipy.viz.skyline.render.renderer import Visualization


class Peak3D(Visualization):
    def __init__(
        self, peaks, *, affine=None, peak_values=1.0, opacity=100, render_callback=None
    ):
        super().__init__(render_callback=render_callback)
        self.peaks = peaks
        self.affine = affine
        self.peak_values = peak_values
        self.opacity = opacity
        self._create_peak_actor()

    def _create_peak_actor(self):
        self._slicer = peaks_slicer(
            self.peaks,
            affine=self.affine,
            peak_values=self.peak_values,
        )
        self.state = self._slicer.cross_section
        self.bounds = self._slicer.bounds

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
                self._slicer.cross_section = self.state
        imgui.spacing()
