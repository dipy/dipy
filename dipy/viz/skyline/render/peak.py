from pathlib import Path

from fury.actor import peaks_slicer
from imgui_bundle import imgui

from dipy.viz.skyline.UI.elements import render_group, thin_slider
from dipy.viz.skyline.render.renderer import Visualization


def create_peak_visualization(
    input,
    idx,
    *,
    peak_values=1.0,
    opacity=100,
    render_callback=None,
):
    """Create peak visualization from input

    Parameters
    ----------
    input : tuple
        Tuple of the (pam, filename) or (pam,)
    idx : int
        Index of the peak for naming purposes if filename is not provided.
    peak_values : float or ndarray, optional
        Peak values to use for scaling.
    opacity : int, optional
        Opacity of the peak rendering.
    render_callback : callable, optional
        Callback function to be called after rendering.

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

    return Peak3D(
        filename,
        pam.peak_dirs,
        affine=pam.affine,
        peak_values=peak_values,
        opacity=opacity,
        render_callback=render_callback,
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
    ):
        super().__init__(name, render_callback)
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
