from fury.actor import show_slices, volume_slicer
import numpy as np

from dipy.viz.skyline.UI.elements import create_numeric_input, create_slider
from dipy.viz.skyline.render.renderer import Visualization


class Slicer(Visualization):
    def __init__(
        self,
        volume,
        *,
        affine=None,
        interpolation="linear",
        render_callback=None,
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
        print(self.state)

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
        changed, new = create_slider(
            "Slice X",
            min_value=int(self.bounds[0][0]),
            max_value=int(self.bounds[1][0]),
            value=self.state[0],
        )
        if changed:
            new = min(new, int(self.bounds[1][0]))
            new = max(new, int(self.bounds[0][0]))
            self.state[0] = int(new)
            show_slices(self.slicer, self.state)
            self.render()
        changed, new = create_numeric_input("Slice Y", value=self.state[1])
        if changed:
            new = min(new, int(self.bounds[1][1]))
            new = max(new, int(self.bounds[0][1]))
            self.state[1] = int(new)
            show_slices(self.slicer, self.state)
            self.render()
        changed, new = create_numeric_input("Slice Z", value=self.state[2])
        if changed:
            new = min(new, int(self.bounds[1][2]))
            new = max(new, int(self.bounds[0][2]))
            self.state[2] = int(new)
            show_slices(self.slicer, self.state)
            self.render()
