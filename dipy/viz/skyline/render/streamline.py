from fury.actor import streamlines, streamtube

from dipy.viz.skyline.render.renderer import Visualization


class Streamline3D(Visualization):
    def __init__(
        self,
        name,
        sft,
        *,
        line_type="line",
        color=(1, 0, 0),
        device=None,
        render_callback=None,
    ):
        super().__init__(name, render_callback)
        self.sft = sft
        self.color = color
        self.device = device
        self._create_streamline_actor(line_type)

    def _create_streamline_actor(self, line_type):
        if line_type == "tube":
            self._actor = streamtube(
                self.sft.streamlines,
                radius=0.1,
                colors=self.color,
                flat_shading=False,
                segments=3,
            )
        elif line_type == "line":
            self._actor = streamlines(
                self.sft.streamlines,
                colors=self.color,
                thickness=1,
                outline_thickness=1,
            )

    @property
    def actor(self):
        return self._actor

    def render_widgets(self):
        pass
