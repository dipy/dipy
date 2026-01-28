from fury.actor import streamlines, streamtube

from dipy.viz.skyline.render.renderer import Visualization


class Streamline3D(Visualization):
    def __init__(
        self, name, sft, *, line_type="tube", color=(1, 0, 0), render_callback=None
    ):
        super().__init__(name, render_callback)
        self.sft = sft
        self.color = color
        self._create_streamline_actor(line_type)

    def _create_streamline_actor(self, line_type):
        if line_type == "tube":
            self._actor = streamtube(
                self.sft.streamlines, radius=0.1, colors=self.color
            )
        elif line_type == "line":
            self._actor = streamlines(self.sft.streamlines, colors=self.color)

    @property
    def actor(self):
        return self._actor

    def render_widgets(self):
        pass
