from fury import distinguishable_colormap
from fury.actor import Group, streamlines, streamtube
import numpy as np

from dipy.segment.clustering import qbx_and_merge
from dipy.tracking.streamline import length as streamline_length
from dipy.viz.skyline.render.renderer import Visualization


def create_streamline(lines, *, color=(1, 0, 0), line_type="line", segments=3):
    if line_type == "tube":
        tubes = streamtube(
            lines=lines,
            radius=0.15,
            colors=color,
            segments=segments,
        )
        tubes.material.side = "front"
        return tubes
    elif line_type == "line":
        return streamlines(
            lines=lines,
            colors=color,
            thickness=1,
            outline_thickness=1,
        )


class Streamline3D(Visualization):
    def __init__(
        self,
        name,
        sft,
        *,
        line_type="line",
        color=(1, 0, 0),
        render_callback=None,
    ):
        super().__init__(name, render_callback)
        self.sft = sft
        self.color = color
        self._create_streamline_actor(line_type)

    def _create_streamline_actor(self, line_type):
        self._actor = create_streamline(
            lines=self.sft.streamlines,
            color=self.color,
            line_type=line_type,
        )

    @property
    def actor(self):
        return self._actor

    def render_widgets(self):
        pass


class ClusterStreamline3D(Visualization):
    def __init__(
        self, name, sft, thr, *, line_type="line", render_callback=None, colormap=None
    ):
        super().__init__(name, render_callback=render_callback)

        self.sft = sft
        self.thr = thr
        self._clusters = None
        self._cluster_state = {}
        self._sizes = []
        self._lengths = []
        self._line_type = line_type
        if colormap is None:
            colormap = distinguishable_colormap()
        self._colormap = colormap
        self._actor = Group()
        self._perform_clustering()

    def _perform_clustering(self):
        self._clusters = qbx_and_merge(self.sft.streamlines, [40, 30, 25, 20, self.thr])
        self._lengths = np.asarray(
            [streamline_length(c) for c in self._clusters.centroids]
        )
        self._sizes = np.asarray([len(c) for c in self._clusters])
        line_widths = np.interp(
            self._sizes, [np.min(self._sizes), np.max(self._sizes)], [0.1, 2.0]
        )
        for idx, centroid in enumerate(self._clusters.centroids):
            color = next(self._colormap)
            centroid_rep = streamtube(
                lines=[centroid],
                radius=line_widths[idx],
                colors=color,
                backend="cpu",
            )
            centroid_rep.add_event_handler(
                lambda event: self._toggle_cluster_selection(event.target),
                "pointer_down",
            )
            self._cluster_state[centroid_rep] = {
                "cluster": idx,
                "size": self._sizes[idx],
                "length": self._lengths[idx],
                "color": color,
                "selected": False,
                "expanded": False,
                "cluster_actor": None,
                "line_actor": None,
            }
            self._actor.add(centroid_rep)

    # Interaction methods
    def _expand_clusters(self):
        for centroid_rep, state in self._cluster_state.items():
            if state["selected"] and not state["expanded"]:
                cluster_idx = state["cluster"]
                cluster_streamlines = self._clusters[cluster_idx]
                color = state["color"]
                streamline_actor = create_streamline(
                    lines=cluster_streamlines,
                    color=color,
                    line_type=self._line_type,
                    segments=3,
                )
                self._actor.add(streamline_actor)
                self._actor.remove(centroid_rep)
                state["cluster_actor"] = streamline_actor
                state["expanded"] = True

    def _collapse_clusters(self):
        for centroid_rep, state in self._cluster_state.items():
            if state["selected"] and state["expanded"]:
                self._actor.add(centroid_rep)
                self._actor.remove(state["cluster_actor"])
                state["cluster_actor"] = None
                state["expanded"] = False

    def _select_all_clusters(self):
        for centroid_rep in self._cluster_state:
            self._update_cluster_state(centroid_rep, True)

    def _deselect_all_clusters(self):
        for centroid_rep in self._cluster_state:
            self._update_cluster_state(centroid_rep, False)

    def _update_cluster_state(self, centroid_rep, selected):
        state = self._cluster_state[centroid_rep]
        state["selected"] = selected
        if selected:
            centroid_rep.material.opacity = 0.5
        else:
            centroid_rep.material.opacity = 1.0

    def _toggle_cluster_selection(self, cluster):
        self._update_cluster_state(
            cluster, not self._cluster_state[cluster]["selected"]
        )

    def _hide_deselected_clusters(self):
        for centroid_rep, state in self._cluster_state.items():
            if not state["selected"]:
                centroid_rep.visible = False

    def _show_all_clusters(self):
        for centroid_rep in self._cluster_state:
            centroid_rep.visible = True

    def handle_key_events(self, event):
        if event.key == "e":
            self._expand_clusters()
        elif event.key == "c":
            self._collapse_clusters()
        elif event.key == "a":
            self._select_all_clusters()
        elif event.key == "d":
            self._deselect_all_clusters()
        elif event.key == "h":
            self._hide_deselected_clusters()
        elif event.key == "s":
            self._show_all_clusters()

    @property
    def actor(self):
        return self._actor

    def render_widgets(self):
        pass
