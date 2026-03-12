import colorsys
from pathlib import Path
import time

from fury import distinguishable_colormap
from fury.actor import Group, streamlines, streamtube
from fury.colormap import line_colors
from fury.ui import TextBlock2D
from imgui_bundle import imgui
import numpy as np

from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.io.utils import split_filename_extension
from dipy.segment.clustering import qbx_and_merge
from dipy.stats.analysis import assignment_map
from dipy.tracking.streamline import (
    length as streamline_length,
)
from dipy.viz.skyline.UI.elements import (
    create_numeric_input,
    downloader,
    open_confirmation_dialog,
    segmented_switch,
    toggle_button,
    uploader,
)
from dipy.viz.skyline.compute import run_async
from dipy.viz.skyline.io import load_npy
from dipy.viz.skyline.render.renderer import Visualization


def apply_buan_colors(
    streamlines, buan_pvals, *, hue=(0.0, 0.1), saturation=(0.8, 0.2)
):
    n = len(buan_pvals)
    indx = assignment_map(streamlines, streamlines, n)
    buan_color_pvals = buan_pvals[indx].astype(np.float32)

    lut = np.zeros((n, 3), dtype=np.float32)

    h = np.interp(np.arange(n), [0, n - 1], hue)
    s = np.interp(np.arange(n), [0, n - 1], saturation)
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb(h[i], s[i], 0.8)
        lut[i] = (r, g, b)

    buan_color_idx = np.interp(
        buan_color_pvals, [buan_pvals.min(), buan_pvals.max()], [0, n - 1]
    ).astype(int)
    buan_colors = lut[buan_color_idx]

    return buan_colors


def create_cluster_help(*, position=(0, 0), size=(200, 180)):
    help_text = (
        "        Cluster Instructions:\n"
        "          Click to select/deselect.\n"
        "          'e' to expand.\n"
        "          'c' to collapse.\n"
        "          'a' to select all.\n"
        "          'd' to deselect all.\n"
        "          'h' to hide.\n"
        "          's' to show.\n"
    )
    text_block = TextBlock2D(
        text=help_text,
        position=position,
        vertical_justification="middle",
        justification="left",
        font_size=16,
        color=(1, 1, 1),
        bg_color=(0.2, 0.2, 0.2),
        size=size,
    )
    return text_block


def create_streamline_visualization(
    input,
    idx,
    *,
    is_cluster=False,
    thr=15.0,
    line_type="line",
    color=(1, 0, 0),
    render_callback=None,
    colormap=None,
    tract_colors=None,
    switch_render_callback=None,
    loader=None,
):
    """Create streamline visualization from input

    Parameters
    ----------
    input : tuple
        Tuple of the (sft, filename) or (sft,)
    idx : int
        Index of the tractogram for naming purposes.
    is_cluster : bool, optional
        Whether to cluster the streamline.
    thr : float, optional
        Clustering distance threshold.
    line_type : str, optional
        The type of line to render ("line" or "tube").
    color : tuple, optional
        Color of the streamline rendering.
    render_callback : callable, optional
        Callback function to be called after rendering.
    colormap : colormap, optional
        Colormap for clustering.
    tract_colors : variable float or str, optional
        Define the colors of the tractograms. Colors can be defined with
        3 values and should be between [0-1].
        String options are 'random' for random colors for each tractogram,
        'direction'  for directionally colored streamlines.
        For example, a value of (1, 0, 0) would mean the red color.
    switch_render_callback : callable, optional
        Callback function to switch rendering type, used for cluster visualization.
    loader : callable, optional
        Callback function to show/hide loader during asynchronous operations.

    Returns
    -------
    Visualization
        The created streamline visualization object.
    """
    if not isinstance(input, tuple) or len(input) not in (1, 2):
        raise ValueError(
            "Input must be a tuple containing (sft, filename) or (sft,) "
            "for streamline visualization."
        )

    if len(input) == 1:
        sft = input[0]
        filename = f"Streamline_{idx}"
    else:
        sft, filename = input
        filename = Path(filename).name if filename is not None else f"Streamline_{idx}"

    if is_cluster:
        return ClusterStreamline3D(
            filename,
            sft,
            thr,
            line_type=line_type,
            render_callback=render_callback,
            switch_render_callback=switch_render_callback,
            loader=loader,
        )

    if tract_colors is not None:
        if tract_colors == "random":
            color = next(colormap)
        elif tract_colors == "direction" or len(tract_colors) in [3, 4]:
            color = tract_colors
        else:
            raise ValueError(
                "Invalid tract_colors value. Must be 'random', 'direction', "
                "or a tuple of 3 or 4 values."
            )

    return Streamline3D(
        filename,
        sft,
        line_type=line_type,
        color=color,
        render_callback=render_callback,
        switch_render_callback=switch_render_callback,
        loader=loader,
    )


def create_streamline(lines, *, color=(1, 0, 0), line_type="line", segments=4):
    if isinstance(color, str) and color == "direction" and lines:
        color = line_colors(lines)
    if line_type == "tube":
        if len(color) != len(lines) and color.ndim == 2:
            points_per_line = [len(line) for line in lines]
            if color.shape[0] == sum(points_per_line):
                color = np.split(color, np.cumsum(points_per_line)[:-1])
        tubes = streamtube(
            lines=lines,
            radius=0.5,
            colors=color,
            segments=segments,
        )
        return tubes
    elif line_type == "line":
        if len(color) == len(lines) and color.ndim == 2:
            color = np.repeat(color, [len(line) for line in lines], axis=0)
        lines = streamlines(
            lines=lines,
            colors=color,
            thickness=5,
            outline_thickness=0.4,
            outline_color=(0.15, 0.15, 0.15),
        )
        lines.material.aa = True
        return lines


class Streamline3D(Visualization):
    def __init__(
        self,
        name,
        sft,
        *,
        line_type="line",
        color=(1, 0, 0),
        render_callback=None,
        switch_render_callback=None,
        loader=None,
    ):
        self.sft = sft
        self.color = color
        self._line_type = line_type
        self._buan_pvals_file = False
        self._buan_pvals_data = None
        self._switch_render_callback = switch_render_callback
        self._loader = loader
        self._show_line_type_confirmation = False
        self._line_type_confirmation_active = False
        self._pending_line_type = None
        self._apply_pending_line_type_next_frame = False
        self._hide_line_type_loader_next_frame = False
        self._create_streamline_actor()
        super().__init__(name, render_callback)

    def _create_streamline_actor(self):
        self._actor = create_streamline(
            lines=self.sft.streamlines,
            color=self.color,
            line_type=self._line_type,
        )

    @property
    def actor(self):
        return self._actor

    def _populate_info(self):
        np.set_printoptions(precision=2, suppress=True)
        info = f"Number of streamlines: {len(self.sft.streamlines)}\n"
        info += f"Min Length: {streamline_length(self.sft.streamlines).min():.0f}\n"
        info += f"Max Length: {streamline_length(self.sft.streamlines).max():.0f}\n"
        np.set_printoptions()
        return info

    def render_widgets(self):
        if (
            self._apply_pending_line_type_next_frame
            and self._pending_line_type is not None
        ):
            self._line_type = self._pending_line_type
            self._create_streamline_actor()
            self._pending_line_type = None
            self._apply_pending_line_type_next_frame = False
            self._hide_line_type_loader_next_frame = True
            self.render()

        if self._hide_line_type_loader_next_frame and self._loader is not None:
            self._loader(False)
            self._hide_line_type_loader_next_frame = False

        changed, is_clustered = toggle_button(False, label="Cluster")
        if changed:
            if self._switch_render_callback is not None:
                self._switch_render_callback(self, is_clustered)
                self.render()

        imgui.spacing()

        changed, new = segmented_switch("Line Type", ["Line", "Tube"], self._line_type)
        if changed:
            requested_line_type = new.lower()
            require_confirmation = len(self.sft.streamlines) > 20000
            if (
                self._line_type == "line"
                and requested_line_type == "tube"
                and require_confirmation
            ):
                if (
                    self._pending_line_type is None
                    and not self._line_type_confirmation_active
                    and not self._apply_pending_line_type_next_frame
                ):
                    self._pending_line_type = requested_line_type
                    self._show_line_type_confirmation = True
            else:
                self._line_type = requested_line_type
                self._create_streamline_actor()
                self.render()

        if self._show_line_type_confirmation:
            self._show_line_type_confirmation = False
            self._line_type_confirmation_active = True
            imgui.open_popup("Line Type Confirmation")
        line_type_dialog_state = open_confirmation_dialog(
            "Line Type Confirmation",
            "Rendering tractograms as tubes may cause performance issues.\n"
            "Do you want to continue?",
            okay_text="Switch to Tube",
            cancel_text="Keep Line",
        )
        if line_type_dialog_state == "okay" and self._pending_line_type == "tube":
            self._line_type_confirmation_active = False
            if self._loader is not None:
                self._loader(True, message="Switching to tubes...")
            self._apply_pending_line_type_next_frame = True
        elif line_type_dialog_state == "cancel":
            self._line_type_confirmation_active = False
            self._pending_line_type = None
            self._apply_pending_line_type_next_frame = False

        imgui.spacing()
        imgui.spacing()
        imgui.spacing()
        imgui.spacing()

        def handle_color_change(fname):
            if fname is not None:
                self._buan_pvals_file = Path(fname[0]).name
                self._buan_pvals_data = load_npy(fname[0])
                self.color = apply_buan_colors(
                    self.sft.streamlines, self._buan_pvals_data
                )
                self._create_streamline_actor()
                self.render()

        uploader(
            "Upload BUAN P values",
            callback=handle_color_change,
            extension="*.npy",
            selected=self._buan_pvals_file,
            type="buan_pvals",
        )


class ClusterStreamline3D(Visualization):
    def __init__(
        self,
        name,
        sft,
        thr,
        *,
        line_type="line",
        render_callback=None,
        switch_render_callback=None,
        loader=None,
        size_threshold=5,
        length_threshold=10.0,
    ):
        self.sft = sft
        self.thr = thr
        self._clusters = []
        self._cluster_state = {}
        self._sizes = np.asarray([])
        self._lengths = np.asarray([])
        self._line_type = line_type
        self._actor = Group()
        self._pending_thr = None
        self._thr_changed_at = None
        self._switch_render_callback = switch_render_callback
        self._recluster_debounce_sec = 0.3
        self._show_confirmation_dialog = False
        self._show_expand_confirmation_dialog = False
        self._expand_confirmation_active = False
        self._loader = loader
        self._is_clustering = False
        self._queued_recluster = False
        self.size = size_threshold
        self.length = length_threshold
        super().__init__(name, render_callback=render_callback)
        self._perform_clustering()

    def _perform_clustering(self):
        if self._is_clustering:
            self._queued_recluster = True
            return

        self._is_clustering = True
        if self._loader is not None:
            self._loader(True, message="Clustering streamlines...")
            self.render()
        run_async(
            self._compute_clustering_data,
            self._apply_clustering_result,
            self.thr,
        )

    def _compute_clustering_data(self, thr):
        clusters = qbx_and_merge(self.sft.streamlines, [40, 30, 25, 20, thr])
        lengths = np.asarray([streamline_length(c) for c in clusters.centroids])
        sizes = np.asarray([len(c) for c in clusters])
        colormap = distinguishable_colormap(nb_colors=len(clusters))
        if sizes.size:
            line_widths = np.interp(sizes, [np.min(sizes), np.max(sizes)], [0.1, 2.0])
        else:
            line_widths = np.asarray([])
        return clusters, lengths, sizes, colormap, line_widths

    def _apply_clustering_result(self, result, exception):
        self._is_clustering = False

        if exception is not None:
            print(f"Error clustering streamlines: {exception}")
        else:
            clusters, lengths, sizes, colormap, line_widths = result
            for actor in list(self._actor.children):
                self._actor.remove(actor)
            self._cluster_state.clear()
            self._clusters = clusters
            self._lengths = lengths
            self._sizes = sizes

            for idx, centroid in enumerate(self._clusters.centroids):
                centroid_rep = streamtube(
                    lines=[centroid],
                    radius=line_widths[idx],
                    colors=colormap[idx],
                    backend="cpu",
                    opacity=0.5,
                )
                centroid_rep.add_event_handler(
                    lambda event: self._toggle_cluster_selection(event.target),
                    "pointer_down",
                )
                self._cluster_state[centroid_rep] = {
                    "cluster": idx,
                    "size": self._sizes[idx],
                    "length": self._lengths[idx],
                    "color": colormap[idx],
                    "selected": False,
                    "expanded": False,
                    "cluster_actor": None,
                }
                self._actor.add(centroid_rep)

            self._refresh_cluster_visibility()
            self._info = self._populate_info()

        if self._queued_recluster:
            self._queued_recluster = False
            self._perform_clustering()
            return

        if self._loader is not None:
            self._loader(False)

    def _refresh_cluster_visibility(self):
        for centroid_rep, state in self._cluster_state.items():
            is_visible = state["size"] >= self.size and state["length"] >= self.length
            if state["expanded"] and state["cluster_actor"] is not None:
                state["cluster_actor"].visible = is_visible
            else:
                centroid_rep.visible = is_visible

    # Interaction methods
    def _create_cluster_streamlines(self, centroid_rep):
        state = self._cluster_state[centroid_rep]
        cluster_idx = state["cluster"]
        cluster_streamlines = self._clusters[cluster_idx]
        color = state["color"]
        streamline_actor = create_streamline(
            lines=cluster_streamlines,
            color=color,
            line_type=self._line_type,
            segments=3,
        )
        return centroid_rep, streamline_actor

    def _selected_unexpanded_clusters(self):
        return [
            centroid_rep
            for centroid_rep, state in self._cluster_state.items()
            if state["selected"] and not state["expanded"]
        ]

    def _expand_clusters(self, *, skip_confirmation=False):
        selected_clusters = self._selected_unexpanded_clusters()
        if not selected_clusters:
            return

        if (
            not skip_confirmation
            and self._line_type == "tube"
            and len(selected_clusters) > 10
        ):
            if (
                self._expand_confirmation_active
                or self._show_expand_confirmation_dialog
            ):
                return
            self._expand_confirmation_active = True
            self._show_expand_confirmation_dialog = True
            return

        for centroid_rep in selected_clusters:
            _, streamline_actor = self._create_cluster_streamlines(centroid_rep)
            state = self._cluster_state[centroid_rep]
            self._actor.add(streamline_actor)
            self._actor.remove(centroid_rep)
            state["cluster_actor"] = streamline_actor
            state["expanded"] = True

    def _collapse_clusters(self):
        for centroid_rep, state in self._cluster_state.items():
            if state["selected"] and state["expanded"]:
                self._actor.add(centroid_rep)
                cluster_actor = state["cluster_actor"]
                if (
                    cluster_actor is not None
                    and cluster_actor in self._actor.children
                ):
                    self._actor.remove(cluster_actor)
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
            centroid_rep.material.opacity = 1.0
        else:
            centroid_rep.material.opacity = 0.5

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

    def _populate_info(self):
        np.set_printoptions(precision=2, suppress=True)
        info = f"Total streamlines: {len(self.sft.streamlines)}\n"
        info += f"Number of clusters: {len(self._clusters)}\n"
        info += (
            f"Max Cluster Size: {(self._sizes.max() if self._sizes.size else 0):.0f}\n"
        )
        info += (
            f"Min Cluster Size: {(self._sizes.min() if self._sizes.size else 0):.0f}\n"
        )
        info += (
            "Max Cluster Length: "
            f"{(self._lengths.max() if self._lengths.size else 0):.0f}\n"
        )
        info += (
            "Min Cluster Length: "
            f"{(self._lengths.min() if self._lengths.size else 0):.0f}\n"
        )
        return info

    def compute_visible_tractogram(self):
        visible_streamlines = []
        for state in self._cluster_state.values():
            if state["selected"]:
                cluster_idx = state["cluster"]
                cluster_streamlines = self._clusters[cluster_idx]
                visible_streamlines.extend(cluster_streamlines)
        return StatefulTractogram.from_sft(visible_streamlines, self.sft)

    def save_tractogram(self, filenames, rois=None, shm_coeffs=None):
        if filenames:
            if isinstance(filenames, (list, tuple)):
                filenames = filenames[0]
            visible_sft = self.compute_visible_tractogram()
            save_tractogram(visible_sft, filenames, bbox_valid_check=False)

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
            self._refresh_cluster_visibility()

    @property
    def actor(self):
        return self._actor

    def render_widgets(self):
        changed, is_clustered = toggle_button(True, label="Cluster")
        if changed:
            if self._switch_render_callback is not None:
                self._switch_render_callback(self, is_clustered)
                self.render()

        imgui.spacing()

        changed, new = segmented_switch("Line Type", ["Line", "Tube"], self._line_type)
        if changed:
            self._line_type = new.lower()
            n_expanded = sum(
                1 for state in self._cluster_state.values() if state["expanded"]
            )
            self._show_confirmation_dialog = (
                self._line_type == "tube" and n_expanded > 10
            )
            if not self._show_confirmation_dialog:
                for centroid_rep, state in self._cluster_state.items():
                    if state["expanded"]:
                        _, new_actor = self._create_cluster_streamlines(centroid_rep)
                        self._actor.remove(state["cluster_actor"])
                        self._actor.add(new_actor)
                        state["cluster_actor"] = new_actor

        if self._show_confirmation_dialog:
            self._show_confirmation_dialog = False
            imgui.open_popup("Cluster Confirmation")
            self._collapse_clusters()
        dialog_state = open_confirmation_dialog(
            "Cluster Confirmation",
            "Rendering many expanded clusters as tubes may cause\n"
            "performance issues. So we will collapse all the clusters.",
            okay_text="Understood",
            cancel_text="Keep Expanded",
        )
        if dialog_state == "cancel":
            self._expand_clusters(skip_confirmation=True)

        if self._show_expand_confirmation_dialog:
            self._show_expand_confirmation_dialog = False
            imgui.open_popup("Expand Cluster Confirmation")
        expand_dialog_state = open_confirmation_dialog(
            "Expand Cluster Confirmation",
            "Expanding more than 10 clusters as tubes may cause\n"
            "performance issues. Do you want to continue?",
            okay_text="Expand Anyway",
            cancel_text="Cancel",
        )
        if expand_dialog_state == "okay":
            self._expand_confirmation_active = False
            self._expand_clusters(skip_confirmation=True)
        elif expand_dialog_state == "cancel":
            self._expand_confirmation_active = False

        imgui.spacing()
        imgui.spacing()

        threshold_value = (
            self._pending_thr if self._pending_thr is not None else self.thr
        )
        changed, new_thr = create_numeric_input(
            "Threshold",
            threshold_value,
            value_type="float",
            step=0.5,
            format="%.1f",
            label_width=100,
            height=26,
        )
        if changed:
            self._pending_thr = max(0.1, float(new_thr))
            self._thr_changed_at = time.monotonic()
        if self._pending_thr is not None and self._thr_changed_at is not None:
            elapsed = time.monotonic() - self._thr_changed_at
            if elapsed >= self._recluster_debounce_sec:
                if not np.isclose(self._pending_thr, self.thr):
                    self.thr = self._pending_thr
                    self._perform_clustering()
                self._pending_thr = None
                self._thr_changed_at = None

        imgui.spacing()
        changed, new_size = create_numeric_input(
            "Size", self.size, value_type="int", step=1, label_width=100, height=26
        )
        if changed:
            self.size = max(0, int(new_size))
            self._refresh_cluster_visibility()

        imgui.spacing()
        changed, new_length = create_numeric_input(
            "Length",
            self.length,
            value_type="float",
            step=1,
            format="%.1f",
            label_width=100,
            height=26,
        )
        if changed:
            self.length = max(0.0, float(new_length))
            self._refresh_cluster_visibility()

        imgui.spacing()
        imgui.spacing()

        downloader(
            "Tractogram",
            callback=self.save_tractogram,
            extension="*.trx *.trk",
            file_name=f"{split_filename_extension(self.name)[0]}_visible_tractogram.trx",
        )

        if imgui.is_item_hovered():
            imgui.set_tooltip("Download the selected cluster tractogram as a .trx file")
