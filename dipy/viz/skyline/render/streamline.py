"""Tractography layers, clustering, and BUAN coloring for Skyline."""

import colorsys
from pathlib import Path
import time

import numpy as np

from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.io.utils import split_filename_extension
from dipy.segment.clustering import qbx_and_merge
from dipy.stats.analysis import assignment_map
from dipy.tracking.streamline import (
    length as streamline_length,
)
from dipy.utils.logging import logger
from dipy.utils.optpkg import optional_package
from dipy.viz.skyline.UI.elements import (
    color_picker,
    colors_equal,
    create_numeric_input,
    downloader,
    normalize_picker_color,
    open_confirmation_dialog,
    segmented_switch,
    thin_slider,
    toggle_button,
    two_disk_slider,
    uploader,
)
from dipy.viz.skyline.compute import run_async
from dipy.viz.skyline.io import load_npy
from dipy.viz.skyline.render.renderer import Visualization

fury_trip_msg = (
    "Skyline requires Fury version 2.0.0 or higher."
    " Please upgrade Fury by `pip install -U fury --pre` to use Skyline."
)
fury, has_fury_v2, _ = optional_package(
    "fury",
    min_version="2.0.0",
    trip_msg=fury_trip_msg,
)
if has_fury_v2:
    from fury import distinguishable_colormap
    from fury.actor import Group, streamlines, streamtube
    from fury.colormap import line_colors
    from fury.ui import TextBlock2D

else:
    actor = fury

imgui_bundle, has_imgui, _ = optional_package(
    "imgui_bundle", min_version="1.92.600", max_version="1.92.801"
)
if has_imgui:
    imgui = imgui_bundle.imgui
    icons_fontawesome_6 = imgui_bundle.icons_fontawesome_6


def create_colormap(n, *, hue=(0.0, 0.1), saturation=(0.8, 0.2), value=0.8):
    """Build an RGB lookup table sampled along HSV space.

    Parameters
    ----------
    n : int
        Number of discrete colors.
    hue : tuple of float, optional
        Hue endpoints passed through ``np.interp`` over the LUT indices.
    saturation : tuple of float, optional
        Saturation endpoints interpolated like ``hue``.
    value : float, optional
        Fixed HSV value (brightness) for every entry.

    Returns
    -------
    ndarray, shape (n, 3)
        Float32 RGB colors in ``[0, 1]``.
    """
    lut = np.zeros((n, 3), dtype=np.float32)
    h = np.interp(np.arange(n), [0, n - 1], hue)
    s = np.interp(np.arange(n), [0, n - 1], saturation)
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb(h[i], s[i], value)
        lut[i] = (r, g, b)
    return lut


def apply_buan_colors(
    streamlines,
    buan_pvals,
    *,
    hue=(0.0, 0.1),
    saturation=(0.8, 0.2),
    value=0.8,
    buan_color_idx=None,
):
    """Assign BUAN p-value-derived RGB colors per streamline or sample.

    Parameters
    ----------
    streamlines : list of ndarray
        Streamline arrays used for assignment mapping when ``buan_color_idx``
        is omitted.
    buan_pvals : ndarray
        Scalar p-values (or statistics) sampled per correspondence bin.
    hue : tuple of float, optional
        Passed to :func:`create_colormap`.
    saturation : tuple of float, optional
        Passed to :func:`create_colormap`.
    value : float, optional
        Passed to :func:`create_colormap`.
    buan_color_idx : ndarray or None, optional
        Precomputed indices into the LUT; if None, computed via ``assignment_map``.

    Returns
    -------
    buan_colors : ndarray, shape (N, 3)
        Per-streamline RGB colors.
    buan_color_idx : ndarray
        Integer LUT indices used for coloring.
    """
    n = len(buan_pvals)
    if n > 1000:
        logger.info("Limiting assignment to 1000 bands for performance reasons.")
        n = 1000
    if buan_color_idx is None:
        _, indx = assignment_map(streamlines, streamlines, n)
        buan_color_pvals = buan_pvals[indx].astype(np.float32)
        buan_color_idx = np.interp(
            buan_color_pvals, [buan_pvals.min(), buan_pvals.max()], [0, n - 1]
        ).astype(int)

    lut = create_colormap(n, hue=hue, saturation=saturation, value=value)

    buan_colors = lut[buan_color_idx]

    return buan_colors, buan_color_idx


def create_cluster_help(*, position=(0, 0), size=(200, 180)):
    """Create a 2D text panel summarizing cluster interaction shortcuts.

    Parameters
    ----------
    position : tuple of float, optional
        Screen-space anchor for the block.
    size : tuple of float, optional
        Pixel width and height of the background rectangle.

    Returns
    -------
    TextBlock2D
        Fury overlay actor ready to be inserted into the scene.
    """
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
    line_type="Line",
    color=(1, 0, 0),
    render_callback=None,
    colormap=None,
    tract_colors=None,
    switch_render_callback=None,
    loader=None,
    size_threshold=None,
    length_threshold=None,
    buan_pvals_file=None,
    async_clustering=True,
):
    """Create a Streamline3D or ClusterStreamline3D from a loaded tractogram.

    Parameters
    ----------
    input : tuple
        Tuple of ``(sft, filename)`` or ``(sft,)``, where ``sft`` is a
        StatefulTractogram.
    idx : int
        Index of the tractogram for naming purposes.
    is_cluster : bool, optional
        Whether to cluster the streamline.
    thr : float, optional
        Clustering distance threshold.
    line_type : str, optional
        The type of line to render ("Line" or "Tube").
    color : tuple, optional
        Color of the streamline rendering.
    render_callback : callable, optional
        Callback function to be called after rendering.
    colormap : colormap, optional
        Colormap for clustering.
    tract_colors : str or tuple of float or None, optional
        ``"random"`` picks the next color from ``colormap``; ``"direction"``
        or a 3- or 4-value tuple in ``[0, 1]`` is used as-is; any other
        string raises ``ValueError``. If None, ``color`` is used unchanged.
    switch_render_callback : callable, optional
        Callback function to switch rendering type, used for cluster visualization.
    loader : callable, optional
        Callback function to show/hide loader during asynchronous operations.
    size_threshold : int, optional
        Minimum number of streamlines in a cluster to be visible.
    length_threshold : float, optional
        Minimum length of streamlines in a cluster to be visible.
    buan_pvals_file : str, optional
        File path to BUAN p-values for coloring streamlines.
    async_clustering : bool, optional
        Whether to perform clustering asynchronously. Set to False to block
        until clustering completes (used in stealth mode).

    Returns
    -------
    Streamline3D or ClusterStreamline3D
        The created streamline visualization; a ClusterStreamline3D when
        ``is_cluster`` is True, otherwise a Streamline3D.

    Raises
    ------
    ValueError
        If ``input`` is not a 1- or 2-element tuple, or if ``tract_colors``
        is a string other than ``"random"``/``"direction"`` and is not a
        3- or 4-value tuple.
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

    if is_cluster:
        return ClusterStreamline3D(
            filename,
            sft,
            thr,
            line_type=line_type,
            render_callback=render_callback,
            switch_render_callback=switch_render_callback,
            loader=loader,
            size_threshold=size_threshold,
            length_threshold=length_threshold,
            async_clustering=async_clustering,
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
        buan_pvals_file=buan_pvals_file,
        loader=loader,
    )


def create_streamline(lines, *, color=(1, 0, 0), line_type="Line", segments=4):
    """Instantiate Fury line or tube geometry for polyline streamlines.

    Parameters
    ----------
    lines : list of ndarray
        Each array is a (N, 3) polyline in world space.
    color : ndarray, tuple, or str, optional
        Per-point, per-line, directional (``"direction"``), or constant RGB colors.
    line_type : {"Line", "Tube"}, optional
        Primitive style passed to Fury.
    segments : int, optional
        Tube tessellation segments when ``line_type`` is ``"Tube"``.

    Returns
    -------
    Actor or None
        Fury actor (line or tube container) ready to parent under a
        ``Group``, or None when ``line_type`` is neither ``"Line"`` nor
        ``"Tube"``.
    """
    if isinstance(color, str) and color == "direction" and lines:
        color = line_colors(lines)
    if line_type == "Tube":
        if (
            isinstance(color, np.ndarray)
            and color.ndim == 2
            and len(color) != len(lines)
        ):
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
    elif line_type == "Line":
        if (
            isinstance(color, np.ndarray)
            and color.ndim == 2
            and len(color) == len(lines)
        ):
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
    """Non-clustered tractography layer rendered as a single line or tube actor.

    Parameters
    ----------
    name : str
        Display name used in the Skyline UI.
    sft : StatefulTractogram
        Tractogram whose streamlines are rendered.
    line_type : str, optional
        The type of line to render ("Line" or "Tube").
    color : tuple(float, float, float), optional
        RGB color of the streamlines in ``[0, 1]``.
    render_callback : callable, optional
        Callback used to request a render/update.
    switch_render_callback : callable, optional
        Callback invoked to switch to the clustered rendering mode.
    buan_pvals_file : str, optional
        File path to BUAN p-values used to color the streamlines on creation.
    loader : callable, optional
        Callback function to show/hide loader during asynchronous operations.
    """

    def __init__(
        self,
        name,
        sft,
        *,
        line_type="Line",
        color=(1, 0, 0),
        render_callback=None,
        switch_render_callback=None,
        buan_pvals_file=None,
        loader=None,
    ):
        """Initialize the non-clustered tractography layer.

        Parameters
        ----------
        name : str
            Display name used in the Skyline UI.
        sft : StatefulTractogram
            Tractogram whose streamlines are rendered.
        line_type : str, optional
            The type of line to render ("Line" or "Tube").
        color : tuple(float, float, float), optional
            RGB color of the streamlines in ``[0, 1]``.
        render_callback : callable, optional
            Callback used to request a render/update.
        switch_render_callback : callable, optional
            Callback invoked to switch to the clustered rendering mode.
        buan_pvals_file : str, optional
            File path to BUAN p-values used to color the streamlines on creation.
        loader : callable, optional
            Callback function to show/hide loader during asynchronous operations.
        """
        self.sft = sft
        self.color = color
        self._original_color = color
        self._draft_color = color
        self._color_picker_open = False
        self._color_picker_popup_id = f"streamline_color_picker_popup##{name}"
        self._hue_low = 0.0
        self._hue_high = 0.1
        self._saturation_high = 0.8
        self._saturation_low = 0.2
        self._value = 0.8
        self._line_type = line_type
        self._buan_pvals_file = buan_pvals_file
        self._buan_pvals_data = None
        self._buan_color_idx = None
        self._show_line_type_confirmation = False
        self._requested_line_type = None
        self._apply_line_change_next_frame = False
        self._switch_render_callback = switch_render_callback
        self._loader = loader

        self._create_streamline_actor()
        super().__init__(name, render_callback)
        if buan_pvals_file is not None:
            self.handle_color_change(buan_pvals_file)

    def _create_streamline_actor(self):
        """Build and store the Fury line or tube actor for the streamlines."""
        self._actor = create_streamline(
            lines=self.sft.streamlines,
            color=self.color,
            line_type=self._line_type,
        )

    @property
    def actor(self):
        """Return the Fury line or tube actor rendering the streamlines.

        Returns
        -------
        Actor
            The line or tube actor built by :func:`create_streamline`.
        """
        return self._actor

    def _populate_info(self):
        """Build the streamline count and length summary shown in the UI.

        Returns
        -------
        str
            Multi-line text with the streamline count and min/max length.
        """
        np.set_printoptions(precision=2, suppress=True)
        info = f"Number of streamlines: {len(self.sft.streamlines)}\n"
        info += f"Min Length: {streamline_length(self.sft.streamlines).min():.0f}\n"
        info += f"Max Length: {streamline_length(self.sft.streamlines).max():.0f}\n"
        np.set_printoptions()
        return info

    def handle_color_change(self, fname):
        """Recolor the streamlines from a BUAN p-values file and re-render.

        Parameters
        ----------
        fname : list of str or None
            Selected file path(s) from the uploader; only ``fname[0]`` is
            used. If None, the color is left unchanged.
        """
        if fname is not None:
            self._buan_pvals_file = Path(fname[0]).name
            self._buan_pvals_data = load_npy(fname[0])
            self.color, self._buan_color_idx = apply_buan_colors(
                self.sft.streamlines,
                self._buan_pvals_data,
                hue=(self._hue_low, self._hue_high),
                saturation=(self._saturation_high, self._saturation_low),
                value=self._value,
            )
            self.apply_scene_op(self._create_streamline_actor)
            self.render()

    def _update_buan_colors_on_sliders(self):
        """Recompute BUAN colors from the current hue/saturation/value sliders."""
        self.color, self._buan_color_idx = apply_buan_colors(
            self.sft.streamlines,
            self._buan_pvals_data,
            buan_color_idx=self._buan_color_idx,
            hue=(self._hue_low, self._hue_high),
            saturation=(self._saturation_high, self._saturation_low),
            value=self._value,
        )
        self.apply_scene_op(self._create_streamline_actor)
        self.render()

    def render_widgets(self):
        """Draw the line-type, color, and BUAN coloring controls for this layer."""
        if self._apply_line_change_next_frame:
            self._apply_line_change_next_frame = False
            self.apply_scene_op(self._create_streamline_actor)
            self._loader(False)
            self.render()

        changed, is_clustered = toggle_button(False, label="Cluster")
        if changed and self._switch_render_callback is not None:
            self._switch_render_callback(self, is_clustered)

        imgui.spacing()

        changed, new = segmented_switch("Line Type", ["Line", "Tube"], self._line_type)
        if changed:
            requested_line_type = new.title()
            require_confirmation = (
                self._line_type == "Line"
                and requested_line_type == "Tube"
                and len(self.sft.streamlines) > 20000
            )
            if require_confirmation:
                self._requested_line_type = requested_line_type
                self._show_line_type_confirmation = True
            else:
                self._loader(True, message="Switching line type...")
                self._line_type = requested_line_type
                self.apply_scene_op(self._create_streamline_actor)
                self._loader(False)
                self.render()

        if self._show_line_type_confirmation:
            self._show_line_type_confirmation = False
            imgui.open_popup("Line Type Confirmation")
        line_type_dialog_state = open_confirmation_dialog(
            "Line Type Confirmation",
            "Rendering tractograms as tubes may cause performance issues.\n"
            "Do you want to continue?",
            okay_text="Switch to Tube",
            cancel_text="Keep Line",
        )
        if line_type_dialog_state == "okay" and self._requested_line_type is not None:
            self._line_type = self._requested_line_type
            self._requested_line_type = None
            self._apply_line_change_next_frame = True
            self._loader(True, message="Switching line type...")
        elif line_type_dialog_state == "cancel":
            self._requested_line_type = None

        imgui.spacing()
        imgui.spacing()
        imgui.spacing()
        imgui.spacing()

        uploader(
            "Upload BUAN P values",
            callback=self.handle_color_change,
            extension="*.npy",
            selected=self._buan_pvals_file,
            type="buan_pvals",
        )

        imgui.same_line(0, 10)
        close_icon = icons_fontawesome_6.ICON_FA_XMARK
        imgui.text(close_icon)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Reset colors")
        if imgui.is_item_clicked():
            self._buan_pvals_file = False
            self._buan_pvals_data = None
            self.color = self._original_color
            self._draft_color = self._original_color
            self.apply_scene_op(self._create_streamline_actor)
            self.render()

        imgui.same_line(0, 10)
        selected_color = normalize_picker_color(self._draft_color)
        changed, new_color, is_open = color_picker(
            selected_color=selected_color,
            tooltip="Pick a color for the streamlines.",
            popup_id=self._color_picker_popup_id,
        )
        if is_open and not self._color_picker_open:
            self._draft_color = normalize_picker_color(self.color)
        if changed:
            self._draft_color = new_color
        if self._color_picker_open and not is_open:
            if not colors_equal(self._draft_color, self.color):
                self.color = self._draft_color
                self.apply_scene_op(self._create_streamline_actor)
                self.render()
            self._draft_color = self.color
        self._color_picker_open = is_open

        if self._buan_pvals_file is not None and self._buan_pvals_data is not None:
            imgui.spacing()
            changed, (hue_low, hue_high) = two_disk_slider(
                "Hue",
                (self._hue_low, self._hue_high),
                0.0,
                1.0,
                text_format=".1f",
                step=0.1,
            )
            if changed:
                self._hue_low = hue_low
                self._hue_high = hue_high
                self._update_buan_colors_on_sliders()

            imgui.spacing()
            changed, (sat_low, sat_high) = two_disk_slider(
                "Saturation",
                (self._saturation_low, self._saturation_high),
                0.0,
                1.0,
                text_format=".1f",
                step=0.01,
            )
            if changed:
                self._saturation_low = sat_low
                self._saturation_high = sat_high
                self._update_buan_colors_on_sliders()

            imgui.spacing()
            changed, value = thin_slider(
                "Value",
                self._value,
                0.0,
                1.0,
                text_format=".1f",
                step=0.01,
            )
            if changed:
                self._value = value
                self._update_buan_colors_on_sliders()


class ClusterStreamline3D(Visualization):
    """Clustered tractography layer that groups streamlines with QuickBundlesX.

    Renders one centroid tube per cluster; clusters can be expanded to show
    their member streamlines, selected, hidden, and filtered by size or
    length. Clustering runs in a background thread unless
    ``async_clustering`` is False.

    Parameters
    ----------
    name : str
        Display name used in the Skyline UI.
    sft : StatefulTractogram
        Tractogram whose streamlines are clustered.
    thr : float
        Initial clustering distance threshold, in mm.
    line_type : str, optional
        The type of line to render ("Line" or "Tube") for expanded clusters.
    render_callback : callable, optional
        Callback used to request a render/update.
    switch_render_callback : callable, optional
        Callback invoked to switch back to the non-clustered rendering mode.
    loader : callable, optional
        Callback function to show/hide loader during asynchronous operations.
    size_threshold : int, optional
        Minimum number of streamlines in a cluster to be visible. If None,
        it is set to 10.
    length_threshold : float, optional
        Minimum length of streamlines in a cluster to be visible. If None,
        it is set to 20.0.
    async_clustering : bool, optional
        Whether to perform clustering asynchronously. Set to False to block
        until clustering completes (used in stealth mode).
    """

    def __init__(
        self,
        name,
        sft,
        thr,
        *,
        line_type="Line",
        render_callback=None,
        switch_render_callback=None,
        loader=None,
        size_threshold=None,
        length_threshold=None,
        async_clustering=True,
    ):
        """Initialize the clustered tractography layer.

        Parameters
        ----------
        name : str
            Display name used in the Skyline UI.
        sft : StatefulTractogram
            Tractogram whose streamlines are clustered.
        thr : float
            Initial clustering distance threshold, in mm.
        line_type : str, optional
            The type of line to render ("Line" or "Tube") for expanded
            clusters.
        render_callback : callable, optional
            Callback used to request a render/update.
        switch_render_callback : callable, optional
            Callback invoked to switch back to the non-clustered rendering
            mode.
        loader : callable, optional
            Callback function to show/hide loader during asynchronous
            operations.
        size_threshold : int, optional
            Minimum number of streamlines in a cluster to be visible. If
            None, it is set to 10.
        length_threshold : float, optional
            Minimum length of streamlines in a cluster to be visible. If
            None, it is set to 20.0.
        async_clustering : bool, optional
            Whether to perform clustering asynchronously. Set to False to
            block until clustering completes (used in stealth mode).
        """
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
        self._loader = loader
        self._is_clustering = False
        self._queued_recluster = False
        self._async_clustering = async_clustering
        self.size = size_threshold if size_threshold is not None else 10
        self.length = length_threshold if length_threshold is not None else 20.0
        super().__init__(name, render_callback=render_callback)
        self._perform_clustering()

    def _perform_clustering(self):
        """Recompute clusters for the current threshold, sync or async.

        Coalesces overlapping requests: if clustering is already running,
        the call is queued and re-run once the in-flight clustering
        finishes.
        """
        if self._is_clustering:
            self._queued_recluster = True
            return

        self._is_clustering = True
        if self._loader is not None:
            self._loader(True, message="Clustering streamlines...")
            self.render()

        if not self._async_clustering:
            result = self._compute_clustering_data(self.thr)
            self._apply_clustering_result(result, None)
            return

        run_async(
            self._compute_clustering_data,
            self._apply_clustering_result,
            self.thr,
        )

    def _compute_clustering_data(self, thr):
        """Cluster the streamlines with QuickBundlesX at the given threshold.

        Parameters
        ----------
        thr : float
            Final clustering distance threshold, in mm, passed to
            :func:`~dipy.segment.clustering.qbx_and_merge` as the last of
            the thresholds ``[40, 30, 25, 20, thr]``.

        Returns
        -------
        clusters : ClusterMapCentroid
            Cluster map returned by ``qbx_and_merge``.
        lengths : ndarray
            Centroid streamline length for each cluster.
        sizes : ndarray
            Number of streamlines in each cluster.
        colormap : list
            Per-cluster RGB color from ``distinguishable_colormap``.
        line_widths : ndarray
            Per-cluster centroid tube radius interpolated from ``sizes``.
        """
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
        """Rebuild cluster actors from ``_compute_clustering_data``'s result.

        Follows the :func:`~dipy.viz.skyline.compute.run_async` callback
        contract: called on the main thread with the return value of the
        clustered function and any exception it raised.

        Parameters
        ----------
        result : tuple
            Clustering outputs ``(clusters, lengths, sizes, colormap,
            line_widths)`` from :meth:`_compute_clustering_data`.
        exception : Exception or None
            Exception raised while clustering, if any. When set, the
            previous cluster actors are left unchanged and the error is
            logged.
        """
        self._is_clustering = False

        if exception is not None:
            logger.error(f"Error clustering streamlines: {exception}")
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
        """Show or hide each cluster actor based on the size/length filters."""
        for centroid_rep, state in self._cluster_state.items():
            is_visible = state["size"] >= self.size and state["length"] >= self.length
            if state["expanded"] and state["cluster_actor"] is not None:
                state["cluster_actor"].visible = is_visible
            else:
                centroid_rep.visible = is_visible

    # Interaction methods
    def _create_cluster_streamlines(self, centroid_rep):
        """Build the expanded line/tube actor for one cluster's streamlines.

        Parameters
        ----------
        centroid_rep : Actor
            Centroid tube actor keying the cluster in ``_cluster_state``.

        Returns
        -------
        centroid_rep : Actor
            The same centroid actor passed in.
        streamline_actor : Actor
            The line or tube actor for the cluster's member streamlines.
        """
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
        """Return the centroid actors that are selected but not expanded.

        Returns
        -------
        list of Actor
            Centroid actors satisfying both conditions.
        """
        return [
            centroid_rep
            for centroid_rep, state in self._cluster_state.items()
            if state["selected"] and not state["expanded"]
        ]

    def _expand_clusters(self):
        """Replace each selected, unexpanded centroid with its streamlines."""
        selected_clusters = self._selected_unexpanded_clusters()
        if not selected_clusters:
            return

        for centroid_rep in selected_clusters:
            _, streamline_actor = self._create_cluster_streamlines(centroid_rep)
            state = self._cluster_state[centroid_rep]
            self._actor.add(streamline_actor)
            self._actor.remove(centroid_rep)
            state["cluster_actor"] = streamline_actor
            state["expanded"] = True

    def _collapse_clusters(self):
        """Replace each selected, expanded cluster's streamlines with its centroid."""
        for centroid_rep, state in self._cluster_state.items():
            if state["selected"] and state["expanded"]:
                self._actor.add(centroid_rep)
                cluster_actor = state["cluster_actor"]
                if cluster_actor is not None and cluster_actor in self._actor.children:
                    self._actor.remove(cluster_actor)
                state["cluster_actor"] = None
                state["expanded"] = False

    def _select_all_clusters(self):
        """Mark every cluster as selected."""
        for centroid_rep in self._cluster_state:
            self._update_cluster_state(centroid_rep, True)

    def _deselect_all_clusters(self):
        """Mark every cluster as not selected."""
        for centroid_rep in self._cluster_state:
            self._update_cluster_state(centroid_rep, False)

    def _update_cluster_state(self, centroid_rep, selected):
        """Set a cluster's selected flag and update its centroid opacity.

        Parameters
        ----------
        centroid_rep : Actor
            Centroid tube actor keying the cluster in ``_cluster_state``.
        selected : bool
            Whether the cluster should be marked as selected.
        """
        state = self._cluster_state[centroid_rep]
        state["selected"] = selected
        if selected:
            centroid_rep.material.opacity = 1.0
        else:
            centroid_rep.material.opacity = 0.5

    def _toggle_cluster_selection(self, cluster):
        """Flip a cluster's selected flag in response to a pointer-down event.

        Parameters
        ----------
        cluster : Actor
            Centroid tube actor that was clicked.
        """
        self.apply_scene_op(
            self._update_cluster_state,
            cluster,
            not self._cluster_state[cluster]["selected"],
        )

    def _hide_deselected_clusters(self):
        """Hide the centroid actor of every cluster that is not selected."""
        for centroid_rep, state in self._cluster_state.items():
            if not state["selected"]:
                centroid_rep.visible = False

    def _show_all_clusters(self):
        """Make every centroid actor visible, ignoring the size/length filters."""
        for centroid_rep in self._cluster_state:
            centroid_rep.visible = True

    def _show_all_clusters_and_refresh(self):
        """Show every cluster, then reapply the size/length visibility filters."""
        self._show_all_clusters()
        self._refresh_cluster_visibility()

    def _apply_cluster_line_type_change(self):
        """Rebuild every expanded cluster's actor with the current line type."""
        for centroid_rep, state in self._cluster_state.items():
            if state["expanded"]:
                _, new_actor = self._create_cluster_streamlines(centroid_rep)
                self._actor.remove(state["cluster_actor"])
                self._actor.add(new_actor)
                state["cluster_actor"] = new_actor

    def _populate_info(self):
        """Build the streamline/cluster count and size/length summary.

        Returns
        -------
        str
            Multi-line text with the streamline count, cluster count, and
            min/max cluster size and length.
        """
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
        """Build a tractogram containing the streamlines of selected clusters.

        Returns
        -------
        StatefulTractogram
            Tractogram with the streamlines from every cluster whose
            ``selected`` state is True, in the same space as ``sft``.
        """
        visible_streamlines = []
        for state in self._cluster_state.values():
            if state["selected"]:
                cluster_idx = state["cluster"]
                cluster_streamlines = self._clusters[cluster_idx]
                visible_streamlines.extend(cluster_streamlines)
        return StatefulTractogram.from_sft(visible_streamlines, self.sft)

    def save_tractogram(self, filenames, *, rois=None, shm_coeffs=None):
        """Save the selected clusters' streamlines to a file.

        Matches the shared download-callback signature used across Skyline
        visualizations; ``rois`` and ``shm_coeffs`` are accepted but unused.

        Parameters
        ----------
        filenames : list of str
            Selected save path(s) from the file dialog; only the first
            entry is used.
        rois : list of str or None, optional
            Unused by this visualization.
        shm_coeffs : list of str or None, optional
            Unused by this visualization.
        """
        if filenames:
            if isinstance(filenames, (list, tuple)):
                filenames = filenames[0]
            visible_sft = self.compute_visible_tractogram()
            save_tractogram(visible_sft, filenames, bbox_valid_check=False)

    def handle_key_events(self, event):
        """Expand, collapse, select, deselect, hide, or show clusters by key.

        Recognizes ``"e"`` (expand), ``"c"`` (collapse), ``"a"`` (select
        all), ``"d"`` (deselect all), ``"h"`` (hide deselected), and
        ``"s"`` (show all) on ``event.key``.

        Parameters
        ----------
        event : Event
            Interaction event from the renderer callback.
        """
        if event.key == "e":
            self.apply_scene_op(self._expand_clusters)
        elif event.key == "c":
            self.apply_scene_op(self._collapse_clusters)
        elif event.key == "a":
            self.apply_scene_op(self._select_all_clusters)
        elif event.key == "d":
            self.apply_scene_op(self._deselect_all_clusters)
        elif event.key == "h":
            self.apply_scene_op(self._hide_deselected_clusters)
        elif event.key == "s":
            self.apply_scene_op(self._show_all_clusters_and_refresh)

    @property
    def actor(self):
        """Return the group containing every centroid and cluster actor.

        Returns
        -------
        Group
            Container actor holding one child per cluster (a centroid tube,
            or its expanded streamline actor).
        """
        return self._actor

    def render_widgets(self):
        """Draw the line-type, threshold, size/length, and download controls."""
        changed, is_clustered = toggle_button(True, label="Cluster")
        if changed and self._switch_render_callback is not None:
            self._switch_render_callback(self, is_clustered)

        imgui.spacing()

        changed, new = segmented_switch("Line Type", ["Line", "Tube"], self._line_type)
        if changed:
            self._line_type = new.title()
            self.apply_scene_op(self._apply_cluster_line_type_change)

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
            self.apply_scene_op(self._refresh_cluster_visibility)

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
            self.apply_scene_op(self._refresh_cluster_visibility)

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


if not has_fury_v2:
    (
        create_cluster_help,
        create_streamline_visualization,
        create_streamline,
        Streamline3D,
        ClusterStreamline3D,
    ) = (fury,) * 5
