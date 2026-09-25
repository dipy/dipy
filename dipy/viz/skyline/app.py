"""Skyline viewer application entry points.

Expose the ``Skyline`` class and the ``skyline``/``skyline_from_files``
functions used to construct and launch the FURY-based multi-modal viewer.
"""

import os
import time

import numpy as np

from dipy.io.utils import split_filename_extension
from dipy.utils.logging import logger
from dipy.utils.optpkg import optional_package
from dipy.viz.skyline.UI.manager import UIWindow
from dipy.viz.skyline.UI.theme import LOGO_SMALL
from dipy.viz.skyline.compute import process_async_callbacks, run_async
from dipy.viz.skyline.io import load_files

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
    from fury.actor import Actor, show_slices
    from fury.colormap import distinguishable_colormap
    from fury.io import load_image_as_wgpu_texture_view
    from fury.window import update_camera

    from dipy.viz.skyline.render.image import Image3D, create_image_visualization
    from dipy.viz.skyline.render.peak import Peak3D, create_peak_visualization
    from dipy.viz.skyline.render.renderer import create_window
    from dipy.viz.skyline.render.roi import ROI3D, create_roi_visualization
    from dipy.viz.skyline.render.sh_slicer import SHGlyph3D, create_shm_visualization
    from dipy.viz.skyline.render.streamline import (
        ClusterStreamline3D,
        Streamline3D,
        create_cluster_help,
        create_streamline_visualization,
    )
    from dipy.viz.skyline.render.surface import Surface, create_surface_visualization
else:
    actor = fury


class Skyline:
    """The Skyline viewer, hosting the FURY scene, UI, and visualizations.

    Parameters
    ----------
    visualizer_type : {"standalone", "gui", "jupyter", "stealth"}, optional
        Kind of window to create. The options map to FURY window types via
        ``create_window``:

        - "standalone": a default interactive window.
        - "gui": a Qt-based window.
        - "jupyter": an inline Jupyter notebook window.
        - "stealth": an offscreen window with no GUI, used for scripted
          snapshots.

        An unrecognized value logs an error and terminates the process.
    images : list of tuple, optional
        Already-loaded image data to show at startup, as ``(data, affine)``
        or ``(data, affine, filename)`` tuples.
    peaks : list of tuple, optional
        Already-loaded peak data to show at startup, as ``(pam,)`` or
        ``(pam, filename)`` tuples, where ``pam`` is a ``PeaksAndMetrics``.
    rois : list of tuple, optional
        Already-loaded ROI data to show at startup, as ``(roi, affine)`` or
        ``(roi, affine, filename)`` tuples.
    surfaces : list of tuple, optional
        Already-loaded surface data to show at startup, as
        ``(vertices, faces)`` or ``(vertices, faces, filename)`` tuples.
    tractograms : list of tuple, optional
        Already-loaded tractogram data to show at startup, as ``(sft,)`` or
        ``(sft, filename)`` tuples, where ``sft`` is a
        ``StatefulTractogram``. Entries with no streamlines are skipped
        with a warning.
    sh_coeffs : list of tuple, optional
        Already-loaded spherical harmonic coefficient data to show at
        startup, as ``(coeffs, affine)``, ``(coeffs, affine, filename)`` or
        ``(coeffs, affine, filename, basis_type)`` tuples. ``coeffs`` must
        be a 4D ndarray, otherwise the entry is skipped with a warning.
    is_cluster : bool, optional
        Whether to cluster the tractograms.
    is_light_version : bool, optional
        Whether to render tractograms as ``"Line"`` instead of ``"Tube"``,
        which improves performance for large tractograms.
    glass_brain : bool, optional
        Whether to render surfaces black with the ``"basic"`` material at
        25% opacity and default the background to white.
    bg_color : tuple of float, optional
        Background color of the scene as an RGB tuple in ``[0, 1]``. If
        None, it is white when ``glass_brain`` is True, otherwise dark
        gray.
    tract_colors : str or tuple of float or None, optional
        Coloring scheme for the tractograms: ``"direction"`` for
        directionally colored streamlines, ``"random"`` for the next color
        from a distinguishable colormap per tractogram, an RGB(A) tuple in
        ``[0, 1]``, or a string of three space-separated numbers parsed to
        such a tuple. If None, ``"direction"`` is used.
    cluster_thr : float, optional
        Final distance threshold, in mm, used by ``qbx_and_merge`` when
        clustering is enabled; small-animal data may need a smaller value
        such as 2.0.
    cluster_size_thr : int, optional
        Clusters with size less than ``cluster_size_thr`` are hidden. If
        None, the 50th percentile of the cluster size distribution is
        used.
    cluster_length_thr : float, optional
        Clusters with average length less than ``cluster_length_thr`` mm
        are hidden. If None, the 25th percentile of the cluster length
        distribution is used.
    buan_pvals : str, optional
        File path for BUAN p-values used for BUAN-based coloring of
        tractograms.
    rgb : bool or None, optional
        ``None``: auto-detect from structured NIfTI ``DT_RGB24``
        dtype; show toggle for other 4D volumes with 3 or 4 channels.
        ``True``: force RGB mode.  ``False``: never treat as RGB.
    initial_filenames : list of str, optional
        File paths loaded asynchronously into the viewer on startup. If
        neither preloaded data nor initial files are given and a UI
        exists, the file dialog opens on start.
    initial_rois : list of str, optional
        ROI file paths loaded asynchronously into the viewer on startup.
    initial_shm_coeffs : list of str, optional
        Spherical harmonic coefficient file paths loaded asynchronously
        into the viewer on startup.
    out_dir : str or Path, optional
        Directory for the stealth-mode output image; created if missing.
        Used only when ``visualizer_type`` is ``"stealth"``.
    out_stealth_png : str, optional
        Output image name, without extension, used as the stealth window
        title. Used only when ``visualizer_type`` is ``"stealth"``.
    """

    def __init__(
        self,
        *,
        visualizer_type="standalone",
        images=None,
        peaks=None,
        rois=None,
        surfaces=None,
        tractograms=None,
        sh_coeffs=None,
        is_cluster=False,
        is_light_version=False,
        glass_brain=False,
        bg_color=None,
        tract_colors=None,
        cluster_thr=15.0,
        cluster_size_thr=None,
        cluster_length_thr=None,
        buan_pvals=None,
        rgb=None,
        initial_filenames=None,
        initial_rois=None,
        initial_shm_coeffs=None,
        out_dir=None,
        out_stealth_png=None,
    ):
        """Initialize the Skyline viewer.

        Blocks in ``self.window.start()`` until the window is closed.

        Parameters
        ----------
        visualizer_type : {"standalone", "gui", "jupyter", "stealth"}, optional
            Kind of window to create. The options map to FURY window types via
            ``create_window``:

            - "standalone": a default interactive window.
            - "gui": a Qt-based window.
            - "jupyter": an inline Jupyter notebook window.
            - "stealth": an offscreen window with no GUI, used for scripted
              snapshots.

            An unrecognized value logs an error and terminates the process.
        images : list of tuple, optional
            Already-loaded image data to show at startup, as ``(data, affine)``
            or ``(data, affine, filename)`` tuples.
        peaks : list of tuple, optional
            Already-loaded peak data to show at startup, as ``(pam,)`` or
            ``(pam, filename)`` tuples, where ``pam`` is a ``PeaksAndMetrics``.
        rois : list of tuple, optional
            Already-loaded ROI data to show at startup, as ``(roi, affine)`` or
            ``(roi, affine, filename)`` tuples.
        surfaces : list of tuple, optional
            Already-loaded surface data to show at startup, as
            ``(vertices, faces)`` or ``(vertices, faces, filename)`` tuples.
        tractograms : list of tuple, optional
            Already-loaded tractogram data to show at startup, as ``(sft,)`` or
            ``(sft, filename)`` tuples, where ``sft`` is a
            ``StatefulTractogram``. Entries with no streamlines are skipped
            with a warning.
        sh_coeffs : list of tuple, optional
            Already-loaded spherical harmonic coefficient data to show at
            startup, as ``(coeffs, affine)``, ``(coeffs, affine, filename)`` or
            ``(coeffs, affine, filename, basis_type)`` tuples. ``coeffs`` must
            be a 4D ndarray, otherwise the entry is skipped with a warning.
        is_cluster : bool, optional
            Whether to cluster the tractograms.
        is_light_version : bool, optional
            Whether to render tractograms as ``"Line"`` instead of ``"Tube"``,
            which improves performance for large tractograms.
        glass_brain : bool, optional
            Whether to render surfaces black with the ``"basic"`` material at
            25% opacity and default the background to white.
        bg_color : tuple of float, optional
            Background color of the scene as an RGB tuple in ``[0, 1]``. If
            None, it is white when ``glass_brain`` is True, otherwise dark
            gray.
        tract_colors : str or tuple of float or None, optional
            Coloring scheme for the tractograms: ``"direction"`` for
            directionally colored streamlines, ``"random"`` for the next color
            from a distinguishable colormap per tractogram, an RGB(A) tuple in
            ``[0, 1]``, or a string of three space-separated numbers parsed to
            such a tuple. If None, ``"direction"`` is used.
        cluster_thr : float, optional
            Final distance threshold, in mm, used by ``qbx_and_merge`` when
            clustering is enabled; small-animal data may need a smaller value
            such as 2.0.
        cluster_size_thr : int, optional
            Clusters with size less than ``cluster_size_thr`` are hidden. If
            None, the 50th percentile of the cluster size distribution is
            used.
        cluster_length_thr : float, optional
            Clusters with average length less than ``cluster_length_thr`` mm
            are hidden. If None, the 25th percentile of the cluster length
            distribution is used.
        buan_pvals : str, optional
            File path for BUAN p-values used for BUAN-based coloring of
            tractograms.
        rgb : bool or None, optional
            ``None``: auto-detect from structured NIfTI ``DT_RGB24``
            dtype; show toggle for other 4D volumes with 3 or 4 channels.
            ``True``: force RGB mode.  ``False``: never treat as RGB.
        initial_filenames : list of str, optional
            File paths loaded asynchronously into the viewer on startup. If
            neither preloaded data nor initial files are given and a UI
            exists, the file dialog opens on start.
        initial_rois : list of str, optional
            ROI file paths loaded asynchronously into the viewer on startup.
        initial_shm_coeffs : list of str, optional
            Spherical harmonic coefficient file paths loaded asynchronously
            into the viewer on startup.
        out_dir : str or Path, optional
            Directory for the stealth-mode output image; created if missing.
            Used only when ``visualizer_type`` is ``"stealth"``.
        out_stealth_png : str, optional
            Output image name, without extension, used as the stealth window
            title. Used only when ``visualizer_type`` is ``"stealth"``.
        """
        self.size = (1200, 1000)
        self.ui_size = (400, self.size[1])
        self._visualizer_type = visualizer_type
        self._direct_load = True
        self._rgb = rgb
        self._cluster_thr = cluster_thr
        self._cluster_size_thr = cluster_size_thr
        self._cluster_length_thr = cluster_length_thr
        self._buan_pvals = buan_pvals
        if self._visualizer_type != "stealth":
            os.environ["FURY_OFFSCREEN"] = "0"
            self.window = create_window(
                visualizer_type=self._visualizer_type,
                size=self.size,
                screen_config=[
                    (self.ui_size[0], 0, self.size[0] - self.ui_size[0], self.size[1]),
                ],
            )
        else:
            os.environ["FURY_OFFSCREEN"] = "1"
            title = "DIPY SKYLINE"
            if out_stealth_png is not None:
                title = split_filename_extension(out_stealth_png)[0]
            if out_dir is not None and out_dir != "":
                os.makedirs(out_dir, exist_ok=True)
                title = os.path.join(out_dir, title)
            self.window = create_window(
                visualizer_type=self._visualizer_type, size=self.size, title=title
            )
        if bg_color is None:
            bg_color = (1, 1, 1) if glass_brain else (0.1, 0.1, 0.1)
        self._bg_color = bg_color
        self.window.screens[0].scene.background = self._bg_color

        if tract_colors is None:
            tract_colors = "direction"
        elif isinstance(tract_colors, str) and len(tract_colors.split(" ")) == 3:
            tract_colors = tuple(map(float, tract_colors.split(" ")))
        self._tract_colors = tract_colors

        self._image_visualizations = []
        self._peak_visualizations = []
        self._roi_visualizations = []
        self._surface_visualizations = []
        self._tractogram_visualizations = []
        self._sh_glyph_visualizations = []
        self._pending_loaded_files = []
        self._pending_tractogram_switches = []
        self._loading_total = 0
        self._loading_done = 0
        self._is_drawing_ui = False
        self._refresh_requested = False
        self._pending_bg_color = None
        self._pending_sync_requests = []
        self._pending_scene_ops = []
        self._is_cluster = is_cluster
        self._is_light_version = is_light_version
        self._glass_brain = glass_brain
        self._tractogram_help = False
        self.window.renderer.add_event_handler(self.handle_key_events, "key_down")
        self.window.resize_callback(self.handle_resize)
        self._color_gen = distinguishable_colormap()
        self.active_image = None
        self._slice_focus_viz = None

        if self._visualizer_type != "stealth":
            gpu_texture = load_image_as_wgpu_texture_view(
                str(LOGO_SMALL), self.window.device
            )
            logo_tex_ref = self.window._imgui.backend.register_texture(gpu_texture)
            self.UI_window = UIWindow(
                "Image Controls",
                size=self.ui_size,
                render_callback=self.request_refresh,
                logo_tex_ref=logo_tex_ref,
                file_dialog_callback=self._append_visualization,
                bg_color_callback=self._update_background_color,
                snapshot_callback=self._save_snapshot,
            )
            self.window._imgui.set_gui(self.draw_ui)
        else:
            self.UI_window = None

        initial_loaded_files = {
            "images": images or [],
            "peaks": peaks or [],
            "rois": rois or [],
            "surfaces": surfaces or [],
            "tractograms": tractograms or [],
            "shm_coeffs": sh_coeffs or [],
        }
        has_initial_visualizations = any(initial_loaded_files.values())
        has_initial_files = any((initial_filenames, initial_rois, initial_shm_coeffs))

        if has_initial_visualizations:
            self._queue_loaded_visualizations(initial_loaded_files)

        if has_initial_files:
            self._append_visualization(
                filenames=initial_filenames,
                rois=initial_rois,
                shm_coeffs=initial_shm_coeffs,
            )
        elif not has_initial_visualizations and self.UI_window is not None:
            self.UI_window.request_file_dialog = True

        if self._visualizer_type == "stealth":
            self._wait_for_loading_in_stealth_mode()

        self.before_render()
        self._direct_load = False
        self.window.start()

    def _wait_for_loading_in_stealth_mode(self):
        """Block until all queued and pending visualizations finish loading.

        Repeatedly runs the async callback queue and drains pending loaded
        files so stealth-mode snapshots are not taken before every
        requested visualization has been created.
        """
        while self._pending_loaded_files or (
            self._loading_total > 0 and self._loading_done < self._loading_total
        ):
            process_async_callbacks()
            self._drain_pending_visualizations()
            time.sleep(0.01)

    def _refresh_actors(self):
        """Sync the main scene's actors with the current visualizations.

        Removes actors that no longer belong to any visualization and adds
        actors for visualizations not yet present in the scene.
        """
        all_actors = [v.actor for v in self.visualizations]

        for actor in list(self.window.screens[0].scene.main_scene.children):
            if not isinstance(actor, Actor):
                continue
            if not any(a == actor for a in all_actors):
                self.window.screens[0].scene.main_scene.remove(actor)
        for a in all_actors:
            if a not in self.window.screens[0].scene.main_scene.children:
                self.window.screens[0].scene.main_scene.add(a)

    def _refresh_ui(self):
        """Drop tracked visualizations whose UI section was closed.

        A visualization removed from the UI (for example via its close
        button) no longer has a matching section id, so it is unregistered
        from the viewer as well.
        """
        for viz in self.visualizations:
            viz_id = f"{viz.path}:{viz.name}"
            if viz_id not in self.UI_window.sections:
                self._remove_visualization(viz)

    def _arrange_image_actors(self):
        """Stagger overlapping image slicer actors to avoid z-fighting.

        Restores the previously active image to its base slice state, then
        offsets the newly active image slightly further along its slice
        axis for each additional loaded image.
        """
        for viz in self._image_visualizations:
            if viz.active:
                show_slices(
                    self.active_image.actor,
                    self.active_image.state,
                )
                self.active_image = viz
        show_slices(
            self.active_image.actor,
            self.active_image.state + (len(self._image_visualizations) * 0.005),
        )

    def _update_tractogram_helper(self, *, remove=False):
        """Show or hide the cluster interaction help overlay.

        The overlay is added when a ``ClusterStreamline3D`` visualization is
        present and removed once none remain, or immediately when
        ``remove`` is True.

        Parameters
        ----------
        remove : bool, optional
            Whether to force-remove the overlay regardless of its current
            visualizations.
        """
        if remove and self._tractogram_help:
            self.window.screens[0].scene.remove(self._tractogram_help)
            self._tractogram_help = False

        if (
            any(
                isinstance(viz, ClusterStreamline3D)
                for viz in self._tractogram_visualizations
            )
            and not self._tractogram_help
        ):
            self._tractogram_help = create_cluster_help(
                position=(self.size[0] - self.ui_size[0] - 200, 0)
            )
            self.window.screens[0].scene.add(self._tractogram_help)
        elif (
            not any(
                isinstance(viz, ClusterStreamline3D)
                for viz in self._tractogram_visualizations
            )
            and self._tractogram_help
        ):
            self.window.screens[0].scene.remove(self._tractogram_help)
            self._tractogram_help = False

    def draw_ui(self):
        """Draw the ImGui overlay for a single frame.

        Invoked as the ImGui GUI callback. Renders the UI window, then
        drains pending tractogram switches, visualizations, synchronization
        requests, and scene operations queued while drawing, applying a
        pending background color change and refreshing if required.
        """
        process_async_callbacks()
        self._is_drawing_ui = True
        try:
            if len(self.visualizations) == 0 and self._loading_total == 0:
                self.UI_window.request_file_dialog = True
            else:
                self.UI_window.request_file_dialog = False
            self.UI_window.render()
        finally:
            self._is_drawing_ui = False

        self._process_tractogram_switches()
        self._drain_pending_visualizations()
        self._flush_pending_sync_requests()
        self._flush_pending_scene_ops()
        if self._pending_bg_color is not None:
            self.window.screens[0].scene.background = self._pending_bg_color
            self._bg_color = self._pending_bg_color
            self._pending_bg_color = None
            self._refresh_requested = True
        self.active_image and self._arrange_image_actors()
        if self._refresh_requested:
            self.before_render()

    def request_refresh(self):
        """Flag the viewer for a refresh on the next UI frame.

        The actual actor sync and render happen later, either at the end of
        the current ``draw_ui`` call or on the next ``before_render`` call.
        """
        self._refresh_requested = True

    def _scene_op_key(self, func):
        """Build a stable key for deferred scene operation coalescing.

        Parameters
        ----------
        func : callable
            Deferred scene operation callable.

        Returns
        -------
        tuple or None
            Comparable key for the operation, or None when unavailable.
        """
        method = getattr(func, "__func__", None)
        owner = getattr(func, "__self__", None)
        if method is not None and owner is not None:
            return (id(owner), method.__name__)
        name = getattr(func, "__name__", None)
        if name is not None:
            return (None, name)
        return None

    def enqueue_scene_op(self, func, *args, **kwargs):
        """Run or defer a scene-mutating callable.

        Runs ``func`` immediately unless the UI is currently drawing, in
        which case the call is queued for ``_flush_pending_scene_ops`` and
        coalesced with any previously queued call sharing the same bound
        method or function name.

        Parameters
        ----------
        func : callable
            Scene-mutating callable to run or defer.
        *args : tuple
            Positional arguments forwarded to ``func``.
        **kwargs : dict
            Keyword arguments forwarded to ``func``.
        """
        if self._is_drawing_ui:
            op_key = self._scene_op_key(func)
            if op_key is not None:
                for idx in range(len(self._pending_scene_ops) - 1, -1, -1):
                    old_func, _, _ = self._pending_scene_ops[idx]
                    if self._scene_op_key(old_func) == op_key:
                        self._pending_scene_ops[idx] = (func, args, kwargs)
                        self.request_refresh()
                        return
            self._pending_scene_ops.append((func, args, kwargs))
            self.request_refresh()
            return
        func(*args, **kwargs)
        self.request_refresh()

    def _perform_refresh(self):
        """Update the cluster helper, UI bookkeeping, and scene actors.

        Does not render; callers that need the change visible must follow
        up with ``_render_window``.
        """
        if self._visualizer_type != "stealth":
            self._update_tractogram_helper()
            self._refresh_ui()
        self._refresh_actors()

    def _perform_refresh_and_render(self):
        """Perform a refresh, clear the refresh flag, and render the window."""
        self._perform_refresh()
        self._refresh_requested = False
        self._render_window()

    def _render_window(self):
        """Render the window, unless the UI is currently being drawn.

        Rendering while ``draw_ui`` is running is skipped because the
        surrounding ImGui frame already triggers a render.
        """
        if self._is_drawing_ui:
            return
        self.window.render()

    def _queue_loaded_visualizations(self, loaded_files, *, message="Loading Files..."):
        """Queue a batch of already-loaded visualization data for creation.

        The batch is consumed later by ``_drain_pending_visualizations``.

        Parameters
        ----------
        loaded_files : dict
            Mapping with the ``"images"``, ``"peaks"``, ``"rois"``,
            ``"surfaces"``, ``"tractograms"``, and ``"shm_coeffs"`` keys,
            each holding a list of loaded-data tuples.
        message : str, optional
            Message text shown to the user.
        """
        self._pending_loaded_files.append(loaded_files)
        self._loading_total += 1
        self._loading_done += 1
        self.loader(True, message=message)

    def _flush_pending_sync_requests(self):
        """Apply state-synchronization requests queued while drawing the UI.

        Requests are re-checked at flush time in case the source
        visualization had synchronization toggled off while queued.
        """
        if not self._pending_sync_requests:
            return
        pending = self._pending_sync_requests.copy()
        self._pending_sync_requests.clear()
        for source_viz, new_state in pending:
            # Re-check source sync at flush time: user may have toggled it off
            # while the request was queued.
            self._synchronize_visualizations_from_source(source_viz, new_state)
        self.active_image and self._arrange_image_actors()
        self._refresh_requested = True

    def _flush_pending_scene_ops(self):
        """Apply scene operations queued while drawing the UI.

        Each queued callable is invoked with its stored arguments; failures
        are logged rather than propagated so one broken operation does not
        block the others.
        """
        if not self._pending_scene_ops:
            return
        pending = self._pending_scene_ops.copy()
        self._pending_scene_ops.clear()
        for func, args, kwargs in pending:
            try:
                func(*args, **kwargs)
            except Exception as e:  # noqa: BLE001
                logger.exception(
                    "Failed to apply deferred scene operation %s: %s",
                    getattr(func, "__qualname__", repr(func)),
                    e,
                )
        self._refresh_requested = True

    def _get_reference_slice_state(self):
        """Return a snapshot of the current slice pose for load-time alignment.

        Returns
        -------
        np.ndarray or list or tuple or None
            Snapshot of slice state from an existing synchronizable visualization,
            or None when no such visualization exists (first load).
        """
        if self.active_image is not None:
            return self._snapshot_state(
                np.asarray(self.active_image.state, dtype=float)
            )
        if self._slice_focus_viz is not None:
            if self._slice_focus_viz not in self.visualizations:
                self._slice_focus_viz = None
            else:
                return self._snapshot_state(
                    np.asarray(self._slice_focus_viz.state, dtype=float)
                )
        for viz in reversed(self.visualizations):
            if isinstance(viz, (Image3D, Peak3D, SHGlyph3D)):
                return self._snapshot_state(np.asarray(viz.state, dtype=float))
        return None

    def _apply_reference_slice_state_to_new_visualizations(
        self, reference_state, n_img_before, n_peak_before, n_sh_before
    ):
        """Apply a pre-load slice pose to visualizations created in this batch.

        Parameters
        ----------
        reference_state : array-like or None
            Snapshot from ``_get_reference_slice_state`` before loading, or None.
        n_img_before : int
            Length of ``_image_visualizations`` before ``_load_visualiations``.
        n_peak_before : int
            Length of ``_peak_visualizations`` before ``_load_visualiations``.
        n_sh_before : int
            Length of ``_sh_glyph_visualizations`` before ``_load_visualiations``.
        """
        if reference_state is None:
            return
        new_visualizations = (
            self._image_visualizations[n_img_before:]
            + self._peak_visualizations[n_peak_before:]
            + self._sh_glyph_visualizations[n_sh_before:]
        )
        for viz in new_visualizations:
            viz.update_state(reference_state)

    def _drain_pending_visualizations(self):
        """Consume one queued batch of loaded files into visualizations.

        Pops the oldest pending batch, creates its visualizations while
        preserving the current slice pose, refreshes the scene actors,
        recenters the camera on the updated scene bounds, and hides the
        loader once every queued batch has been consumed.
        """
        if self._pending_loaded_files:
            loaded_files = self._pending_loaded_files.pop(0)
            n_img_before = len(self._image_visualizations)
            n_peak_before = len(self._peak_visualizations)
            n_sh_before = len(self._sh_glyph_visualizations)
            reference_slice_state = self._get_reference_slice_state()
            self._load_visualiations(
                loaded_files["images"],
                loaded_files["peaks"],
                loaded_files["rois"],
                loaded_files["surfaces"],
                loaded_files["tractograms"],
                loaded_files["shm_coeffs"],
                is_cluster=loaded_files.get("is_cluster_override"),
                async_clustering=loaded_files.get("async_clustering_override"),
            )
            self._apply_reference_slice_state_to_new_visualizations(
                reference_slice_state,
                n_img_before,
                n_peak_before,
                n_sh_before,
            )

            if self._visualizer_type != "stealth":
                self._update_tractogram_helper()

            self._refresh_actors()
            if (
                self.window.screens[0].scene.main_scene.get_world_bounding_sphere()
                is not None
            ):
                update_camera(
                    self.window.screens[0].camera,
                    None,
                    self.window.screens[0].scene,
                )

        if (
            self._loading_total > 0
            and self._loading_done >= self._loading_total
            and not self._pending_loaded_files
        ):
            self.loader(False)
            self._loading_total = 0
            self._loading_done = 0

    def before_render(self):
        """Refresh and render, or defer to a request if mid-UI-draw.

        Called after the constructor's initial load and whenever a change
        needs to be shown outside of the ``draw_ui`` frame callback.
        """
        if self._is_drawing_ui:
            self.request_refresh()
            return
        self._perform_refresh_and_render()

    def handle_resize(self, size):
        """Update cached layout state after the window is resized.

        Registered as the window's resize callback.

        Parameters
        ----------
        size : tuple of int
            New window size, in pixels, as ``(width, height)``.
        """
        self.size = size
        self.ui_size = (400, self.size[1])
        self.UI_window.size = (self.ui_size[0], size[1])
        self.window._screen_config = [
            (self.ui_size[0], 0, self.size[0] - self.ui_size[0], self.size[1])
        ]
        self.UI_window.size = (self.ui_size[0], size[1])
        self._update_tractogram_helper(remove=True)
        self._render_window()

    def handle_key_events(self, event):
        """Forward a key event to clustered tractogram visualizations.

        Registered as the renderer's ``"key_down"`` event handler.

        Parameters
        ----------
        event : Event
            Interaction event from the renderer callback.
        """
        for viz in self._tractogram_visualizations:
            if isinstance(viz, ClusterStreamline3D):
                viz.handle_key_events(event)

    def _add_visualization(self, viz):
        """Register a visualization in its per-type list and in the UI.

        Skips registration and logs a warning if a visualization with the
        same path/name id is already present.

        Parameters
        ----------
        viz : Visualization
            Visualization instance to register.

        Raises
        ------
        TypeError
            If ``viz`` is not an instance of a supported visualization type.
        """
        viz_id = f"{viz.path}:{viz.name}"
        if self.UI_window is not None and viz_id in self.UI_window.sections:
            logger.warning(
                f"Visualization with id '{viz_id}' already exists. Skipping."
            )
            return
        if isinstance(viz, Image3D):
            self._image_visualizations.append(viz)
        elif isinstance(viz, Peak3D):
            self._peak_visualizations.append(viz)
        elif isinstance(viz, ROI3D):
            self._roi_visualizations.append(viz)
        elif isinstance(viz, Surface):
            self._surface_visualizations.append(viz)
        elif isinstance(viz, (Streamline3D, ClusterStreamline3D)):
            self._tractogram_visualizations.append(viz)
        elif isinstance(viz, SHGlyph3D):
            self._sh_glyph_visualizations.append(viz)
        else:
            raise TypeError("Unsupported visualization type")
        viz._scene_op_callback = self.enqueue_scene_op
        if self.UI_window is not None:
            self.UI_window.add(viz_id, viz.renderer, viz_type=viz.viz_type)

    def _load_visualiations(
        self,
        images,
        peaks,
        rois,
        surfaces,
        tractograms,
        sh_coeffs,
        *,
        is_cluster=None,
        async_clustering=None,
    ):
        """Create and register visualizations from batches of loaded data.

        Each argument is a list of loaded-data tuples in the shape produced
        by ``io.load_files`` (see ``Skyline`` for the tuple forms). Sets the
        last loaded image active and opens the file dialog if no
        visualization ends up loaded.

        Parameters
        ----------
        images : list of tuple, optional
            Loaded image data, as ``(data, affine)`` or
            ``(data, affine, filename)`` tuples.
        peaks : list of tuple, optional
            Loaded peak data, as ``(pam,)`` or ``(pam, filename)`` tuples.
        rois : list of tuple, optional
            Loaded ROI data, as ``(roi, affine)`` or
            ``(roi, affine, filename)`` tuples.
        surfaces : list of tuple, optional
            Loaded surface data, as ``(vertices, faces)`` or
            ``(vertices, faces, filename)`` tuples.
        tractograms : list of tuple, optional
            Loaded tractogram data, as ``(sft,)`` or ``(sft, filename)``
            tuples. Entries with no streamlines are skipped with a warning.
        sh_coeffs : list of tuple, optional
            Loaded spherical harmonic coefficient data, as
            ``(coeffs, affine)``, ``(coeffs, affine, filename)`` or
            ``(coeffs, affine, filename, basis_type)`` tuples. Entries whose
            ``coeffs`` is not a 4D ndarray are skipped with a warning.
        is_cluster : bool, optional
            Overrides ``self._is_cluster`` for the tractograms in this batch.
        async_clustering : bool, optional
            Overrides the default async-clustering choice for the
            tractograms in this batch.
        """
        for idx, input in enumerate(images or []):
            image3d = create_image_visualization(
                input,
                idx,
                render_callback=self.request_refresh,
                sync_callabck=self._synchronize_visualizations,
                rgb=self._rgb,
            )
            self._add_visualization(image3d)
        for idx, input in enumerate(peaks or []):
            peak3d = create_peak_visualization(
                input,
                idx,
                render_callback=self.request_refresh,
                sync_callabck=self._synchronize_visualizations,
            )
            self._add_visualization(peak3d)
        for idx, input in enumerate(rois or []):
            color = next(self._color_gen)
            roi3d = create_roi_visualization(
                input,
                idx,
                color=color,
                render_callback=self.request_refresh,
            )
            self._add_visualization(roi3d)
        for idx, input in enumerate(surfaces or []):
            color = next(self._color_gen) if not self._glass_brain else (0, 0, 0)
            opacity = 25 if self._glass_brain else 100
            surface3d = create_surface_visualization(
                input,
                idx,
                color=color,
                material="basic" if self._glass_brain else "phong",
                opacity=opacity,
                render_callback=self.request_refresh,
            )
            self._add_visualization(surface3d)
        for idx, input in enumerate(tractograms or []):
            if isinstance(input, tuple):
                sft = input[0]
                filename = f"Streamlines {idx}"
                if len(input) == 2:
                    filename = input[1]
                if len(sft.streamlines) <= 0:
                    logger.warning(
                        f"The provide file: {filename} does not "
                        "contain any streamlines."
                    )
                    continue
            tractogram3d = create_streamline_visualization(
                input,
                idx,
                is_cluster=is_cluster if is_cluster is not None else self._is_cluster,
                thr=self._cluster_thr,
                line_type="Line" if self._is_light_version else "Tube",
                render_callback=self.request_refresh,
                colormap=self._color_gen,
                tract_colors=self._tract_colors,
                switch_render_callback=self._update_tractogram_rendering,
                loader=self.loader,
                size_threshold=self._cluster_size_thr,
                length_threshold=self._cluster_length_thr,
                buan_pvals_file=self._buan_pvals,
                async_clustering=(
                    async_clustering
                    if async_clustering is not None
                    else self._direct_load and self._visualizer_type != "stealth"
                ),
            )
            self._add_visualization(tractogram3d)
        for idx, input in enumerate(sh_coeffs or []):
            if isinstance(input, tuple):
                coeffs = input[0]
                filename = f"ODFs {idx}"
                if len(input) >= 3:
                    filename = input[2]
                if not isinstance(coeffs, np.ndarray) or len(coeffs.shape) != 4:
                    logger.warning(
                        f"The provide file: {filename} does not "
                        "contain any SH coefficients or is not a 4D array."
                    )
                    continue
            sh3d = create_shm_visualization(
                input,
                idx,
                render_callback=self.request_refresh,
                scale=1.0,
                l_max=8,
                sync_callback=self._synchronize_visualizations,
            )
            self._add_visualization(sh3d)

        if self._image_visualizations:
            self._image_visualizations[-1].active = True
            self.active_image = self._image_visualizations[-1]
            self._arrange_image_actors()

        if len(self.visualizations) == 0 and self.UI_window is not None:
            self.UI_window.request_file_dialog = True

    def _append_visualization(self, *, filenames=None, rois=None, shm_coeffs=None):
        """Load files from disk asynchronously and queue them for display.

        Each path is loaded in its own background task via
        ``io.load_files``; each completed task's result is appended to
        ``self._pending_loaded_files`` for ``_drain_pending_visualizations``
        to consume on a later frame.

        Parameters
        ----------
        filenames : list of str, optional
            Paths to images, peaks, surfaces, or tractograms to load.
        rois : list of str, optional
            Paths to ROI files to load.
        shm_coeffs : list of str, optional
            Paths to spherical harmonic coefficient files to load.
        """
        total_files = len(filenames or []) + len(rois or []) + len(shm_coeffs or [])
        if total_files == 0:
            return

        self._loading_total = total_files
        self._loading_done = 0

        def load_files_task(filenames, rois, shm_coeffs):
            return load_files(filenames, rois=rois, shm_coeffs=shm_coeffs)

        def on_files_loaded(loaded_files, exception):
            self._loading_done += 1
            if exception is None and loaded_files is not None:
                self._pending_loaded_files.append(loaded_files)

        self.loader(True, message="Loading Files...")
        for filename in filenames or []:
            run_async(
                load_files_task,
                on_files_loaded,
                filenames=[filename],
                rois=[],
                shm_coeffs=[],
            )
        for roi in rois or []:
            run_async(
                load_files_task,
                on_files_loaded,
                filenames=[],
                rois=[roi],
                shm_coeffs=[],
            )
        for shm in shm_coeffs or []:
            run_async(
                load_files_task,
                on_files_loaded,
                filenames=[],
                rois=[],
                shm_coeffs=[shm],
            )

    def _remove_visualization(self, viz):
        """Unregister a visualization from its per-type tracking list.

        Also clears ``self._slice_focus_viz`` if it pointed at ``viz`` and
        opens the file dialog if no visualization remains.

        Parameters
        ----------
        viz : Visualization
            Visualization instance to unregister.

        Raises
        ------
        TypeError
            If ``viz`` is not an instance of a supported visualization type.
        """
        if isinstance(viz, Image3D):
            self._image_visualizations.remove(viz)
        elif isinstance(viz, Peak3D):
            self._peak_visualizations.remove(viz)
        elif isinstance(viz, ROI3D):
            self._roi_visualizations.remove(viz)
        elif isinstance(viz, Surface):
            self._surface_visualizations.remove(viz)
        elif isinstance(viz, (Streamline3D, ClusterStreamline3D)):
            self._tractogram_visualizations.remove(viz)
        elif isinstance(viz, SHGlyph3D):
            self._sh_glyph_visualizations.remove(viz)
        else:
            raise TypeError("Unsupported visualization type")

        if viz is self._slice_focus_viz:
            self._slice_focus_viz = None

        if len(self.visualizations) == 0 and self.UI_window is not None:
            self.UI_window.request_file_dialog = True

    @staticmethod
    def _snapshot_state(new_state):
        """Copy a slice state so later mutation cannot affect the snapshot.

        Parameters
        ----------
        new_state : array-like
            New synchronized state for this visualization.

        Returns
        -------
        np.ndarray
            The snapshot state of the visualization.
        """
        if hasattr(new_state, "copy"):
            return new_state.copy()
        if isinstance(new_state, list):
            return list(new_state)
        if isinstance(new_state, tuple):
            return tuple(new_state)
        return new_state

    def _synchronize_visualizations_from_source(self, source_viz, new_state):
        """Push a new slice state from ``source_viz`` to other visualizations.

        Parameters
        ----------
        source_viz : Visualization
            Visualization whose state change is being propagated.
        new_state : array-like
            New synchronized state for this visualization.
        """
        # Source-side guard: only push if this view has sync enabled.
        if not getattr(source_viz, "_synchronize", True):
            return

        for viz in self.visualizations:
            if viz is not source_viz and isinstance(viz, (Image3D, Peak3D, SHGlyph3D)):
                # Target-side guard is inside each viz's update_state.
                viz.update_state(new_state)

    def _synchronize_visualizations(self, source_viz, new_state):
        """Propagate a slice-state change reported by a visualization.

        Called by visualizations as their state-change callback. Updates
        the slice-focus visualization, then either queues the request for
        ``_flush_pending_sync_requests`` when the UI is mid-draw, or
        propagates it immediately otherwise.

        Parameters
        ----------
        source_viz : Visualization
            Visualization whose state change is being reported.
        new_state : array-like
            New synchronized state for this visualization.
        """
        if not getattr(source_viz, "_synchronize", True):
            return

        if isinstance(source_viz, (Image3D, Peak3D, SHGlyph3D)):
            self._slice_focus_viz = source_viz

        new_state = self._snapshot_state(new_state)

        if self._is_drawing_ui:
            self._pending_sync_requests.append((source_viz, new_state))
            self.request_refresh()
            return
        self._synchronize_visualizations_from_source(source_viz, new_state)
        self.active_image and self._arrange_image_actors()

    def _update_background_color(self, new_color):
        """Apply or defer a background color change from the UI.

        Parameters
        ----------
        new_color : tuple of float
            New scene background color as an RGB tuple in ``[0, 1]``.
        """
        if self._is_drawing_ui:
            self._pending_bg_color = new_color
            self.request_refresh()
            return
        self._bg_color = new_color
        self.window.screens[0].scene.background = self._bg_color
        self._render_window()

    def _process_tractogram_switches(self):
        """Apply queued clustered/unclustered tractogram mode switches.

        For each queued switch, removes the old visualization from the
        scene and UI, then asynchronously re-creates it in the requested
        mode via a deferred ``run_async`` call.
        """
        if not self._pending_tractogram_switches:
            return
        pending = self._pending_tractogram_switches.copy()
        self._pending_tractogram_switches.clear()
        for viz, is_clustered in pending:
            if viz not in self._tractogram_visualizations:
                continue
            viz_id = f"{viz.path}:{viz.name}"
            sft = viz.sft
            path = viz.path

            if self.UI_window is not None:
                self.UI_window.remove(viz_id)
            # Remove the old visualization object immediately so mode switches
            # replace state instead of coexisting under the same UI id.
            self._remove_visualization(viz)
            self.request_refresh()

            self._loading_total = 1
            self._loading_done = 0

            def _delay():
                pass

            def _on_delay_done(
                _, exception, *, _sft=sft, _path=path, _is_clustered=is_clustered
            ):
                self._loading_done += 1
                self._pending_loaded_files.append(
                    {
                        "images": [],
                        "peaks": [],
                        "rois": [],
                        "surfaces": [],
                        "tractograms": [(_sft, _path)],
                        "shm_coeffs": [],
                        "is_cluster_override": _is_clustered,
                        "async_clustering_override": True,
                    }
                )

            run_async(_delay, _on_delay_done)

    def _update_tractogram_rendering(self, streamline_viz, is_clustered):
        """Queue a clustered/unclustered mode switch for a tractogram.

        Parameters
        ----------
        streamline_viz : Visualization
            Streamline visualization whose rendering mode changed.
        is_clustered : bool
            Whether the visualization should switch to clustered mode.
        """
        for viz in self._tractogram_visualizations:
            if viz is streamline_viz and isinstance(
                viz, (Streamline3D, ClusterStreamline3D)
            ):
                self._pending_tractogram_switches.append((viz, is_clustered))
                break

    def loader(self, show, *, message=None):
        """Show or hide the UI's loading indicator.

        Parameters
        ----------
        show : bool
            Whether to show the UI element/loader.
        message : str, optional
            Message text shown to the user.
        """
        if self.UI_window is not None:
            self.UI_window.update_loader(show=show, message=message)

    def _save_snapshot(self, snapshot_path):
        """Save a snapshot image using the ShowManager.

        Parameters
        ----------
        snapshot_path : str
            Target file path where the PNG snapshot will be saved.
        """
        _, extension = split_filename_extension(snapshot_path)
        if extension == "":
            snapshot_path = f"{snapshot_path}.png"

        logger.info(f"Saving snapshot to {snapshot_path}")
        self.enqueue_scene_op(self.window.snapshot, fname=snapshot_path)

    @property
    def visualizations(self):
        """Return every visualization currently tracked by the viewer.

        Returns
        -------
        list
            The list of visualizations in the Skyline viewer.
        """
        return (
            self._image_visualizations
            + self._peak_visualizations
            + self._roi_visualizations
            + self._surface_visualizations
            + self._tractogram_visualizations
            + self._sh_glyph_visualizations
        )


def skyline_from_files(
    fnames,
    *,
    rois=None,
    shm_coeffs=None,
    is_cluster=False,
    is_light_version=False,
    glass_brain=False,
    bg_color=None,
    tract_colors=None,
    cluster_thr=15.0,
    cluster_size_thr=None,
    cluster_length_thr=None,
    buan_pvals=None,
    stealth=False,
    rgb=None,
    out_dir=None,
    out_stealth_png=None,
):
    """Launch the Skyline GUI from file paths.

    Loads every path in the background and constructs the corresponding
    ``Skyline`` viewer, forwarding ``fnames``/``rois``/``shm_coeffs`` as
    ``initial_filenames``/``initial_rois``/``initial_shm_coeffs``.

    Parameters
    ----------
    fnames : list of str
        File paths to be loaded into the Skyline viewer.

        Supported file types include:

        - NIfTI images (.nii, .nii.gz)
        - Peaks (.pam5)
        - Surfaces (.pial, .gii, .gii.gz)
        - Tractograms (.trk, .trx, .dpy, .tck, .vtk, .vtp, .fib)

        Unsupported extensions are logged and skipped; ``.npy`` entries are
        ignored.
    rois : list of str, optional
        File paths for ROIs to be loaded into the Skyline viewer. Only
        NIfTI images (.nii, .nii.gz) are supported; other extensions are
        logged and skipped.
    shm_coeffs : list of str, optional
        File paths for spherical harmonics coefficients to be loaded into
        the Skyline viewer. Only ``.pam5`` files are supported; other
        extensions are silently skipped.
    is_cluster : bool, optional
        Whether to cluster the tractograms.
    is_light_version : bool, optional
        Whether to render tractograms as ``"Line"`` instead of ``"Tube"``,
        which improves performance for large tractograms.
    glass_brain : bool, optional
        Whether to render surfaces black with the ``"basic"`` material at
        25% opacity and default the background to white.
    bg_color : tuple of float, optional
        Background color of the scene as an RGB tuple in ``[0, 1]``. If
        None, it is white when ``glass_brain`` is True, otherwise dark
        gray.
    tract_colors : str or tuple of float or None, optional
        Coloring scheme for the tractograms: ``"direction"`` for
        directionally colored streamlines, ``"random"`` for the next color
        from a distinguishable colormap per tractogram, an RGB(A) tuple in
        ``[0, 1]``, or a string of three space-separated numbers parsed to
        such a tuple. If None, ``"direction"`` is used.
    cluster_thr : float, optional
        Final distance threshold, in mm, used by ``qbx_and_merge`` when
        clustering is enabled; small-animal data may need a smaller value
        such as 2.0.
    cluster_size_thr : int, optional
        Clusters with size less than ``cluster_size_thr`` are hidden. If
        None, the 50th percentile of the cluster size distribution is
        used.
    cluster_length_thr : float, optional
        Clusters with average length less than ``cluster_length_thr`` mm
        are hidden. If None, the 25th percentile of the cluster length
        distribution is used.
    buan_pvals : str, optional
        File path for BUAN p-values used for BUAN-based coloring of
        tractograms.
    stealth : bool, optional
        Whether to render offscreen and save a snapshot instead of opening
        an interactive window; sets ``visualizer_type`` to ``"stealth"``.
    rgb : bool or None, optional
        ``None``: auto-detect from structured NIfTI ``DT_RGB24``
        dtype; show toggle for other 4D volumes with 3 or 4 channels.
        ``True``: force RGB mode.  ``False``: never treat as RGB.
    out_dir : str or Path, optional
        Directory for the stealth-mode output image; created if missing.
        Used only when ``stealth`` is True.
    out_stealth_png : str, optional
        Output image name, without extension, used as the stealth window
        title. Used only when ``stealth`` is True.

    Returns
    -------
    Skyline
        The constructed viewer, returned once construction returns from
        its blocking ``self.window.start()`` call.
    """
    visualizer_type = "stealth" if stealth else "standalone"

    return skyline(
        visualizer_type=visualizer_type,
        initial_filenames=fnames,
        initial_rois=rois,
        initial_shm_coeffs=shm_coeffs,
        is_cluster=is_cluster,
        is_light_version=is_light_version,
        glass_brain=glass_brain,
        bg_color=bg_color,
        tract_colors=tract_colors,
        cluster_thr=cluster_thr,
        cluster_size_thr=cluster_size_thr,
        cluster_length_thr=cluster_length_thr,
        buan_pvals=buan_pvals,
        rgb=rgb,
        out_dir=out_dir,
        out_stealth_png=out_stealth_png,
    )


def skyline(
    *,
    visualizer_type="standalone",
    images=None,
    peaks=None,
    rois=None,
    surfaces=None,
    tractograms=None,
    sh_coeffs=None,
    is_cluster=False,
    is_light_version=False,
    glass_brain=False,
    bg_color=None,
    tract_colors=None,
    cluster_thr=15.0,
    cluster_size_thr=None,
    cluster_length_thr=None,
    buan_pvals=None,
    rgb=None,
    initial_filenames=None,
    initial_rois=None,
    initial_shm_coeffs=None,
    out_dir=None,
    out_stealth_png=None,
):
    """Launch the Skyline GUI.

    Constructs and returns a ``Skyline`` viewer with the given data.

    Parameters
    ----------
    visualizer_type : {"standalone", "gui", "jupyter", "stealth"}, optional
        Kind of window to create. The options map to FURY window types via
        ``create_window``:

        - "standalone": a default interactive window.
        - "gui": a Qt-based window.
        - "jupyter": an inline Jupyter notebook window.
        - "stealth": an offscreen window with no GUI, used for scripted
          snapshots.

        An unrecognized value logs an error and terminates the process.
    images : list of tuple, optional
        Already-loaded image data to show at startup, as ``(data, affine)``
        or ``(data, affine, filename)`` tuples.
    peaks : list of tuple, optional
        Already-loaded peak data to show at startup, as ``(pam,)`` or
        ``(pam, filename)`` tuples, where ``pam`` is a ``PeaksAndMetrics``.
    rois : list of tuple, optional
        Already-loaded ROI data to show at startup, as ``(roi, affine)`` or
        ``(roi, affine, filename)`` tuples.
    surfaces : list of tuple, optional
        Already-loaded surface data to show at startup, as
        ``(vertices, faces)`` or ``(vertices, faces, filename)`` tuples.
    tractograms : list of tuple, optional
        Already-loaded tractogram data to show at startup, as ``(sft,)`` or
        ``(sft, filename)`` tuples, where ``sft`` is a
        ``StatefulTractogram``. Entries with no streamlines are skipped
        with a warning.
    sh_coeffs : list of tuple, optional
        Already-loaded spherical harmonic coefficient data to show at
        startup, as ``(coeffs, affine)``, ``(coeffs, affine, filename)`` or
        ``(coeffs, affine, filename, basis_type)`` tuples. ``coeffs`` must
        be a 4D ndarray, otherwise the entry is skipped with a warning.
    is_cluster : bool, optional
        Whether to cluster the tractograms.
    is_light_version : bool, optional
        Whether to render tractograms as ``"Line"`` instead of ``"Tube"``,
        which improves performance for large tractograms.
    glass_brain : bool, optional
        Whether to render surfaces black with the ``"basic"`` material at
        25% opacity and default the background to white.
    bg_color : tuple of float, optional
        Background color of the scene as an RGB tuple in ``[0, 1]``. If
        None, it is white when ``glass_brain`` is True, otherwise dark
        gray.
    tract_colors : str or tuple of float or None, optional
        Coloring scheme for the tractograms: ``"direction"`` for
        directionally colored streamlines, ``"random"`` for the next color
        from a distinguishable colormap per tractogram, an RGB(A) tuple in
        ``[0, 1]``, or a string of three space-separated numbers parsed to
        such a tuple. If None, ``"direction"`` is used.
    cluster_thr : float, optional
        Final distance threshold, in mm, used by ``qbx_and_merge`` when
        clustering is enabled; small-animal data may need a smaller value
        such as 2.0.
    cluster_size_thr : int, optional
        Clusters with size less than ``cluster_size_thr`` are hidden. If
        None, the 50th percentile of the cluster size distribution is
        used.
    cluster_length_thr : float, optional
        Clusters with average length less than ``cluster_length_thr`` mm
        are hidden. If None, the 25th percentile of the cluster length
        distribution is used.
    buan_pvals : str, optional
        File path for BUAN p-values used for BUAN-based coloring of
        tractograms.
    rgb : bool or None, optional
        ``None``: auto-detect from structured NIfTI ``DT_RGB24``
        dtype; show toggle for other 4D volumes with 3 or 4 channels.
        ``True``: force RGB mode.  ``False``: never treat as RGB.
    initial_filenames : list of str, optional
        File paths loaded asynchronously into the viewer on startup. If
        neither preloaded data nor initial files are given and a UI
        exists, the file dialog opens on start.
    initial_rois : list of str, optional
        ROI file paths loaded asynchronously into the viewer on startup.
    initial_shm_coeffs : list of str, optional
        Spherical harmonic coefficient file paths loaded asynchronously
        into the viewer on startup.
    out_dir : str or Path, optional
        Directory for the stealth-mode output image; created if missing.
        Used only when ``visualizer_type`` is ``"stealth"``.
    out_stealth_png : str, optional
        Output image name, without extension, used as the stealth window
        title. Used only when ``visualizer_type`` is ``"stealth"``.

    Returns
    -------
    Skyline
        The constructed viewer, returned once construction returns from
        its blocking ``self.window.start()`` call.
    """
    return Skyline(
        visualizer_type=visualizer_type,
        images=images,
        peaks=peaks,
        rois=rois,
        surfaces=surfaces,
        tractograms=tractograms,
        sh_coeffs=sh_coeffs,
        is_cluster=is_cluster,
        is_light_version=is_light_version,
        glass_brain=glass_brain,
        bg_color=bg_color,
        tract_colors=tract_colors,
        cluster_thr=cluster_thr,
        cluster_size_thr=cluster_size_thr,
        cluster_length_thr=cluster_length_thr,
        buan_pvals=buan_pvals,
        rgb=rgb,
        initial_filenames=initial_filenames,
        initial_rois=initial_rois,
        initial_shm_coeffs=initial_shm_coeffs,
        out_dir=out_dir,
        out_stealth_png=out_stealth_png,
    )


if not has_fury_v2:
    Skyline = skyline = skyline_from_files = fury
