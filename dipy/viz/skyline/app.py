from fury.actor import Actor, show_slices
from fury.colormap import distinguishable_colormap
from fury.io import load_image_as_wgpu_texture_view
from fury.window import update_camera

from dipy.viz.skyline.UI.manager import UIWindow
from dipy.viz.skyline.UI.theme import LOGO
from dipy.viz.skyline.compute import run_async
from dipy.viz.skyline.io import load_files
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


class Skyline:
    def __init__(
        self,
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
        initial_filenames=None,
        initial_rois=None,
        initial_shm_coeffs=None,
    ):
        self.size = (1200, 1000)
        self.ui_size = (400, self.size[1])

        self.window = create_window(
            visualizer_type=visualizer_type,
            size=self.size,
            screen_config=[
                (self.ui_size[0], 0, self.size[0] - self.ui_size[0], self.size[1]),
            ],
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
        self._loading_total = 0
        self._loading_done = 0
        self._is_cluster = is_cluster
        self._is_light_version = is_light_version
        self._glass_brain = glass_brain
        self._tractogram_help = False
        gpu_texture = load_image_as_wgpu_texture_view(str(LOGO), self.window.device)
        logo_tex_ref = self.window._imgui.backend.register_texture(gpu_texture)
        self.window.renderer.add_event_handler(self.handle_key_events, "key_down")

        self.UI_window = UIWindow(
            "Image Controls",
            size=self.ui_size,
            render_callback=self.before_render,
            logo_tex_ref=logo_tex_ref,
            file_dialog_callback=self._append_visualization,
            bg_color_callback=self._update_background_color,
        )
        self.window.resize_callback(self.handle_resize)
        self._color_gen = distinguishable_colormap()

        self.active_image = None
        self.window._imgui.set_gui(self.draw_ui)
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
        elif not has_initial_visualizations:
            self.UI_window.request_file_dialog = True

        self.before_render()
        self.window.start()

    def _refresh_actors(self):
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
        for viz in self.visualizations:
            if viz.name not in self.UI_window.sections:
                self._remove_visualization(viz)

    def _arrange_image_actors(self):
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
        self.UI_window.render()
        self._drain_pending_visualizations()
        self.active_image and self._arrange_image_actors()

    def _queue_loaded_visualizations(self, loaded_files, *, message="Loading Files..."):
        self._pending_loaded_files.append(loaded_files)
        self._loading_total += 1
        self._loading_done += 1
        self.loader(True, message=message)

    def _drain_pending_visualizations(self):
        if self._pending_loaded_files:
            loaded_files = self._pending_loaded_files.pop(0)
            self._load_visualiations(
                loaded_files["images"],
                loaded_files["peaks"],
                loaded_files["rois"],
                loaded_files["surfaces"],
                loaded_files["tractograms"],
                loaded_files["shm_coeffs"],
            )

            if self.active_image is not None:
                self._synchronize_visualizations_from_source(
                    self.active_image, self.active_image.state
                )

            self._update_tractogram_helper()
            self._refresh_actors()
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
        self._update_tractogram_helper()
        self._refresh_ui()
        self._refresh_actors()
        self.window.render()

    def handle_resize(self, size):
        self.size = size
        self.ui_size = (400, self.size[1])
        self.UI_window.size = (self.ui_size[0], size[1])
        self.window._screen_config = [
            (self.ui_size[0], 0, self.size[0] - self.ui_size[0], self.size[1])
        ]
        self.UI_window.size = (self.ui_size[0], size[1])
        self._update_tractogram_helper(remove=True)
        self.window.render()

    def handle_key_events(self, event):
        for viz in self._tractogram_visualizations:
            if isinstance(viz, ClusterStreamline3D):
                viz.handle_key_events(event)

    def _add_visualization(self, viz):
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
            raise ValueError("Unsupported visualization type")
        self.UI_window.add(viz.name, viz.renderer)

    def _load_visualiations(
        self, images, peaks, rois, surfaces, tractograms, sh_coeffs
    ):
        for idx, input in enumerate(images or []):
            image3d = create_image_visualization(
                input,
                idx,
                render_callback=self.before_render,
                sync_callabck=self._synchronize_visualizations,
            )
            self._add_visualization(image3d)
        for idx, input in enumerate(peaks or []):
            peak3d = create_peak_visualization(
                input,
                idx,
                render_callback=self.before_render,
                sync_callabck=self._synchronize_visualizations,
            )
            self._add_visualization(peak3d)
        for idx, input in enumerate(rois or []):
            color = next(self._color_gen)
            roi3d = create_roi_visualization(
                input,
                idx,
                color=color,
                render_callback=self.before_render,
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
                render_callback=self.before_render,
            )
            self._add_visualization(surface3d)
        for idx, input in enumerate(tractograms or []):
            tractogram3d = create_streamline_visualization(
                input,
                idx,
                is_cluster=self._is_cluster,
                line_type="line" if self._is_light_version else "tube",
                render_callback=self.before_render,
                colormap=self._color_gen,
                tract_colors=self._tract_colors,
                switch_render_callback=self._update_tractogram_rendering,
                loader=self.loader,
            )
            self._add_visualization(tractogram3d)
        for idx, input in enumerate(sh_coeffs or []):
            sh3d = create_shm_visualization(
                input,
                idx,
                render_callback=self.before_render,
                scale=1.0,
                l_max=8,
                sync_callback=self._synchronize_visualizations,
            )
            self._add_visualization(sh3d)

        if self._image_visualizations:
            self._image_visualizations[-1].active = True
            self.active_image = self._image_visualizations[-1]
            self._arrange_image_actors()

        if len(self.visualizations) == 0:
            self.UI_window.request_file_dialog = True

    def _append_visualization(self, *, filenames=None, rois=None, shm_coeffs=None):
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
            raise ValueError("Unsupported visualization type")

        if len(self.visualizations) == 0:
            self.UI_window.request_file_dialog = True

    def _synchronize_visualizations_from_source(self, source_viz, new_state):
        for viz in self.visualizations:
            if viz is not source_viz and isinstance(viz, (Image3D, Peak3D, SHGlyph3D)):
                viz.update_state(new_state)

    def _synchronize_visualizations(self, source_viz, new_state):
        self._synchronize_visualizations_from_source(source_viz, new_state)
        self.active_image and self._arrange_image_actors()
        self.window.render()

    def _update_background_color(self, new_color):
        self._bg_color = new_color
        self.window.screens[0].scene.background = self._bg_color
        self.window.render()

    def _update_tractogram_rendering(self, streamline_viz, is_clustered):
        for idx, viz in enumerate(self._tractogram_visualizations):
            if viz is streamline_viz and isinstance(
                viz, (Streamline3D, ClusterStreamline3D)
            ):
                new_viz = create_streamline_visualization(
                    (viz.sft, viz.name),
                    idx,
                    is_cluster=is_clustered,
                    line_type=viz._line_type,
                    render_callback=self.before_render,
                    colormap=self._color_gen,
                    tract_colors=self._tract_colors,
                    switch_render_callback=self._update_tractogram_rendering,
                    loader=self.loader,
                )
                self._tractogram_visualizations[idx] = new_viz
                self.UI_window._sections[viz.name] = new_viz.renderer

    def loader(self, show, *, message=None):
        self.UI_window.update_loader(show=show, message=message)

    @property
    def visualizations(self):
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
):
    """Launch Skyline GUI from files.

    Parameters
    ----------
    fnames : list
        List of file paths to be loaded into the Skyline viewer.
        Supported file types include:
        - NIfTI images (.nii, .nii.gz)
        - Peaks (.pam5)
        - Surfaces (.pial, .gii, .gii.gz)
        - Tractograms (.trx, .trk, .dpy, .tck, .vtk, .vtp, .fib)
    rois : list, optional
        List of file paths for ROIs to be loaded into the Skyline viewer.
        Supported file types include NIfTI images (.nii, .nii.gz).
    shm_coeffs : list, optional
        List of file paths for spherical harmonics coefficients to be loaded into the
        Skyline viewer. Supported file types include .pam5 files containing SH
        coefficients.
    is_cluster : bool, optional
        Whether to cluster the tractograms.
    is_light_version : bool, optional
        Whether to use the light version of the tractogram rendering. This will render
        tractograms as lines instead of tubes, which can improve performance for large
        tractograms.
    glass_brain : bool, optional
        Whether to use glass brain mode. This will overwrite the background color
        to white if not explicitly set by the user.
    bg_color : variable float, optional
        Define the background color of the scene. Colors can be defined with
        3 values and should be between [0-1].
        For example, a value of (0, 0, 0) would mean the black color.
    tract_colors : variable float or str, optional
        Define the colors of the tractograms. Colors can be defined with
        3 values and should be between [0-1].
        String options are 'random' for random colors for each tractogram,
        'direction'  for directionally colored streamlines.
        For example, a value of (1, 0, 0) would mean the red color.
    """
    return skyline(
        initial_filenames=fnames,
        initial_rois=rois,
        initial_shm_coeffs=shm_coeffs,
        is_cluster=is_cluster,
        is_light_version=is_light_version,
        glass_brain=glass_brain,
        bg_color=bg_color,
        tract_colors=tract_colors,
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
    initial_filenames=None,
    initial_rois=None,
    initial_shm_coeffs=None,
):
    """Launch Skyline GUI.

    Parameters
    ----------
    visualizer_type : str, optional
        Type of visualizer to create. The options are:
        - "standalone": A standalone window with full interactivity.
        - "gui": A Qt-based GUI window.
        - "jupyter": An inline Jupyter notebook visualizer.
        - "stealth": An offscreen visualizer without GUI.
    images : list, optional
        List of path for each image to be added to the Skyline viewer.
    peaks : list, optional
        List of path for each peak to be added to the Skyline viewer.
    rois : list, optional
        List of path for each ROI to be added to the Skyline viewer.
    surfaces : list, optional
        List of path for each surface to be added to the Skyline viewer.
    tractograms : list, optional
        List of path for each tractogram to be added to the Skyline viewer.
    is_cluster : bool, optional
        Whether to cluster the tractograms.
    is_light_version : bool, optional
        Whether to use the light version of the tractogram rendering. This will render
        tractograms as lines instead of tubes, which can improve performance for large
        tractograms.
    glass_brain : bool, optional
        Whether to use glass brain mode. This will overwrite the background color
        to white if not explicitly set by the user.
    bg_color : variable float, optional
        Define the background color of the scene. Colors can be defined with
        3 values and should be between [0-1].
        For example, a value of (0, 0, 0) would mean the black color.
    tract_colors : variable float or str, optional
        Define the colors of the tractograms. Colors can be defined with
        3 values and should be between [0-1].
        String options are 'random' for random colors for each tractogram,
        'direction'  for directionally colored streamlines.
        For example, a value of (1, 0, 0) would mean the red color.
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
        initial_filenames=initial_filenames,
        initial_rois=initial_rois,
        initial_shm_coeffs=initial_shm_coeffs,
    )
