from fury.actor import Actor, show_slices
from fury.colormap import distinguishable_colormap
from fury.io import load_image_as_wgpu_texture_view

from dipy.viz.skyline.UI.manager import UIWindow
from dipy.viz.skyline.UI.theme import LOGO
from dipy.viz.skyline.io import load_files
from dipy.viz.skyline.render.image import Image3D, create_image_visualization
from dipy.viz.skyline.render.peak import Peak3D, create_peak_visualization
from dipy.viz.skyline.render.renderer import create_window
from dipy.viz.skyline.render.roi import ROI3D, create_roi_visualization
from dipy.viz.skyline.render.streamline import (
    ClusterStreamline3D,
    Streamline3D,
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
        is_cluster=False,
        is_light_version=False,
        glass_brain=False,
        bg_color=None,
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
            bg_color = (1, 1, 1) if glass_brain else (0, 0, 0)
        self.window.screens[0].scene.background = bg_color

        self._image_visualizations = []
        self._peak_visualizations = []
        self._roi_visualizations = []
        self._surface_visualizations = []
        self._tractogram_visualizations = []
        gpu_texture = load_image_as_wgpu_texture_view(str(LOGO), self.window.device)
        logo_tex_ref = self.window._imgui.backend.register_texture(gpu_texture)
        self.window.renderer.add_event_handler(self.handle_key_events, "key_down")

        self.UI_window = UIWindow(
            "Image Controls",
            size=self.ui_size,
            render_callback=self.before_render,
            logo_tex_ref=logo_tex_ref,
        )
        self.window.resize_callback(self.handle_resize)
        self._color_gen = distinguishable_colormap()

        if images is not None:
            for idx, input in enumerate(images):
                image3d = create_image_visualization(
                    input,
                    idx,
                    render_callback=self.before_render,
                )
                self._add_visualization(image3d)
        if peaks is not None:
            for idx, input in enumerate(peaks):
                peak3d = create_peak_visualization(
                    input, idx, render_callback=self.before_render
                )
                self._add_visualization(peak3d)
        if rois is not None:
            for idx, input in enumerate(rois):
                color = next(self._color_gen)
                roi3d = create_roi_visualization(
                    input,
                    idx,
                    color=color,
                    render_callback=self.before_render,
                )
                self._add_visualization(roi3d)
        if surfaces is not None:
            for idx, input in enumerate(surfaces):
                color = next(self._color_gen) if not glass_brain else (0, 0, 0)
                opacity = 25 if glass_brain else 100
                surface3d = create_surface_visualization(
                    input,
                    idx,
                    color=color,
                    material="basic" if glass_brain else "phong",
                    opacity=opacity,
                    render_callback=self.before_render,
                )
                self._add_visualization(surface3d)
        if tractograms is not None:
            for idx, input in enumerate(tractograms):
                tractogram3d = create_streamline_visualization(
                    input,
                    idx,
                    is_cluster=is_cluster,
                    line_type="line" if is_light_version else "tube",
                    render_callback=self.before_render,
                    colormap=self._color_gen,
                )
                self._add_visualization(tractogram3d)
        self.active_image = None
        if self._image_visualizations:
            self._image_visualizations[-1].active = True
            self.active_image = self._image_visualizations[-1]
            self._arrange_image_actors()
        self.window._imgui.set_gui(self.draw_ui)
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
                self.visualizations.remove(viz)

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
            self.active_image.state + (len(self._image_visualizations) * 0.1),
        )

    def draw_ui(self):
        self.UI_window.render()
        self.active_image and self._arrange_image_actors()

    def before_render(self):
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
        else:
            raise ValueError("Unsupported visualization type")
        self.UI_window.add(viz.name, viz.renderer)

    @property
    def visualizations(self):
        return (
            self._image_visualizations
            + self._peak_visualizations
            + self._roi_visualizations
            + self._surface_visualizations
            + self._tractogram_visualizations
        )


def skyline_from_files(
    fnames,
    rois=None,
    is_cluster=False,
    is_light_version=False,
    glass_brain=False,
    bg_color=None,
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
    """
    loaded_files = load_files(fnames, rois=rois)
    return skyline(
        images=loaded_files["images"],
        peaks=loaded_files["peaks"],
        rois=loaded_files["rois"],
        surfaces=loaded_files["surfaces"],
        tractograms=loaded_files["tractograms"],
        is_cluster=is_cluster,
        is_light_version=is_light_version,
        glass_brain=glass_brain,
        bg_color=bg_color,
    )


def skyline(
    *,
    visualizer_type="standalone",
    images=None,
    peaks=None,
    rois=None,
    surfaces=None,
    tractograms=None,
    is_cluster=False,
    is_light_version=False,
    glass_brain=False,
    bg_color=None,
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
    """
    return Skyline(
        visualizer_type=visualizer_type,
        images=images,
        peaks=peaks,
        rois=rois,
        surfaces=surfaces,
        tractograms=tractograms,
        is_cluster=is_cluster,
        is_light_version=is_light_version,
        glass_brain=glass_brain,
        bg_color=bg_color,
    )
