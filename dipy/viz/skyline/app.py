from pathlib import Path

from fury.actor import Actor
from fury.colormap import distinguishable_colormap
from fury.io import load_image_as_wgpu_texture_view

from dipy.viz.skyline.UI.manager import UIWindow
from dipy.viz.skyline.UI.theme import LOGO
from dipy.viz.skyline.render.image import Image3D
from dipy.viz.skyline.render.peak import Peak3D
from dipy.viz.skyline.render.renderer import create_window
from dipy.viz.skyline.render.roi import ROI3D
from dipy.viz.skyline.render.surface import Surface


class Skyline:
    def __init__(
        self,
        visualizer_type="standalone",
        images=None,
        peaks=None,
        rois=None,
        surfaces=None,
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

        self.visualizations = []
        gpu_texture = load_image_as_wgpu_texture_view(str(LOGO), self.window.device)
        logo_tex_ref = self.window._imgui.backend.register_texture(gpu_texture)

        self.UI_window = UIWindow(
            "Image Controls",
            size=self.ui_size,
            render_callback=self.before_render,
            logo_tex_ref=logo_tex_ref,
        )
        self.window.resize_callback(self.handle_resize)
        self._color_gen = distinguishable_colormap()

        if images is not None:
            for idx, (img, affine, path) in enumerate(images):
                fname = Path(path).name if path is not None else f"Image {idx}"
                image3d = Image3D(
                    fname,
                    img,
                    affine=affine,
                    render_callback=self.before_render,
                    interpolation="linear",
                )
                self.visualizations.append(image3d)
                self.UI_window.add(fname, image3d.renderer)
        if peaks is not None:
            for idx, (pam, path) in enumerate(peaks):
                fname = Path(path).name if path is not None else f"Peaks {idx}"
                peak3d = Peak3D(
                    fname,
                    pam.peak_dirs,
                    affine=pam.affine,
                    render_callback=self.before_render,
                )
                self.visualizations.append(peak3d)
                self.UI_window.add(fname, peak3d.renderer)
        if rois is not None:
            for idx, (roi, affine, path) in enumerate(rois):
                color = next(self._color_gen)
                fname = Path(path).name if path is not None else f"ROI {idx}"
                roi3d = ROI3D(
                    fname,
                    roi,
                    affine=affine,
                    color=color,
                    render_callback=self.before_render,
                )
                self.visualizations.append(roi3d)
                self.UI_window.add(fname, roi3d.renderer)
        if surfaces is not None:
            for idx, (verts, faces, path) in enumerate(surfaces):
                color = next(self._color_gen)
                fname = Path(path).name if path is not None else f"Surface {idx}"
                surface3d = Surface(
                    fname,
                    verts,
                    faces,
                    color=color,
                    render_callback=self.before_render,
                )
                self.visualizations.append(surface3d)
                self.UI_window.add(fname, surface3d.renderer)
        self.window._imgui.set_gui(self.UI_window.render)
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


def skyline(
    *, visualizer_type="standalone", images=None, peaks=None, rois=None, surfaces=None
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
    """
    Skyline(
        visualizer_type=visualizer_type,
        images=images,
        peaks=peaks,
        rois=rois,
        surfaces=surfaces,
    )
