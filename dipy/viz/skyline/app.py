from pathlib import Path

from fury.actor import Actor
from fury.colormap import distinguishable_colormap

from dipy.viz.skyline.UI.manager import UIWindow
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
        self.windows = []
        self.size = (1200, 1000)
        self.ui_size = (400, self.size[1])
        self.windows.append(
            create_window(
                visualizer_type=visualizer_type,
                size=self.size,
                screen_config=[
                    (self.ui_size[0], 0, self.size[0] - self.ui_size[0], self.size[1]),
                ],
            )
        )
        self.visualizations = [{"images": [], "peaks": [], "rois": [], "surfaces": []}]

        self.UI_window = UIWindow("Image Controls", size=self.ui_size)
        self.windows[0].resize_callback(self.handle_resize)
        self._color_gen = distinguishable_colormap()

        if images is not None:
            for idx, (img, affine, path) in enumerate(images):
                image3d = Image3D(
                    img,
                    affine=affine,
                    render_callback=self.before_render,
                    interpolation="linear",
                )
                self.visualizations[0]["images"].append(image3d)
                fname = Path(path).name if path is not None else f"Image {idx}"
                self.UI_window.add(fname, image3d.render_widgets)
        if peaks is not None:
            for idx, (pam, path) in enumerate(peaks):
                peak3d = Peak3D(
                    pam.peak_dirs,
                    affine=pam.affine,
                    render_callback=self.before_render,
                )
                self.visualizations[0]["peaks"].append(peak3d)
                fname = Path(path).name if path is not None else f"Peaks {idx}"
                self.UI_window.add(fname, peak3d.render_widgets)
        if rois is not None:
            for idx, (roi, affine, path) in enumerate(rois):
                color = next(self._color_gen)
                roi3d = ROI3D(
                    roi,
                    affine=affine,
                    color=color,
                    render_callback=self.before_render,
                )
                self.visualizations[0]["rois"].append(roi3d)
                fname = Path(path).name if path is not None else f"ROI {idx}"
                self.UI_window.add(fname, roi3d.render_widgets)
        if surfaces is not None:
            for idx, (verts, faces, path) in enumerate(surfaces):
                color = next(self._color_gen)
                surface3d = Surface(
                    verts,
                    faces,
                    color=color,
                    render_callback=self.before_render,
                )
                self.visualizations[0]["surfaces"].append(surface3d)
                fname = Path(path).name if path is not None else f"Surface {idx}"
                self.UI_window.add(fname, surface3d.render_widgets)
        self.windows[0]._imgui.set_gui(self.UI_window.render)
        self.before_render()
        self.windows[0].start()

    def _refresh_actors(self):
        for actor in list(self.windows[0].screens[0].scene.main_scene.children):
            if not isinstance(actor, Actor):
                continue
            if not any(viz.actor == actor for viz in self.visualizations[0]):
                self.windows[0].screens[0].scene.main_scene.remove(actor)
        for viz in self.visualizations[0]:
            if viz.actor not in self.windows[0].screens[0].scene.main_scene.children:
                self.windows[0].screens[0].scene.main_scene.add(viz.actor)

    def before_render(self):
        self._refresh_actors()
        self.windows[0].render()

    def handle_resize(self, size):
        self.size = size
        self.ui_size = (400, self.size[1])
        self.UI_window.size = (self.ui_size[0], size[1])
        self.windows[0]._screen_config = [
            (self.ui_size[0], 0, self.size[0] - self.ui_size[0], self.size[1])
        ]
        self.UI_window.size = (self.ui_size[0], size[1])
        self.windows[0].render()
