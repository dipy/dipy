from pathlib import Path

from fury import update_camera
from fury.actor import Actor
from fury.window import update_viewports

from dipy.viz.skyline.UI.manager import UIWindow
from dipy.viz.skyline.render.image import Image3D
from dipy.viz.skyline.render.renderer import create_window


class Skyline:
    def __init__(self, visualizer_type="standalone", images=None):
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
        self.visualizations = [[]]

        self.UI_window = UIWindow("Image Controls", size=self.ui_size)
        self.windows[0].resize_callback(self.handle_resize)

        if images is not None:
            for idx, (img, affine, path) in enumerate(images):
                image3d = Image3D(
                    img,
                    affine=affine,
                    render_callback=self.before_render,
                    interpolation="linear",
                )
                self.visualizations[0].append(image3d)
                fname = Path(path).name if path is not None else f"Image {idx}"
                self.UI_window.add(fname, image3d.render_widgets)
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
        update_viewports(
            self.windows[0].screens,
            [
                (0, 0, self.ui_size[0], size[1]),
                (self.ui_size[0], 0, size[0] - self.ui_size[0], size[1]),
            ],
        )
        self.UI_window.size = (self.ui_size[0], size[1])
        camera = self.windows[0].screens[0].camera
        scene = self.windows[0].screens[0].scene
        update_camera(camera, None, scene)
        self.windows[0].render()
