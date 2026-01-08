from pathlib import Path

from fury.lib import EventType
from fury.window import update_viewports

from dipy.viz.skyline.UI.manager import UIWindow
from dipy.viz.skyline.render.renderer import create_window
from dipy.viz.skyline.render.slicer import Slicer


class Skyline:
    def __init__(self, visualizer_type="standalone", images=None):
        self.windows = []
        size = (1200, 1000)
        ui_size = (400, size[1])
        self.windows.append(
            create_window(
                visualizer_type=visualizer_type,
                size=size,
                screen_config=[
                    (0, 0, ui_size[0], ui_size[1]),
                    (ui_size[0], 0, size[0] - ui_size[0], size[1]),
                ],
            )
        )

        self.UI_window = UIWindow("Slicer Controls", size=ui_size)
        self.windows[0].renderer.add_event_handler(
            lambda event: self.handle_resize((event.width, event.height)),
            EventType.RESIZE,
        )

        if images is not None:
            for idx, (img, affine, path) in enumerate(images):
                slicer = Slicer(
                    img,
                    affine=affine,
                    render_callback=self.windows[0].render,
                    interpolation="linear",
                )
                self.windows[0].screens[1].scene.add(slicer.actor)
                fname = Path(path).name if path is not None else f"Image {idx}"
                self.UI_window.add(fname, slicer.render_widgets)
        self.windows[0]._imgui.set_gui(self.UI_window.render)
        self.windows[0].start()

    def handle_resize(self, size):
        update_viewports(
            self.windows[0].screens,
            [(0, 0, 400, size[1]), (400, 0, size[0] - 400, size[1])],
        )
        self.UI_window.size = (400, size[1])
        self.windows[0].render()
