from pathlib import Path

from dipy.viz.skyline.UI.manager import UIWindow
from dipy.viz.skyline.render.renderer import create_window
from dipy.viz.skyline.render.slicer import Slicer


class Skyline:
    def __init__(self, visualizer_type="standalone", images=None):
        self.windows = []
        self.windows.append(create_window(visualizer_type=visualizer_type))

        self.UI_window = UIWindow("Slicer Controls")

        if images is not None:
            for idx, (img, affine, path) in enumerate(images):
                slicer = Slicer(
                    img,
                    affine=affine,
                    render_callback=self.windows[0].render,
                    interpolation="linear",
                )
                self.windows[0].screens[0].scene.add(slicer.actor)
                fname = Path(path).stem if path is not None else f"Image {idx}"
                self.UI_window.add(fname, slicer.render_widgets)
        self.windows[0]._imgui.set_gui(self.UI_window.render)
        self.windows[0].start()
