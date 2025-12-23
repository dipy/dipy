from dipy.io.image import load_nifti
from dipy.viz.skyline.UI.manager import UIWindow
from dipy.viz.skyline.render.renderer import create_window
from dipy.viz.skyline.render.slicer import Slicer


class Skyline:
    def __init__(self, visualizer_type="standalone", images=None):
        self.windows = []
        self.windows.append(create_window(visualizer_type=visualizer_type))

        self.UI_window = UIWindow("Slicer Controls")

        if images is not None:
            for img, affine, _ in images:
                slicer = Slicer(
                    img,
                    affine=affine,
                    render_callback=self.windows[0].render,
                    interpolation="nearest",
                )
                self.windows[0].screens[0].scene.add(slicer.actor)
                self.UI_window.add("HARDI Slicer", slicer.render_widgets)
        self.windows[0]._imgui.set_gui(self.UI_window.render)
        self.windows[0].start()


if __name__ == "__main__":
    file_path = "~/.dipy/stanford_hardi/HARDI150.nii.gz"
    img, affine = load_nifti(file_path)
    print("Affine:\n", affine)
    print("Data shape: ", img.shape)
    skyline = Skyline(
        visualizer_type="standalone", images=[(img[..., 0], affine, file_path)]
    )
