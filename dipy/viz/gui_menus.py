import os

from dipy.utils.optpkg import optional_package

# Allow import, but disable doctests if we don't have vtk.
from dipy.viz.gui import UI
from dipy.viz.gui_2d import Panel2D, FileSelect2D, Text2D, TextBox2D

vtk, have_vtk, setup_module = optional_package('vtk')

if have_vtk:
    vtkInteractorStyleUser = vtk.vtkInteractorStyleUser
    version = vtk.vtkVersion.GetVTKSourceVersion().split(' ')[-1]
    major_version = vtk.vtkVersion.GetVTKMajorVersion()
else:
    vtkInteractorStyleUser = object

numpy_support, have_ns, _ = optional_package('vtk.util.numpy_support')


class FileSaveMenu(UI):
    def __init__(self, size, position):
        """

        Parameters
        ----------
        size: (float, float)
        position: (float, float)
        """
        super(FileSaveMenu, self).__init__()
        self.center = position
        self.size = size
        self.panel = Panel2D(center=self.center, size=self.size, color=(0, 0, 0.9), opacity=0.9)  # type: Panel2D
        self.file_select = FileSelect2D(size=(self.size[0] * 0.9, self.size[1] * 0.6), font_size=12, position=(100, 100))
        self.build()

    def add_to_renderer(self, ren):
        """ Adds the actor to renderer.

        Parameters
        ----------
        ren : renderer
        """
        self.panel.add_to_renderer(ren)

    def build(self):
        # Needs the following
        # A TextActor2D for title
        # A file select class
        # A TextActor2D for selected filename
        # A textbox for filename
        # 2 buttons for save and close
        title = Text2D(message=os.getcwd())
        text_box = TextBox2D(width=20, height=1, text="FileName")

        self.panel.add_element(element=title, relative_position=(0.05, 0.9))
        self.panel.add_element(element=self.file_select, relative_position=(0.5, 0.5))
        self.panel.add_element(element=text_box, relative_position=(0.05, 0.05))


