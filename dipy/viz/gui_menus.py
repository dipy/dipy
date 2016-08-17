from dipy.utils.optpkg import optional_package

# Allow import, but disable doctests if we don't have vtk.
from dipy.viz.gui import UI
from dipy.viz.gui_2d import Panel2D

vtk, have_vtk, setup_module = optional_package('vtk')

if have_vtk:
    vtkInteractorStyleUser = vtk.vtkInteractorStyleUser
    version = vtk.vtkVersion.GetVTKSourceVersion().split(' ')[-1]
    major_version = vtk.vtkVersion.GetVTKMajorVersion()
else:
    vtkInteractorStyleUser = object

numpy_support, have_ns, _ = optional_package('vtk.util.numpy_support')


class FileSaveMenu(UI):
    def __init__(self, size):
        """

        Parameters
        ----------
        size: (float, float)
        """
        super(FileSaveMenu, self).__init__()
        self.center = (size[0]/2, size[1]/2)
        self.size = size
        self.panel = Panel2D(self.center, self.size)  # type: Panel2D
        self.build()

    def build(self):
        # Needs the following
        # A TextActor2D for title
        # A file select class
        # A TextActor2D for selected filename
        # A textbox for filename
        # 2 buttons for save and close
        pass

