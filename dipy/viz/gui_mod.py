from dipy.utils.optpkg import optional_package

# Allow import, but disable doctests if we don't have vtk.
from dipy.viz.gui_2d import DiskSlider2D, LineSlider2D

vtk, have_vtk, setup_module = optional_package('vtk')

if have_vtk:
    vtkInteractorStyleUser = vtk.vtkInteractorStyleUser
    version = vtk.vtkVersion.GetVTKSourceVersion().split(' ')[-1]
    major_version = vtk.vtkVersion.GetVTKMajorVersion()
else:
    vtkInteractorStyleUser = object

numpy_support, have_ns, _ = optional_package('vtk.util.numpy_support')

# --------------------------------------------------------------------
#                                  W I P
# --------------------------------------------------------------------


class DiskSlider2DMod(DiskSlider2D):

    def __init__(self):
        super(DiskSlider2DMod, self).__init__()

    def move_move_disk(self, click_position):
        super(DiskSlider2DMod, self).move_move_disk(click_position)
        if self.angle_state < 180:
            r = self.angle_state/360
        else:
            r = (360-self.angle_state) / 360
        base_disk = vtk.vtkDiskSource()
        base_disk.SetInnerRadius(self.base_disk_inner_radius - r*15)
        base_disk.SetOuterRadius(self.base_disk_outer_radius + r*15)
        base_disk.SetRadialResolution(10)
        base_disk.SetCircumferentialResolution(50)
        base_disk.Update()

        base_disk_mapper = vtk.vtkPolyDataMapper2D()
        base_disk_mapper.SetInputConnection(base_disk.GetOutputPort())
        self.base_disk.SetMapper(base_disk_mapper)


class LineSliderMod(LineSlider2D):
    def __init__(self):

        super(LineSliderMod, self).__init__()
