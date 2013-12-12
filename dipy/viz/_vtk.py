from ..utils.optpkg import optional_package


def setup_vtk():
    vtk, have_vtk, setup_module = optional_package('vtk')

    if have_vtk:
        version = vtk.vtkVersion.GetVTKSourceVersion().split(' ')[-1]
        major_version = vtk.vtkVersion.GetVTKMajorVersion()

        # Create a text mapper and actor to display the results of picking.
        textMapper = vtk.vtkTextMapper()
        tprop = textMapper.GetTextProperty()
        tprop.SetFontFamilyToArial()
        tprop.SetFontSize(10)
        # tprop.BoldOn()
        # tprop.ShadowOn()
        tprop.SetColor(1, 0, 0)
        textActor = vtk.vtkActor2D()
        textActor.VisibilityOff()
        textActor.SetMapper(textMapper)
    else:
        version = '0.0.0'
        major_version = int

    return vtk, have_vtk, version, major_version
