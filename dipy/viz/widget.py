

# Conditional import machinery for vtk
from dipy.utils.optpkg import optional_package

#import vtk
# Allow import, but disable doctests if we don't have vtk
vtk, have_vtk, setup_module = optional_package('vtk')
colors, have_vtk_colors, _ = optional_package('vtk.util.colors')
numpy_support, have_ns, _ = optional_package('vtk.util.numpy_support')


def slider(iren, callback, min_value=0, max_value=255, value=125,
           label="Slider",
           coord1=(0.8, 0.5), coord2=(0.9, 0.5),
           length=0.04, width=0.02,
           cap_length=0.01, cap_width=0.01,
           tube_width=0.005,
           label_format="%0.0lf"):
    """ Create a 2D slider with normalized window coordinates
    """

    slider_rep  = vtk.vtkSliderRepresentation2D()
    slider_rep.SetMinimumValue(min_value)
    slider_rep.SetMaximumValue(max_value)
    slider_rep.SetValue(value)
    slider_rep.SetTitleText(label)
    slider_rep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
    slider_rep.GetPoint1Coordinate().SetValue(*coord1)
    slider_rep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
    slider_rep.GetPoint2Coordinate().SetValue(*coord2)
    slider_rep.SetSliderLength(length)
    slider_rep.SetSliderWidth(length)
    slider_rep.SetEndCapLength(cap_length)
    slider_rep.SetEndCapWidth(cap_width)
    slider_rep.SetTubeWidth(tube_width)

    slider_rep.SetLabelFormat(label_format)

    slider = vtk.vtkSliderWidget()
    slider.SetInteractor(iren)
    slider.SetRepresentation(slider_rep)
    slider.SetAnimationModeToAnimate()
    slider.KeyPressActivationOff()
    slider.AddObserver("InteractionEvent", callback)
    slider.SetEnabled(True)
    return slider
