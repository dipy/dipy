

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


def compute_bounds(renderer, normalized_display_position, size):
    upperRight = vtk.vtkCoordinate()
    upperRight.SetCoordinateSystemToNormalizedDisplay()
    upperRight.SetValue(normalized_display_position[0], normalized_display_position[1])
    bds = [0.0]*6
    bds[0] = upperRight.GetComputedDisplayValue(renderer)[0] - size[0]
    bds[1] = bds[0] + size[0]
    bds[2] = upperRight.GetComputedDisplayValue(renderer)[1] - size[1]
    bds[3] = bds[2] + size[1]
    return bds


def button(iren, callback):

    image1 = vtk.vtkPNGReader()
    image1.SetFileName('/home/eleftherios/Downloads/dipy-running-high-res.png')
    image1.Update()

    image2 = vtk.vtkPNGReader()
    image2.SetFileName('/home/eleftherios/Downloads/dipy-running-high-res2.png')
    image2.Update()

    #button_rep = vtk.vtkProp3DButtonRepresentation()
    button_rep = vtk.vtkTexturedButtonRepresentation2D()
    button_rep.SetNumberOfStates(1)
    button_rep.SetButtonTexture(0, image1.GetOutput())
    #button_rep.SetButtonTexture(1, image2.GetOutput())

    button_rep.SetPlaceFactor(1)



    #bds = compute_bounds(ren, (1.5, 1.5), (100, 100))
    #print(bds)
    button_rep.PlaceWidget([0., 100, 50, 500, 0, 0])

    #button_rep.PlaceWidget(bds)

    button = vtk.vtkButtonWidget()
    button.SetInteractor(iren)
    button.SetRepresentation(button_rep)
    button.On()
    button.AddObserver(vtk.vtkCommand.StateChangedEvent, callback)

    #ren = iren.GetRenderWindow().GetRenderers().GetFirstRenderer()
    #bds = compute_bounds(ren, (.5, .5), (100, 100))
    #button_rep.PlaceWidget(bds)

    #button_rep.SetPlaceFactor(1)

    #button_rep.PlaceWidget((0.75, 0, 0), (250, 450))
    # see state changed
    #http://vtk.org/gitweb?p=VTK.git;a=blob;f=Interaction/Widgets/Testing/Cxx/TestButtonWidget.cxx
    #http://vtk.org/Wiki/VTK/Examples/Cxx/Widgets/TexturedButtonWidget
    button.SetEnabled(True)

    return button