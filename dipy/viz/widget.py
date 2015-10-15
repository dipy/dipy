# Widgets are different than actors in that they can interact with events
# To do so they need as input a vtkRenderWindowInteractor also known as iren.

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
    """ A 2D slider

    Parameters
    ----------
    iren : vtkRenderWindowInteractor
        Can also be given by the ``ShowManager``as ``iren``. Used to process
        events and handle them to the slider.
    ren :  vtkRenderer or Renderer
        Used to update the slider's position when the window changes.
    callback : function
        Function that has at least ``obj`` and ``event`` as parameters and
        can be called when a specific event is being triggered.
    min_value : float
        Minimum value of slider.
    max_value : float
        Maximum value of slider.
    value :
        Default value of slider.
    label : str
        Slider's caption.
    right_normalized_pos : tuple
        2d tuple holding the normalized right (X, Y) position of the slider.
    size: tuple
        2d tuple holding the size of the slider in pixels.
    label_format: str
        Formating in which the slider's value will appear for example "%0.2lf"
        allows for 2 decimal values.

    Returns
    -------

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
    bds = [0.0] * 6
    bds[0] = upperRight.GetComputedDisplayValue(renderer)[0] - size[0]
    print(upperRight.GetComputedDisplayValue(renderer)[0])
    print(upperRight.GetComputedDisplayValue(renderer)[1])
    bds[1] = bds[0] + size[0]
    bds[2] = upperRight.GetComputedDisplayValue(renderer)[1] - size[1]
    bds[3] = bds[2] + size[1]

    print(bds)
    return bds


def button(iren, callback, fname, button_norm_coords, button_size):

    image1 = vtk.vtkPNGReader()
    image1.SetFileName(fname)
    image1.Update()


    #button_rep = vtk.vtkProp3DButtonRepresentation()
    button_rep = vtk.vtkTexturedButtonRepresentation2D()
    button_rep.SetNumberOfStates(2)
    button_rep.SetButtonTexture(0, image1.GetOutput())
    button_rep.SetButtonTexture(1, image1.GetOutput())
    #button_rep.SetButtonTexture(1, image2.GetOutput())

    button_rep.SetPlaceFactor(1)

    #READ THIS!
    #http://www.vtk.org/Wiki/VTK/Examples/Cxx/Widgets/TexturedButtonWidget

    class ButtonWidget(vtk.vtkButtonWidget):

        def place(self, renderer):

            bds = compute_bounds(renderer, button_norm_coords, button_size)
            self.GetRepresentation().PlaceWidget(bds)
            self.On()

    button = ButtonWidget() # vtk.vtkButtonWidget()
    button.SetInteractor(iren)
    button.SetRepresentation(button_rep)
    button.AddObserver(vtk.vtkCommand.StateChangedEvent, callback)

    #http://vtk.org/gitweb?p=VTK.git;a=blob;f=Interaction/Widgets/Testing/Cxx/TestButtonWidget.cxx
    #http://vtk.org/Wiki/VTK/Examples/Cxx/Widgets/TexturedButtonWidget

    return button


