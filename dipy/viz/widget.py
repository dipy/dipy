

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
    #1/0
    #test = slider_rep.GetPoint1Coordinate().GetDisplayValue()
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


def button_display_coordinates(renderer, normalized_display_position, size):
    upperRight = vtk.vtkCoordinate()
    upperRight.SetCoordinateSystemToNormalizedDisplay()
    upperRight.SetValue(normalized_display_position[0], normalized_display_position[1])
    bds = [0.0] * 6
    #1/0
    bds[0] = upperRight.GetComputedDisplayValue(renderer)[0] - size[0]
    print(upperRight.GetComputedDisplayValue(renderer)[0],
          upperRight.GetComputedDisplayValue(renderer)[1])
    print(renderer.GetSize())
    bds[1] = bds[0] + size[0]
    bds[2] = upperRight.GetComputedDisplayValue(renderer)[1] - size[1]
    bds[3] = bds[2] + size[1]
    # print(bds)
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

    # http://www.vtk.org/Wiki/VTK/Examples/Cxx/Widgets/TexturedButtonWidget

    class ButtonWidget(vtk.vtkButtonWidget):

        def place(self, renderer):

            bds = button_display_coordinates(renderer, button_norm_coords, button_size)
            self.GetRepresentation().SetPlaceFactor(1)
            self.GetRepresentation().PlaceWidget(bds)
            self.On()

    button = ButtonWidget()
    button.SetInteractor(iren)
    button.SetRepresentation(button_rep)
    button.AddObserver(vtk.vtkCommand.StateChangedEvent, callback)

    #http://vtk.org/gitweb?p=VTK.git;a=blob;f=Interaction/Widgets/Testing/Cxx/TestButtonWidget.cxx
    #http://vtk.org/Wiki/VTK/Examples/Cxx/Widgets/TexturedButtonWidget

    return button


def text(iren, callback, message="Accelerating computational anatomy...",
         coord1=(.2, .2), coord2=(.7, .2),
         color=(.9, .9, .9),
         opacity=1.,
         selectable=True,
         border=True):

    # Create the TextActor
    text_actor = vtk.vtkTextActor()
    text_actor.SetInput(message)
    text_actor.GetTextProperty().SetColor(color)
    text_actor.GetTextProperty().SetOpacity(opacity)

    # Create the text representation. Used for positioning the text_actor
    text_representation = vtk.vtkTextRepresentation()
    text_representation.GetPositionCoordinate().SetValue(*coord1)
    text_representation.GetPosition2Coordinate().SetValue(*coord2)
    if border:
        text_representation.SetShowBorderToOn()
    else:
        text_representation.SetShowBorderToOff()

    # Create the TextWidget
    # Note that the SelectableOff method MUST be invoked!
    # According to the documentation :
    #
    # SelectableOn/Off indicates whether the interior region of the widget can
    # be selected or not. If not, then events (such as left mouse down) allow
    # the user to "move" the widget, and no selection is possible. Otherwise
    # the SelectRegion() method is invoked.

    text_widget = vtk.vtkTextWidget()
    text_widget.SetRepresentation(text_representation)
    text_widget.SetInteractor(iren)
    text_widget.SetTextActor(text_actor)
    if selectable:
        text_widget.SelectableOn()
    else:
        text_widget.SelectableOff()
    text_widget.On()

    text_widget.AddObserver(vtk.vtkCommand.InteractionEvent, callback)

    # This is a hack for avoiding not plotting the text widget when
    # backface culling in On on a different actor
    ss = vtk.vtkSphereSource()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(ss.GetOutputPort())
    actor = vtk.vtkActor()
    actor.GetProperty().BackfaceCullingOff()

    return text_widget
