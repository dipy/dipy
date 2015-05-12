

# Conditional import machinery for vtk
from dipy.utils.optpkg import optional_package

#import vtk
# Allow import, but disable doctests if we don't have vtk
vtk, have_vtk, setup_module = optional_package('vtk')
colors, have_vtk_colors, _ = optional_package('vtk.util.colors')
numpy_support, have_ns, _ = optional_package('vtk.util.numpy_support')

def slider(iren, ren, callback, min_value=0, max_value=255, value=125,
           label="Slider",
           right_normalized_pos=(0.9, 0.5),
           size=(50, 0),
           label_format="%0.0lf"):
    """ Create a 2D slider with normalized window coordinates
    """

    slider_rep  = vtk.vtkSliderRepresentation2D()
    slider_rep.SetMinimumValue(min_value)
    slider_rep.SetMaximumValue(max_value)
    slider_rep.SetValue(value)
    slider_rep.SetTitleText(label)

    slider_rep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
    slider_rep.GetPoint2Coordinate().SetValue(*right_normalized_pos)

    coord2_display = slider_rep.GetPoint2Coordinate().GetComputedDisplayValue(ren)
    slider_rep.GetPoint1Coordinate().SetCoordinateSystemToDisplay()
    slider_rep.GetPoint1Coordinate().SetValue(coord2_display[0] - size[0], coord2_display[1] - size[1])

    length=0.04
    width=0.04
    cap_length=0.01
    cap_width=0.01
    tube_width=0.005

    slider_rep.SetSliderLength(length)
    slider_rep.SetSliderWidth(width)
    slider_rep.SetEndCapLength(cap_length)
    slider_rep.SetEndCapWidth(cap_width)
    slider_rep.SetTubeWidth(tube_width)

    slider_rep.SetLabelFormat(label_format)

    class SliderWidget(vtk.vtkSliderWidget):

        def place(self, renderer):

            slider_rep = self.GetRepresentation()
            slider_rep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
            slider_rep.GetPoint2Coordinate().SetValue(*right_normalized_pos)

            coord2_display = slider_rep.GetPoint2Coordinate().GetComputedDisplayValue(renderer)
            slider_rep.GetPoint1Coordinate().SetCoordinateSystemToDisplay()
            slider_rep.GetPoint1Coordinate().SetValue(coord2_display[0] - size[0], coord2_display[1] - size[1])

            # slider_rep.SetLabelFormat(label_format)


    slider = SliderWidget()
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


def text(iren, ren, callback, message="DIPY",
         left_down_pos=(0.8, 0.5),
         right_top_pos=(0.9, 0.5),
         color=(1., .5, .0),
         opacity=1.,
         font_size=10.,
         border=False):

    # Create the TextActor
    text_actor = vtk.vtkTextActor()
    text_actor.SetInput(message)
    text_actor.GetTextProperty().SetColor(color)
    text_actor.GetTextProperty().SetOpacity(opacity)
    #text_actor.GetTextProperty().SetJustificationToLeft()
    #text_actor.GetTextProperty().SetFontSize(int(font_size))
    #text_actor.GetTextProperty().SetFontFamilyToArial()

    # Create the text representation. Used for positioning the text_actor
    text_rep = vtk.vtkTextRepresentation()
    text_rep.SetPlaceFactor(1)

    text_rep.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
    text_rep.GetPositionCoordinate().SetValue(*left_down_pos)

    text_rep.GetPosition2Coordinate().SetCoordinateSystemToNormalizedDisplay()
    text_rep.GetPosition2Coordinate().SetValue(*right_top_pos)


    if border:
        text_rep.SetShowBorderToOn()
    else:
        text_rep.SetShowBorderToOff()

    # SelectableOn/Off indicates whether the interior region of the widget can
    # be selected or not. If not, then events (such as left mouse down) allow
    # the user to "move" the widget, and no selection is possible. Otherwise
    # the SelectRegion() method is invoked.

    class TextWidget(vtk.vtkTextWidget):

        def place(self, renderer):

            text_rep = self.GetRepresentation()

            text_rep.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
            text_rep.GetPositionCoordinate().SetValue(*left_down_pos)

            text_rep.GetPosition2Coordinate().SetCoordinateSystemToNormalizedDisplay()
            text_rep.GetPosition2Coordinate().SetValue(*right_top_pos)

            #text_rep.SetPlaceFactor(1)
            self.SelectableOn()
            self.ResizableOff()
            text_rep.ProportionalResizeOn()

            self.On()


    text_widget = TextWidget()
    text_widget.SetRepresentation(text_rep)
    text_widget.SetInteractor(iren)
    text_widget.SetTextActor(text_actor)
    text_widget.SelectableOn()
    text_widget.ResizableOff()
    text_widget.GetRepresentation().ProportionalResizeOn()

    #1/0

    # text_widget.AddObserver(vtk.vtkCommand.InteractionEvent, callback)
    text_widget.AddObserver(vtk.vtkCommand.WidgetActivateEvent, callback)
    # text_widget.AddObserver(vtk.vtkCommand.KeyPressEvent, callback)

    text_widget.On()

    # This is a hack for avoiding not plotting the text widget when
    # backface culling in On on a different actor
    ss = vtk.vtkSphereSource()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(ss.GetOutputPort())
    actor = vtk.vtkActor()
    actor.GetProperty().BackfaceCullingOff()

    return text_widget
