# Widgets are different than actors in that they can interact with events
# To do so they need as input a vtkRenderWindowInteractor also known as iren.

import numpy as np

# Conditional import machinery for vtk
from dipy.utils.optpkg import optional_package

# Allow import, but disable doctests if we don't have vtk
vtk, have_vtk, setup_module = optional_package('vtk')
colors, have_vtk_colors, _ = optional_package('vtk.util.colors')
numpy_support, have_ns, _ = optional_package('vtk.util.numpy_support')


def slider(iren, ren, callback, min_value=0, max_value=255, value=125,
           label="Slider",
           right_normalized_pos=(0.9, 0.5),
           size=(50, 0),
           label_format="%0.0lf",
           color=(0.5, 0.5, 0.5),
           selected_color=(0.9, 0.2, 0.1)):
    """ A 2D slider widget

    Parameters
    ----------
    iren : vtkRenderWindowInteractor
        Used to process events and handle them to the slider. Can also be given
        by the attribute ``ShowManager.iren``.
    ren :  vtkRenderer or Renderer
        Used to update the slider's position when the window changes. Can also
        be given by the ``ShowManager.ren`` attribute.
    callback : function
        Function that has at least ``obj`` and ``event`` as parameters. It will
        be called when the slider's bar has changed.
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
    slider : SliderObject
        This object inherits from vtkSliderWidget and has additional method
        called ``place`` which allows to update the position of the slider
        when for example the window is resized.
    """

    slider_rep = vtk.vtkSliderRepresentation2D()
    slider_rep.SetMinimumValue(min_value)
    slider_rep.SetMaximumValue(max_value)
    slider_rep.SetValue(value)
    slider_rep.SetTitleText(label)

    slider_rep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
    slider_rep.GetPoint2Coordinate().SetValue(*right_normalized_pos)

    coord2 = slider_rep.GetPoint2Coordinate().GetComputedDisplayValue(ren)
    slider_rep.GetPoint1Coordinate().SetCoordinateSystemToDisplay()
    slider_rep.GetPoint1Coordinate().SetValue(coord2[0] - size[0],
                                              coord2[1] - size[1])

    initial_window_size = ren.GetSize()
    length = 0.04
    width = 0.04
    cap_length = 0.01
    cap_width = 0.01
    tube_width = 0.005

    slider_rep.SetSliderLength(length)
    slider_rep.SetSliderWidth(width)
    slider_rep.SetEndCapLength(cap_length)
    slider_rep.SetEndCapWidth(cap_width)
    slider_rep.SetTubeWidth(tube_width)
    slider_rep.SetLabelFormat(label_format)

    slider_rep.GetLabelProperty().SetColor(*color)
    slider_rep.GetTubeProperty().SetColor(*color)
    slider_rep.GetCapProperty().SetColor(*color)
    slider_rep.GetTitleProperty().SetColor(*color)
    slider_rep.GetSelectedProperty().SetColor(*selected_color)
    slider_rep.GetSliderProperty().SetColor(*color)

    slider_rep.GetLabelProperty().SetShadow(0)
    slider_rep.GetTitleProperty().SetShadow(0)

    class SliderWidget(vtk.vtkSliderWidget):

        def place(self, ren):

            slider_rep = self.GetRepresentation()
            coord2_norm = slider_rep.GetPoint2Coordinate()
            coord2_norm.SetCoordinateSystemToNormalizedDisplay()
            coord2_norm.SetValue(*right_normalized_pos)

            coord2 = coord2_norm.GetComputedDisplayValue(ren)
            slider_rep.GetPoint1Coordinate().SetCoordinateSystemToDisplay()
            slider_rep.GetPoint1Coordinate().SetValue(coord2[0] - size[0],
                                                      coord2[1] - size[1])

            window_size = ren.GetSize()
            length = initial_window_size[0] * 0.04 / window_size[0]
            width = initial_window_size[1] * 0.04 / window_size[1]

            slider_rep.SetSliderLength(length)
            slider_rep.SetSliderWidth(width)

        def set_value(self, value):
            return self.GetSliderRepresentation().SetValue(value)

        def get_value(self):
            return self.GetSliderRepresentation().GetValue()

    slider = SliderWidget()
    slider.SetInteractor(iren)
    slider.SetRepresentation(slider_rep)
    slider.SetAnimationModeToAnimate()
    slider.KeyPressActivationOff()
    slider.AddObserver("InteractionEvent", callback)
    slider.SetEnabled(True)

    # Place widget after window resizing.
    def _place_widget(obj, event):
        slider.place(ren)

    iren.GetRenderWindow().AddObserver(
        vtk.vtkCommand.StartEvent, _place_widget)
    iren.GetRenderWindow().AddObserver(
        vtk.vtkCommand.ModifiedEvent, _place_widget)

    return slider


def button_display_coordinates(renderer, normalized_display_position, size):
    upperRight = vtk.vtkCoordinate()
    upperRight.SetCoordinateSystemToNormalizedDisplay()
    upperRight.SetValue(normalized_display_position[0],
                        normalized_display_position[1])
    bds = [0.0] * 6
    bds[0] = upperRight.GetComputedDisplayValue(renderer)[0] - size[0]
    bds[1] = bds[0] + size[0]
    bds[2] = upperRight.GetComputedDisplayValue(renderer)[1] - size[1]
    bds[3] = bds[2] + size[1]

    return bds


def button(iren, ren, callback, fname, right_normalized_pos=(.98, .9),
           size=(50, 50)):
    """ A textured two state button widget

    Parameters
    ----------
    iren : vtkRenderWindowInteractor
        Used to process events and handle them to the button. Can also be given
        by the attribute ``ShowManager.iren``.
    ren :  vtkRenderer or Renderer
        Used to update the slider's position when the window changes. Can also
        be given by the ``ShowManager.ren`` attribute.
    callback : function
        Function that has at least ``obj`` and ``event`` as parameters. It will
        be called when the button is pressed.
    fname : str
        PNG file path of the icon used for the button.
    right_normalized_pos : tuple
        2d tuple holding the normalized right (X, Y) position of the slider.
    size: tuple
        2d tuple holding the size of the slider in pixels.

    Returns
    -------
    button : ButtonWidget
        This object inherits from vtkButtonWidget and has an additional  method
        called ``place`` which allows to update the position of the slider
        if necessary. For example when the renderer size changes.

    Notes
    ------
    The button and slider widgets have similar positioning system. This enables
    the developers to create a HUD-like collections of buttons and sliders on
    the right side of the window that always stays in place when the dimensions
    of the window change.
    """

    image1 = vtk.vtkPNGReader()
    image1.SetFileName(fname)
    image1.Update()

    button_rep = vtk.vtkTexturedButtonRepresentation2D()
    button_rep.SetNumberOfStates(2)
    button_rep.SetButtonTexture(0, image1.GetOutput())
    button_rep.SetButtonTexture(1, image1.GetOutput())

    class ButtonWidget(vtk.vtkButtonWidget):

        def place(self, renderer):

            bds = button_display_coordinates(renderer, right_normalized_pos,
                                             size)
            self.GetRepresentation().SetPlaceFactor(1)
            self.GetRepresentation().PlaceWidget(bds)
            self.On()

    button = ButtonWidget()
    button.SetInteractor(iren)
    button.SetRepresentation(button_rep)
    button.AddObserver(vtk.vtkCommand.StateChangedEvent, callback)

    # Place widget after window resizing.
    def _place_widget(obj, event):
        button.place(ren)

    iren.GetRenderWindow().AddObserver(
        vtk.vtkCommand.StartEvent, _place_widget)
    iren.GetRenderWindow().AddObserver(
        vtk.vtkCommand.ModifiedEvent, _place_widget)

    return button


def text(iren, ren, callback, message="DIPY",
         left_down_pos=(0.8, 0.5), right_top_pos=(0.9, 0.5),
         color=(1., .5, .0), opacity=1., border=False):
    """ 2D text that can be clicked and process events

    Parameters
    ----------
    iren : vtkRenderWindowInteractor
        Used to process events and handle them to the button. Can also be given
        by the attribute ``ShowManager.iren``.
    ren :  vtkRenderer or Renderer
        Used to update the slider's position when the window changes. Can also
        be given by the ``ShowManager.ren`` attribute.
    callback : function
        Function that has at least ``obj`` and ``event`` as parameters. It will
        be called when the button is pressed.
    message : str
        Message to be shown in the text widget
    left_down_pos : tuple
        Coordinates for left down corner of text. If float are provided,
        the normalized coordinate system is used, otherwise the coordinates
        represent pixel positions. Default is (0.8, 0.5).
    right_top_pos : tuple
        Coordinates for right top corner of text. If float are provided,
        the normalized coordinate system is used, otherwise the coordinates
        represent pixel positions. Default is (0.9, 0.5).
    color : tuple
        Foreground RGB color of text. Default is (1., .5, .0).
    opacity : float
        Takes values from 0 to 1. Default is 1.
    border : bool
        Show text border. Default is False.

    Returns
    -------
    text : TextWidget
        This object inherits from ``vtkTextWidget`` has an additional method
        called ``place`` which allows to update the position of the text if
        necessary.
    """

    # Create the TextActor
    text_actor = vtk.vtkTextActor()
    text_actor.SetInput(message)
    text_actor.GetTextProperty().SetColor(color)
    text_actor.GetTextProperty().SetOpacity(opacity)

    # Create the text representation. Used for positioning the text_actor
    text_rep = vtk.vtkTextRepresentation()
    text_rep.SetTextActor(text_actor)

    if border:
        text_rep.SetShowBorderToOn()
    else:
        text_rep.SetShowBorderToOff()

    class TextWidget(vtk.vtkTextWidget):

        def place(self, renderer):
            text_rep = self.GetRepresentation()

            position = text_rep.GetPositionCoordinate()
            position2 = text_rep.GetPosition2Coordinate()

            # The dtype of `left_down_pos` determines coordinate system type.
            if np.issubdtype(np.asarray(left_down_pos).dtype, np.integer):
                position.SetCoordinateSystemToDisplay()
            else:
                position.SetCoordinateSystemToNormalizedDisplay()

            # The dtype of `right_top_pos` determines coordinate system type.
            if np.issubdtype(np.asarray(right_top_pos).dtype, np.integer):
                position2.SetCoordinateSystemToDisplay()
            else:
                position2.SetCoordinateSystemToNormalizedDisplay()

            position.SetValue(*left_down_pos)
            position2.SetValue(*right_top_pos)

    text_widget = TextWidget()
    text_widget.SetRepresentation(text_rep)
    text_widget.SetInteractor(iren)
    text_widget.SelectableOn()
    text_widget.ResizableOff()

    text_widget.AddObserver(vtk.vtkCommand.WidgetActivateEvent, callback)

    # Place widget after window resizing.
    def _place_widget(obj, event):
        text_widget.place(ren)

    iren.GetRenderWindow().AddObserver(
        vtk.vtkCommand.StartEvent, _place_widget)
    iren.GetRenderWindow().AddObserver(
        vtk.vtkCommand.ModifiedEvent, _place_widget)

    text_widget.On()

    return text_widget
