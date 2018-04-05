from __future__ import division
from _warnings import warn

import os
import glob
import numpy as np

from dipy.data import read_viz_icons
from dipy.viz.interactor import CustomInteractorStyle

from dipy.utils.optpkg import optional_package

# Allow import, but disable doctests if we don't have vtk.
vtk, have_vtk, setup_module = optional_package('vtk')

if have_vtk:
    version = vtk.vtkVersion.GetVTKVersion()
    major_version = vtk.vtkVersion.GetVTKMajorVersion()

TWO_PI = 2 * np.pi


class UI(object):
    """ An umbrella class for all UI elements.

    While adding UI elements to the renderer, we go over all the sub-elements
    that come with it and add those to the renderer automatically.

    Attributes
    ----------
    ui_param : object
        This is an attribute that can be passed to the UI object by the
        interactor.
    ui_list : list of :class:`UI`
        This is used when there are more than one UI elements inside
        a UI element. They're all automatically added to the renderer at the
        same time as this one.
    parent_ui: UI
        Reference to the parent UI element. This is useful of there is a parent
        UI element and its reference needs to be passed down to the child.
    on_left_mouse_button_pressed: function
        Callback function for when the left mouse button is pressed.
    on_left_mouse_button_released: function
        Callback function for when the left mouse button is released.
    on_left_mouse_button_clicked: function
        Callback function for when clicked using the left mouse button
        (i.e. pressed -> released).
    on_left_mouse_button_dragged: function
        Callback function for when dragging using the left mouse button.
    on_right_mouse_button_pressed: function
        Callback function for when the right mouse button is pressed.
    on_right_mouse_button_released: function
        Callback function for when the right mouse button is released.
    on_right_mouse_button_clicked: function
        Callback function for when clicking using the right mouse button
        (i.e. pressed -> released).
    on_right_mouse_button_dragged: function
        Callback function for when dragging using the right mouse button.

    """

    def __init__(self):
        self.ui_param = None
        self.ui_list = list()

        self.parent_ui = None
        self._callbacks = []

        self.left_button_state = "released"
        self.right_button_state = "released"

        self.on_left_mouse_button_pressed = lambda i_ren, obj, element: None
        self.on_left_mouse_button_dragged = lambda i_ren, obj, element: None
        self.on_left_mouse_button_released = lambda i_ren, obj, element: None
        self.on_left_mouse_button_clicked = lambda i_ren, obj, element: None
        self.on_right_mouse_button_pressed = lambda i_ren, obj, element: None
        self.on_right_mouse_button_released = lambda i_ren, obj, element: None
        self.on_right_mouse_button_clicked = lambda i_ren, obj, element: None
        self.on_right_mouse_button_dragged = lambda i_ren, obj, element: None
        self.on_key_press = lambda i_ren, obj, element: None

    def get_actors(self):
        """ Returns the actors that compose this UI component.

        """
        msg = "Subclasses of UI must implement `get_actors(self)`."
        raise NotImplementedError(msg)

    def add_to_renderer(self, ren):
        """ Allows UI objects to add their own props to the renderer.

        Parameters
        ----------
        ren : renderer

        """
        ren.add(*self.get_actors())

        # Get a hold on the current interactor style.
        iren = ren.GetRenderWindow().GetInteractor().GetInteractorStyle()

        for callback in self._callbacks:
            if not isinstance(iren, CustomInteractorStyle):
                msg = ("The ShowManager requires `CustomInteractorStyle` in"
                       " order to use callbacks.")
                raise TypeError(msg)

            iren.add_callback(*callback, args=[self])

    def add_callback(self, prop, event_type, callback, priority=0):
        """ Adds a callback to a specific event for this UI component.

        Parameters
        ----------
        prop : vtkProp
            The prop on which is callback is to be added.
        event_type : string
            The event code.
        callback : function
            The callback function.
        priority : int
            Higher number is higher priority.

        """
        # Actually since we need an interactor style we will add the callback
        # only when this UI component is added to the renderer.
        self._callbacks.append((prop, event_type, callback, priority))

    def set_center(self, position):
        """ Sets the center of the UI component

        Parameters
        ----------
        position : (float, float)
            These are the x and y coordinates respectively, with the
            origin at the bottom left.

        """
        msg = "Subclasses of UI must implement `set_center(self, position)`."
        raise NotImplementedError(msg)

    def set_visibility(self, visibility):
        """ Sets visibility of this UI component and all its sub-components.

        """
        for actor in self.get_actors():
            actor.SetVisibility(visibility)

    def handle_events(self, actor):
        self.add_callback(actor, "LeftButtonPressEvent", self.left_button_click_callback)
        self.add_callback(actor, "LeftButtonReleaseEvent", self.left_button_release_callback)
        self.add_callback(actor, "RightButtonPressEvent", self.right_button_click_callback)
        self.add_callback(actor, "RightButtonReleaseEvent", self.right_button_release_callback)
        self.add_callback(actor, "MouseMoveEvent", self.mouse_move_callback)
        self.add_callback(actor, "KeyPressEvent", self.key_press_callback)

    @staticmethod
    def left_button_click_callback(i_ren, obj, self):
        self.left_button_state = "pressing"
        self.on_left_mouse_button_pressed(i_ren, obj, self)
        i_ren.event.abort()

    @staticmethod
    def left_button_release_callback(i_ren, obj, self):
        if self.left_button_state == "pressing":
            self.on_left_mouse_button_clicked(i_ren, obj, self)
        self.left_button_state = "released"
        self.on_left_mouse_button_released(i_ren, obj, self)

    @staticmethod
    def right_button_click_callback(i_ren, obj, self):
        self.right_button_state = "pressing"
        self.on_right_mouse_button_pressed(i_ren, obj, self)
        i_ren.event.abort()

    @staticmethod
    def right_button_release_callback(i_ren, obj, self):
        if self.right_button_state == "pressing":
            self.on_right_mouse_button_clicked(i_ren, obj, self)
        self.right_button_state = "released"
        self.on_right_mouse_button_released(i_ren, obj, self)

    @staticmethod
    def mouse_move_callback(i_ren, obj, self):
        if self.left_button_state == "pressing" or self.left_button_state == "dragging":
            self.left_button_state = "dragging"
            self.on_left_mouse_button_dragged(i_ren, obj, self)
        elif self.right_button_state == "pressing" or self.right_button_state == "dragging":
            self.right_button_state = "dragging"
            self.on_right_mouse_button_dragged(i_ren, obj, self)
        else:
            pass

    @staticmethod
    def key_press_callback(i_ren, obj, self):
        self.on_key_press(i_ren, obj, self)


class Button2D(UI):
    """ A 2D overlay button and is of type vtkTexturedActor2D.
    Currently supports:
    - Multiple icons.
    - Switching between icons.

    Attributes
    ----------
    size: (float, float)
        Button size (width, height) in pixels.

    """

    def __init__(self, icon_fnames, size=(30, 30)):
        """
        Parameters
        ----------
        size : 2-tuple of int, optional
            Button size.
        icon_fnames : dict
            {iconname : filename, iconname : filename, ...}

        """
        super(Button2D, self).__init__()
        self.icon_extents = dict()
        self.icons = self.__build_icons(icon_fnames)
        self.icon_names = list(self.icons.keys())
        self.current_icon_id = 0
        self.current_icon_name = self.icon_names[self.current_icon_id]
        self.actor = self.build_actor(self.icons[self.current_icon_name])
        self.size = size
        self.handle_events(self.actor)

    def __build_icons(self, icon_fnames):
        """ Converts file names to vtkImageDataGeometryFilters.

        A pre-processing step to prevent re-read of file names during every
        state change.

        Parameters
        ----------
        icon_fnames : dict
            {iconname: filename, iconname: filename, ...}

        Returns
        -------
        icons : dict
            A dictionary of corresponding vtkImageDataGeometryFilters.

        """
        icons = {}
        for icon_name, icon_fname in icon_fnames.items():
            if icon_fname.split(".")[-1] not in ["png", "PNG"]:
                error_msg = "A specified icon file is not in the PNG format. SKIPPING."
                warn(Warning(error_msg))
            else:
                png = vtk.vtkPNGReader()
                png.SetFileName(icon_fname)
                png.Update()
                icons[icon_name] = png.GetOutput()

        return icons

    @property
    def size(self):
        """ Gets the button size.

        """
        return self._size

    @size.setter
    def size(self, size):
        """ Sets the button size.

        Parameters
        ----------
        size : (float, float)
            Button size (width, height) in pixels.

        """
        self._size = np.asarray(size)

        # Update actor.
        self.texture_points.SetPoint(0, 0, 0, 0.0)
        self.texture_points.SetPoint(1, size[0], 0, 0.0)
        self.texture_points.SetPoint(2, size[0], size[1], 0.0)
        self.texture_points.SetPoint(3, 0, size[1], 0.0)
        self.texture_polydata.SetPoints(self.texture_points)

    @property
    def color(self):
        """ Gets the button's color.

        """
        color = self.actor.GetProperty().GetColor()
        return np.asarray(color)

    @color.setter
    def color(self, color):
        """ Sets the button's color.

        Parameters
        ----------
        color : (float, float, float)
            RGB. Must take values in [0, 1].

        """
        self.actor.GetProperty().SetColor(*color)

    def scale(self, size):
        """ Scales the button.

        Parameters
        ----------
        size : (float, float)
            Scaling factor (width, height) in pixels.

        """
        self.size *= size

    def build_actor(self, icon):
        """ Return an image as a 2D actor with a specific position.

        Parameters
        ----------
        icon : :class:`vtkImageData`

        Returns
        -------
        :class:`vtkTexturedActor2D`

        """
        # This is highly inspired by
        # https://github.com/Kitware/VTK/blob/c3ec2495b183e3327820e927af7f8f90d34c3474\
        # /Interaction/Widgets/vtkBalloonRepresentation.cxx#L47

        self.texture_polydata = vtk.vtkPolyData()
        self.texture_points = vtk.vtkPoints()
        self.texture_points.SetNumberOfPoints(4)
        self.size = icon.GetExtent()

        polys = vtk.vtkCellArray()
        polys.InsertNextCell(4)
        polys.InsertCellPoint(0)
        polys.InsertCellPoint(1)
        polys.InsertCellPoint(2)
        polys.InsertCellPoint(3)
        self.texture_polydata.SetPolys(polys)

        tc = vtk.vtkFloatArray()
        tc.SetNumberOfComponents(2)
        tc.SetNumberOfTuples(4)
        tc.InsertComponent(0, 0, 0.0)
        tc.InsertComponent(0, 1, 0.0)
        tc.InsertComponent(1, 0, 1.0)
        tc.InsertComponent(1, 1, 0.0)
        tc.InsertComponent(2, 0, 1.0)
        tc.InsertComponent(2, 1, 1.0)
        tc.InsertComponent(3, 0, 0.0)
        tc.InsertComponent(3, 1, 1.0)
        self.texture_polydata.GetPointData().SetTCoords(tc)

        texture_mapper = vtk.vtkPolyDataMapper2D()
        if major_version <= 5:
            texture_mapper.SetInput(self.texture_polydata)
        else:
            texture_mapper.SetInputData(self.texture_polydata)

        button = vtk.vtkTexturedActor2D()
        button.SetMapper(texture_mapper)

        self.texture = vtk.vtkTexture()
        button.SetTexture(self.texture)

        button_property = vtk.vtkProperty2D()
        button_property.SetOpacity(1.0)
        button.SetProperty(button_property)

        self.set_icon(icon)
        return button

    def get_actors(self):
        """ Returns the actors that compose this UI component.

        """
        return [self.actor]

    def set_icon(self, icon):
        """ Modifies the icon used by the vtkTexturedActor2D.

        Parameters
        ----------
        icon : imageDataGeometryFilter

        """
        if major_version <= 5:
            self.texture.SetInput(icon)
        else:
            self.texture.SetInputData(icon)

    def next_icon_name(self):
        """ Returns the next icon name while cycling through icons.

        """
        self.current_icon_id += 1
        if self.current_icon_id == len(self.icons):
            self.current_icon_id = 0
        self.current_icon_name = self.icon_names[self.current_icon_id]

    def next_icon(self):
        """ Increments the state of the Button.

            Also changes the icon.

        """
        self.next_icon_name()
        self.set_icon(self.icons[self.current_icon_name])

    def set_center(self, position):
        """ Sets the icon center to position.

        Parameters
        ----------
        position : (float, float)
            The new center of the button (x, y).

        """
        new_position = np.asarray(position) - self.size / 2.
        self.actor.SetPosition(*new_position)


class Rectangle2D(UI):
    """ A 2D rectangle sub-classed from UI.
    Uses vtkPolygon.

    Attributes
    ----------
    size : (float, float)
        The size of the rectangle (height, width) in pixels.

    """

    def __init__(self, size, center=(0, 0), color=(1, 1, 1), opacity=1.0):
        """ Initializes a rectangle.

        Parameters
        ----------
        size : (float, float)
            The size of the rectangle (height, width) in pixels.
        center : (float, float)
            The center of the rectangle (x, y).
        color : (float, float, float)
            Must take values in [0, 1].
        opacity : float
            Must take values in [0, 1].

        """
        super(Rectangle2D, self).__init__()
        self.size = size
        self.actor = self.build_actor(size=size)
        self.color = color
        self.set_center(center)
        self.opacity = opacity
        self.handle_events(self.actor)

    def get_actors(self):
        """ Returns the actors that compose this UI component.

        """
        return [self.actor]

    def build_actor(self, size):
        """ Builds the text actor.

        Parameters
        ----------
        size : (float, float)
            The size of the rectangle (height, width) in pixels.

        Returns
        -------
        :class:`vtkActor2D`

        """
        # Setup four points
        points = vtk.vtkPoints()
        points.InsertNextPoint(0, 0, 0)
        points.InsertNextPoint(size[0], 0, 0)
        points.InsertNextPoint(size[0], size[1], 0)
        points.InsertNextPoint(0, size[1], 0)

        # Create the polygon
        polygon = vtk.vtkPolygon()
        polygon.GetPointIds().SetNumberOfIds(4)  # make a quad
        polygon.GetPointIds().SetId(0, 0)
        polygon.GetPointIds().SetId(1, 1)
        polygon.GetPointIds().SetId(2, 2)
        polygon.GetPointIds().SetId(3, 3)

        # Add the polygon to a list of polygons
        polygons = vtk.vtkCellArray()
        polygons.InsertNextCell(polygon)

        # Create a PolyData
        polygonPolyData = vtk.vtkPolyData()
        polygonPolyData.SetPoints(points)
        polygonPolyData.SetPolys(polygons)

        # Create a mapper and actor
        mapper = vtk.vtkPolyDataMapper2D()
        if vtk.VTK_MAJOR_VERSION <= 5:
            mapper.SetInput(polygonPolyData)
        else:
            mapper.SetInputData(polygonPolyData)

        actor = vtk.vtkActor2D()
        actor.SetMapper(mapper)

        return actor

    def set_position(self, position):
        self.actor.SetPosition(*position)

    def set_center(self, position):
        """ Sets the center to position.

        Parameters
        ----------
        position : (float, float)
            The new center of the rectangle (x, y).

        """
        self.actor.SetPosition(position[0] - self.size[0] / 2,
                               position[1] - self.size[1] / 2)

    @property
    def color(self):
        """ Gets the rectangle's color.

        """
        color = self.actor.GetProperty().GetColor()
        return np.asarray(color)

    @color.setter
    def color(self, color):
        """ Sets the rectangle's color.

        Parameters
        ----------
        color : (float, float, float)
            RGB. Must take values in [0, 1].

        """
        self.actor.GetProperty().SetColor(*color)

    @property
    def opacity(self):
        """ Gets the rectangle's opacity.

        """
        return self.actor.GetProperty().GetOpacity()

    @opacity.setter
    def opacity(self, opacity):
        """ Sets the rectangle's opacity.

        Parameters
        ----------
        opacity : float
            Degree of transparency. Must be between [0, 1].

        """
        self.actor.GetProperty().SetOpacity(opacity)

    @property
    def position(self):
        """ Gets text actor position.

        Returns
        -------
        (float, float)
            The current actor position. (x, y) in pixels.

        """
        return self.actor.GetPosition()

    @position.setter
    def position(self, position):
        """ Set text actor position.

        Parameters
        ----------
        position : (float, float)
            The new position. (x, y) in pixels.

        """
        self.actor.SetPosition(*position)


class Panel2D(UI):
    """ A 2D UI Panel.

    Can contain one or more UI elements.

    Attributes
    ----------
    center : (float, float)
        The center of the panel (x, y).
    size : (float, float)
        The size of the panel (width, height) in pixels.
    alignment : [left, right]
        Alignment of the panel with respect to the overall screen.

    """

    def __init__(self, center, size, color=(0.1, 0.1, 0.1), opacity=0.7, align="left"):
        """
        Parameters
        ----------
        center : (float, float)
            The center of the panel (x, y).
        size : (float, float)
            The size of the panel (width, height) in pixels.
        color : (float, float, float)
            Must take values in [0, 1].
        opacity : float
            Must take values in [0, 1].
        align : [left, right]
            Alignment of the panel with respect to the overall screen.

        """
        super(Panel2D, self).__init__()
        self.center = center
        self.size = size
        self.lower_limits = (self.center[0] - self.size[0] / 2,
                             self.center[1] - self.size[1] / 2)

        self.panel = Rectangle2D(size=size, center=center, color=color,
                                 opacity=opacity)

        self.element_positions = []
        self.element_positions.append([self.panel, 'relative', 0.5, 0.5])
        self.alignment = align

        self.handle_events(self.panel.actor)

        self.on_left_mouse_button_pressed = self.left_button_pressed
        self.on_left_mouse_button_dragged = self.left_button_dragged

    def add_to_renderer(self, ren):
        """ Allows UI objects to add their own props to the renderer.

        Here, we add only call add_to_renderer for the additional components.

        Parameters
        ----------
        ren : renderer

        """
        super(Panel2D, self).add_to_renderer(ren)
        for ui_item in self.ui_list:
            ui_item.add_to_renderer(ren)

    def get_actors(self):
        """ Returns the panel actor.

        """
        return [self.panel.actor]

    def add_element(self, element, position_type, position):
        """ Adds an element to the panel.

        The center of the rectangular panel is its bottom lower position.

        Parameters
        ----------
        element : UI
            The UI item to be added.
        position_type: string
            'absolute' or 'relative'
        position : (float, float)
            Absolute for absolute and relative for relative

        """
        self.ui_list.append(element)
        if position_type == 'relative':
            self.element_positions.append([element, position_type, position[0], position[1]])
            element.set_center((self.lower_limits[0] + position[0] * self.size[0],
                                self.lower_limits[1] + position[1] * self.size[1]))
        elif position_type == 'absolute':
            self.element_positions.append([element, position_type, position[0], position[1]])
            element.set_center((position[0], position[1]))
        else:
            raise ValueError("Position can only be absolute or relative")

    def set_center(self, position):
        """ Sets the panel center to position.

        The center of the rectangular panel is its bottom lower position.

        Parameters
        ----------
        position : (float, float)
            The new center of the panel (x, y).

        """
        shift = [position[0] - self.center[0], position[1] - self.center[1]]
        self.center = position
        self.lower_limits = (position[0] - self.size[0] / 2, position[1] - self.size[1] / 2)
        for ui_element in self.element_positions:
            if ui_element[1] == 'relative':
                ui_element[0].set_center((self.lower_limits[0] + ui_element[2] * self.size[0],
                                          self.lower_limits[1] + ui_element[3] * self.size[1]))
            elif ui_element[1] == 'absolute':
                ui_element[2] += shift[0]
                ui_element[3] += shift[1]
                ui_element[0].set_center((ui_element[2], ui_element[3]))

    @staticmethod
    def left_button_pressed(i_ren, obj, panel2d_object):
        click_position = i_ren.event.position
        panel2d_object.ui_param = (click_position[0] -
                                   panel2d_object.panel.actor.GetPosition()[0] -
                                   panel2d_object.panel.size[0] / 2,
                                   click_position[1] -
                                   panel2d_object.panel.actor.GetPosition()[1] -
                                   panel2d_object.panel.size[1] / 2)
        i_ren.event.abort()  # Stop propagating the event.

    @staticmethod
    def left_button_dragged(i_ren, obj, panel2d_object):
        click_position = i_ren.event.position
        if panel2d_object.ui_param is not None:
            panel2d_object.set_center((click_position[0] - panel2d_object.ui_param[0],
                                       click_position[1] - panel2d_object.ui_param[1]))
        i_ren.force_render()

    def re_align(self, window_size_change):
        """ Re-organises the elements in case the window size is changed.

        Parameters
        ----------
        window_size_change : (int, int)
            New window size (width, height) in pixels.

        """
        if self.alignment == "left":
            pass
        elif self.alignment == "right":
            self.set_center((self.center[0] + window_size_change[0],
                             self.center[1] + window_size_change[1]))
        else:
            raise ValueError("You can only left-align or right-align objects in a panel.")


class TextBlock2D(UI):
    """ Wraps over the default vtkTextActor and helps setting the text.

    Contains member functions for text formatting.

    Attributes
    ----------
    actor : :class:`vtkTextActor`
        The text actor.
    message : str
        The initial text while building the actor.
    position : (float, float)
        (x, y) in pixels.
    color : (float, float, float)
        RGB: Values must be between 0-1.
    bg_color : (float, float, float)
        RGB: Values must be between 0-1.
    font_size : int
        Size of the text font.
    font_family : str
        Currently only supports Arial.
    justification : str
        left, right or center.
    vertical_justification : str
        bottom, middle or top.
    bold : bool
        Makes text bold.
    italic : bool
        Makes text italicised.
    shadow : bool
        Adds text shadow.
    """

    def __init__(self, text="Text Block", font_size=18, font_family='Arial',
                 justification='left', vertical_justification="bottom",
                 bold=False, italic=False, shadow=False,
                 color=(1, 1, 1), bg_color=None, position=(0, 0)):
        """
        Parameters
        ----------
        text : str
            The initial text while building the actor.
        position : (float, float)
            (x, y) in pixels.
        color : (float, float, float)
            RGB: Values must be between 0-1.
        bg_color : (float, float, float)
            RGB: Values must be between 0-1.
        font_size : int
            Size of the text font.
        font_family : str
            Currently only supports Arial.
        justification : str
            left, right or center.
        vertical_justification : str
            bottom, middle or top.
        bold : bool
            Makes text bold.
        italic : bool
            Makes text italicised.
        shadow : bool
            Adds text shadow.
        """
        super(TextBlock2D, self).__init__()
        self.actor = vtk.vtkTextActor()

        self._background = None  # For VTK < 7
        self.position = position
        self.color = color
        self.background_color = bg_color
        self.font_size = font_size
        self.font_family = font_family
        self.justification = justification
        self.bold = bold
        self.italic = italic
        self.shadow = shadow
        self.vertical_justification = vertical_justification
        self.message = text

    def get_actor(self):
        """ Returns the actor composing this element.

        Returns
        -------
        :class:`vtkTextActor`
            The actor composing this class.
        """
        return self.actor

    def get_actors(self):
        """ Returns the actors that compose this UI component.

        """
        if self._background is not None:
            return [self._background, self.actor]

        return [self.actor]

    @property
    def message(self):
        """ Gets message from the text.

        Returns
        -------
        str
            The current text message.

        """
        return self.actor.GetInput()

    @message.setter
    def message(self, text):
        """ Sets the text message.

        Parameters
        ----------
        text : str
            The message to be set.

        """
        self.actor.SetInput(text)

    @property
    def font_size(self):
        """ Gets text font size.

        Returns
        ----------
        int
            Text font size.

        """
        return self.actor.GetTextProperty().GetFontSize()

    @font_size.setter
    def font_size(self, size):
        """ Sets font size.

        Parameters
        ----------
        size : int
            Text font size.

        """
        self.actor.GetTextProperty().SetFontSize(size)

    @property
    def font_family(self):
        """ Gets font family.

        Returns
        ----------
        str
            Text font family.

        """
        return self.actor.GetTextProperty().GetFontFamilyAsString()

    @font_family.setter
    def font_family(self, family='Arial'):
        """ Sets font family.

        Currently Arial and Courier are supported.

        Parameters
        ----------
        family : str
            The font family.

        """
        if family == 'Arial':
            self.actor.GetTextProperty().SetFontFamilyToArial()
        elif family == 'Courier':
            self.actor.GetTextProperty().SetFontFamilyToCourier()
        else:
            raise ValueError("Font not supported yet: {}.".format(family))

    @property
    def justification(self):
        """ Gets text justification.

        Returns
        -------
        str
            Text justification.

        """
        justification = self.actor.GetTextProperty().GetJustificationAsString()
        if justification == 'Left':
            return "left"
        elif justification == 'Centered':
            return "center"
        elif justification == 'Right':
            return "right"

    @justification.setter
    def justification(self, justification):
        """ Justifies text.

        Parameters
        ----------
        justification : str
            Possible values are left, right, center.

        """
        text_property = self.actor.GetTextProperty()
        if justification == 'left':
            text_property.SetJustificationToLeft()
        elif justification == 'center':
            text_property.SetJustificationToCentered()
        elif justification == 'right':
            text_property.SetJustificationToRight()
        else:
            raise ValueError("Text can only be justified left, right and center.")

    @property
    def vertical_justification(self):
        """ Gets text vertical justification.

        Returns
        -------
        str
            Text vertical justification.

        """
        text_property = self.actor.GetTextProperty()
        vjustification = text_property.GetVerticalJustificationAsString()
        if vjustification == 'Bottom':
            return "bottom"
        elif vjustification == 'Centered':
            return "middle"
        elif vjustification == 'Top':
            return "top"

    @vertical_justification.setter
    def vertical_justification(self, vertical_justification):
        """ Justifies text vertically.

        Parameters
        ----------
        vertical_justification : str
            Possible values are bottom, middle, top.

        """
        text_property = self.actor.GetTextProperty()
        if vertical_justification == 'bottom':
            text_property.SetVerticalJustificationToBottom()
        elif vertical_justification == 'middle':
            text_property.SetVerticalJustificationToCentered()
        elif vertical_justification == 'top':
            text_property.SetVerticalJustificationToTop()
        else:
            msg = "Vertical justification must be: bottom, middle or top."
            raise ValueError(msg)

    @property
    def bold(self):
        """ Returns whether the text is bold.

        Returns
        -------
        bool
            Text is bold if True.

        """
        return self.actor.GetTextProperty().GetBold()

    @bold.setter
    def bold(self, flag):
        """ Bolds/un-bolds text.

        Parameters
        ----------
        flag : bool
            Sets text bold if True.

        """
        self.actor.GetTextProperty().SetBold(flag)

    @property
    def italic(self):
        """ Returns whether the text is italicised.

        Returns
        -------
        bool
            Text is italicised if True.

        """
        return self.actor.GetTextProperty().GetItalic()

    @italic.setter
    def italic(self, flag):
        """ Italicises/un-italicises text.

        Parameters
        ----------
        flag : bool
            Italicises text if True.

        """
        self.actor.GetTextProperty().SetItalic(flag)

    @property
    def shadow(self):
        """ Returns whether the text has shadow.

        Returns
        -------
        bool
            Text is shadowed if True.

        """
        return self.actor.GetTextProperty().GetShadow()

    @shadow.setter
    def shadow(self, flag):
        """ Adds/removes text shadow.

        Parameters
        ----------
        flag : bool
            Shadows text if True.

        """
        self.actor.GetTextProperty().SetShadow(flag)

    @property
    def color(self):
        """ Gets text color.

        Returns
        -------
        (float, float, float)
            Returns text color in RGB.

        """
        return self.actor.GetTextProperty().GetColor()

    @color.setter
    def color(self, color=(1, 0, 0)):
        """ Set text color.

        Parameters
        ----------
        color : (float, float, float)
            RGB: Values must be between 0-1.

        """
        self.actor.GetTextProperty().SetColor(*color)

    @property
    def background_color(self):
        """ Gets background color.

        Returns
        -------
        (float, float, float) or None
            If None, there no background color.
            Otherwise, background color in RGB.

        """
        if major_version < 7:
            if self._background is None:
                return None

            return self._background.GetProperty().GetColor()

        if self.actor.GetTextProperty().GetBackgroundOpacity() == 0:
            return None

        return self.actor.GetTextProperty().GetBackgroundColor()

    @background_color.setter
    def background_color(self, color):
        """ Set text color.

        Parameters
        ----------
        color : (float, float, float) or None
            If None, remove background.
            Otherwise, RGB values (must be between 0-1).

        """

        if color is None:
            # Remove background.
            if major_version < 7:
                self._background = None
            else:
                self.actor.GetTextProperty().SetBackgroundOpacity(0.)

        else:
            if major_version < 7:
                self._background = vtk.vtkActor2D()
                self._background.GetProperty().SetColor(*color)
                self._background.GetProperty().SetOpacity(1)
                self._background.SetMapper(self.actor.GetMapper())
                self._background.SetPosition(*self.actor.GetPosition())

            else:
                self.actor.GetTextProperty().SetBackgroundColor(*color)
                self.actor.GetTextProperty().SetBackgroundOpacity(1.)

    @property
    def position(self):
        """ Gets text actor position.

        Returns
        -------
        (float, float)
            The current actor position. (x, y) in pixels.

        """
        return self.actor.GetPosition()

    @position.setter
    def position(self, position):
        """ Set text actor position.

        Parameters
        ----------
        position : (float, float)
            The new position. (x, y) in pixels.

        """
        self.actor.SetPosition(*position)
        if self._background is not None:
            self._background.SetPosition(*self.actor.GetPosition())

    def set_center(self, position):
        """ Sets the text center to position.

        Parameters
        ----------
        position : (float, float)

        """
        self.position = position


class TextBox2D(UI):
    """ An editable 2D text box that behaves as a UI component.

    Currently supports:
    - Basic text editing.
    - Cursor movements.
    - Single and multi-line text boxes.
    - Pre text formatting (text needs to be formatted beforehand).

    Attributes
    ----------
    text : str
        The current text state.
    actor : :class:`vtkActor2d`
        The text actor.
    width : int
        The number of characters in a single line of text.
    height : int
        The number of lines in the textbox.
    window_left : int
        Left limit of visible text in the textbox.
    window_right : int
        Right limit of visible text in the textbox.
    caret_pos : int
        Position of the caret in the text.
    init : bool
        Flag which says whether the textbox has just been initialized.

    """
    def __init__(self, width, height, text="Enter Text", position=(100, 10),
                 color=(0, 0, 0), font_size=18, font_family='Arial',
                 justification='left', bold=False,
                 italic=False, shadow=False):
        """
        Parameters
        ----------
        width : int
            The number of characters in a single line of text.
        height : int
            The number of lines in the textbox.
        text : str
            The initial text while building the actor.
        position : (float, float)
            (x, y) in pixels.
        color : (float, float, float)
            RGB: Values must be between 0-1.
        font_size : int
            Size of the text font.
        font_family : str
            Currently only supports Arial.
        justification : str
            left, right or center.
        bold : bool
            Makes text bold.
        italic : bool
            Makes text italicised.
        shadow : bool
            Adds text shadow.

        """
        super(TextBox2D, self).__init__()
        self.text = text
        self.actor = self.build_actor(self.text, position, color, font_size,
                                      font_family, justification, bold, italic, shadow)
        self.width = width
        self.height = height
        self.window_left = 0
        self.window_right = 0
        self.caret_pos = 0
        self.init = True

        self.handle_events(self.actor.get_actor())

        self.on_left_mouse_button_pressed = self.left_button_press
        self.on_key_press = self.key_press

    def build_actor(self, text, position, color, font_size,
                    font_family, justification, bold, italic, shadow):

        """ Builds a text actor.

        Parameters
        ----------
        text : str
            The initial text while building the actor.
        position : (float, float)
            (x, y) in pixels.
        color : (float, float, float)
            RGB: Values must be between 0-1.
        font_size : int
            Size of the text font.
        font_family : str
            Currently only supports Arial.
        justification : str
            left, right or center.
        bold : bool
            Makes text bold.
        italic : bool
            Makes text italicised.
        shadow : bool
            Adds text shadow.

        Returns
        -------
        :class:`TextBlock2D`

        """
        text_block = TextBlock2D()
        text_block.position = position
        text_block.message = text
        text_block.font_size = font_size
        text_block.font_family = font_family
        text_block.justification = justification
        text_block.bold = bold
        text_block.italic = italic
        text_block.shadow = shadow

        if major_version >= 7:
            text_block.actor.GetTextProperty().SetBackgroundColor(1, 1, 1)
            text_block.actor.GetTextProperty().SetBackgroundOpacity(1.0)
            text_block.color = color

        return text_block

    def set_message(self, message):
        """ Set custom text to textbox.

        Parameters
        ----------
        message: str
            The custom message to be set.

        """
        self.text = message
        self.actor.message = message
        self.init = False
        self.window_right = len(self.text)
        self.window_left = 0
        self.caret_pos = self.window_right

    def get_actors(self):
        """ Returns the actors that compose this UI component.

        """
        return [self.actor.get_actor()]

    def width_set_text(self, text):
        """ Adds newlines to text where necessary.

        This is needed for multi-line text boxes.

        Parameters
        ----------
        text : str
            The final text to be formatted.

        Returns
        -------
        str
            A multi line formatted text.

        """
        multi_line_text = ""
        for i in range(len(text)):
            multi_line_text += text[i]
            if (i + 1) % self.width == 0:
                multi_line_text += "\n"
        return multi_line_text.rstrip("\n")

    def handle_character(self, character):
        """ Main driving function that handles button events.

        # TODO: Need to handle all kinds of characters like !, +, etc.

        Parameters
        ----------
        character : str

        """
        if character.lower() == "return":
            self.render_text(False)
            return True
        if character.lower() == "backspace":
            self.remove_character()
        elif character.lower() == "left":
            self.move_left()
        elif character.lower() == "right":
            self.move_right()
        else:
            self.add_character(character)
        self.render_text()
        return False

    def move_caret_right(self):
        """ Moves the caret towards right.

        """
        self.caret_pos = min(self.caret_pos + 1, len(self.text))

    def move_caret_left(self):
        """ Moves the caret towards left.

        """
        self.caret_pos = max(self.caret_pos - 1, 0)

    def right_move_right(self):
        """ Moves right boundary of the text window right-wards.

        """
        if self.window_right <= len(self.text):
            self.window_right += 1

    def right_move_left(self):
        """ Moves right boundary of the text window left-wards.

        """
        if self.window_right > 0:
            self.window_right -= 1

    def left_move_right(self):
        """ Moves left boundary of the text window right-wards.

        """
        if self.window_left <= len(self.text):
            self.window_left += 1

    def left_move_left(self):
        """ Moves left boundary of the text window left-wards.

        """
        if self.window_left > 0:
            self.window_left -= 1

    def add_character(self, character):
        """ Inserts a character into the text and moves window and caret accordingly.

        Parameters
        ----------
        character : str

        """
        if len(character) > 1 and character.lower() != "space":
            return
        if character.lower() == "space":
            character = " "
        self.text = (self.text[:self.caret_pos] +
                     character +
                     self.text[self.caret_pos:])
        self.move_caret_right()
        if (self.window_right -
                self.window_left == self.height * self.width - 1):
            self.left_move_right()
        self.right_move_right()

    def remove_character(self):
        """ Removes a character from the text and moves window and caret accordingly.

        """
        if self.caret_pos == 0:
            return
        self.text = self.text[:self.caret_pos - 1] + self.text[self.caret_pos:]
        self.move_caret_left()
        if len(self.text) < self.height * self.width - 1:
            self.right_move_left()
        if (self.window_right -
                self.window_left == self.height * self.width - 1):
            if self.window_left > 0:
                self.left_move_left()
                self.right_move_left()

    def move_left(self):
        """ Handles left button press.

        """
        self.move_caret_left()
        if self.caret_pos == self.window_left - 1:
            if (self.window_right -
                    self.window_left == self.height * self.width - 1):
                self.left_move_left()
                self.right_move_left()

    def move_right(self):
        """ Handles right button press.

        """
        self.move_caret_right()
        if self.caret_pos == self.window_right + 1:
            if (self.window_right -
                    self.window_left == self.height * self.width - 1):
                self.left_move_right()
                self.right_move_right()

    def showable_text(self, show_caret):
        """ Chops out text to be shown on the screen.

        Parameters
        ----------
        show_caret : bool
            Whether or not to show the caret.

        """
        if show_caret:
            ret_text = (self.text[:self.caret_pos] +
                        "_" +
                        self.text[self.caret_pos:])
        else:
            ret_text = self.text
        ret_text = ret_text[self.window_left:self.window_right + 1]
        return ret_text

    def render_text(self, show_caret=True):
        """ Renders text after processing.

        Parameters
        ----------
        show_caret : bool
            Whether or not to show the caret.

        """
        text = self.showable_text(show_caret)
        if text == "":
            text = "Enter Text"
        self.actor.message = self.width_set_text(text)

    def edit_mode(self):
        """ Turns on edit mode.

        """
        if self.init:
            self.text = ""
            self.init = False
            self.caret_pos = 0
        self.render_text()

    def set_center(self, position):
        """ Sets the text center to position.

        Parameters
        ----------
        position : (float, float)

        """
        self.actor.position = position

    @staticmethod
    def left_button_press(i_ren, obj, textbox_object):
        """ Left button press handler for textbox

        Parameters
        ----------
        i_ren: :class:`CustomInteractorStyle`
        obj: :class:`vtkActor`
            The picked actor
        textbox_object: :class:`TextBox2D`

        """
        i_ren.add_active_prop(textbox_object.actor.get_actor())
        textbox_object.edit_mode()
        i_ren.force_render()

    @staticmethod
    def key_press(i_ren, obj, textbox_object):
        """ Key press handler for textbox

        Parameters
        ----------
        i_ren: :class:`CustomInteractorStyle`
        obj: :class:`vtkActor`
            The picked actor
        textbox_object: :class:`TextBox2D`

        """
        key = i_ren.event.key
        is_done = textbox_object.handle_character(key)
        if is_done:
            i_ren.remove_active_prop(textbox_object.actor.get_actor())

        i_ren.force_render()


class LineSlider2D(UI):
    """ A 2D Line Slider.

    A sliding ring on a line with a percentage indicator.

    Currently supports:
    - A disk on a line (a thin rectangle).
    - Setting disk position.

    Attributes
    ----------
    line_width : int
        Width of the line on which the disk will slide.
    inner_radius : int
        Inner radius of the disk (ring).
    outer_radius : int
        Outer radius of the disk.
    center : (float, float)
        Center of the slider.
    length : int
        Length of the slider.
    slider_line : :class:`vtkActor`
        The line on which the slider disk moves.
    slider_disk : :class:`vtkActor`
        The moving slider disk.
    text : :class:`TextBlock2D`
        The text that shows percentage.

    """
    def __init__(self, line_width=5, inner_radius=0, outer_radius=10,
                 center=(450, 300), length=200, initial_value=50,
                 min_value=0, max_value=100, text_size=16,
                 text_template="{value:.1f} ({ratio:.0%})"):
        """
        Parameters
        ----------
        line_width : int
            Width of the line on which the disk will slide.
        inner_radius : int
            Inner radius of the disk (ring).
        outer_radius : int
            Outer radius of the disk.
        center : (float, float)
            Center of the slider.
        length : int
            Length of the slider.
        initial_value : float
            Initial value of the slider.
        min_value : float
            Minimum value of the slider.
        max_value : float
            Maximum value of the slider.
        text_size : int
            Size of the text to display alongside the slider (pt).
        text_template : str, callable
            If str, text template can contain one or multiple of the
            replacement fields: `{value:}`, `{ratio:}`.
            If callable, this instance of `:class:LineSlider2D` will be
            passed as argument to the text template function.

        """
        super(LineSlider2D, self).__init__()

        self.length = length
        self.min_value = min_value
        self.max_value = max_value

        self.text_template = text_template

        self.line_width = line_width
        self.center = center
        self.current_state = center[0]
        self.left_x_position = center[0] - length / 2
        self.right_x_position = center[0] + length / 2
        self._ratio = (self.current_state - self.left_x_position) / length

        self.slider_line = None
        self.slider_disk = None
        self.text = None

        self.build_actors(inner_radius=inner_radius,
                          outer_radius=outer_radius, text_size=text_size)

        # Setting the disk position will also update everything.
        self.value = initial_value
        # self.update()

        self.handle_events(None)

    def build_actors(self, inner_radius, outer_radius, text_size):
        """ Builds required actors.

        Parameters
        ----------
        inner_radius: int
            The inner radius of the sliding disk.
        outer_radius: int
            The outer radius of the sliding disk.
        text_size: int
            Size of the text that displays percentage.

        """
        # Slider Line
        self.slider_line = Rectangle2D(size=(self.length, self.line_width),
                                       center=self.center).actor
        self.slider_line.GetProperty().SetColor(1, 0, 0)
        # /Slider Line

        # Slider Disk
        # Create source
        disk = vtk.vtkDiskSource()
        disk.SetInnerRadius(inner_radius)
        disk.SetOuterRadius(outer_radius)
        disk.SetRadialResolution(10)
        disk.SetCircumferentialResolution(50)
        disk.Update()

        # Mapper
        mapper = vtk.vtkPolyDataMapper2D()
        mapper.SetInputConnection(disk.GetOutputPort())

        # Actor
        self.slider_disk = vtk.vtkActor2D()
        self.slider_disk.SetMapper(mapper)
        # /Slider Disk

        # Slider Text
        self.text = TextBlock2D()
        self.text.position = (self.left_x_position - 50, self.center[1] - 10)
        self.text.font_size = text_size
        # /Slider Text

    def get_actors(self):
        """ Returns the actors that compose this UI component.

        """
        return [self.slider_line, self.slider_disk, self.text.get_actor()]

    def set_position(self, position):
        """ Sets the disk's position.

        Parameters
        ----------
        position : (float, float)
            The absolute position of the disk (x, y).

        """
        x_position = position[0]

        if x_position < self.center[0] - self.length/2:
            x_position = self.center[0] - self.length/2

        if x_position > self.center[0] + self.length/2:
            x_position = self.center[0] + self.length/2

        self.current_state = x_position
        self.update()

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        value_range = self.max_value - self.min_value
        self.ratio = (value - self.min_value) / value_range

    @property
    def ratio(self):
        return self._ratio

    @ratio.setter
    def ratio(self, ratio):
        position_x = self.left_x_position + ratio*self.length
        self.set_position((position_x, None))

    def format_text(self):
        """ Returns formatted text to display along the slider. """
        if callable(self.text_template):
            return self.text_template(self)

        return self.text_template.format(ratio=self.ratio, value=self.value)

    def update(self):
        """ Updates the slider. """

        # Compute the ratio determined by the position of the slider disk.
        length = float(self.right_x_position - self.left_x_position)
        assert length == self.length
        self._ratio = (self.current_state - self.left_x_position) / length

        # Compute the selected value considering min_value and max_value.
        value_range = self.max_value - self.min_value
        self._value = self.min_value + self.ratio*value_range

        # Update text disk actor.
        self.slider_disk.SetPosition(self.current_state, self.center[1])

        # Update text.
        text = self.format_text()
        self.text.message = text
        offset_x = 8 * len(text) / 2.
        offset_y = 30
        self.text.position = (self.current_state - offset_x,
                              self.center[1] - offset_y)

    def set_center(self, position):
        """ Sets the center of the slider to position.

        Parameters
        ----------
        position : (float, float)
            The new center of the whole slider (x, y).

        """
        self.slider_line.SetPosition(position[0] - self.length / 2,
                                     position[1] - self.line_width / 2)

        x_change = position[0] - self.center[0]
        self.current_state += x_change
        self.center = position
        self.left_x_position = position[0] - self.length / 2
        self.right_x_position = position[0] + self.length / 2
        self.set_position((self.current_state, self.center[1]))

    @staticmethod
    def line_click_callback(i_ren, obj, slider):
        """ Update disk position and grab the focus.

        Parameters
        ----------
        i_ren : :class:`CustomInteractorStyle`
        obj : :class:`vtkActor`
            The picked actor
        slider : :class:`LineSlider2D`

        """
        position = i_ren.event.position
        slider.set_position(position)
        i_ren.force_render()
        i_ren.event.abort()  # Stop propagating the event.

    @staticmethod
    def disk_press_callback(i_ren, obj, slider):
        """ Only need to grab the focus.

        Parameters
        ----------
        i_ren : :class:`CustomInteractorStyle`
        obj : :class:`vtkActor`
            The picked actor
        slider : :class:`LineSlider2D`

        """
        i_ren.event.abort()  # Stop propagating the event.

    @staticmethod
    def disk_move_callback(i_ren, obj, slider):
        """ Actual disk movement.

        Parameters
        ----------
        i_ren : :class:`CustomInteractorStyle`
        obj : :class:`vtkActor`
            The picked actor
        slider : :class:`LineSlider2D`

        """
        position = i_ren.event.position
        slider.set_position(position)
        i_ren.force_render()
        i_ren.event.abort()  # Stop propagating the event.

    def handle_events(self, actor):
        """ Handle all events for the LineSlider.
        Base method needs to be overridden due to multiple actors.

        """
        self.add_callback(self.slider_line, "LeftButtonPressEvent",
                          self.line_click_callback, 1)
        self.add_callback(self.slider_disk, "LeftButtonPressEvent",
                          self.disk_press_callback)
        self.add_callback(self.slider_disk, "MouseMoveEvent",
                          self.disk_move_callback)
        self.add_callback(self.slider_line, "MouseMoveEvent",
                          self.disk_move_callback)


class DiskSlider2D(UI):
    """ A disk slider.

    A disk moves along the boundary of a ring.
    Goes from 0-360 degrees.

    Attributes
    ----------
    base_disk_center: (float, float)
        Position of the system.
    slider_inner_radius: int
        Inner radius of the base disk.
    slider_outer_radius: int
        Outer radius of the base disk.
    slider_radius: float
        Average radius of the base disk.
    handle_outer_radius: int
        Outer radius of the slider's handle.
    handle_inner_radius: int
        Inner radius of the slider's handle.
    previous_value: float
        Value of Rotation of the actor before the current value.
    initial_value: float
        Initial Value of Rotation of the actor assigned on creation of object.

    """
    def __init__(self, position=(0, 0),
                 initial_value=180, min_value=0, max_value=360,
                 slider_inner_radius=40, slider_outer_radius=44,
                 handle_inner_radius=10, handle_outer_radius=0,
                 text_size=16,
                 text_template="{ratio:.0%}"):

        """
        Parameters
        ----------
        position : (float, float)
            Position (x, y) of the slider's center.
        initial_value : float
            Initial value of the slider.
        min_value : float
            Minimum value of the slider.
        max_value : float
            Maximum value of the slider.
        slider_inner_radius : int
            Inner radius of the base disk.
        slider_outer_radius : int
            Outer radius of the base disk.
        handle_outer_radius : int
            Outer radius of the slider's handle.
        handle_inner_radius : int
            Inner radius of the slider's handle.
        text_size : int
            Size of the text to display alongside the slider (pt).
        text_template : str, callable
            If str, text template can contain one or multiple of the
            replacement fields: `{value:}`, `{ratio:}`, `{angle:}`.
            If callable, this instance of `:class:DiskSlider2D` will be
            passed as argument to the text template function.

        """
        super(DiskSlider2D, self).__init__()
        self.center = np.array(position)
        self.min_value = min_value
        self.max_value = max_value
        self.initial_value = initial_value
        self.slider_inner_radius = slider_inner_radius
        self.slider_outer_radius = slider_outer_radius
        self.handle_inner_radius = handle_inner_radius
        self.handle_outer_radius = handle_outer_radius
        self.slider_radius = (slider_inner_radius + slider_outer_radius) / 2.

        self.handle = None
        self.base_disk = None

        self.text = None
        self.text_size = text_size
        self.text_template = text_template

        self.build_actors()

        # By setting the value, it also updates everything.
        self.value = initial_value
        self.previous_value = initial_value
        self.handle_events(None)

    def build_actors(self):
        """ Builds actors for the system.

        """
        base_disk = vtk.vtkDiskSource()
        base_disk.SetInnerRadius(self.slider_inner_radius)
        base_disk.SetOuterRadius(self.slider_outer_radius)
        base_disk.SetRadialResolution(10)
        base_disk.SetCircumferentialResolution(50)
        base_disk.Update()

        base_disk_mapper = vtk.vtkPolyDataMapper2D()
        base_disk_mapper.SetInputConnection(base_disk.GetOutputPort())

        self.base_disk = vtk.vtkActor2D()
        self.base_disk.SetMapper(base_disk_mapper)
        self.base_disk.GetProperty().SetColor(1, 0, 0)
        self.base_disk.SetPosition(self.center)

        handle = vtk.vtkDiskSource()
        handle.SetInnerRadius(self.handle_inner_radius)
        handle.SetOuterRadius(self.handle_outer_radius)
        handle.SetRadialResolution(10)
        handle.SetCircumferentialResolution(50)
        handle.Update()

        handle_mapper = vtk.vtkPolyDataMapper2D()
        handle_mapper.SetInputConnection(handle.GetOutputPort())

        self.handle = vtk.vtkActor2D()
        self.handle.SetMapper(handle_mapper)

        self.text = TextBlock2D()
        offset = np.array((16., 8.))
        self.text.position = self.center - offset
        self.text.font_size = self.text_size

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        value_range = self.max_value - self.min_value
        self.ratio = (value - self.min_value) / value_range

    @property
    def previous_value(self):
        return self._previous_value

    @previous_value.setter
    def previous_value(self, previous_value):
        self._previous_value = previous_value

    @property
    def ratio(self):
        return self._ratio

    @ratio.setter
    def ratio(self, ratio):
        self.angle = ratio * TWO_PI

    @property
    def angle(self):
        """ Angle (in rad) the handle makes with x-axis """
        return self._angle

    @angle.setter
    def angle(self, angle):
        self._angle = angle % TWO_PI  # Wraparound
        self.update()

    def format_text(self):
        """ Returns formatted text to display along the slider. """
        if callable(self.text_template):
            return self.text_template(self)

        return self.text_template.format(ratio=self.ratio, value=self.value,
                                         angle=np.rad2deg(self.angle))

    def update(self):
        """ Updates the slider. """

        # Compute the ratio determined by the position of the slider disk.
        self._ratio = self.angle / TWO_PI

        # Compute the selected value considering min_value and max_value.
        value_range = self.max_value - self.min_value
        try:
            self._previous_value = self.value
        except:
            self._previous_value = self.initial_value
        self._value = self.min_value + self.ratio*value_range

        # Update text disk actor.
        x = self.slider_radius * np.cos(self.angle) + self.center[0]
        y = self.slider_radius * np.sin(self.angle) + self.center[1]
        self.handle.SetPosition(x, y)

        # Update text.
        text = self.format_text()
        self.text.message = text

    def get_actors(self):
        """ Returns the actors that compose this UI component.

        """
        return [self.base_disk, self.handle, self.text.get_actor()]

    def move_handle(self, click_position):
        """Moves the slider's handle.

        Parameters
        ----------
        click_position: (float, float)
            Position of the mouse click.

        """
        x, y = np.array(click_position) - self.center
        angle = np.arctan2(y, x)
        if angle < 0:
            angle += TWO_PI

        self.angle = angle

    def set_center(self, position):
        """ Changes the slider's center position.

        Parameters
        ----------
        position : (float, float)
            New position (x, y).

        """
        position = np.array(position)
        offset = position - self.center
        self.base_disk.SetPosition(position)
        self.handle.SetPosition(*(offset + self.handle.GetPosition()))
        self.text.position += offset
        self.center = position

    @staticmethod
    def base_disk_click_callback(i_ren, obj, slider):
        """ Update disk position and grab the focus.

        Parameters
        ----------
        i_ren : :class:`CustomInteractorStyle`
        obj : :class:`vtkActor`
            The picked actor
        slider : :class:`DiskSlider2D`

        """
        click_position = i_ren.event.position
        slider.move_handle(click_position=click_position)
        i_ren.force_render()
        i_ren.event.abort()  # Stop propagating the event.

    @staticmethod
    def handle_move_callback(i_ren, obj, slider):
        """ Move the slider's handle.

        Parameters
        ----------
        i_ren : :class:`CustomInteractorStyle`
        obj : :class:`vtkActor`
            The picked actor
        slider : :class:`DiskSlider2D`

        """
        click_position = i_ren.event.position
        slider.move_handle(click_position=click_position)
        i_ren.force_render()
        i_ren.event.abort()  # Stop propagating the event.

    @staticmethod
    def handle_press_callback(i_ren, obj, slider):
        """ This is only needed to grab the focus.

        Parameters
        ----------
        i_ren : :class:`CustomInteractorStyle`
        obj : :class:`vtkActor`
            The picked actor
        slider : :class:`DiskSlider2D`

        """
        i_ren.event.abort()  # Stop propagating the event.

    def handle_events(self, actor):
        """ Handle all default slider events.

        """
        self.add_callback(self.base_disk, "LeftButtonPressEvent",
                          self.base_disk_click_callback, 1)
        self.add_callback(self.handle, "LeftButtonPressEvent",
                          self.handle_press_callback)
        self.add_callback(self.base_disk, "MouseMoveEvent",
                          self.handle_move_callback)
        self.add_callback(self.handle, "MouseMoveEvent",
                          self.handle_move_callback)


class FileSelectMenu2D(UI):
    """ A menu to select files in the current folder.

    Can go to new folder, previous folder and select a file
    and keep it in a variable.

    Attributes
    ----------
    n_text_actors: int
        The number of text actors. Calculated dynamically.
    selected_file: string
        Current selected file.
    text_item_list: list(:class:`FileSelectMenuText2D`)
        List of FileSelectMenuText2Ds - both visible and invisible.
    window_offset: int
        Used for scrolling.
        Tells you the index of the first visible FileSelectMenuText2D
        object.
    size: (float, float)
        The size of the system (x, y) in pixels.
    font_size: int
        The font size in pixels.
    line_spacing: float
        Distance between menu text items in pixels.
    parent_ui: :class:`UI`
        The UI component this object belongs to.
    extensions: list(string)
            List of extensions to be shown as files.

    """

    def __init__(self, size, font_size, position, parent, extensions,
                 directory_path, reverse_scrolling=False, line_spacing=1.4):
        """
        Parameters
        ----------
        size: (float, float)
            The size of the system (x, y) in pixels.
        font_size: int
            The font size in pixels.
        parent: :class:`UI`
            The UI component this object belongs to.
            This will be useful when this UI element is used as a
            part of other UI elements, like a file save dialog.
        position: (float, float)
            The initial position (x, y) in pixels.
        reverse_scrolling: {True, False}
            If True, scrolling up will move the list of files down.
        line_spacing: float
            Distance between menu text items in pixels.
        extensions: list(string)
            List of extensions to be shown as files.
        directory_path: string
            Path of the directory where this dialog should open.
            Example: os.getcwd()

        """
        super(FileSelectMenu2D, self).__init__()

        self.size = size
        self.font_size = font_size
        self.parent_ui = parent
        self.reverse_scrolling = reverse_scrolling
        self.line_spacing = line_spacing
        self.extensions = extensions

        self.n_text_actors = 0  # Initialisation Value
        self.text_item_list = []
        self.selected_file = ""
        self.window_offset = 0
        self.current_directory = directory_path
        self.buttons = dict()

        self.menu = self.build_actors(position)

        self.fill_text_actors()
        self.handle_events(None)

    def add_to_renderer(self, ren):
        self.menu.add_to_renderer(ren)
        super(FileSelectMenu2D, self).add_to_renderer(ren)
        for menu_text in self.text_item_list:
            menu_text.add_to_renderer(ren)

    def get_actors(self):
        """ Returns the actors that compose this UI component.

        """
        return [self.buttons["up"], self.buttons["down"]]

    def build_actors(self, position):
        """ Builds the number of text actors that will fit in the given size.

        Allots them positions in the panel, which is only there to allot positions,
        otherwise the panel itself is invisible.

        Parameters
        ----------
        position: (float, float)
            Position of the panel (x, y) in pixels.

        """
        # Calculating the number of text actors.
        self.n_text_actors = int(self.size[1]/(self.font_size*self.line_spacing))

        # This panel is just to facilitate the addition of actors at the right positions
        panel = Panel2D(center=position, size=self.size, color=(1, 1, 1))

        # Initialisation of empty text actors
        for i in range(self.n_text_actors):

            text = FileSelectMenuText2D(position=(0, 0), font_size=self.font_size,
                                        file_select=self)
            text.parent_UI = self.parent_ui
            self.ui_list.append(text)
            self.text_item_list.append(text)

            panel.add_element(text, 'relative',
                              (0.1,
                               float(self.n_text_actors-i - 1) /
                               float(self.n_text_actors)))

        up_button = Button2D({"up": read_viz_icons(fname="arrow-up.png")})
        panel.add_element(up_button, 'relative', (0.95, 0.95))
        self.buttons["up"] = up_button

        down_button = Button2D({"down": read_viz_icons(fname="arrow-down.png")})
        panel.add_element(down_button, 'relative', (0.95, 0.05))
        self.buttons["down"] = down_button

        return panel

    @staticmethod
    def up_button_callback(i_ren, obj, file_select_menu):
        """ Pressing up button scrolls up in the menu.

        Parameters
        ----------
        i_ren: :class:`CustomInteractorStyle`
        obj: :class:`vtkActor`
            The picked actor
        file_select_menu: :class:`FileSelectMenu2D`

        """
        all_file_names = file_select_menu.get_all_file_names()

        if (file_select_menu.n_text_actors +
                file_select_menu.window_offset) <= len(all_file_names):
            if file_select_menu.window_offset > 0:
                file_select_menu.window_offset -= 1
                file_select_menu.fill_text_actors()

        i_ren.force_render()
        i_ren.event.abort()  # Stop propagating the event.

    @staticmethod
    def down_button_callback(i_ren, obj, file_select_menu):
        """ Pressing down button scrolls down in the menu.

        Parameters
        ----------
        i_ren: :class:`CustomInteractorStyle`
        obj: :class:`vtkActor`
            The picked actor
        file_select_menu: :class:`FileSelectMenu2D`

        """
        all_file_names = file_select_menu.get_all_file_names()

        if (file_select_menu.n_text_actors +
                file_select_menu.window_offset) < len(all_file_names):
            file_select_menu.window_offset += 1
            file_select_menu.fill_text_actors()

        i_ren.force_render()
        i_ren.event.abort()  # Stop propagating the event.

    def fill_text_actors(self):
        """ Fills file/folder names to text actors.

        The list is truncated if the number of file/folder names is greater
        than the available number of text actors.

        """
        # Flush all the text actors
        for text_item in self.text_item_list:
            text_item.text_actor.message = ""
            text_item.text_actor.actor.SetVisibility(False)

        all_file_names = self.get_all_file_names()

        clipped_file_names = all_file_names[self.window_offset:self.n_text_actors + self.window_offset]

        # Allot file names as in the above list
        i = 0
        for file_name in clipped_file_names:
            self.text_item_list[i].text_actor.actor.SetVisibility(True)
            self.text_item_list[i].set_attributes(file_name[0], file_name[1])
            if file_name[0] == self.selected_file:
                self.text_item_list[i].mark_selected()
            i += 1

    def get_all_file_names(self):
        """ Gets file and directory names.

        Returns
        -------
        all_file_names: list(string)
            List of all file and directory names as string.

        """
        all_file_names = []

        directory_names = self.get_directory_names()
        for directory_name in directory_names:
            all_file_names.append((directory_name, "directory"))

        file_names = self.get_file_names()
        for file_name in file_names:
            all_file_names.append((file_name, "file"))

        return all_file_names

    def get_directory_names(self):
        """ Re-allots file names to the text actors.

        Uses FileSelectMenuText2D for selecting files and folders.

        Returns
        -------
        directory_names: list(string)
            List of all directory names as string.

        """
        # A list of directory names in the current directory
        directory_names = next(os.walk(self.current_directory))[1]
        directory_names = [os.path.basename(os.path.abspath(dn)) for dn in directory_names]
        directory_names = ["../"] + directory_names

        return directory_names

    def get_file_names(self):
        """ Re-allots file names to the text actors.

        Uses FileSelectMenuText2D for selecting files and folders.

        Returns
        -------
        file_names: list(string)
            List of all file names as string.

        """
        # A list of file names with extension in the current directory
        file_names = []
        for extension in self.extensions:
            file_names += glob.glob(self.current_directory + "/*." + extension)
        file_names = [os.path.basename(os.path.abspath(fn)) for fn in file_names]
        return file_names

    def select_file(self, file_name):
        """ Changes the selected file name.

        Parameters
        ----------
        file_name: string
            Name of the file.

        """
        self.selected_file = file_name

    def set_center(self, position):
        """ Sets the elements center.

        Parameters
        ----------
        position: (float, float)
            New position (x, y) in pixels.

        """
        self.menu.set_center(position=position)

    def handle_events(self, actor):
        self.add_callback(self.buttons["up"].actor, "LeftButtonPressEvent",
                          self.up_button_callback)
        self.add_callback(self.buttons["down"].actor, "LeftButtonPressEvent",
                          self.down_button_callback)

        # Handle mouse wheel events
        up_event = "MouseWheelForwardEvent"
        down_event = "MouseWheelBackwardEvent"
        if self.reverse_scrolling:
            up_event, down_event = down_event, up_event  # Swap events

        self.add_callback(self.menu.get_actors()[0], up_event,
                          self.up_button_callback)
        self.add_callback(self.menu.get_actors()[0], down_event,
                          self.down_button_callback)

        for text_ui in self.text_item_list:
            self.add_callback(text_ui.text_actor.get_actors()[0], up_event,
                              self.up_button_callback)
            self.add_callback(text_ui.text_actor.get_actors()[0], down_event,
                              self.down_button_callback)


class FileSelectMenuText2D(UI):
    """ The text to select folder in a file select menu.

    Provides a callback to change the directory.

    Attributes
    ----------
    file_name: string
        The name of the file the text is displaying.
    file_type: string
        Whether the file is a file or directory.
    file_select: :class:`FileSelect2D`
        The FileSelectMenu2D reference this text belongs to.

    """

    def __init__(self, font_size, position, file_select):
        """
        Parameters
        ----------
        font_size: int
            The font size of the text in pixels.
        position: (float, float)
            Absolute text position (x, y) in pixels.
        file_select: :class:`FileSelect2D`
            The FileSelectMenu2D reference this text belongs to.

        """
        super(FileSelectMenuText2D, self).__init__()

        self.file_name = ""
        self.file_type = ""
        self.file_select = file_select

        self.text_actor = self.build_actor(position=position, font_size=font_size)

        self.handle_events(self.text_actor.get_actor())

        self.on_left_mouse_button_clicked = self.left_button_clicked

    def build_actor(self, position, text="Text", color=(1, 1, 1), font_family='Arial',
                    justification='left', bold=False, italic=False,
                    shadow=False, font_size='14'):
        """ Builds a text actor.

        Parameters
        ----------
        text: string
            The initial text while building the actor.
        position: (float, float)
            The text position (x, y) in pixels.
        color: (float, float, float)
            Values must be between 0-1 (RGB).
        font_family: string
            Currently only supports Arial.
        justification: string
            Text justification - left, right or center.
        bold: bool
            Whether or not the text is bold.
        italic: bool
            Whether or not the text is italicized.
        shadow: bool
            Whether or not the text has shadow.
        font_size: int
            The font size of the text in pixels.

        Returns
        -------
        text_actor: :class:`TextBlock2D`
            The base text actor.

        """
        text_actor = TextBlock2D()
        text_actor.position = position
        text_actor.message = text
        text_actor.font_size = font_size
        text_actor.font_family = font_family
        text_actor.justification = justification
        text_actor.bold = bold
        text_actor.italic = italic
        text_actor.shadow = shadow
        text_actor.color = color

        if vtk.vtkVersion.GetVTKVersion() <= "6.2.0":
            pass
        else:
            text_actor.actor.GetTextProperty().SetBackgroundColor(1, 1, 1)
            text_actor.actor.GetTextProperty().SetBackgroundOpacity(1.0)

        text_actor.actor.GetTextProperty().SetColor(0, 0, 0)
        text_actor.actor.GetTextProperty().SetLineSpacing(1)

        return text_actor

    def get_actors(self):
        """ Returns the actors that compose this UI component.

        """
        return [self.text_actor.get_actor()]

    def set_attributes(self, file_name, file_type):
        """  Set attributes (file name and type) of this component.

        This function is for use by a FileSelectMenu2D to set the
        current file_name and file_type for this FileSelectMenuText2D
        component.

        Parameters
        ----------
        file_name: string
            The name of the file.
        file_type: string
            File type = directory or file.

        """
        self.file_name = file_name
        self.file_type = file_type
        self.text_actor.message = file_name

        if vtk.vtkVersion.GetVTKVersion() <= "6.2.0":
            self.text_actor.get_actor().GetTextProperty().SetColor(1, 1, 1)
            if file_type != "file":
                self.text_actor.get_actor().GetTextProperty().SetBold(True)

        else:
            if file_type == "file":
                self.text_actor.get_actor().GetTextProperty().SetBackgroundColor(0, 0, 0)
                self.text_actor.get_actor().GetTextProperty().SetColor(1, 1, 1)
            else:
                self.text_actor.get_actor().GetTextProperty().SetBackgroundColor(1, 1, 1)
                self.text_actor.get_actor().GetTextProperty().SetColor(0, 0, 0)

    def mark_selected(self):
        """ Changes the background color of the actor.

        """
        if vtk.vtkVersion.GetVTKVersion() <= "6.2.0":
            self.text_actor.actor.GetTextProperty().SetColor(1, 0, 0)
        else:
            self.text_actor.actor.GetTextProperty().SetBackgroundColor(1, 0, 0)
            self.text_actor.actor.GetTextProperty().SetBackgroundOpacity(1.0)

    @staticmethod
    def left_button_clicked(i_ren, obj, file_select_text):
        """ A callback to handle left click for this UI element.

        Parameters
        ----------
        i_ren: :class:`CustomInteractorStyle`
        obj: :class:`vtkActor`
            The picked actor
        file_select_text: :class:`FileSelectMenuText2D`

        """

        if file_select_text.file_type == "directory":
            file_select_text.file_select.select_file(file_name="")
            file_select_text.file_select.window_offset = 0
            file_select_text.file_select.current_directory = os.path.abspath(
                os.path.join(file_select_text.file_select.current_directory,
                             file_select_text.text_actor.message))
            file_select_text.file_select.window = 0
            file_select_text.file_select.fill_text_actors()
        else:
            file_select_text.file_select.select_file(
                file_name=file_select_text.file_name)
            file_select_text.file_select.fill_text_actors()
            file_select_text.mark_selected()

        i_ren.force_render()
        i_ren.event.abort()  # Stop propagating the event.

    def set_center(self, position):
        """ Sets the text center to position.

        Parameters
        ----------
        position: (float, float)
            The new position (x, y) in pixels.
        """
        self.text_actor.position = position
