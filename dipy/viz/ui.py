from _warnings import warn

import numpy as np

from dipy.viz.interactor import CustomInteractorStyle

from dipy.utils.optpkg import optional_package

# Allow import, but disable doctests if we don't have vtk.
vtk, have_vtk, setup_module = optional_package('vtk')

if have_vtk:
    version = vtk.vtkVersion.GetVTKSourceVersion().split(' ')[-1]
    major_version = vtk.vtkVersion.GetVTKMajorVersion()
    vtkTextActor = vtk.vtkTextActor
else:
    vtkTextActor = object


class UI(object):
    """ An umbrella class for all UI elements.

    While adding UI elements to the renderer, we go over all the sub-elements
    that come with it and add those to the renderer automatically.

    Attributes
    ----------
    ui_param : object
        This is an attribute that can be passed to the UI object by the interactor.
    ui_list : list of :class:`UI`
        This is used when there are more than one UI elements inside
        a UI element. They're all automatically added to the renderer at the same time
        as this one.
    parent_ui: UI
        Reference to the parent UI element. This is useful of there is a parent
        UI element and its reference needs to be passed down to the child.
    on_left_mouse_button_pressed: function
        Callback function for when the left mouse button is pressed.
    on_left_mouse_button_drag: function
        Callback function for when the left mouse button is dragged.
    on_right_mouse_button_pressed: function
        Callback function for when the right mouse button is pressed.
    on_right_mouse_button_drag: function
        Callback function for when the right mouse button is dragged.

    """

    def __init__(self):
        self.ui_param = None
        self.ui_list = list()

        self.parent_ui = None
        self._callbacks = []

        self.left_button_state = "released"
        self.right_button_state = "released"

        self.handle_events()

        self.on_left_mouse_button_pressed = lambda i_ren, obj, element: None
        self.on_left_mouse_button_drag = lambda i_ren, obj, element: None
        self.on_right_mouse_button_pressed = lambda i_ren, obj, element: None
        self.on_right_mouse_button_drag = lambda i_ren, obj, element: None
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

    def handle_events(self):
        self.add_callback("LeftButtonPressEvent", self.left_button_click_callback)
        self.add_callback("LeftButtonReleaseEvent", self.left_button_release_callback)
        self.add_callback("RightButtonPressEvent", self.right_button_click_callback)
        self.add_callback("RightButtonReleaseEvent", self.right_button_release_callback)
        self.add_callback("MouseMoveEvent", self.mouse_move_callback)
        self.add_callback("KeyPressEvent", self.key_press_callback)

    @staticmethod
    def left_button_click_callback(i_ren, obj, self):
        self.left_button_state = "clicked"
        i_ren.event.abort()

    @staticmethod
    def left_button_release_callback(i_ren, obj, self):
        if self.left_button_state == "clicked":
            self.on_left_mouse_button_pressed(i_ren, obj, self)
        self.left_button_state = "released"

    @staticmethod
    def right_button_click_callback(i_ren, obj, self):
        self.right_button_state = "clicked"
        i_ren.event.abort()

    @staticmethod
    def right_button_release_callback(i_ren, obj, self):
        if self.right_button_state == "clicked":
            self.on_right_mouse_button_pressed(i_ren, obj, self)
        self.right_button_state = "released"

    @staticmethod
    def mouse_move_callback(i_ren, obj, self):
        if self.left_button_state == "clicked" or self.left_button_state == "dragging":
            self.left_button_state = "dragging"
            self.on_left_mouse_button_drag(i_ren, obj, self)
        elif self.right_button_state == "clicked" or self.right_button_state == "dragging":
            self.right_button_state = "dragging"
            self.on_right_mouse_button_drag(i_ren, obj, self)
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
        self.icon_extents = dict()
        self.icons = self.__build_icons(icon_fnames)
        self.icon_names = list(self.icons.keys())
        self.current_icon_id = 0
        self.current_icon_name = self.icon_names[self.current_icon_id]
        self.actor = self.build_actor(self.icons[self.current_icon_name])
        self.size = size
        super(Button2D, self).__init__()

    def __build_icons(self, icon_fnames):
        """ Converts file names to vtkImageDataGeometryFilters.

        A pre-processing step to prevent re-read of file names during every state change.

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

    def add_callback(self, event_type, callback):
        """ Adds events to button actor.

        Parameters
        ----------
        event_type : string
            event code
        callback : function
            callback function

        """
        super(Button2D, self).add_callback(self.actor, event_type, callback)

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
        self.size = size
        self.actor = self.build_actor(size=size, center=center,
                                      color=color, opacity=opacity)
        super(Rectangle2D, self).__init__()

    def get_actors(self):
        """ Returns the actors that compose this UI component.

        """
        return [self.actor]

    def add_callback(self, event_type, callback):
        """ Adds events to rectangle actor.

        Parameters
        ----------
        event_type : string
            event code
        callback : function
            callback function

        """
        super(Rectangle2D, self).add_callback(self.actor, event_type, callback)

    def build_actor(self, size, center, color, opacity):
        """ Builds the text actor.

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
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetOpacity(opacity)
        actor.SetPosition(center[0] - self.size[0] / 2, center[1] - self.size[1] / 2)

        return actor

    def set_center(self, position):
        """ Sets the center to position.

        Parameters
        ----------
        position : (float, float)
            The new center of the rectangle (x, y).

        """
        self.actor.SetPosition(position[0] - self.size[0] / 2, position[1] - self.size[1] / 2)


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
        self.center = center
        self.size = size
        self.lower_limits = (self.center[0] - self.size[0] / 2,
                             self.center[1] - self.size[1] / 2)

        self.panel = Rectangle2D(size=size, center=center, color=color,
                                 opacity=opacity)

        self.element_positions = []
        self.element_positions.append([self.panel, 'relative', 0.5, 0.5])
        self.alignment = align
        super(Panel2D, self).__init__()
        self.on_left_mouse_button_pressed = self.left_button_press
        self.on_left_mouse_button_drag = self.left_button_drag

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

    def add_callback(self, event_type, callback):
        """ Adds events to an actor.

        Parameters
        ----------
        event_type : string
            event code
        callback : function
            callback function

        """
        super(Panel2D, self).add_callback(self.panel.actor, event_type, callback)

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
    def left_button_press(i_ren, obj, panel2d_object):
        click_position = i_ren.event.position
        panel2d_object.ui_param = (click_position[0] - panel2d_object.panel.actor.GetPosition()[0]
                                   - panel2d_object.panel.size[0] / 2,
                                   click_position[1] - panel2d_object.panel.actor.GetPosition()[1]
                                   - panel2d_object.panel.size[1] / 2)
        i_ren.event.abort()  # Stop propagating the event.

    @staticmethod
    def left_button_drag(i_ren, obj, panel2d_object):
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


class TextActor2D(object):
    """ Wraps over the default vtkTextActor and helps setting the text.

    Contains member functions for text formatting.

    Attributes
    ----------
    actor : :class:`vtkTextActor`

    """

    def __init__(self):
        self.actor = vtkTextActor()

    def get_actor(self):
        """ Returns the actor composing this element.

        Returns
        -------
        :class:`vtkTextActor`
            The actor composing this class.
        """
        return self.actor

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

        Currently defaults to Arial.
        # ToDo: Add other font families.

        Parameters
        ----------
        family : str
            The font family.

        """
        if family == 'Arial':
            self.actor.GetTextProperty().SetFontFamilyToArial()
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
        return self.actor.GetTextProperty().GetJustificationAsString()

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
        self.actor.SetDisplayPosition(*position)


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
        self.text = text
        self.actor = self.build_actor(self.text, position, color, font_size,
                                      font_family, justification, bold, italic, shadow)
        self.width = width
        self.height = height
        self.window_left = 0
        self.window_right = 0
        self.caret_pos = 0
        self.init = True
        super(TextBox2D, self).__init__()
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
        :class:`vtkActor2d`

        """
        text_actor = TextActor2D()
        text_actor.position = position
        text_actor.message = text
        text_actor.font_size = font_size
        text_actor.font_family = font_family
        text_actor.justification = justification
        text_actor.bold = bold
        text_actor.italic = italic
        text_actor.shadow = shadow
        if vtk.vtkVersion.GetVTKSourceVersion().split(' ')[-1] <= "6.2.0":
            pass
        else:
            text_actor.actor.GetTextProperty().SetBackgroundColor(1, 1, 1)
            text_actor.actor.GetTextProperty().SetBackgroundOpacity(1.0)
            text_actor.color = color

        return text_actor

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

    def add_callback(self, event_type, callback):
        """ Adds events to the text actor.

        Parameters
        ----------
        event_type : str
            event code
        callback : function
            callback function

        """
        super(TextBox2D, self).add_callback(self.actor.get_actor(), event_type, callback)

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
        self.text = self.text[:self.caret_pos] + character + self.text[self.caret_pos:]
        self.move_caret_right()
        if self.window_right - self.window_left == self.height * self.width - 1:
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
        if self.window_right - self.window_left == self.height * self.width - 1:
            if self.window_left > 0:
                self.left_move_left()
                self.right_move_left()

    def move_left(self):
        """ Handles left button press.

        """
        self.move_caret_left()
        if self.caret_pos == self.window_left - 1:
            if self.window_right - self.window_left == self.height * self.width - 1:
                self.left_move_left()
                self.right_move_left()

    def move_right(self):
        """ Handles right button press.

        """
        self.move_caret_right()
        if self.caret_pos == self.window_right + 1:
            if self.window_right - self.window_left == self.height * self.width - 1:
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
            ret_text = self.text[:self.caret_pos] + "_" + self.text[self.caret_pos:]
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
