import numpy as np

from dipy.viz.interactor import CustomInteractorStyle

from dipy.utils.optpkg import optional_package

# Allow import, but disable doctests if we don't have vtk.
vtk, have_vtk, setup_module = optional_package('vtk')

if have_vtk:
    version = vtk.vtkVersion.GetVTKSourceVersion().split(' ')[-1]
    major_version = vtk.vtkVersion.GetVTKMajorVersion()
else:
    vtkInteractorStyleUser = object


class UI(object):
    """ An umbrella class for all UI elements.
    While adding UI elements to the renderer, we go over all the sub-elements
    that come with it and add those to the renderer automatically.

    Attributes
    ----------
    ui_param : object
        This is an attribute that can be passed to the UI object by the interactor.
    ui_list : list(UI)
        This is used when there are more than one UI elements inside
        a UI element. They're all automatically added to the renderer at the same time
        as this one.
    parent_UI: UI
        Reference to the parent UI element. This is useful of there is a parent
        UI element and its reference needs to be passed down to the child.
    on_left_mouse_button_pressed: function
    on_left_mouse_button_drag: function
    on_right_mouse_button_pressed: function
    on_right_mouse_button_drag: function
    """

    def __init__(self):
        self.ui_param = None
        self.ui_list = list()

        self.parent_UI = None
        self._callbacks = []

        self.left_button_state = "released"
        self.right_button_state = "released"

        self.handle_events()

        self.on_left_mouse_button_pressed = lambda i_ren, obj, element: None
        self.on_left_mouse_button_drag = lambda i_ren, obj, element: None
        self.on_right_mouse_button_pressed = lambda i_ren, obj, element: None
        self.on_right_mouse_button_drag = lambda i_ren, obj, element: None

    def get_actors(self):
        """ Returns the actors that compose this UI component. """
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
        """
        # Actually since we need an interactor style we will add the callback
        # only when this UI component is added to the renderer.
        self._callbacks.append((prop, event_type, callback, priority))

    def set_center(self, position):
        """ Sets the center of the UI component

        Parameters
        ----------
        position : (float, float)
        """
        msg = "Subclasses of UI must implement `set_center(self, position)`."
        raise NotImplementedError(msg)

    def set_visibility(self, visibility):
        """ Sets visibility of this UI component and all its sub-components. """
        for actor in self.get_actors():
            actor.SetVisibility(visibility)

    def handle_events(self):
        self.add_callback("LeftButtonPressEvent", self.left_button_click_callback)
        self.add_callback("LeftButtonReleaseEvent", self.left_button_release_callback)
        self.add_callback("RightButtonPressEvent", self.right_button_click_callback)
        self.add_callback("RightButtonReleaseEvent", self.right_button_release_callback)
        self.add_callback("MouseMoveEvent", self.mouse_move_callback)

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


class Button2D(UI):
    """A 2D overlay button and is of type vtkTexturedActor2D.
    Currently supports:
    - Multiple icons.
    - Switching between icons.

    Attributes
    ----------
    size: (float, float)
        Button Size.
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
        self.icons = self.build_icons(icon_fnames)
        self.icon_names = list(self.icons.keys())
        self.current_icon_id = 0
        self.current_icon_name = self.icon_names[self.current_icon_id]
        self.actor = self.build_actor(self.icons[self.current_icon_name])
        self.size = size
        super(Button2D, self).__init__()

    def build_icons(self, icon_fnames):
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
                print("Warning: A specified icon file is not in the PNG format. SKIPPING.")
            else:
                png = vtk.vtkPNGReader()
                png.SetFileName(icon_fname)
                png.Update()
                icons[icon_name] = png.GetOutput()

        return icons

    @property
    def size(self):
        """Gets the button size."""
        return self._size

    @size.setter
    def size(self, size):
        """Sets the button size.

        Parameters
        ----------
        size : (float, float)
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
        """Gets the button's color."""
        color = self.actor.GetProperty().GetColor()
        return np.asarray(color)

    @color.setter
    def color(self, color):
        """Sets the button's color.

        Parameters
        ----------
        color : (float, float, float)
        """
        self.actor.GetProperty().SetColor(*color)

    def scale(self, size):
        """Scales the button.

        Parameters
        ----------
        size : (float, float)
        """
        self.size *= size

    def build_actor(self, icon):
        """ Return an image as a 2D actor with a specific position.

        Parameters
        ----------
        icon : vtkImageData
        Returns
        -------
        button : vtkTexturedActor2D
        """
        # This is highly inspired by
        # https://github.com/Kitware/VTK/blob/c3ec2495b183e3327820e927af7f8f90d34c3474/Interaction/Widgets/vtkBalloonRepresentation.cxx#L47

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
        """ Returns the actors that compose this UI component. """
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
        """
        new_position = np.asarray(position) - self.size / 2.
        self.actor.SetPosition(*new_position)


class Rectangle2D(UI):
    """A 2D rectangle sub-classed from UI.
    Uses vtkPolygon.

    Attributes
    ----------
    size : (float, float)
        The size of the rectangle.
    """

    def __init__(self, size, center=(0, 0), color=(1, 1, 1), opacity=1.0):
        """
        Initializes a rectangle.

        Parameters
        ----------
        size : (float, float)
        center : (float, float)
        color : (float, float, float)
            Must take values between 0-1.
        opacity : float
        """
        self.size = size
        self.actor = self.build_actor(size=size, center=center, color=color, opacity=opacity)
        super(Rectangle2D, self).__init__()

    def get_actors(self):
        """ Returns the actors that compose this UI component. """
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
        center : (float, float)
        color : (float, float, float)
            Must be between 0-1
        opacity : float

        Returns
        -------
        actor : vtkActor2D
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
        """
        self.actor.SetPosition(position[0] - self.size[0] / 2, position[1] - self.size[1] / 2)


class Panel2D(UI):
    """ A 2D UI Panel.
    Can contain one or more UI elements.

    Attributes
    ----------
    center : (float, float)
    size : (float, float)
    alignment : [left, right]
        Alignment of the panel with respect to the overall screen.
    """

    def __init__(self, center, size, color=(0.1, 0.1, 0.1), opacity=0.7, align="left"):
        """
        Parameters
        ----------
        center : (float, float)
        size : (float, float)
        color : (float, float, float)
            Values must be between 0-1
        opacity : float
        align : [left, right]
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
        """ Returns the panel actor. """
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
    def left_button_press(i_ren, obj, element):
        click_position = i_ren.event.position
        element.ui_param = (click_position[0] - element.panel.actor.GetPosition()[0] - element.panel.size[0] / 2,
                            click_position[1] - element.panel.actor.GetPosition()[1] - element.panel.size[1] / 2)
        i_ren.event.abort()  # Stop propagating the event.

    @staticmethod
    def left_button_drag(i_ren, obj, element):
        click_position = i_ren.event.position
        if element.ui_param is not None:
            element.set_center((click_position[0] - element.ui_param[0], click_position[1] - element.ui_param[1]))
        i_ren.force_render()

    def re_align(self, window_size_change):
        """ Re-organises the elements in case the
        window size is changed

        Parameters
        ----------
        window_size_change : (int, int)
        """
        if self.alignment == "left":
            pass
        elif self.alignment == "right":
            self.set_center((self.center[0] + window_size_change[0], self.center[1] + window_size_change[1]))
        else:
            raise ValueError("You can only left-align or right-align objects in a panel.")
