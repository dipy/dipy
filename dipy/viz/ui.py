from __future__ import division
from _warnings import warn

import numpy as np

from dipy.data import read_viz_icons
from dipy.viz.interactor import CustomInteractorStyle
from dipy.viz.utils import set_input

from dipy.utils.optpkg import optional_package

# Allow import, but disable doctests if we don't have vtk.
vtk, have_vtk, setup_module = optional_package('vtk')

if have_vtk:
    version = vtk.vtkVersion.GetVTKVersion()
    VTK_MAJOR_VERSION = vtk.vtkVersion.GetVTKMajorVersion()

TWO_PI = 2 * np.pi


class UI(object):
    """ An umbrella class for all UI elements.

    While adding UI elements to the renderer, we go over all the sub-elements
    that come with it and add those to the renderer automatically.

    Attributes
    ----------
    position : (float, float)
        Absolute coordinates (x, y) of the lower-left corner of this
        UI component.
    center : (float, float)
        Absolute coordinates (x, y) of the center of this UI component.
    size : (int, int)
        Width and height in pixels of this UI component.
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
    on_key_press: function
        Callback function for when a keyboard key is pressed.
    """

    def __init__(self, position=(0, 0)):
        """
        Parameters
        ----------
        position : (float, float)
            Absolute coordinates (x, y) of the lower-left corner of this
            UI component.
        """
        self._position = np.array([0, 0])
        self._callbacks = []

        self._setup()  # Setup needed actors and sub UI components.
        self.position = position

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

    def _setup(self):
        """ Setup this UI component.

        This is where you should create all your needed actors and sub UI
        components.
        """
        msg = "Subclasses of UI must implement `_setup(self)`."
        raise NotImplementedError(msg)

    def _get_actors(self):
        """ Get the actors composing this UI component.
        """
        msg = "Subclasses of UI must implement `_get_actors(self)`."
        raise NotImplementedError(msg)

    @property
    def actors(self):
        """ Actors composing this UI component. """
        return self._get_actors()

    def _add_to_renderer(self, ren):
        """ Add all subcomponents or VTK props that compose this UI component.

        Parameters
        ----------
        ren : renderer
        """
        msg = "Subclasses of UI must implement `_add_to_renderer(self, ren)`."
        raise NotImplementedError(msg)

    def add_to_renderer(self, ren):
        """ Allows UI objects to add their own props to the renderer.

        Parameters
        ----------
        ren : renderer
        """
        self._add_to_renderer(ren)

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

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, coords):
        coords = np.asarray(coords)
        self._set_position(coords)
        self._position = coords

    def _set_position(self, coords):
        """ Position the lower-left corner of this UI component.

        Parameters
        ----------
        coords: (float, float)
            Absolute pixel coordinates (x, y).
        """
        msg = "Subclasses of UI must implement `_set_position(self, coords)`."
        raise NotImplementedError(msg)

    @property
    def size(self):
        return np.asarray(self._get_size(), dtype=int)

    def _get_size(self):
        msg = "Subclasses of UI must implement property `size`."
        raise NotImplementedError(msg)

    @property
    def center(self):
        return self.position + self.size / 2.

    @center.setter
    def center(self, coords):
        """ Position the center of this UI component.

        Parameters
        ----------
        coords: (float, float)
            Absolute pixel coordinates (x, y).
        """
        if not hasattr(self, "size"):
            msg = "Subclasses of UI must implement the `size` property."
            raise NotImplementedError(msg)

        new_center = np.array(coords)
        size = np.array(self.size)
        new_lower_left_corner = new_center - size / 2.
        self.position = new_lower_left_corner

    def set_visibility(self, visibility):
        """ Sets visibility of this UI component.
        """
        for actor in self.actors:
            actor.SetVisibility(visibility)

    def handle_events(self, actor):
        self.add_callback(actor, "LeftButtonPressEvent",
                          self.left_button_click_callback)
        self.add_callback(actor, "LeftButtonReleaseEvent",
                          self.left_button_release_callback)
        self.add_callback(actor, "RightButtonPressEvent",
                          self.right_button_click_callback)
        self.add_callback(actor, "RightButtonReleaseEvent",
                          self.right_button_release_callback)
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
        left_pressing_or_dragging = (self.left_button_state == "pressing" or
                                     self.left_button_state == "dragging")

        right_pressing_or_dragging = (self.right_button_state == "pressing" or
                                      self.right_button_state == "dragging")
        if left_pressing_or_dragging:
            self.left_button_state = "dragging"
            self.on_left_mouse_button_dragged(i_ren, obj, self)
        elif right_pressing_or_dragging:
            self.right_button_state = "dragging"
            self.on_right_mouse_button_dragged(i_ren, obj, self)

    @staticmethod
    def key_press_callback(i_ren, obj, self):
        self.on_key_press(i_ren, obj, self)


class Button2D(UI):
    """ A 2D overlay button and is of type vtkTexturedActor2D.

    Currently supports:
    - Multiple icons.
    - Switching between icons.
    """

    def __init__(self, icon_fnames, position=(0, 0), size=(30, 30)):
        """
        Parameters
        ----------
        icon_fnames : List(string, string)
            ((iconname, filename), (iconname, filename), ....)
        position : (float, float), optional
            Absolute coordinates (x, y) of the lower-left corner of the button.
        size : (int, int), optional
            Width and height in pixels of the button.

        """
        super(Button2D, self).__init__(position)

        self.icon_extents = dict()
        self.icons = self._build_icons(icon_fnames)
        self.icon_names = [icon[0] for icon in self.icons]
        self.current_icon_id = 0
        self.current_icon_name = self.icon_names[self.current_icon_id]
        self.set_icon(self.icons[self.current_icon_id][1])
        self.resize(size)

    def _get_size(self):
        lower_left_corner = self.texture_points.GetPoint(0)
        upper_right_corner = self.texture_points.GetPoint(2)
        size = np.array(upper_right_corner) - np.array(lower_left_corner)
        return abs(size[:2])

    def _build_icons(self, icon_fnames):
        """ Converts file names to vtkImageDataGeometryFilters.

        A pre-processing step to prevent re-read of file names during every
        state change.

        Parameters
        ----------
        icon_fnames : List(string, string)
            ((iconname, filename), (iconname, filename), ....)

        Returns
        -------
        icons : List
            A list of corresponding vtkImageDataGeometryFilters.

        """
        icons = []
        for icon_name, icon_fname in icon_fnames:
            if icon_fname.split(".")[-1] not in ["png", "PNG"]:
                error_msg = "Skipping {}: not in the PNG format."
                warn(Warning(error_msg.format(icon_fname)))
            else:
                png = vtk.vtkPNGReader()
                png.SetFileName(icon_fname)
                png.Update()
                icons.append((icon_name, png.GetOutput()))

        return icons

    def _setup(self):
        """ Setup this UI component.

        Creating the button actor used internally.
        """
        # This is highly inspired by
        # https://github.com/Kitware/VTK/blob/c3ec2495b183e3327820e927af7f8f90d34c3474/Interaction/Widgets/vtkBalloonRepresentation.cxx#L47

        self.texture_polydata = vtk.vtkPolyData()
        self.texture_points = vtk.vtkPoints()
        self.texture_points.SetNumberOfPoints(4)

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
        texture_mapper = set_input(texture_mapper, self.texture_polydata)

        button = vtk.vtkTexturedActor2D()
        button.SetMapper(texture_mapper)

        self.texture = vtk.vtkTexture()
        button.SetTexture(self.texture)

        button_property = vtk.vtkProperty2D()
        button_property.SetOpacity(1.0)
        button.SetProperty(button_property)
        self.actor = button

        # Add default events listener to the VTK actor.
        self.handle_events(self.actor)

    def _get_actors(self):
        """ Get the actors composing this UI component.
        """
        return [self.actor]

    def _add_to_renderer(self, ren):
        """ Add all subcomponents or VTK props that compose this UI component.

        Parameters
        ----------
        ren : renderer
        """
        ren.add(self.actor)

    def resize(self, size):
        """ Resize the button.

        Parameters
        ----------
        size : (float, float)
            Button size (width, height) in pixels.
        """
        # Update actor.
        self.texture_points.SetPoint(0, 0, 0, 0.0)
        self.texture_points.SetPoint(1, size[0], 0, 0.0)
        self.texture_points.SetPoint(2, size[0], size[1], 0.0)
        self.texture_points.SetPoint(3, 0, size[1], 0.0)
        self.texture_polydata.SetPoints(self.texture_points)

    def _set_position(self, coords):
        """ Position the lower-left corner of this UI component.

        Parameters
        ----------
        coords: (float, float)
            Absolute pixel coordinates (x, y).
        """
        self.actor.SetPosition(*coords)

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

    def scale(self, factor):
        """ Scales the button.

        Parameters
        ----------
        factor : (float, float)
            Scaling factor (width, height) in pixels.
        """
        self.resize(self.size * factor)

    def set_icon(self, icon):
        """ Modifies the icon used by the vtkTexturedActor2D.

        Parameters
        ----------
        icon : imageDataGeometryFilter
        """
        self.texture = set_input(self.texture, icon)

    def next_icon_id(self):
        """ Sets the next icon ID while cycling through icons.
        """
        self.current_icon_id += 1
        if self.current_icon_id == len(self.icons):
            self.current_icon_id = 0
        self.current_icon_name = self.icon_names[self.current_icon_id]

    def next_icon(self):
        """ Increments the state of the Button.

            Also changes the icon.
        """
        self.next_icon_id()
        self.set_icon(self.icons[self.current_icon_id][1])


class Rectangle2D(UI):
    """ A 2D rectangle sub-classed from UI.
    """

    def __init__(self, size=(0, 0), position=(0, 0), color=(1, 1, 1),
                 opacity=1.0):
        """ Initializes a rectangle.

        Parameters
        ----------
        size : (int, int)
            The size of the rectangle (width, height) in pixels.
        position : (float, float)
            Coordinates (x, y) of the lower-left corner of the rectangle.
        color : (float, float, float)
            Must take values in [0, 1].
        opacity : float
            Must take values in [0, 1].
        """
        super(Rectangle2D, self).__init__(position)
        self.color = color
        self.opacity = opacity
        self.resize(size)

    def _setup(self):
        """ Setup this UI component.

        Creating the polygon actor used internally.
        """
        # Setup four points
        size = (1, 1)
        self._points = vtk.vtkPoints()
        self._points.InsertNextPoint(0, 0, 0)
        self._points.InsertNextPoint(size[0], 0, 0)
        self._points.InsertNextPoint(size[0], size[1], 0)
        self._points.InsertNextPoint(0, size[1], 0)

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
        self._polygonPolyData = vtk.vtkPolyData()
        self._polygonPolyData.SetPoints(self._points)
        self._polygonPolyData.SetPolys(polygons)

        # Create a mapper and actor
        mapper = vtk.vtkPolyDataMapper2D()
        mapper = set_input(mapper, self._polygonPolyData)

        self.actor = vtk.vtkActor2D()
        self.actor.SetMapper(mapper)

        # Add default events listener to the VTK actor.
        self.handle_events(self.actor)

    def _get_actors(self):
        """ Get the actors composing this UI component.
        """
        return [self.actor]

    def _add_to_renderer(self, ren):
        """ Add all subcomponents or VTK props that compose this UI component.

        Parameters
        ----------
        ren : renderer
        """
        ren.add(self.actor)

    def _get_size(self):
        # Get 2D coordinates of two opposed corners of the rectangle.
        lower_left_corner = np.array(self._points.GetPoint(0)[:2])
        upper_right_corner = np.array(self._points.GetPoint(2)[:2])
        size = abs(upper_right_corner - lower_left_corner)
        return size

    @property
    def width(self):
        return self._points.GetPoint(2)[0]

    @width.setter
    def width(self, width):
        self.resize((width, self.height))

    @property
    def height(self):
        return self._points.GetPoint(2)[1]

    @height.setter
    def height(self, height):
        self.resize((self.width, height))

    def resize(self, size):
        """ Sets the button size.

        Parameters
        ----------
        size : (float, float)
            Button size (width, height) in pixels.
        """
        self._points.SetPoint(0, 0, 0, 0.0)
        self._points.SetPoint(1, size[0], 0, 0.0)
        self._points.SetPoint(2, size[0], size[1], 0.0)
        self._points.SetPoint(3, 0, size[1], 0.0)
        self._polygonPolyData.SetPoints(self._points)

    def _set_position(self, coords):
        """ Position the lower-left corner of this UI component.

        Parameters
        ----------
        coords: (float, float)
            Absolute pixel coordinates (x, y).
        """
        self.actor.SetPosition(*coords)

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


class Disk2D(UI):
    """ A 2D disk UI component.
    """

    def __init__(self, outer_radius, inner_radius=0, center=(0, 0),
                 color=(1, 1, 1), opacity=1.0):
        """ Initializes a rectangle.

        Parameters
        ----------
        outer_radius : int
            Outer radius of the disk.
        inner_radius : int, optional
            Inner radius of the disk. A value > 0, makes a ring.
        center : (float, float), optional
            Coordinates (x, y) of the center of the disk.
        color : (float, float, float), optional
            Must take values in [0, 1].
        opacity : float, optional
            Must take values in [0, 1].
        """
        super(Disk2D, self).__init__()
        self.outer_radius = outer_radius
        self.inner_radius = inner_radius
        self.color = color
        self.opacity = opacity
        self.center = center

    def _setup(self):
        """ Setup this UI component.

        Creating the disk actor used internally.
        """
        # Setting up disk actor.
        self._disk = vtk.vtkDiskSource()
        self._disk.SetRadialResolution(10)
        self._disk.SetCircumferentialResolution(50)
        self._disk.Update()

        # Mapper
        mapper = vtk.vtkPolyDataMapper2D()
        mapper = set_input(mapper, self._disk.GetOutputPort())

        # Actor
        self.actor = vtk.vtkActor2D()
        self.actor.SetMapper(mapper)

        # Add default events listener to the VTK actor.
        self.handle_events(self.actor)

    def _get_actors(self):
        """ Get the actors composing this UI component.
        """
        return [self.actor]

    def _add_to_renderer(self, ren):
        """ Add all subcomponents or VTK props that compose this UI component.

        Parameters
        ----------
        ren : renderer
        """
        ren.add(self.actor)

    def _get_size(self):
        diameter = 2 * self.outer_radius
        size = (diameter, diameter)
        return size

    def _set_position(self, coords):
        """ Position the lower-left corner of this UI component's bounding box.

        Parameters
        ----------
        coords: (float, float)
            Absolute pixel coordinates (x, y).
        """
        # Disk actor are positioned with respect to their center.
        self.actor.SetPosition(*coords + self.outer_radius)

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
    def inner_radius(self):
        return self._disk.GetInnerRadius()

    @inner_radius.setter
    def inner_radius(self, radius):
        self._disk.SetInnerRadius(radius)
        self._disk.Update()

    @property
    def outer_radius(self):
        return self._disk.GetOuterRadius()

    @outer_radius.setter
    def outer_radius(self, radius):
        self._disk.SetOuterRadius(radius)
        self._disk.Update()


class Panel2D(UI):
    """ A 2D UI Panel.

    Can contain one or more UI elements.

    Attributes
    ----------
    alignment : [left, right]
        Alignment of the panel with respect to the overall screen.
    """

    def __init__(self, size, position=(0, 0), color=(0.1, 0.1, 0.1),
                 opacity=0.7, align="left"):
        """
        Parameters
        ----------
        size : (int, int)
            Size (width, height) in pixels of the panel.
        position : (float, float)
            Absolute coordinates (x, y) of the lower-left corner of the panel.
        color : (float, float, float)
            Must take values in [0, 1].
        opacity : float
            Must take values in [0, 1].
        align : [left, right]
            Alignment of the panel with respect to the overall screen.
        """
        super(Panel2D, self).__init__(position)
        self.resize(size)
        self.alignment = align
        self.color = color
        self.opacity = opacity
        self.position = position
        self._drag_offset = None

    def _setup(self):
        """ Setup this UI component.

        Create the background (Rectangle2D) of the panel.
        """
        self._elements = []
        self.element_offsets = []
        self.background = Rectangle2D()
        self.add_element(self.background, (0, 0))

        # Add default events listener for this UI component.
        self.background.on_left_mouse_button_pressed = self.left_button_pressed
        self.background.on_left_mouse_button_dragged = self.left_button_dragged

    def _get_actors(self):
        """ Get the actors composing this UI component.
        """
        actors = []
        for element in self._elements:
            actors += element.actors

        return actors

    def _add_to_renderer(self, ren):
        """ Add all subcomponents or VTK props that compose this UI component.

        Parameters
        ----------
        ren : renderer
        """
        for element in self._elements:
            element.add_to_renderer(ren)

    def _get_size(self):
        return self.background.size

    def resize(self, size):
        """ Sets the panel size.

        Parameters
        ----------
        size : (float, float)
            Panel size (width, height) in pixels.
        """
        self.background.resize(size)

    def _set_position(self, coords):
        """ Position the lower-left corner of this UI component.

        Parameters
        ----------
        coords: (float, float)
            Absolute pixel coordinates (x, y).
        """
        coords = np.array(coords)
        for element, offset in self.element_offsets:
            element.position = coords + offset

    @property
    def color(self):
        return self.background.color

    @color.setter
    def color(self, color):
        self.background.color = color

    @property
    def opacity(self):
        return self.background.opacity

    @opacity.setter
    def opacity(self, opacity):
        self.background.opacity = opacity

    def add_element(self, element, coords, anchor="position"):
        """ Adds a UI component to the panel.

        The coordinates represent an offset from the lower left corner of the
        panel.

        Parameters
        ----------
        element : UI
            The UI item to be added.
        coords : (float, float) or (int, int)
            If float, normalized coordinates are assumed and they must be
            between [0,1].
            If int, pixels coordinates are assumed and it must fit within the
            panel's size.
        """
        coords = np.array(coords)

        if np.issubdtype(coords.dtype, np.floating):
            if np.any(coords < 0) or np.any(coords > 1):
                raise ValueError("Normalized coordinates must be in [0,1].")

            coords = coords * self.size

        if anchor == "center":
            element.center = self.position + coords
        elif anchor == "position":
            element.position = self.position + coords
        else:
            msg = ("Unknown anchor {}. Supported anchors are 'position'"
                   " and 'center'.")
            raise ValueError(msg)

        self._elements.append(element)
        offset = element.position - self.position
        self.element_offsets.append((element, offset))

    def left_button_pressed(self, i_ren, obj, panel2d_object):
        click_pos = np.array(i_ren.event.position)
        self._drag_offset = click_pos - panel2d_object.position
        i_ren.event.abort()  # Stop propagating the event.

    def left_button_dragged(self, i_ren, obj, panel2d_object):
        if self._drag_offset is not None:
            click_position = np.array(i_ren.event.position)
            new_position = click_position - self._drag_offset
            self.position = new_position
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
            self.position += np.array(window_size_change)
        else:
            msg = "You can only left-align or right-align objects in a panel."
            raise ValueError(msg)


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
        super(TextBlock2D, self).__init__(position=position)
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

    def _setup(self):
        self.actor = vtk.vtkTextActor()
        self._background = None  # For VTK < 7
        self.handle_events(self.actor)

    def _get_actors(self):
        """ Get the actors composing this UI component.
        """
        if self._background is not None:
            return [self.actor, self._background]

        return [self.actor]

    def _add_to_renderer(self, ren):
        """ Add all subcomponents or VTK props that compose this UI component.

        Parameters
        ----------
        ren : renderer
        """
        if self._background is not None:
            ren.add(self._background)

        ren.add(self.actor)

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
            msg = "Text can only be justified left, right and center."
            raise ValueError(msg)

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
        if VTK_MAJOR_VERSION < 7:
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
            if VTK_MAJOR_VERSION < 7:
                self._background = None
            else:
                self.actor.GetTextProperty().SetBackgroundOpacity(0.)

        else:
            if VTK_MAJOR_VERSION < 7:
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
        super(TextBox2D, self).__init__(position=position)

        self.message = text
        self.text.message = text
        self.text.font_size = font_size
        self.text.font_family = font_family
        self.text.justification = justification
        self.text.bold = bold
        self.text.italic = italic
        self.text.shadow = shadow
        self.text.color = color
        self.text.background_color = (1, 1, 1)

        self.width = width
        self.height = height
        self.window_left = 0
        self.window_right = 0
        self.caret_pos = 0
        self.init = True

    def _setup(self):
        """ Setup this UI component.

        Create the TextBlock2D component used for the textbox.
        """
        self.text = TextBlock2D()

        # Add default events listener for this UI component.
        self.text.on_left_mouse_button_pressed = self.left_button_press
        self.text.on_key_press = self.key_press

    def _get_actors(self):
        """ Get the actors composing this UI component.
        """
        return self.text.actors

    def _add_to_renderer(self, ren):
        """ Add all subcomponents or VTK props that compose this UI component.

        Parameters
        ----------
        ren : renderer
        """
        self.text.add_to_renderer(ren)

    def _set_position(self, coords):
        """ Position the lower-left corner of this UI component.

        Parameters
        ----------
        coords: (float, float)
            Absolute pixel coordinates (x, y).
        """
        self.text.position = coords

    def set_message(self, message):
        """ Set custom text to textbox.

        Parameters
        ----------
        message: str
            The custom message to be set.
        """
        self.message = message
        self.text.message = message
        self.init = False
        self.window_right = len(self.message)
        self.window_left = 0
        self.caret_pos = self.window_right

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
        self.caret_pos = min(self.caret_pos + 1, len(self.message))

    def move_caret_left(self):
        """ Moves the caret towards left.
        """
        self.caret_pos = max(self.caret_pos - 1, 0)

    def right_move_right(self):
        """ Moves right boundary of the text window right-wards.
        """
        if self.window_right <= len(self.message):
            self.window_right += 1

    def right_move_left(self):
        """ Moves right boundary of the text window left-wards.
        """
        if self.window_right > 0:
            self.window_right -= 1

    def left_move_right(self):
        """ Moves left boundary of the text window right-wards.
        """
        if self.window_left <= len(self.message):
            self.window_left += 1

    def left_move_left(self):
        """ Moves left boundary of the text window left-wards.
        """
        if self.window_left > 0:
            self.window_left -= 1

    def add_character(self, character):
        """ Inserts a character into the text and moves window and caret.

        Parameters
        ----------
        character : str
        """
        if len(character) > 1 and character.lower() != "space":
            return
        if character.lower() == "space":
            character = " "
        self.message = (self.message[:self.caret_pos] +
                        character +
                        self.message[self.caret_pos:])
        self.move_caret_right()
        if (self.window_right -
                self.window_left == self.height * self.width - 1):
            self.left_move_right()
        self.right_move_right()

    def remove_character(self):
        """ Removes a character and moves window and caret accordingly.
        """
        if self.caret_pos == 0:
            return
        self.message = (self.message[:self.caret_pos - 1] +
                        self.message[self.caret_pos:])
        self.move_caret_left()
        if len(self.message) < self.height * self.width - 1:
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
            ret_text = (self.message[:self.caret_pos] +
                        "_" +
                        self.message[self.caret_pos:])
        else:
            ret_text = self.message
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
        self.text.message = self.width_set_text(text)

    def edit_mode(self):
        """ Turns on edit mode.
        """
        if self.init:
            self.message = ""
            self.init = False
            self.caret_pos = 0
        self.render_text()

    def left_button_press(self, i_ren, obj, textbox_object):
        """ Left button press handler for textbox

        Parameters
        ----------
        i_ren: :class:`CustomInteractorStyle`
        obj: :class:`vtkActor`
            The picked actor
        textbox_object: :class:`TextBox2D`
        """
        i_ren.add_active_prop(self.text.actor)
        self.edit_mode()
        i_ren.force_render()

    def key_press(self, i_ren, obj, textbox_object):
        """ Key press handler for textbox

        Parameters
        ----------
        i_ren: :class:`CustomInteractorStyle`
        obj: :class:`vtkActor`
            The picked actor
        textbox_object: :class:`TextBox2D`
        """
        key = i_ren.event.key
        is_done = self.handle_character(key)
        if is_done:
            i_ren.remove_active_prop(self.text.actor)

        i_ren.force_render()


class LineSlider2D(UI):
    """ A 2D Line Slider.

    A sliding handle on a line with a percentage indicator.

    Attributes
    ----------
    line_width : int
        Width of the line on which the disk will slide.
    length : int
        Length of the slider.
    track : :class:`Rectangle2D`
        The line on which the slider's handle moves.
    handle : :class:`Disk2D`
        The moving part of the slider.
    text : :class:`TextBlock2D`
        The text that shows percentage.
    shape : string
        Describes the shape of the handle.
        Currently supports 'disk' and 'square'.
    default_color : (float, float, float)
        Color of the handle when in unpressed state.
    active_color : (float, float, float)
        Color of the handle when it is pressed.
    """
    def __init__(self, center=(0, 0),
                 initial_value=50, min_value=0, max_value=100,
                 length=200, line_width=5,
                 inner_radius=0, outer_radius=10, handle_side=20,
                 font_size=16,
                 text_template="{value:.1f} ({ratio:.0%})", shape="disk"):
        """
        Parameters
        ----------
        center : (float, float)
            Center of the slider's center.
        initial_value : float
            Initial value of the slider.
        min_value : float
            Minimum value of the slider.
        max_value : float
            Maximum value of the slider.
        length : int
            Length of the slider.
        line_width : int
            Width of the line on which the disk will slide.
        inner_radius : int
            Inner radius of the handles (if disk).
        outer_radius : int
            Outer radius of the handles (if disk).
        handle_side : int
            Side length of the handles (if sqaure).
        font_size : int
            Size of the text to display alongside the slider (pt).
        text_template : str, callable
            If str, text template can contain one or multiple of the
            replacement fields: `{value:}`, `{ratio:}`.
            If callable, this instance of `:class:LineSlider2D` will be
            passed as argument to the text template function.
        shape : string
            Describes the shape of the handle.
            Currently supports 'disk' and 'square'.
        """
        self.shape = shape
        self.default_color = (1, 1, 1)
        self.active_color = (0, 0, 1)
        super(LineSlider2D, self).__init__()

        self.track.width = length
        self.track.height = line_width
        if shape == "disk":
            self.handle.inner_radius = inner_radius
            self.handle.outer_radius = outer_radius
        elif shape == "square":
            self.handle.width = handle_side
            self.handle.height = handle_side
        self.center = center

        self.min_value = min_value
        self.max_value = max_value
        self.text.font_size = font_size
        self.text_template = text_template

        # Offer some standard hooks to the user.
        self.on_change = lambda ui: None

        self.value = initial_value
        self.update()

    def _setup(self):
        """ Setup this UI component.

        Create the slider's track (Rectangle2D), the handle (Disk2D) and
        the text (TextBlock2D).
        """
        # Slider's track
        self.track = Rectangle2D()
        self.track.color = (1, 0, 0)

        # Slider's handle
        if self.shape == "disk":
            self.handle = Disk2D(outer_radius=1)
        elif self.shape == "square":
            self.handle = Rectangle2D(size=(1, 1))
        self.handle.color = self.default_color

        # Slider Text
        self.text = TextBlock2D(justification="center",
                                vertical_justification="top")

        # Add default events listener for this UI component.
        self.track.on_left_mouse_button_pressed = self.track_click_callback
        self.track.on_left_mouse_button_dragged = self.handle_move_callback
        self.track.on_left_mouse_button_released = \
            self.handle_release_callback
        self.handle.on_left_mouse_button_dragged = self.handle_move_callback
        self.handle.on_left_mouse_button_released = \
            self.handle_release_callback

    def _get_actors(self):
        """ Get the actors composing this UI component.
        """
        return self.track.actors + self.handle.actors + self.text.actors

    def _add_to_renderer(self, ren):
        """ Add all subcomponents or VTK props that compose this UI component.

        Parameters
        ----------
        ren : renderer
        """
        self.track.add_to_renderer(ren)
        self.handle.add_to_renderer(ren)
        self.text.add_to_renderer(ren)

    def _get_size(self):
        # Consider the handle's size when computing the slider's size.
        width = self.track.width + self.handle.size[0]
        height = max(self.track.height, self.handle.size[1])
        return np.array([width, height])

    def _set_position(self, coords):
        """ Position the lower-left corner of this UI component.

        Parameters
        ----------
        coords: (float, float)
            Absolute pixel coordinates (x, y).
        """
        # Offset the slider line by the handle's radius.
        track_position = coords + self.handle.size / 2.
        # Offset the slider line height by half the slider line width.
        track_position[1] -= self.track.size[1] / 2.
        self.track.position = track_position
        self.handle.position = self.handle.position.astype('float64')
        self.handle.position += coords - self.position
        # Position the text below the handle.
        self.text.position = (self.handle.center[0],
                              self.handle.position[1] - 10)

    @property
    def left_x_position(self):
        return self.track.position[0]

    @property
    def right_x_position(self):
        return self.track.position[0] + self.track.size[0]

    def set_position(self, position):
        """ Sets the disk's position.

        Parameters
        ----------
        position : (float, float)
            The absolute position of the disk (x, y).
        """
        x_position = position[0]
        x_position = max(x_position, self.left_x_position)
        x_position = min(x_position, self.right_x_position)

        # Move slider disk.
        self.handle.center = (x_position, self.track.center[1])
        self.update()  # Update information.

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
        position_x = self.left_x_position + ratio * self.track.width
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
        assert length == self.track.width
        disk_position_x = self.handle.center[0]
        self._ratio = (disk_position_x - self.left_x_position) / length

        # Compute the selected value considering min_value and max_value.
        value_range = self.max_value - self.min_value
        self._value = self.min_value + self.ratio * value_range

        # Update text.
        text = self.format_text()
        self.text.message = text

        # Move the text below the slider's handle.
        self.text.position = (disk_position_x, self.text.position[1])

        self.on_change(self)

    def track_click_callback(self, i_ren, vtkactor, slider):
        """ Update disk position and grab the focus.

        Parameters
        ----------
        i_ren : :class:`CustomInteractorStyle`
        vtkactor : :class:`vtkActor`
            The picked actor
        slider : :class:`LineSlider2D`
        """
        position = i_ren.event.position
        self.set_position(position)
        i_ren.force_render()
        i_ren.event.abort()  # Stop propagating the event.

    def handle_move_callback(self, i_ren, vtkactor, slider):
        """ Actual handle movement.

        Parameters
        ----------
        i_ren : :class:`CustomInteractorStyle`
        vtkactor : :class:`vtkActor`
            The picked actor
        slider : :class:`LineSlider2D`
        """
        self.handle.color = self.active_color
        position = i_ren.event.position
        self.set_position(position)
        i_ren.force_render()
        i_ren.event.abort()  # Stop propagating the event.

    def handle_release_callback(self, i_ren, vtkactor, slider):
        """ Change color when handle is released.

        Parameters
        ----------
        i_ren : :class:`CustomInteractorStyle`
        vtkactor : :class:`vtkActor`
            The picked actor
        slider : :class:`LineSlider2D`
        """
        self.handle.color = self.default_color
        i_ren.force_render()
        i_ren.event.abort()  # Stop propagating the event.


class LineDoubleSlider2D(UI):
    """ A 2D Line Slider with two sliding rings.
    Useful for setting min and max values for something.

    Currently supports:
    - Setting positions of both disks.

    Attributes
    ----------
    line_width : int
        Width of the line on which the disk will slide.
    length : int
        Length of the slider.
    track : :class:`vtkActor`
        The line on which the handles move.
    handles : [:class:`vtkActor`, :class:`vtkActor`]
        The moving slider disks.
    text : [:class:`TextBlock2D`, :class:`TextBlock2D`]
        The texts that show the values of the disks.
    shape : string
        Describes the shape of the handle.
        Currently supports 'disk' and 'square'.
    default_color : (float, float, float)
        Color of the handles when in unpressed state.
    active_color : (float, float, float)
        Color of the handles when they are pressed.

    """
    def __init__(self, line_width=5, inner_radius=0, outer_radius=10,
                 handle_side=20, center=(450, 300), length=200,
                 initial_values=(0, 100), min_value=0, max_value=100,
                 font_size=16, text_template="{value:.1f}", shape="disk"):
        """
        Parameters
        ----------
        line_width : int
            Width of the line on which the disk will slide.
        inner_radius : int
            Inner radius of the handles (if disk).
        outer_radius : int
            Outer radius of the handles (if disk).
        handle_side : int
            Side length of the handles (if sqaure).
        center : (float, float)
            Center of the slider.
        length : int
            Length of the slider.
        initial_values : (float, float)
            Initial values of the two handles.
        min_value : float
            Minimum value of the slider.
        max_value : float
            Maximum value of the slider.
        font_size : int
            Size of the text to display alongside the slider (pt).
        text_template : str, callable
            If str, text template can contain one or multiple of the
            replacement fields: `{value:}`, `{ratio:}`.
            If callable, this instance of `:class:LineDoubleSlider2D` will be
            passed as argument to the text template function.
        shape : string
            Describes the shape of the handle.
            Currently supports 'disk' and 'square'.

        """
        self.shape = shape
        self.default_color = (1, 1, 1)
        self.active_color = (0, 0, 1)
        super(LineDoubleSlider2D, self).__init__()

        self.track.width = length
        self.track.height = line_width
        self.center = center
        if shape == "disk":
            self.handles[0].inner_radius = inner_radius
            self.handles[0].outer_radius = outer_radius
            self.handles[1].inner_radius = inner_radius
            self.handles[1].outer_radius = outer_radius
        elif shape == "square":
            self.handles[0].width = handle_side
            self.handles[0].height = handle_side
            self.handles[1].width = handle_side
            self.handles[1].height = handle_side

        self.min_value = min_value
        self.max_value = max_value
        self.text[0].font_size = font_size
        self.text[1].font_size = font_size
        self.text_template = text_template

        # Setting the handle positions will also update everything.
        self._values = [initial_values[0], initial_values[1]]
        self._ratio = [None, None]
        self.left_disk_value = initial_values[0]
        self.right_disk_value = initial_values[1]

    def _setup(self):
        """ Setup this UI component.

        Create the slider's track (Rectangle2D), the handles (Disk2D) and
        the text (TextBlock2D).
        """
        # Slider's track
        self.track = Rectangle2D()
        self.track.color = (1, 0, 0)

        # Handles
        self.handles = []
        if self.shape == "disk":
            self.handles.append(Disk2D(outer_radius=1))
            self.handles.append(Disk2D(outer_radius=1))
        elif self.shape == "square":
            self.handles.append(Rectangle2D(size=(1, 1)))
            self.handles.append(Rectangle2D(size=(1, 1)))
        self.handles[0].color = self.default_color
        self.handles[1].color = self.default_color

        # Slider Text
        self.text = [TextBlock2D(justification="center",
                                 vertical_justification="top"),
                     TextBlock2D(justification="center",
                                 vertical_justification="top")
                     ]

        # Add default events listener for this UI component.
        self.track.on_left_mouse_button_dragged = self.handle_move_callback
        self.handles[0].on_left_mouse_button_dragged = \
            self.handle_move_callback
        self.handles[1].on_left_mouse_button_dragged = \
            self.handle_move_callback
        self.handles[0].on_left_mouse_button_released = \
            self.handle_release_callback
        self.handles[1].on_left_mouse_button_released = \
            self.handle_release_callback

    def _get_actors(self):
        """ Get the actors composing this UI component.
        """
        return (self.track.actors + self.handles[0].actors +
                self.handles[1].actors + self.text[0].actors +
                self.text[1].actors)

    def _add_to_renderer(self, ren):
        """ Add all subcomponents or VTK props that compose this UI component.

        Parameters
        ----------
        ren : renderer
        """
        self.track.add_to_renderer(ren)
        self.handles[0].add_to_renderer(ren)
        self.handles[1].add_to_renderer(ren)
        self.text[0].add_to_renderer(ren)
        self.text[1].add_to_renderer(ren)

    def _get_size(self):
        # Consider the handle's size when computing the slider's size.
        width = self.track.width + 2 * self.handles[0].size[0]
        height = max(self.track.height, self.handles[0].size[1])
        return np.array([width, height])

    def _set_position(self, coords):
        """ Position the lower-left corner of this UI component.

        Parameters
        ----------
        coords: (float, float)
            Absolute pixel coordinates (x, y).
        """
        # Offset the slider line by the handle's radius.
        track_position = coords + self.handles[0].size / 2.
        # Offset the slider line height by half the slider line width.
        track_position[1] -= self.track.size[1] / 2.
        self.track.position = track_position
        self.handles[0].position = self.handles[0].position.astype('float64')
        self.handles[1].position = self.handles[1].position.astype('float64')
        self.handles[0].position += coords - self.position
        self.handles[1].position += coords - self.position
        # Position the text below the handles.
        self.text[0].position = (self.handles[0].center[0],
                                 self.handles[0].position[1] - 20)
        self.text[1].position = (self.handles[1].center[0],
                                 self.handles[1].position[1] - 20)

    @property
    def left_x_position(self):
        return self.track.position[0]

    @property
    def right_x_position(self):
        return self.track.position[0] + self.track.size[0]

    def value_to_ratio(self, value):
        """ Converts the value of a disk to the ratio

        Parameters
        ----------
        value : float
        """
        value_range = self.max_value - self.min_value
        return (value - self.min_value) / value_range

    def ratio_to_coord(self, ratio):
        """ Converts the ratio to the absolute coordinate.

        Parameters
        ----------
        ratio : float
        """
        return self.left_x_position + ratio * self.track.width

    def coord_to_ratio(self, coord):
        """ Converts the x coordinate of a disk to the ratio

        Parameters
        ----------
        coord : float
        """
        return (coord - self.left_x_position) / self.track.width

    def ratio_to_value(self, ratio):
        """ Converts the ratio to the value of the disk.

        Parameters
        ----------
        ratio : float
        """
        value_range = self.max_value - self.min_value
        return self.min_value + ratio * value_range

    def set_position(self, position, disk_number):
        """ Sets the disk's position.

        Parameters
        ----------
        position : (float, float)
            The absolute position of the disk (x, y).
        disk_number : int
            The index of disk being moved.
        """
        x_position = position[0]

        if disk_number == 0 and x_position >= self.handles[1].center[0]:
            x_position = self.ratio_to_coord(
                self.value_to_ratio(self._values[1] - 1))

        if disk_number == 1 and x_position <= self.handles[0].center[0]:
            x_position = self.ratio_to_coord(
                self.value_to_ratio(self._values[0] + 1))

        x_position = max(x_position, self.left_x_position)
        x_position = min(x_position, self.right_x_position)

        self.handles[disk_number].center = (x_position, self.track.center[1])
        self.update(disk_number)

    @property
    def left_disk_value(self):
        """ Returns the value of the left disk. """
        return self._values[0]

    @left_disk_value.setter
    def left_disk_value(self, left_disk_value):
        """ Sets the value of the left disk.

        Parameters
        ----------
        left_disk_value : New value for the left disk.
        """
        self.left_disk_ratio = self.value_to_ratio(left_disk_value)

    @property
    def right_disk_value(self):
        """ Returns the value of the right disk. """
        return self._values[1]

    @right_disk_value.setter
    def right_disk_value(self, right_disk_value):
        """ Sets the value of the right disk.

        Parameters
        ----------
        right_disk_value : New value for the right disk.
        """
        self.right_disk_ratio = self.value_to_ratio(right_disk_value)

    @property
    def left_disk_ratio(self):
        """ Returns the ratio of the left disk. """
        return self._ratio[0]

    @left_disk_ratio.setter
    def left_disk_ratio(self, left_disk_ratio):
        """ Sets the ratio of the left disk.

        Parameters
        ----------
        left_disk_ratio : New ratio for the left disk.
        """
        position_x = self.ratio_to_coord(left_disk_ratio)
        self.set_position((position_x, None), 0)

    @property
    def right_disk_ratio(self):
        """ Returns the ratio of the right disk. """
        return self._ratio[1]

    @right_disk_ratio.setter
    def right_disk_ratio(self, right_disk_ratio):
        """ Sets the ratio of the right disk.

        Parameters
        ----------
        right_disk_ratio : New ratio for the right disk.
        """
        position_x = self.ratio_to_coord(right_disk_ratio)
        self.set_position((position_x, None), 1)

    def format_text(self, disk_number):
        """ Returns formatted text to display along the slider.

        Parameters
        ----------
        disk_number : Index of the disk.
        """
        if callable(self.text_template):
            return self.text_template(self)

        return self.text_template.format(value=self._values[disk_number])

    def update(self, disk_number):
        """ Updates the slider.

        Parameters
        ----------
        disk_number : Index of the disk to be updated.
        """
        # Compute the ratio determined by the position of the slider disk.
        self._ratio[disk_number] = self.coord_to_ratio(
            self.handles[disk_number].center[0])

        # Compute the selected value considering min_value and max_value.
        self._values[disk_number] = self.ratio_to_value(
            self._ratio[disk_number])

        # Update text.
        text = self.format_text(disk_number)
        self.text[disk_number].message = text

        self.text[disk_number].position = (
            self.handles[disk_number].center[0],
            self.text[disk_number].position[1])

    def handle_move_callback(self, i_ren, vtkactor, slider):
        """ Actual handle movement.

        Parameters
        ----------
        i_ren : :class:`CustomInteractorStyle`
        vtkactor : :class:`vtkActor`
            The picked actor
        slider : :class:`LineDoubleSlider2D`
        """
        position = i_ren.event.position
        if vtkactor == self.handles[0].actors[0]:
            self.set_position(position, 0)
            self.handles[0].color = self.active_color
        elif vtkactor == self.handles[1].actors[0]:
            self.set_position(position, 1)
            self.handles[1].color = self.active_color
        i_ren.force_render()
        i_ren.event.abort()  # Stop propagating the event.

    def handle_release_callback(self, i_ren, vtkactor, slider):
        """ Change color when handle is released.

        Parameters
        ----------
        i_ren : :class:`CustomInteractorStyle`
        vtkactor : :class:`vtkActor`
            The picked actor
        slider : :class:`LineDoubleSlider2D`
        """
        if vtkactor == self.handles[0].actors[0]:
            self.handles[0].color = self.default_color
        elif vtkactor == self.handles[1].actors[0]:
            self.handles[1].color = self.default_color
        i_ren.force_render()
        i_ren.event.abort()  # Stop propagating the event.


class RingSlider2D(UI):
    """ A disk slider.

    A disk moves along the boundary of a ring.
    Goes from 0-360 degrees.

    Attributes
    ----------
    mid_track_radius: float
        Distance from the center of the slider to the middle of the track.
    previous_value: float
        Value of Rotation of the actor before the current value.
    track : :class:`Disk2D`
        The circle on which the slider's handle moves.
    handle : :class:`Disk2D`
        The moving part of the slider.
    text : :class:`TextBlock2D`
        The text that shows percentage.
    default_color : (float, float, float)
        Color of the handle when in unpressed state.
    active_color : (float, float, float)
        Color of the handle when it is pressed.
    """
    def __init__(self, center=(0, 0),
                 initial_value=180, min_value=0, max_value=360,
                 slider_inner_radius=40, slider_outer_radius=44,
                 handle_inner_radius=0, handle_outer_radius=10,
                 font_size=16,
                 text_template="{ratio:.0%}"):
        """
        Parameters
        ----------
        center : (float, float)
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
        font_size : int
            Size of the text to display alongside the slider (pt).
        text_template : str, callable
            If str, text template can contain one or multiple of the
            replacement fields: `{value:}`, `{ratio:}`, `{angle:}`.
            If callable, this instance of `:class:RingSlider2D` will be
            passed as argument to the text template function.
        """
        self.default_color = (1, 1, 1)
        self.active_color = (0, 0, 1)
        super(RingSlider2D, self).__init__()

        self.track.inner_radius = slider_inner_radius
        self.track.outer_radius = slider_outer_radius
        self.handle.inner_radius = handle_inner_radius
        self.handle.outer_radius = handle_outer_radius
        self.center = center

        self.min_value = min_value
        self.max_value = max_value
        self.text.font_size = font_size
        self.text_template = text_template

        # Offer some standard hooks to the user.
        self.on_change = lambda ui: None

        self._value = initial_value
        self.value = initial_value

    def _setup(self):
        """ Setup this UI component.

        Create the slider's circle (Disk2D), the handle (Disk2D) and
        the text (TextBlock2D).
        """
        # Slider's track.
        self.track = Disk2D(outer_radius=1)
        self.track.color = (1, 0, 0)

        # Slider's handle.
        self.handle = Disk2D(outer_radius=1)
        self.handle.color = self.default_color

        # Slider Text
        self.text = TextBlock2D(justification="center",
                                vertical_justification="middle")

        # Add default events listener for this UI component.
        self.track.on_left_mouse_button_pressed = self.track_click_callback
        self.track.on_left_mouse_button_dragged = self.handle_move_callback
        self.track.on_left_mouse_button_released = \
            self.handle_release_callback
        self.handle.on_left_mouse_button_dragged = self.handle_move_callback
        self.handle.on_left_mouse_button_released = \
            self.handle_release_callback

    def _get_actors(self):
        """ Get the actors composing this UI component.
        """
        return self.track.actors + self.handle.actors + self.text.actors

    def _add_to_renderer(self, ren):
        """ Add all subcomponents or VTK props that compose this UI component.

        Parameters
        ----------
        ren : renderer
        """
        self.track.add_to_renderer(ren)
        self.handle.add_to_renderer(ren)
        self.text.add_to_renderer(ren)

    def _get_size(self):
        return self.track.size + self.handle.size

    def _set_position(self, coords):
        """ Position the lower-left corner of this UI component.

        Parameters
        ----------
        coords: (float, float)
            Absolute pixel coordinates (x, y).
        """
        self.track.position = coords + self.handle.size / 2.
        self.handle.position += coords - self.position
        # Position the text in the center of the slider's track.
        self.text.position = coords + self.size / 2.

    @property
    def mid_track_radius(self):
        return (self.track.inner_radius + self.track.outer_radius) / 2.

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
        self._previous_value = self.value
        self._value = self.min_value + self.ratio * value_range

        # Update text disk actor.
        x = self.mid_track_radius * np.cos(self.angle) + self.center[0]
        y = self.mid_track_radius * np.sin(self.angle) + self.center[1]
        self.handle.center = (x, y)

        # Update text.
        text = self.format_text()
        self.text.message = text

        self.on_change(self)  # Call hook.

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

    def track_click_callback(self, i_ren, obj, slider):
        """ Update disk position and grab the focus.

        Parameters
        ----------
        i_ren : :class:`CustomInteractorStyle`
        obj : :class:`vtkActor`
            The picked actor
        slider : :class:`RingSlider2D`
        """
        click_position = i_ren.event.position
        self.move_handle(click_position=click_position)
        i_ren.force_render()
        i_ren.event.abort()  # Stop propagating the event.

    def handle_move_callback(self, i_ren, obj, slider):
        """ Move the slider's handle.

        Parameters
        ----------
        i_ren : :class:`CustomInteractorStyle`
        obj : :class:`vtkActor`
            The picked actor
        slider : :class:`RingSlider2D`
        """
        click_position = i_ren.event.position
        self.handle.color = self.active_color
        self.move_handle(click_position=click_position)
        i_ren.force_render()
        i_ren.event.abort()  # Stop propagating the event.

    def handle_release_callback(self, i_ren, obj, slider):
        """ Change color when handle is released.

        Parameters
        ----------
        i_ren : :class:`CustomInteractorStyle`
        vtkactor : :class:`vtkActor`
            The picked actor
        slider : :class:`RingSlider2D`
        """
        self.handle.color = self.default_color
        i_ren.force_render()
        i_ren.event.abort()  # Stop propagating the event.


class RangeSlider(UI):

    """ A set of a LineSlider2D and a LineDoubleSlider2D.
    The double slider is used to set the min and max value
    for the LineSlider2D

    Attributes
    ----------
    range_slider_center : (float, float)
        Center of the LineDoubleSlider2D object.
    value_slider_center : (float, float)
        Center of the LineSlider2D object.
    range_slider : :class:`LineDoubleSlider2D`
        The line slider which sets the min and max values
    value_slider : :class:`LineSlider2D`
        The line slider which sets the value

    """
    def __init__(self, line_width=5, inner_radius=0, outer_radius=10,
                 handle_side=20, range_slider_center=(450, 400),
                 value_slider_center=(450, 300), length=200, min_value=0,
                 max_value=100, font_size=16, range_precision=1,
                 value_precision=2, shape="disk"):
        """
        Parameters
        ----------
        line_width : int
            Width of the slider tracks
        inner_radius : int
            Inner radius of the handles.
        outer_radius : int
            Outer radius of the handles.
        handle_side : int
            Side length of the handles (if sqaure).
        range_slider_center : (float, float)
            Center of the LineDoubleSlider2D object.
        value_slider_center : (float, float)
            Center of the LineSlider2D object.
        length : int
            Length of the sliders.
        min_value : float
            Minimum value of the double slider.
        max_value : float
            Maximum value of the double slider.
        font_size : int
            Size of the text to display alongside the sliders (pt).
        range_precision : int
            Number of decimal places to show the min and max values set.
        value_precision : int
            Number of decimal places to show the value set on slider.
        shape : string
            Describes the shape of the handle.
            Currently supports 'disk' and 'square'.
        """
        self.min_value = min_value
        self.max_value = max_value
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.handle_side = handle_side
        self.length = length
        self.line_width = line_width
        self.font_size = font_size
        self.shape = shape

        self.range_slider_text_template = \
            "{value:." + str(range_precision) + "f}"
        self.value_slider_text_template = \
            "{value:." + str(value_precision) + "f}"

        self.range_slider_center = range_slider_center
        self.value_slider_center = value_slider_center
        super(RangeSlider, self).__init__()

    def _setup(self):
        """ Setup this UI component.
        """
        self.range_slider = \
            LineDoubleSlider2D(line_width=self.line_width,
                               inner_radius=self.inner_radius,
                               outer_radius=self.outer_radius,
                               handle_side=self.handle_side,
                               center=self.range_slider_center,
                               length=self.length, min_value=self.min_value,
                               max_value=self.max_value,
                               initial_values=(self.min_value,
                                               self.max_value),
                               font_size=self.font_size, shape=self.shape,
                               text_template=self.range_slider_text_template)

        self.value_slider = \
            LineSlider2D(line_width=self.line_width, length=self.length,
                         inner_radius=self.inner_radius,
                         outer_radius=self.outer_radius,
                         handle_side=self.handle_side,
                         center=self.value_slider_center,
                         min_value=self.min_value, max_value=self.max_value,
                         initial_value=(self.min_value + self.max_value) / 2,
                         font_size=self.font_size, shape=self.shape,
                         text_template=self.value_slider_text_template)

        # Add default events listener for this UI component.
        self.range_slider.handles[0].on_left_mouse_button_dragged = \
            self.range_slider_handle_move_callback
        self.range_slider.handles[1].on_left_mouse_button_dragged = \
            self.range_slider_handle_move_callback

    def _get_actors(self):
        """ Get the actors composing this UI component.
        """
        return self.range_slider.actors + self.value_slider.actors

    def _add_to_renderer(self, ren):
        """ Add all subcomponents or VTK props that compose this UI component.

        Parameters
        ----------
        ren : renderer
        """
        self.range_slider.add_to_renderer(ren)
        self.value_slider.add_to_renderer(ren)

    def _get_size(self):
        return self.range_slider.size + self.value_slider.size

    def _set_position(self, coords):
        pass

    def range_slider_handle_move_callback(self, i_ren, obj, slider):
        """ Actual movement of range_slider's handles.

        Parameters
        ----------
        i_ren : :class:`CustomInteractorStyle`
        obj : :class:`vtkActor`
            The picked actor
        slider : :class:`RangeSlider`

        """
        position = i_ren.event.position
        if obj == self.range_slider.handles[0].actors[0]:
            self.range_slider.handles[0].color = \
                self.range_slider.active_color
            self.range_slider.set_position(position, 0)
            self.value_slider.min_value = self.range_slider.left_disk_value
            self.value_slider.update()
        elif obj == self.range_slider.handles[1].actors[0]:
            self.range_slider.handles[1].color = \
                self.range_slider.active_color
            self.range_slider.set_position(position, 1)
            self.value_slider.max_value = self.range_slider.right_disk_value
            self.value_slider.update()
        i_ren.force_render()
        i_ren.event.abort()  # Stop propagating the event.


class ImageContainer2D(UI):
    """ A 2D container to hold an image.
    Currently Supports:
    - png and jpg/jpeg images

    Attributes
    ----------
    size: (float, float)
        Image size (width, height) in pixels.
    img : vtkImageDataGeometryFilters
        The image loaded from the specified path.

    """

    def __init__(self, img_path, position=(0, 0), size=(100, 100)):
        """
        Parameters
        ----------
        img_path : string
            Path of the image
        position : (float, float), optional
            Absolute coordinates (x, y) of the lower-left corner of the image.
        size : (int, int), optional
            Width and height in pixels of the image.
        """
        super(ImageContainer2D, self).__init__(position)
        self.img = self._build_image(img_path)
        self.set_img(self.img)
        self.resize(size)

    def _build_image(self, img_path):
        """ Converts image path to vtkImageDataGeometryFilters.

        A pre-processing step to prevent re-read of image during every
        state change.

        Parameters
        ----------
        img_path : string
            Path of the image

        Returns
        -------
        img : vtkImageDataGeometryFilters
            The corresponding image .
        """
        imgExt = img_path.split(".")[-1].lower()
        if imgExt == "png":
            png = vtk.vtkPNGReader()
            png.SetFileName(img_path)
            png.Update()
            img = png.GetOutput()
        elif imgExt in ["jpg", "jpeg"]:
            jpeg = vtk.vtkJPEGReader()
            jpeg.SetFileName(img_path)
            jpeg.Update()
            img = jpeg.GetOutput()
        else:
            error_msg = "This file format is not supported by the Image Holder"
            warn(Warning(error_msg))
        return img

    def _get_size(self):
        lower_left_corner = self.texture_points.GetPoint(0)
        upper_right_corner = self.texture_points.GetPoint(2)
        size = np.array(upper_right_corner) - np.array(lower_left_corner)
        return abs(size[:2])

    def _setup(self):
        """ Setup this UI Component.
        Return an image as a 2D actor with a specific position.

        Returns
        -------
        :class:`vtkTexturedActor2D`
        """
        self.texture_polydata = vtk.vtkPolyData()
        self.texture_points = vtk.vtkPoints()
        self.texture_points.SetNumberOfPoints(4)

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
        texture_mapper = set_input(texture_mapper, self.texture_polydata)

        image = vtk.vtkTexturedActor2D()
        image.SetMapper(texture_mapper)

        self.texture = vtk.vtkTexture()
        image.SetTexture(self.texture)

        image_property = vtk.vtkProperty2D()
        image_property.SetOpacity(1.0)
        image.SetProperty(image_property)
        self.actor = image

        # Add default events listener to the VTK actor.
        self.handle_events(self.actor)

    def _get_actors(self):
        """ Returns the actors that compose this UI component.
        """
        return [self.actor]

    def _add_to_renderer(self, ren):
        """ Add all subcomponents or VTK props that compose this UI component.

        Parameters
        ----------
        ren : renderer
        """
        ren.add(self.actor)

    def resize(self, size):
        """ Resize the image.

        Parameters
        ----------
        size : (float, float)
            image size (width, height) in pixels.
        """
        # Update actor.
        self.texture_points.SetPoint(0, 0, 0, 0.0)
        self.texture_points.SetPoint(1, size[0], 0, 0.0)
        self.texture_points.SetPoint(2, size[0], size[1], 0.0)
        self.texture_points.SetPoint(3, 0, size[1], 0.0)
        self.texture_polydata.SetPoints(self.texture_points)

    def _set_position(self, coords):
        """ Position the lower-left corner of this UI component.

        Parameters
        ----------
        coords: (float, float)
            Absolute pixel coordinates (x, y).
        """
        self.actor.SetPosition(*coords)

    def scale(self, factor):
        """ Scales the image.

        Parameters
        ----------
        factor : (float, float)
            Scaling factor (width, height) in pixels.
        """
        self.resize(self.size * factor)

    def set_img(self, img):
        """ Modifies the image used by the vtkTexturedActor2D.

        Parameters
        ----------
        img : imageDataGeometryFilter

        """
        self.texture = set_input(self.texture, img)


class ListBox2D(UI):
    """ UI component that allows the user to select items from a list.

    Attributes
    ----------
    on_change: function
        Callback function for when the selected items have changed.
    """

    def __init__(self, values, position=(0, 0), size=(100, 300),
                 multiselection=True, reverse_scrolling=False,
                 font_size=20, line_spacing=1.4):
        """
        Parameters
        ----------
        values: list of objects
            Values used to populate this listbox. Objects must be castable
            to string.
        position : (float, float)
            Absolute coordinates (x, y) of the lower-left corner of this
            UI component.
        size : (int, int)
            Width and height in pixels of this UI component.
        multiselection: {True, False}
            Whether multiple values can be selected at once.
        reverse_scrolling: {True, False}
            If True, scrolling up will move the list of files down.
        font_size: int
            The font size in pixels.
        line_spacing: float
            Distance between listbox's items in pixels.

        """
        self.view_offset = 0
        self.slots = []
        self.selected = []

        self.panel_size = size
        self.font_size = font_size
        self.line_spacing = line_spacing

        # self.panel.resize(size)
        self.values = values
        self.multiselection = multiselection
        self.reverse_scrolling = reverse_scrolling
        super(ListBox2D, self).__init__()

        self.position = position
        self.update()

        # Offer some standard hooks to the user.
        self.on_change = lambda: None

    def _setup(self):
        """ Setup this UI component.

        Create the ListBox (Panel2D) filled with empty slots (ListBoxItem2D).
        """
        margin = 10
        size = self.panel_size
        font_size = 20
        line_spacing = 1.4
        # Calculating the number of slots.
        slot_height = int(font_size * line_spacing)
        nb_slots = int((size[1] - 2 * margin) // slot_height)

        # This panel facilitates adding slots at the right position.
        self.panel = Panel2D(size=size, color=(1, 1, 1))

        # Add up and down buttons
        arrow_up = read_viz_icons(fname="arrow-up.png")
        self.up_button = Button2D({"up": arrow_up})
        pos = self.panel.size - self.up_button.size // 2 - margin
        self.panel.add_element(self.up_button, pos, anchor="center")

        arrow_down = read_viz_icons(fname="arrow-down.png")
        self.down_button = Button2D({"down": arrow_down})
        pos = (pos[0], self.up_button.size[1] // 2 + margin)
        self.panel.add_element(self.down_button, pos, anchor="center")

        # Initialisation of empty text actors
        slot_width = size[0] - self.up_button.size[0] - 2 * margin - margin
        x = margin
        y = size[1] - margin
        for _ in range(nb_slots):
            y -= slot_height
            item = ListBoxItem2D(list_box=self, size=(slot_width, slot_height))
            item.textblock.font_size = font_size
            item.textblock.color = (0, 0, 0)
            self.slots.append(item)
            self.panel.add_element(item, (x, y + margin))

        # Add default events listener for this UI component.
        self.up_button.on_left_mouse_button_pressed = self.up_button_callback
        self.down_button.on_left_mouse_button_pressed = self.down_button_callback

        # Handle mouse wheel events on the panel.
        up_event = "MouseWheelForwardEvent"
        down_event = "MouseWheelBackwardEvent"
        if self.reverse_scrolling:
            up_event, down_event = down_event, up_event  # Swap events

        self.add_callback(self.panel.background.actor, up_event,
                          self.up_button_callback)
        self.add_callback(self.panel.background.actor, down_event,
                          self.down_button_callback)

        # Handle mouse wheel events on the slots.
        for slot in self.slots:
            self.add_callback(slot.background.actor, up_event,
                              self.up_button_callback)
            self.add_callback(slot.background.actor, down_event,
                              self.down_button_callback)
            self.add_callback(slot.textblock.actor, up_event,
                              self.up_button_callback)
            self.add_callback(slot.textblock.actor, down_event,
                              self.down_button_callback)

    def resize(self, size):
        pass

    def _get_actors(self):
        """ Get the actors composing this UI component.
        """
        return self.panel.actors

    def _add_to_renderer(self, ren):
        """ Add all subcomponents or VTK props that compose this UI component.

        Parameters
        ----------
        ren : renderer
        """
        self.panel.add_to_renderer(ren)

    def _get_size(self):
        return self.panel.size

    def _set_position(self, coords):
        """ Position the lower-left corner of this UI component.

        Parameters
        ----------
        coords: (float, float)
            Absolute pixel coordinates (x, y).
        """
        self.panel.position = coords

    def up_button_callback(self, i_ren, obj, list_box):
        """ Pressing up button scrolls up in the combo box.

        Parameters
        ----------
        i_ren: :class:`CustomInteractorStyle`
        obj: :class:`vtkActor`
            The picked actor
        list_box: :class:`ListBox2D`

        """
        if self.view_offset > 0:
            self.view_offset -= 1
            self.update()

        i_ren.force_render()
        i_ren.event.abort()  # Stop propagating the event.

    def down_button_callback(self, i_ren, obj, list_box):
        """ Pressing down button scrolls down in the combo box.

        Parameters
        ----------
        i_ren: :class:`CustomInteractorStyle`
        obj: :class:`vtkActor`
            The picked actor
        list_box: :class:`ListBox2D`

        """
        view_end = self.view_offset + len(self.slots)
        if view_end < len(self.values):
            self.view_offset += 1
            self.update()

        i_ren.force_render()
        i_ren.event.abort()  # Stop propagating the event.

    def update(self):
        """ Refresh listbox's content. """
        view_start = self.view_offset
        view_end = view_start + len(self.slots)
        values_to_show = self.values[view_start:view_end]

        # Populate slots according to the view.
        for i, choice in enumerate(values_to_show):
            slot = self.slots[i]
            slot.element = choice
            slot.set_visibility(True)
            if slot.element in self.selected:
                slot.select()
            else:
                slot.deselect()

        # Flush remaining slots.
        for slot in self.slots[len(values_to_show):]:
            slot.element = None
            slot.set_visibility(False)
            slot.deselect()

    def clear_selection(self):
        del self.selected[:]

    def select(self, item, multiselect=False, range_select=False):
        """ Select the item.

        Parameters
        ----------
        item: ListBoxItem2D's object
            Item to select.
        multiselect: {True, False}
            If True and multiselection is allowed, the item is added to the
            selection.
            Otherwise, the selection will only contain the provided item unless
            range_select is True.
        range_select: {True, False}
            If True and multiselection is allowed, all items between the last
            selected item and the current one will be added to the selection.
            Otherwise, the selection will only contain the provided item unless
            multi_select is True.

        """
        selection_idx = self.values.index(item.element)
        if self.multiselection and range_select:
            self.clear_selection()
            step = 1 if selection_idx >= self.last_selection_idx else -1
            for i in range(self.last_selection_idx, selection_idx + step, step):
                self.selected.append(self.values[i])

        elif self.multiselection and multiselect:
            if item.element in self.selected:
                self.selected.remove(item.element)
            else:
                self.selected.append(item.element)
            self.last_selection_idx = selection_idx

        else:
            self.clear_selection()
            self.selected.append(item.element)
            self.last_selection_idx = selection_idx

        self.on_change()  # Call hook.
        self.update()


class ListBoxItem2D(UI):
    """ The text displayed in a listbox. """

    def __init__(self, list_box, size):
        """
        Parameters
        ----------
        list_box: :class:`ListBox`
            The ListBox reference this text belongs to.
        size: int
            The size of the listbox item.
        """
        super(ListBoxItem2D, self).__init__()
        self._element = None
        self.list_box = list_box
        self.background.resize(size)
        self.deselect()

    def _setup(self):
        """ Setup this UI component.

        Create the ListBoxItem2D with its background (Rectangle2D) and its
        label (TextBlock2D).
        """
        self.background = Rectangle2D()
        self.textblock = TextBlock2D(justification="left",
                                     vertical_justification="middle")

        # Add default events listener for this UI component.
        self.textblock.on_left_mouse_button_clicked = self.left_button_clicked
        self.background.on_left_mouse_button_clicked = self.left_button_clicked

    def _get_actors(self):
        """ Get the actors composing this UI component.
        """
        return self.background.actors + self.textblock.actors

    def _add_to_renderer(self, ren):
        """ Add all subcomponents or VTK props that compose this UI component.

        Parameters
        ----------
        ren : renderer
        """
        self.background.add_to_renderer(ren)
        self.textblock.add_to_renderer(ren)

    def _get_size(self):
        return self.background.size

    def _set_position(self, coords):
        """ Position the lower-left corner of this UI component.

        Parameters
        ----------
        coords: (float, float)
            Absolute pixel coordinates (x, y).
        """
        self.textblock.position = coords
        # Center background underneath the text.
        position = coords
        self.background.position = (position[0],
                                    position[1] - self.background.size[1] / 2.)

    def deselect(self):
        self.background.color = (0.9, 0.9, 0.9)
        self.textblock.bold = False
        self.selected = False

    def select(self):
        self.textblock.bold = True
        self.background.color = (0, 1, 1)
        self.selected = True

    @property
    def element(self):
        return self._element

    @element.setter
    def element(self, element):
        self._element = element
        self.textblock.message = "" if self._element is None else str(element)

    def left_button_clicked(self, i_ren, obj, list_box_item):
        """ A callback to handle left click for this UI element.

        Parameters
        ----------
        i_ren: :class:`CustomInteractorStyle`
        obj: :class:`vtkActor`
            The picked actor
        list_box_item: :class:`ListBoxItem2D`

        """
        multiselect = i_ren.event.ctrl_key
        range_select = i_ren.event.shift_key
        self.list_box.select(self, multiselect, range_select)
        i_ren.force_render()
        i_ren.event.abort()  # Stop propagating the event.
