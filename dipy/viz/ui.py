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
        return np.asarray(self._get_size())

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
    """

    def __init__(self, icon_fnames, position=(0, 0), size=(30, 30)):
        """
        Parameters
        ----------
        icon_fnames : dict
            {iconname : filename, iconname : filename, ...}
        position : (float, float), optional
            Absolute coordinates (x, y) of the lower-left corner of the button.
        size : (int, int), optional
            Width and height in pixels of the button.
        """
        super(Button2D, self).__init__(position)

        self.icon_extents = dict()
        self.icons = self._build_icons(icon_fnames)
        self.icon_names = list(self.icons.keys())
        self.current_icon_id = 0
        self.current_icon_name = self.icon_names[self.current_icon_id]
        self.set_icon(self.icons[self.current_icon_name])
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

    def _setup(self):
        """ Setup this UI component.

        Creating the button actor used internally.
        """
        # This is highly inspired by
        # https://github.com/Kitware/VTK/blob/c3ec2495b183e3327820e927af7f8f90d34c3474\
        # /Interaction/Widgets/vtkBalloonRepresentation.cxx#L47

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
        if vtk.VTK_MAJOR_VERSION <= 5:
            mapper.SetInput(self._polygonPolyData)
        else:
            mapper.SetInputData(self._polygonPolyData)

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
        mapper.SetInputConnection(self._disk.GetOutputPort())

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
        self.element_positions = []
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
        for element, offset in self.element_positions:
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

    def add_element(self, element, coords):
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

        self._elements.append(element)
        self.element_positions.append((element, coords))
        element.position = self.position + coords

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
        """ Inserts a character into the text and moves window and caret accordingly.

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
        self.message = self.message[:self.caret_pos - 1] + \
                       self.message[self.caret_pos:]
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
    """
    def __init__(self, center=(0, 0),
                 initial_value=50, min_value=0, max_value=100,
                 length=200, line_width=5,
                 inner_radius=0, outer_radius=10,
                 font_size=16,
                 text_template="{value:.1f} ({ratio:.0%})"):
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
            Inner radius of the slider's handle.
        outer_radius : int
            Outer radius of the slider's handle.
        font_size : int
            Size of the text to display alongside the slider (pt).
        text_template : str, callable
            If str, text template can contain one or multiple of the
            replacement fields: `{value:}`, `{ratio:}`.
            If callable, this instance of `:class:LineSlider2D` will be
            passed as argument to the text template function.
        """
        super(LineSlider2D, self).__init__()

        self.track.width = length
        self.track.height = line_width
        self.handle.inner_radius = inner_radius
        self.handle.outer_radius = outer_radius
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
        self.handle = Disk2D(outer_radius=1)
        self.handle.color = (1, 1, 1)

        # Slider Text
        self.text = TextBlock2D(justification="center",
                                vertical_justification="top")

        # Add default events listener for this UI component.
        self.track.on_left_mouse_button_pressed = self.track_click_callback
        self.track.on_left_mouse_button_dragged = self.handle_move_callback
        self.handle.on_left_mouse_button_dragged = self.handle_move_callback

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
        position = i_ren.event.position
        self.set_position(position)
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
    initial_value: float
        Initial Value of Rotation of the actor assigned on creation of object.
    track : :class:`Disk2D`
        The circle on which the slider's handle moves.
    handle : :class:`Disk2D`
        The moving part of the slider.
    text : :class:`TextBlock2D`
        The text that shows percentage.
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

        self.initial_value = initial_value
        self.value = initial_value
        self.previous_value = initial_value

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
        self.handle.color = (1, 1, 1)

        # Slider Text
        self.text = TextBlock2D(justification="center",
                                vertical_justification="middle")

        # Add default events listener for this UI component.
        self.track.on_left_mouse_button_pressed = self.track_click_callback
        self.track.on_left_mouse_button_dragged = self.handle_move_callback
        self.handle.on_left_mouse_button_dragged = self.handle_move_callback

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
        self.move_handle(click_position=click_position)
        i_ren.force_render()
        i_ren.event.abort()  # Stop propagating the event.
