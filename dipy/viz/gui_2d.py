# Conditional import machinery for vtk.
import glob
import math
import os

from dipy.utils.optpkg import optional_package

from ipdb import set_trace

from dipy.viz.gui import UI, TextActor2D

# Allow import, but disable doctests if we don't have vtk.
vtk, have_vtk, setup_module = optional_package('vtk')

if have_vtk:
    vtkInteractorStyleUser = vtk.vtkInteractorStyleUser
    version = vtk.vtkVersion.GetVTKSourceVersion().split(' ')[-1]
    major_version = vtk.vtkVersion.GetVTKMajorVersion()
else:
    vtkInteractorStyleUser = object

numpy_support, have_ns, _ = optional_package('vtk.util.numpy_support')


class Panel2D(UI):
    """ A 2D UI Panel.
    Can contain one or more UI elements.
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
        super(Panel2D, self).__init__()
        self.center = center
        self.size = size
        self.lower_limits = (self.center[0] - self.size[0] / 2, self.center[1] - self.size[1] / 2)

        self.panel = Rectangle2D(size=size, center=center, color=color, opacity=opacity)  # type: Rectangle2D
        self.ui_list.append(self.panel)

        self.element_positions = []
        self.element_positions.append((self.panel, 0.5, 0.5))
        self.alignment = align

    def add_to_renderer(self, ren):
        # Should be a recursive function, but we never go more than 2 levels down (by design)
        """ Add props to renderer

        Parameters
        ----------
        ren : renderer
        """
        for ui_item in self.ui_list:
            ui_item.add_to_renderer(ren=ren)

    def add_element(self, element, relative_position):
        """ Adds an elements to the panel.
        The center of the rectangular panel is its bottom lower position.

        Parameters
        ----------
        element : UI
            The UI item to be added.
        relative_position : (float, float)
        """
        self.ui_list.append(element)
        self.element_positions.append((element, relative_position[0], relative_position[1]))
        element.set_center((self.lower_limits[0] + relative_position[0]*self.size[0],
                            self.lower_limits[1] + relative_position[1]*self.size[1]))

    def add_callback(self, event_type, callback, component):
        """ Adds events to an actor.

        Parameters
        ----------
        event_type : string
            event code
        callback : function
            callback function
        component : UI
            component
        """
        super(Panel2D, self).add_callback(component.actor, event_type, callback)

    def set_center(self, position):
        """ Sets the panel center to position.
        The center of the rectangular panel is its bottom lower position.

        Parameters
        ----------
        position : (float, float)
        """
        self.center = position
        self.lower_limits = (position[0] - self.size[0] / 2, position[1] - self.size[1] / 2)
        for ui_element in self.element_positions:
            ui_element[0].set_center((self.lower_limits[0] + ui_element[1]*self.size[0],
                                      self.lower_limits[1] + ui_element[2]*self.size[1]))

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
            pass


class Button2D(UI):
    """A 2D overlay button and is of type vtkTexturedActor2D.
    Currently supports:
    - Multiple icons.
    - Switching between icons.
    """

    def __init__(self, icon_fnames):
        """

        Parameters
        ----------
        icon_fnames : dict
            {iconname : filename, iconname : filename, ...}
        """
        super(Button2D, self).__init__()
        self.icon_extents = dict()
        self.icons = self.build_icons(icon_fnames)
        self.icon_names = list(self.icons.keys())
        self.current_icon_id = 0
        self.current_icon_name = self.icon_names[self.current_icon_id]
        self.actor = self.build_actor(self.icons[self.current_icon_name])

        # self.ui_list.append(self)

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
            A dictionary of corresponding vtkImageDataGeometryFilters
        """
        icons = {}
        for icon_name, icon_fname in icon_fnames.items():
            png = vtk.vtkPNGReader()
            png.SetFileName(icon_fname)
            png.Update()

            self.icon_extents[icon_name] = png.GetOutput().GetExtent()
            # Convert the image to a polydata
            image_data_geometry_filter = vtk.vtkImageDataGeometryFilter()
            image_data_geometry_filter.SetInputConnection(png.GetOutputPort())
            image_data_geometry_filter.Update()

            icons[icon_name] = image_data_geometry_filter

        return icons

    def build_actor(self, icon, center=None):
        """ Return an image as a 2D actor with a specific position.

        Parameters
        ----------
        icon : imageDataGeometryFilter
        center : (float, float)

        Returns
        -------
        button : vtkTexturedActor2D
        """

        mapper = vtk.vtkPolyDataMapper2D()
        mapper.SetInputConnection(icon.GetOutputPort())

        button = vtk.vtkTexturedActor2D()
        button.SetMapper(mapper)

        if center is not None:
            button.SetCenter(*center)

        return button

    def add_to_renderer(self, ren):
        """ Adds the button actor to renderer.

        Parameters
        ----------
        ren : renderer
        """
        ren.add(self.actor)

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
        self.actor.GetMapper().SetInputConnection(icon.GetOutputPort())

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
        extent = self.icon_extents[self.current_icon_name]
        self.actor.SetPosition(position[0] - (extent[1] - extent[0])/2, position[1] - (extent[3] - extent[2])/2)


class TextBox2D(UI):
    """ An editable 2D text box that behaves as a UI component.
    Currently supports:
    - Basic text editing.
    - Cursor movements.
    - Single and multi-line text boxes.
    - Pre text formatting (text needs to be formatted beforehand).
    """

    def __init__(self, width, height, text="Enter Text"):
        """

        Parameters
        ----------
        width : int
            The number of characters in a single line of text.
        height : int
            The number of lines in the textbox.
        text : string
            Initial text while placing the element.
        """
        super(TextBox2D, self).__init__()
        self.text = text
        self.actor = self.build_actor(self.text)
        self.width = width
        self.height = height
        self.window_left = 0
        self.window_right = 0
        self.caret_pos = 0
        self.init = True

        # self.ui_list.append(self)

    def set_message(self, message):
        """ Set custom text to textbox.

        Parameters
        ----------
        message: string
        """
        self.text = message
        self.actor.set_message(message)
        self.init = False
        self.window_right = len(self.text)
        self.window_left = 0
        self.caret_pos = self.window_right

    def build_actor(self, text, position=(100, 10), color=(1, 1, 1),
                    font_size=18, font_family='Arial', justification='left',
                    bold=False, italic=False, shadow=False):

        """ Builds a text actor.

        Parameters
        ----------
        text : string
            The initial text while building the actor.
        position : (float, float)
        color : (float, float, float)
            Values must be between 0-1.
        font_size : int
        font_family : string
            Currently only supports Ariel.
        justification : string
            left, right or center.
        bold : bool
        italic : bool
        shadow : bool

        Returns
        -------
        text_actor : actor2d

        """
        text_actor = TextActor2D()
        text_actor.set_position(position)
        text_actor.message(text)
        text_actor.font_size(font_size)
        text_actor.font_family(font_family)
        text_actor.justification(justification)
        text_actor.font_style(bold, italic, shadow)
        text_actor.color(color)
        # text_actor.GetTextProperty().SetBackgroundColor(1, 1, 1)
        # text_actor.GetTextProperty().SetBackgroundOpacity(1.0)

        return text_actor

    def add_to_renderer(self, ren):
        """ Adds the text actor to the renderer.

        Parameters
        ----------
        ren : renderer
        """
        ren.add(self.actor)

    def add_callback(self, event_type, callback):
        """ Adds events to the text actor.

        Parameters
        ----------
        event_type : string
            event code
        callback : string
            callback function
        """
        super(TextBox2D, self).add_callback(self.actor, event_type, callback)

    def width_set_text(self, text):
        """ Adds newlines to text where necessary.
        This is needed for multi-line text boxes.

        Parameters
        ----------
        text : string
            The final text to be formatted.

        Returns
        -------
        multi_line_text : string

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
        character : string
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
        self.caret_pos += 1
        if self.caret_pos > len(self.text):
            self.caret_pos = len(self.text)

    def move_caret_left(self):
        """ Moves the caret towards left.

        """
        self.caret_pos -= 1
        if self.caret_pos < 0:
            self.caret_pos = 0

    def right_move_right(self):
        """ Moves right window right.

        """
        if self.window_right <= len(self.text):
            self.window_right += 1

    def right_move_left(self):
        """ Moves right window left.

        """
        if self.window_right > 0:
            self.window_right -= 1

    def left_move_right(self):
        """ Moves left window right.

        """
        if self.window_left <= len(self.text):
            self.window_left += 1

    def left_move_left(self):
        """ Moves left window left.

        """
        if self.window_left > 0:
            self.window_left -= 1

    def add_character(self, character):
        """ Inserts a character into the text and moves window and caret accordingly.

        Parameters
        ----------
        character : string
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
        """ Finally renders text.

        Parameters
        ----------
        show_caret : bool
            Whether or not to show the caret.
        """
        text = self.showable_text(show_caret)
        if text == "":
            text = "Enter Text"
        self.actor.set_message(self.width_set_text(text))

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
        self.actor.SetPosition(position)


class Rectangle2D(UI):
    """A 2D rectangle sub-classed from UI.
    Uses vtkPolygon.
    """

    def __init__(self, size, center=(0, 0), color=(1, 1, 1), opacity=1.0):
        """

        Parameters
        ----------
        size : (float, float)
        center : (float, float)
        color : (float, float, float)
            Must take values between 0-1.
        opacity : float
        """
        super(Rectangle2D, self).__init__()
        self.size = size
        self.actor = self.build_actor(size=size, center=center, color=color, opacity=opacity)

        # self.ui_list.append(self)

    def add_to_renderer(self, ren):
        """ Adds the actor to the renderer.

        Parameters
        ----------
        ren : renderer
        """
        ren.add(self.actor)

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
        self.actor.SetPosition(position[0] - self.size[0]/2, position[1] - self.size[1]/2)


class LineSlider2D(UI):
    """ A 2D Line Slider.
    Currently supports:
    - A disk on a line (a thin rectangle).
    - Setting disk position.
    """
    def __init__(self, line_width=5, inner_radius=0, outer_radius=10, center=(450, 20), length=200):

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
        """
        super(LineSlider2D, self).__init__()

        self.length = length
        self.line_width = line_width
        self.center = center
        self.current_state = center[0]
        self.left_x_position = center[0] - length / 2
        self.right_x_position = center[0] + length / 2

        self.slider_line = None
        self.slider_disk = None
        self.text = None

        self.build_actors(inner_radius=inner_radius, outer_radius=outer_radius)

    def build_actors(self, inner_radius, outer_radius):
        """

        Parameters
        ----------
        inner_radius: int
        outer_radius: int
        """
        self.slider_line = Rectangle2D(size=(self.length, self.line_width), center=self.center).actor
        self.slider_line.GetProperty().SetColor(1, 0, 0)

        # create source
        disk = vtk.vtkDiskSource()
        disk.SetInnerRadius(inner_radius)
        disk.SetOuterRadius(outer_radius)
        disk.SetRadialResolution(10)
        disk.SetCircumferentialResolution(50)
        disk.Update()

        # mapper
        mapper = vtk.vtkPolyDataMapper2D()
        mapper.SetInputConnection(disk.GetOutputPort())

        # actor
        self.slider_disk = vtk.vtkActor2D()
        self.slider_disk.SetMapper(mapper)

        self.slider_disk.SetPosition(self.center[0], self.center[1])

        self.text = TextActor2D()

        self.text.set_position(position=(self.left_x_position-50, self.center[1]-10))
        percentage = self.calculate_percentage(current_val=self.current_state)
        self.text.set_message(text=percentage)
        self.text.font_size(size=16)

    def add_to_renderer(self, ren):
        """ Adds the actors to the renderer.

        Parameters
        ----------
        ren : renderer
        """
        ren.add(self.slider_line)
        ren.add(self.slider_disk)
        ren.add(self.text)

    def add_callback(self, event_type, callback, component):
        """ Adds events to an actor.

        Parameters
        ----------
        event_type : string
            event code
        callback : function
            callback function
        component : UI
            component
        """
        super(LineSlider2D, self).add_callback(component, event_type, callback)

    def set_position(self, position):
        """ Sets the disk's position.

        Parameters
        ----------
        position : (float, float)
        """
        x_position = position[0]
        if x_position < self.center[0] - self.length/2:
            x_position = self.center[0] - self.length/2
        if x_position > self.center[0] + self.length/2:
            x_position = self.center[0] + self.length/2
        self.slider_disk.SetPosition(x_position, self.center[1])
        self.current_state = x_position

    def calculate_percentage(self, current_val):
        """ Calculates the percentage to be displayed.

        Parameters
        ----------
        current_val : int
        """
        percentage = int(((current_val-self.left_x_position)*100)/(self.right_x_position-self.left_x_position))
        if percentage < 0:
            percentage = 0
        if percentage > 100:
            percentage = 100
        return str(percentage) + "%"

    def set_percentage(self, current_val):
        """ Sets text percentage.

        Parameters
        ----------
        current_val : int
            This is the x-position of the slider in the 2D coordinate space
            and not the percentage on the base scale.
        """
        self.current_state = current_val
        percentage = self.calculate_percentage(current_val=current_val)
        self.text.set_message(text=percentage)

    def set_center(self, position):
        """ Sets the center of the slider to position.

        Parameters
        ----------
        position : (float, float)
        """
        self.slider_line.SetPosition(position[0] - self.length / 2, position[1] - self.line_width / 2)

        x_change = position[0] - self.center[0]
        self.current_state += x_change
        self.center = position
        self.set_position((self.current_state, self.center[1]))

        self.left_x_position = position[0] - self.length / 2
        self.right_x_position = position[0] + self.length / 2
        self.text.SetPosition(position[0] - self.length / 2 - 40, position[1] - 10)
        self.set_percentage(int(self.current_state))


class DiskSlider2D(UI):
    """ A disk slider.
    A disk moves along the boundary of a ring.
    Goes from 0-360.

    """
    def __init__(self, base_disk_inner_radius=40, base_disk_outer_radius=44, base_disk_position=(450, 100),
                 move_disk_outer_radius=10, move_disk_inner_radius=0):
        """

        Parameters
        ----------
        base_disk_inner_radius : int
            Inner radius of the base disk.
        base_disk_outer_radius : int
            Outer radius of the base disk.
        base_disk_position : (float, float)
            Position of the system.
        move_disk_outer_radius : int
            Outer radius of the moving disk.
        move_disk_inner_radius : int
            Inner radius of the moving disk.
        """
        super(DiskSlider2D, self).__init__()
        self.move_disk_inner_radius = move_disk_inner_radius
        self.move_disk_outer_radius = move_disk_outer_radius
        self.base_disk_inner_radius = base_disk_inner_radius
        self.base_disk_outer_radius = base_disk_outer_radius
        self.base_disk_radius = base_disk_inner_radius + (base_disk_outer_radius - base_disk_inner_radius) / 2
        self.base_disk_center = base_disk_position

        self.angle_state = 0

        self.base_disk = None
        self.move_disk = None
        self.text = None

        self.build_actors()

    def build_actors(self):
        # Base Disk
        base_disk = vtk.vtkDiskSource()
        base_disk.SetInnerRadius(self.base_disk_inner_radius)
        base_disk.SetOuterRadius(self.base_disk_outer_radius)
        base_disk.SetRadialResolution(10)
        base_disk.SetCircumferentialResolution(50)
        base_disk.Update()

        base_disk_mapper = vtk.vtkPolyDataMapper2D()
        base_disk_mapper.SetInputConnection(base_disk.GetOutputPort())

        self.base_disk = vtk.vtkActor2D()
        self.base_disk.SetMapper(base_disk_mapper)
        self.base_disk.GetProperty().SetColor(1, 0, 0)
        self.base_disk.SetPosition(self.base_disk_center)
        # /Base Disk

        # Move Disk
        move_disk = vtk.vtkDiskSource()
        move_disk.SetInnerRadius(self.move_disk_inner_radius)
        move_disk.SetOuterRadius(self.move_disk_outer_radius)
        move_disk.SetRadialResolution(10)
        move_disk.SetCircumferentialResolution(50)
        move_disk.Update()

        move_disk_mapper = vtk.vtkPolyDataMapper2D()
        move_disk_mapper.SetInputConnection(move_disk.GetOutputPort())

        self.move_disk = vtk.vtkActor2D()
        self.move_disk.SetMapper(move_disk_mapper)
        self.move_disk.GetProperty().SetColor(1, 1, 1)
        self.move_disk.SetPosition((self.base_disk_center[0] + self.base_disk_radius, self.base_disk_center[1]))
        # /Move Disk

        # Text
        self.text = TextActor2D()

        self.text.set_position(position=(self.base_disk_center[0] - 16, self.base_disk_center[1] - 8))
        percentage = self.calculate_percentage(current_val=0)
        self.text.set_message(text=percentage)
        self.text.font_size(size=16)
        # /Text

    def set_percentage(self, current_val):
        """ Sets the text percentage.

        Parameters
        ----------
        current_val : float
        """
        percentage = self.calculate_percentage(current_val=current_val)
        self.text.set_message(text=percentage)

    def calculate_percentage(self, current_val):
        """ Calculate percentage of completion

        Parameters
        ----------
        current_val : float
        """
        percentage = int((current_val/360)*100)
        if len(str(percentage)) == 1:
            percentage_string = "0" + str(percentage)
        else:
            percentage_string = str(percentage)
        return percentage_string + "%"

    def add_to_renderer(self, ren):
        """ Adds the actors to the renderer.

        Parameters
        ----------
        ren : renderer
        """
        ren.add(self.base_disk)
        ren.add(self.move_disk)
        ren.add(self.text)

    def add_callback(self, event_type, callback, component):
        """ Adds events to an actor

        Parameters
        ----------
        event_type : string
            event code
        callback : function
            callback function
        component : UI
            component
        """
        super(DiskSlider2D, self).add_callback(component, event_type, callback)

    def get_poi(self, coordinates):
        """ Finds point of intersection between the line joining the mouse
        position and the center with the base disk.

        Parameters
        ----------
        coordinates : (float, float)

        Returns
        -------
        x, y : (float, float)

        """
        radius = self.base_disk_radius
        center = self.base_disk_center
        point = coordinates

        dx = point[0] - center[0]
        dy = point[1] - center[1]

        x1 = float(center[0]) + float(radius*dx)/float(math.sqrt(float(dx*dx) + float(dy*dy)))
        x2 = float(center[0]) - float(radius*dx)/float(math.sqrt(float(dx*dx) + float(dy*dy)))

        if x1 == x2:
            y1 = center[1] + radius
            y2 = center[1] - radius
        else:
            y1 = float(center[1]) + float(float(dy) / float(dx)) * float(x1 - center[0])
            y2 = float(center[1]) + float(float(dy) / float(dx)) * float(x2 - center[0])

        d1 = (x1 - point[0])*(x1 - point[0]) + (y1 - point[1])*(y1 - point[1])
        d2 = (x2 - point[0])*(x2 - point[0]) + (y2 - point[1])*(y2 - point[1])

        if d1 < d2:
            return x1, y1
        else:
            return x2, y2

    def get_angle(self, coordinates):
        """ Gets the angle made with the X-Axis for calculating
        the percentage. Varies between 0-360.

        Parameters
        ----------
        coordinates : (float, float)

        Returns
        -------
        angle : float

        """
        center = self.base_disk_center

        perpendicular = -center[1] + coordinates[1]
        base = -center[0] + coordinates[0]

        angle = math.degrees(math.atan2(float(perpendicular), float(base)))
        if angle < 0:
            angle += 360

        return angle

    def move_move_disk(self, click_position):
        intersection_coordinate = self.get_poi(click_position)
        self.set_position(intersection_coordinate)
        angle = self.get_angle(intersection_coordinate)
        self.set_percentage(angle)
        self.angle_state = angle

    def set_position(self, position):
        """ Sets the move disk's position.

        Parameters
        ----------
        position : (float, float)
        """
        self.move_disk.SetPosition(position)

    def set_center(self, position):
        """ Sets the center of the slider to position.

        Parameters
        ----------
        position : (float, float)
        """
        self.base_disk_center = position

        self.base_disk.SetPosition(position)

        if self.angle_state > 180:
            self.angle_state -= 360
        self.set_position((position[0] + self.base_disk_radius * math.cos(math.radians(self.angle_state)),
                           position[1] + self.base_disk_radius * math.sin(math.radians(self.angle_state))))

        self.text.SetPosition(position[0] - 16, position[1] - 8)


class Text2D(UI):
    """ A text actor in 2D.
    Ability to set and reset message.
    """
    def __init__(self, message):
        """

        Parameters
        ----------
        message: string
        """
        super(Text2D, self).__init__()
        self.actor = self.build_actor(text=message)

        # self.ui_list.append(self)

    def build_actor(self, text, position=(100, 10), color=(1, 1, 1),
                    font_size=12, font_family='Arial', justification='left',
                    bold=False, italic=False, shadow=False):

        """ Builds a text actor.

        Parameters
        ----------
        text : string
            The initial text while building the actor.
        position : (float, float)
        color : (float, float, float)
            Values must be between 0-1.
        font_size : int
        font_family : string
            Currently only supports Ariel.
        justification : string
            left, right or center.
        bold : bool
        italic : bool
        shadow : bool

        Returns
        -------
        text_actor : actor2d

        """
        text_actor = TextActor2D()
        text_actor.set_position(position)
        text_actor.message(text)
        text_actor.font_size(font_size)
        text_actor.font_family(font_family)
        text_actor.justification(justification)
        text_actor.font_style(bold, italic, shadow)
        text_actor.color(color)
        # text_actor.GetTextProperty().SetBackgroundColor(0, 1, 1)
        # text_actor.GetTextProperty().SetBackgroundOpacity(1.0)

        return text_actor

    def set_message(self, message):
        self.actor.set_message(message)

    def add_to_renderer(self, ren):
        """ Adds the actor to renderer.

        Parameters
        ----------
        ren : renderer
        """
        ren.add(self.actor)

    def add_callback(self, event_type, callback):
        """ Adds events to actor.

        Parameters
        ----------
        event_type : string
            event code
        callback : function
            callback function
        """
        super(Text2D, self).add_callback(self.actor, event_type, callback)

    def set_center(self, position):
        """ Sets the text center to position.

        Parameters
        ----------
        position : (float, float)
        """
        self.actor.SetPosition(position)
