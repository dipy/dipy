# Conditional import machinery for vtk.
import math

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

    def __init__(self, center, size, color=(0.1, 0.1, 0.1), opacity=0.7):
        super(Panel2D, self).__init__()
        self.center = center
        self.size = size
        self.panel = Rectangle2D(size=size, center=center, color=color, opacity=opacity)
        self.lower_limits = (self.center[0] - self.size[0]/2, self.center[1] - self.size[1]/2)

        self.ui_list.append(self.panel)

    def add_to_renderer(self, ren):
        # Should be a recursive function, but we never go more than 2 levels down (by design)
        for ui_item_list in self.ui_list:
            for ui_item in ui_item_list.ui_list:
                ren.add(ui_item.actor)

    def add_element(self, element, relative_position):
        self.ui_list.append(element)
        element.set_center((self.lower_limits[0] + relative_position[0]*self.size[0],
                            self.lower_limits[1] + relative_position[1]*self.size[1]))


class Button2D(UI):
    """A 2D overlay button and is of type vtkTexturedActor2D.

    """

    def __init__(self, icon_fnames):
        """

        Parameters
        ----------
        icon_fnames
        """
        super(Button2D, self).__init__()
        self.icons = self.build_icons(icon_fnames)
        self.icon_names = list(self.icons.keys())
        self.current_icon_id = 0
        self.current_icon_name = self.icon_names[self.current_icon_id]
        self.actor = self.build_actor(self.icons[self.current_icon_name])

        self.ui_list.append(self)

    def build_icons(self, icon_fnames):
        """ Converts filenames to vtkImageDataGeometryFilters
        A peprocessing step to prevent re-read of filenames during every state change

        Parameters
        ----------
        icon_fnames : A list of filenames

        Returns
        -------
        icons : A list of corresponding vtkImageDataGeometryFilters
        """
        icons = {}
        for icon_name, icon_fname in icon_fnames.items():
            png = vtk.vtkPNGReader()
            png.SetFileName(icon_fname)
            png.Update()

            # print png.GetOutput().GetExtent()

            # Convert the image to a polydata
            imageDataGeometryFilter = vtk.vtkImageDataGeometryFilter()
            imageDataGeometryFilter.SetInputConnection(png.GetOutputPort())
            imageDataGeometryFilter.Update()

            icons[icon_name] = imageDataGeometryFilter

        return icons

    def build_actor(self, icon, center=None):
        """ Return an image as a 2D actor with a specific position

        Parameters
        ----------
        icon : imageDataGeometryFilter
        center : a two tuple

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
        ren.add(self.actor)

    def add_callback(self, event_type, callback):
        """ Adds events to button actor

        Parameters
        ----------
        event_type: event code
        callback: callback function
        """
        super(Button2D, self).add_callback(self.actor, event_type, callback)

    def set_icon(self, icon):
        """ Modifies the icon used by the vtkTexturedActor2D

        Parameters
        ----------
        icon : imageDataGeometryFilter
        """
        self.actor.GetMapper().SetInputConnection(icon.GetOutputPort())

    def next_icon_name(self):
        """ Returns the next icon name while cycling through icons

        """
        self.current_icon_id += 1
        if self.current_icon_id == len(self.icons):
            self.current_icon_id = 0
        self.current_icon_name = self.icon_names[self.current_icon_id]

    def next_icon(self):
        """ Increments the state of the Button
            Also changes the icon
        """
        self.next_icon_name()
        self.set_icon(self.icons[self.current_icon_name])

    def set_center(self, position):
        self.actor.SetPosition(position)


class TextBox2D(UI):
    def __init__(self, width, height, text="Enter Text"):
        """

        Parameters
        ----------
        width
        height
        text
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

        self.ui_list.append(self)

    def build_actor(self, text, position=(100, 10), color=(1, 1, 1),
                    font_size=18, font_family='Arial', justification='left',
                    bold=False, italic=False, shadow=False):

        """ Builds a text actor

        Parameters
        ----------
        text
        position
        color
        font_size
        font_family
        justification
        bold
        italic
        shadow

        Returns
        -------
        text_actor

        """
        text_actor = TextActor2D()
        text_actor.set_position(position)
        text_actor.message(text)
        text_actor.font_size(font_size)
        text_actor.font_family(font_family)
        text_actor.justification(justification)
        text_actor.font_style(bold, italic, shadow)
        text_actor.color(color)

        return text_actor

    def add_to_renderer(self, ren):
        ren.add(self.actor)

    def add_callback(self, event_type, callback):
        """ Adds events to the text actor

        Parameters
        ----------
        event_type: event code
        callback: callback function
        """
        super(TextBox2D, self).add_callback(self.actor, event_type, callback)

    def width_set_text(self, text):
        """ Adds newlines to text where necessary

        Parameters
        ----------
        text

        Returns
        -------
        multi_line_text

        """
        multi_line_text = ""
        for i in range(len(text)):
            multi_line_text += text[i]
            if (i + 1) % self.width == 0:
                multi_line_text += "\n"
        return multi_line_text.rstrip("\n")

    def handle_character(self, character):
        """ Main driving function that handles button events

        Parameters
        ----------
        character
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
        """ Moves the caret towards right

        """
        self.caret_pos += 1
        if self.caret_pos > len(self.text):
            self.caret_pos = len(self.text)

    def move_caret_left(self):
        """ Moves the caret towards left

        """
        self.caret_pos -= 1
        if self.caret_pos < 0:
            self.caret_pos = 0

    def right_move_right(self):
        """ Moves right window right

        """
        if self.window_right <= len(self.text):
            self.window_right += 1

    def right_move_left(self):
        """ Moves right window left

        """
        if self.window_right > 0:
            self.window_right -= 1

    def left_move_right(self):
        """ Moves left window right

        """
        if self.window_left <= len(self.text):
            self.window_left += 1

    def left_move_left(self):
        """ Moves left window left

        """
        if self.window_left > 0:
            self.window_left -= 1

    def add_character(self, character):
        """ Inserts a character into the text and moves window and caret accordingly

        Parameters
        ----------
        character

        Returns
        -------

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
        """ Removes a character from the text and moves window and caret accordingly

        Returns
        -------

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
        """ Handles left button press

        """
        self.move_caret_left()
        if self.caret_pos == self.window_left - 1:
            if self.window_right - self.window_left == self.height * self.width - 1:
                self.left_move_left()
                self.right_move_left()

    def move_right(self):
        """ Handles right button press

        """
        self.move_caret_right()
        if self.caret_pos == self.window_right + 1:
            if self.window_right - self.window_left == self.height * self.width - 1:
                self.left_move_right()
                self.right_move_right()

    def showable_text(self, show_caret):
        """ Chops out text to be shown on the screen

        Parameters
        ----------
        show_caret

        Returns
        -------

        """
        if show_caret:
            ret_text = self.text[:self.caret_pos] + "_" + self.text[self.caret_pos:]
        else:
            ret_text = self.text
        ret_text = ret_text[self.window_left:self.window_right + 1]
        return ret_text

    def render_text(self, show_caret=True):
        """ Renders text

        Parameters
        ----------
        show_caret
        """
        text = self.showable_text(show_caret)
        if text == "":
            text = "Enter Text"
        self.actor.set_message(self.width_set_text(text))

    def edit_mode(self):
        """ Turns on edit mode

        """
        if self.init:
            self.text = ""
            self.init = False
            self.caret_pos = 0
        self.render_text()

    def set_center(self, position):
        self.actor.SetPosition(position)


class Rectangle2D(UI):
    def __init__(self, size, center=(0, 0), color=(1, 1, 1), opacity=1.0):
        """

        Parameters
        ----------
        size
        """
        super(Rectangle2D, self).__init__()
        self.size = size
        self.actor = self.build_actor(size=size, center=center, color=color, opacity=opacity)

        self.ui_list.append(self)

    def add_to_renderer(self, ren):
        ren.add(self.actor)

    def build_actor(self, size, center, color, opacity):
        """

        Parameters
        ----------
        size
        center
        color
        opacity

        Returns
        -------
        actor

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
        self.actor.SetPosition(position[0] - self.size[0]/2, position[1] - self.size[1]/2)


class LineSlider2D(UI):
    def __init__(self, start_point=(350, 20), end_point=(550, 20), line_width=5, inner_radius=0,
                 outer_radius=10, center=(450, 20), length=200):
        """

        Parameters
        ----------
        inner_radius
        outer_radius
        start_point
        end_point
        line_width
        """
        super(LineSlider2D, self).__init__()
        self.slider_line = LineSlider2DBase(line_width=line_width, center=center, length=length)
        self.slider_disk = LineSlider2DDisk(inner_radius=inner_radius, outer_radius=outer_radius, center=center,
                                            length=length)
        self.text = LineSlider2DText(center=center, length=length,
                                     current_val=(start_point[0] + (end_point[0] - start_point[0])/2))

        self.ui_list.append(self.slider_line)
        self.ui_list.append(self.slider_disk)
        self.ui_list.append(self.text)

    def add_to_renderer(self, ren):
        ren.add(self.slider_line.actor)
        ren.add(self.slider_disk.actor)
        ren.add(self.text.actor)

    def add_callback(self, event_type, callback, component):
        """ Adds events to an actor

        Parameters
        ----------
        event_type: event code
        callback: callback function
        component: component
        """
        super(LineSlider2D, self).add_callback(component.actor, event_type, callback)

    def set_center(self, position):
        self.slider_disk.set_center(position)
        self.slider_line.set_center(position)
        self.text.set_center(position)


class LineSlider2DBase(UI):

    def __init__(self, line_width, center, length):
        """

        Parameters
        ----------
        center
        length
        line_width
        """
        super(LineSlider2DBase, self).__init__()
        self.actor = self.build_actor(line_width=line_width, center=center, length=length)
        self.length = length
        self.line_width = line_width
        self.ui_list.append(self)

    def build_actor(self, line_width, center, length):
        """

        Parameters
        ----------
        center
        length
        line_width

        Returns
        -------
        actor

        """
        actor = Rectangle2D(size=(length, line_width), center=center).actor
        actor.GetProperty().SetColor(1, 0, 0)
        return actor

    def set_center(self, position):
        self.actor.SetPosition(position[0] - self.length/2, position[1] - self.line_width/2)


class LineSlider2DDisk(UI):

    def __init__(self, inner_radius, outer_radius, center, length):
        """

        Parameters
        ----------
        center
        length
        inner_radius
        outer_radius
        """
        super(LineSlider2DDisk, self).__init__()
        self.center = center
        self.length = length
        self.actor = self.build_actor(inner_radius=inner_radius, outer_radius=outer_radius)

        self.ui_list.append(self)

    def build_actor(self, inner_radius, outer_radius):
        """

        Parameters
        ----------
        inner_radius
        outer_radius

        Returns
        -------
        actor

        """
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
        actor = vtk.vtkActor2D()
        actor.SetMapper(mapper)

        actor.SetPosition(self.center[0], self.center[1])

        return actor

    def set_position(self, position):
        """ Sets the disk's position

        Parameters
        ----------
        position
        """
        x_position = position[0]
        if x_position < self.center[0] - self.length/2:
            x_position = self.center[0] - self.length/2
        if x_position > self.center[0] + self.length/2:
            x_position = self.center[0] + self.length/2
        self.actor.SetPosition(x_position, self.center[1])

    def set_center(self, position):
        self.center = position
        self.set_position(position)


class LineSlider2DText(UI):

    def __init__(self, center, length, current_val):
        """

        Parameters
        ----------
        limits
        current_val
        center
        length
        """
        super(LineSlider2DText, self).__init__()
        self.left_x_position = center[0] - length/2
        self.right_x_position = center[0] + length/2
        self.length = length

        self.actor = self.build_actor(current_val=current_val, position=(self.left_x_position-50, center[1]))

        self.ui_list.append(self)

    def calculate_percentage(self, current_val):
        """

        Parameters
        ----------
        current_val

        Returns
        -------

        """
        percentage = ((current_val-self.left_x_position)*100)/(self.right_x_position-self.left_x_position)
        if percentage < 0:
            percentage = 0
        if percentage > 100:
            percentage = 100
        return str(percentage) + "%"

    def build_actor(self, current_val, position):
        """

        Parameters
        ----------
        current_val
        position

        Returns
        -------
        actor

        """
        actor = TextActor2D()

        actor.set_position(position=position)
        percentage = self.calculate_percentage(current_val=current_val)
        actor.set_message(text=percentage)
        actor.font_size(size=16)

        return actor

    def set_percentage(self, current_val):
        """

        Parameters
        ----------
        current_val
        """
        percentage = self.calculate_percentage(current_val=current_val)
        self.actor.set_message(text=percentage)

    def set_center(self, position):
        self.actor.SetPosition(position[0]-self.length/2-40, position[1]-10)


class DiskSlider2D(UI):
    def __init__(self, outer_inner_radius=40, outer_outer_radius=44, outer_position=(450, 100), inner_outer_radius=10,
                 inner_inner_radius=0):
        """

        Parameters
        ----------
        outer_inner_radius
        outer_outer_radius
        outer_position
        inner_outer_radius
        inner_inner_radius
        """
        super(DiskSlider2D, self).__init__()
        self.outer_disk_radius = outer_inner_radius + (outer_outer_radius - outer_inner_radius) / 2
        self.outer_disk_center = outer_position
        self.slider_outer_disk = DiskSlider2DBase(inner_radius=outer_inner_radius, outer_radius=outer_outer_radius,
                                                  disk_position=outer_position)
        self.slider_inner_disk = DiskSlider2DDisk(inner_radius=inner_inner_radius, outer_radius=inner_outer_radius,
                                                  disk_position=(outer_position[0] + self.outer_disk_radius,
                                                                 outer_position[1]))
        self.slider_text = DiskSlider2DText(position=outer_position, current_val=0)

        self.ui_list.append(self.slider_outer_disk)
        self.ui_list.append(self.slider_inner_disk)
        self.ui_list.append(self.slider_text)

    def add_to_renderer(self, ren):
        ren.add(self.slider_outer_disk.actor)
        ren.add(self.slider_inner_disk.actor)
        ren.add(self.slider_text.actor)

    def add_callback(self, event_type, callback, component):
        """ Adds events to an actor

        Parameters
        ----------
        event_type: event code
        callback: callback function
        component: component
        """
        super(DiskSlider2D, self).add_callback(component.actor, event_type, callback)

    def get_poi(self, coordinates):
        """

        Parameters
        ----------
        coordinates

        Returns
        -------
        x, y

        """
        radius = self.outer_disk_radius
        center = self.outer_disk_center
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
        """

        Parameters
        ----------
        coordinates

        Returns
        -------
        angle

        """
        center = self.outer_disk_center

        perpendicular = -center[1] + coordinates[1]
        base = -center[0] + coordinates[0]

        angle = math.degrees(math.atan2(float(perpendicular), float(base)))
        if angle < 0:
            angle += 360

        return angle

    def set_center(self, position):
        self.slider_outer_disk.set_center(position)
        self.slider_text.set_center(position)
        self.outer_disk_center = position
        self.slider_inner_disk.set_position((position[0]+self.outer_disk_radius, position[1]))


class DiskSlider2DBase(UI):
    def __init__(self, inner_radius, outer_radius, disk_position):
        """

        Parameters
        ----------
        inner_radius
        outer_radius
        disk_position
        """
        super(DiskSlider2DBase, self).__init__()
        self.actor = self.build_actor(inner_radius=inner_radius, outer_radius=outer_radius, disk_position=disk_position)

        self.ui_list.append(self)

    def build_actor(self, inner_radius, outer_radius, disk_position):
        """

        Parameters
        ----------
        inner_radius
        outer_radius
        disk_position

        Returns
        -------

        """
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
        actor = vtk.vtkActor2D()
        actor.SetMapper(mapper)

        actor.GetProperty().SetColor(1, 0, 0)

        actor.SetPosition(disk_position[0], disk_position[1])

        return actor

    def set_center(self, position):
        self.actor.SetPosition(position)


class DiskSlider2DDisk(UI):
    def __init__(self, inner_radius, outer_radius, disk_position):
        """

        Parameters
        ----------
        inner_radius
        outer_radius
        disk_position
        """
        super(DiskSlider2DDisk, self).__init__()
        self.actor = self.build_actor(inner_radius=inner_radius, outer_radius=outer_radius, disk_position=disk_position)

        self.ui_list.append(self)

    def build_actor(self, inner_radius, outer_radius, disk_position):
        # create source
        """

        Parameters
        ----------
        inner_radius
        outer_radius
        disk_position

        Returns
        -------

        """
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
        actor = vtk.vtkActor2D()
        actor.SetMapper(mapper)

        actor.SetPosition(disk_position[0], disk_position[1])

        return actor

    def set_position(self, position):
        """ Sets the disk's position

        Parameters
        ----------
        position
        """
        self.actor.SetPosition(position)


class DiskSlider2DText(UI):

    def __init__(self, position, current_val):
        """

        Parameters
        ----------
        position
        current_val
        """
        super(DiskSlider2DText, self).__init__()

        self.actor = self.build_actor(current_val=current_val, position=position)

        self.ui_list.append(self)

    def calculate_percentage(self, current_val):
        """

        Parameters
        ----------
        current_val

        Returns
        -------

        """
        percentage = int((current_val/360)*100)
        if len(str(percentage)) == 1:
            percentage_string = "0" + str(percentage)
        else:
            percentage_string = str(percentage)
        return percentage_string + "%"

    def build_actor(self, current_val, position):
        """

        Parameters
        ----------
        current_val
        position

        Returns
        -------
        actor

        """
        actor = TextActor2D()

        actor.set_position(position=(position[0]-16, position[1]-8))
        percentage = self.calculate_percentage(current_val=current_val)
        actor.set_message(text=percentage)
        actor.font_size(size=16)

        return actor

    def set_percentage(self, current_val):
        """

        Parameters
        ----------
        current_val
        """
        percentage = self.calculate_percentage(current_val=current_val)
        self.actor.set_message(text=percentage)

    def set_center(self, position):
        self.actor.SetPosition(position[0]-16, position[1]-8)
