# Conditional import machinery for vtk.
from dipy.utils.optpkg import optional_package

from ipdb import set_trace

import math

# Allow import, but disable doctests if we don't have vtk.
vtk, have_vtk, setup_module = optional_package('vtk')

if have_vtk:
    vtkInteractorStyleUser = vtk.vtkInteractorStyleUser
    version = vtk.vtkVersion.GetVTKSourceVersion().split(' ')[-1]
    major_version = vtk.vtkVersion.GetVTKMajorVersion()
else:
    vtkInteractorStyleUser = object

numpy_support, have_ns, _ = optional_package('vtk.util.numpy_support')


class UI(object):

    def __init__(self):
        self.ui_param = None

    def set_ui_param(self, ui_param):
        self.ui_param = ui_param


class Button(UI):
    """ Currently implements a 2D overlay button and is of type vtkTexturedActor2D. 

    """

    def __init__(self, icon_fnames):
        self.icons = self.build_icons(icon_fnames)
        self.icon_names = list(self.icons.keys())
        self.current_icon_id = 0
        self.current_icon_name = self.icon_names[self.current_icon_id]
        self.actor = self.build_actor(self.icons[self.current_icon_name])

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

            # Convert the image to a polydata
            imageDataGeometryFilter = vtk.vtkImageDataGeometryFilter()
            imageDataGeometryFilter.SetInputConnection(png.GetOutputPort())
            imageDataGeometryFilter.Update()

            icons[icon_name] = imageDataGeometryFilter

        return icons

    def build_actor(self, icon, position=(0, 0), center=None):
        """ Return an image as a 2D actor with a specific position

        Parameters
        ----------
        icon : imageDataGeometryFilter
        position : a two tuple
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
        button.SetPosition(position[0], position[1])

        return button

    def add_callback(self, event_type, callback):
        """ Adds events to button actor

        Parameters
        ----------
        event_type: event code
        callback: callback function
        """
        self.actor.AddObserver(event_type, callback)

    def set_icon(self, icon):
        """ Modifies the icon used by the vtkTexturedActor2D

        Parameters
        ----------
        icon : imageDataGeometryFilter
        """
        self.actor.GetMapper().SetInputConnection(icon.GetOutputPort())

    def next_icon_name(self):
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


class TextActor(vtk.vtkTextActor):

    def message(self, text):
        self.SetInput(text)

    def set_message(self, text):
        self.SetInput(text)

    def get_message(self):
        return self.GetInput()

    def font_size(self, size):
        self.GetTextProperty().SetFontSize(size)

    def font_family(self, family='Arial'):
        self.GetTextProperty().SetFontFamilyToArial()

    def justification(self, justification):
        tprop = self.GetTextProperty()
        if justification == 'left':
            tprop.SetJustificationToLeft()
        if justification == 'center':
            tprop.SetJustificationToCentered()
        if justification == 'right':
            tprop.SetJustificationToRight()

    def font_style(self, bold=False, italic=False, shadow=False):
        tprop = self.GetTextProperty()
        if bold:
            tprop.BoldOn()
        else:
            tprop.BoldOff()
        if italic:
            tprop.ItalicOn()
        else:
            tprop.ItalicOff()
        if shadow:
            tprop.ShadowOn()
        else:
            tprop.ShadowOff()

    def color(self, color):
        self.GetTextProperty().SetColor(*color)

    def set_position(self, position):
        self.SetDisplayPosition(*position)

    def get_position(self, position):
        return self.GetDisplayPosition()


class TextBox(UI):

    def __init__(self, width, height, text="Enter Text"):
        self.text = text
        self.actor = self.build_actor(self.text)
        self.width = width
        self.height = height
        self.window_left = 0
        self.window_right = 0
        self.caret_pos = 0
        self.init = True 

    def build_actor(self, text, position=(100, 10), color=(1, 1, 1),
                 font_size=18, font_family='Arial', justification='left',
                 bold=False, italic=False, shadow=False):

        text_actor = TextActor()
        text_actor.set_position(position)
        text_actor.message(text)
        text_actor.font_size(font_size)
        text_actor.font_family(font_family)
        text_actor.justification(justification)
        text_actor.font_style(bold, italic, shadow)
        text_actor.color(color)

        return text_actor

    def add_callback(self, event_type, callback):
        """ Adds events to the text actor

        Parameters
        ----------
        event_type: event code
        callback: callback function
        """
        self.actor.AddObserver(event_type, callback)

    def width_set_text(self, text):
        multi_line_text = ""
        for i in range(len(text)):
            multi_line_text += text[i]
            if (i+1)%self.width == 0:
                multi_line_text += "\n"
        return multi_line_text.rstrip("\n")

    def handle_character(self, character):
        if character.lower() == "return":
            self.render_text(False)
        else:
            if character.lower() == "backspace" :
                self.remove_character()
            elif character.lower() == "left":
                self.move_left()
            elif character.lower() == "right":
                self.move_right()
            else:
                self.add_character(character)
            self.render_text()

    def move_caret_right(self):
        self.caret_pos += 1
        if self.caret_pos > len(self.text):
            self.caret_pos = len(self.text)

    def move_caret_left(self):
        self.caret_pos -= 1
        if self.caret_pos < 0:
            self.caret_pos = 0
    
    def right_move_right(self):
        if self.window_right <= len(self.text):
            self.window_right += 1

    def right_move_left(self):
        if self.window_right > 0:
            self.window_right -= 1

    def left_move_right(self):
        if self.window_left <= len(self.text):
            self.window_left += 1

    def left_move_left(self):
        if self.window_left > 0:
            self.window_left -= 1

    def add_character(self, character):
        if len(character) > 1 and character.lower() != "space":
            return
        if character.lower() == "space":
            character = " "
        self.text = self.text[:self.caret_pos] + character + self.text[self.caret_pos:]
        self.move_caret_right()
        if self.window_right-self.window_left == self.height*self.width-1:
            self.left_move_right()
        self.right_move_right()

    def remove_character(self):
        if self.caret_pos == 0:
            return
        self.text = self.text[:self.caret_pos-1] + self.text[self.caret_pos:]
        self.move_caret_left()
        if len(self.text) < self.height*self.width-1:
            self.right_move_left()
        if self.window_right-self.window_left == self.height*self.width-1:
            if self.window_left > 0:
                self.left_move_left()
                self.right_move_left()

    def move_left(self):
        self.move_caret_left()
        if self.caret_pos == self.window_left-1:
            if self.window_right-self.window_left == self.height*self.width-1:
                self.left_move_left()
                self.right_move_left()

    def move_right(self):
        self.move_caret_right()
        if self.caret_pos == self.window_right+1:
            if self.window_right-self.window_left == self.height*self.width-1:
                self.left_move_right()
                self.right_move_right()

    def showable_text(self, show_caret):
        if show_caret:
            ret_text = self.text[:self.caret_pos] + "|" + self.text[self.caret_pos:]
        else:
            ret_text = self.text
        ret_text = ret_text[self.window_left:self.window_right+1]
        return ret_text

    def render_text(self, show_caret=True):
        text = self.showable_text(show_caret)
        self.actor.set_message(self.width_set_text(text))

    def edit_mode(self):
        if self.init:
            self.text = ""
            self.init = False
            self.caret_pos = 0
        self.render_text()