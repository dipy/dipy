import glob
import os

from dipy.data import read_viz_icons
from dipy.utils.optpkg import optional_package

# Allow import, but disable doctests if we don't have vtk.
from dipy.viz.gui import UI, TextActor2D
from dipy.viz.gui_2d import Panel2D, Text2D, TextBox2D, Button2D

vtk, have_vtk, setup_module = optional_package('vtk')

if have_vtk:
    vtkInteractorStyleUser = vtk.vtkInteractorStyleUser
    version = vtk.vtkVersion.GetVTKSourceVersion().split(' ')[-1]
    major_version = vtk.vtkVersion.GetVTKMajorVersion()
else:
    vtkInteractorStyleUser = object

numpy_support, have_ns, _ = optional_package('vtk.util.numpy_support')


class FileSaveMenu(UI):
    def __init__(self, size, position):
        """

        Parameters
        ----------
        size: (float, float)
        position: (float, float)
        """
        super(FileSaveMenu, self).__init__()
        self.center = position
        self.size = size

        self.panel = Panel2D(center=self.center, size=self.size, color=(1, 1, 1), opacity=0.7)  # type: Panel2D

        self.title = Text2D(message=os.getcwd())
        self.file_select = FileSelect2D(size=(self.size[0] * 0.9, self.size[1] * 0.6), font_size=12, position=(0, 0),
                                        parent=self)
        self.text_box = TextBox2D(width=20, height=1, text="FileName")
        self.save_button = Button2D({"save": read_viz_icons(fname="floppy-disk.png")})
        self.cancel_button = Button2D({"cancel": read_viz_icons(fname="cross.png")})

        self.build()

    def add_to_renderer(self, ren):
        """ Adds the actor to renderer.

        Parameters
        ----------
        ren : renderer
        """
        self.panel.add_to_renderer(ren)

    def add_callback(self, event_type, callback, component):
        """ Adds events to a component.

        Parameters
        ----------
        event_type : string
            event code
        callback : function
            callback function
        component : UI
            component
        """
        component.add_callback(event_type, callback)

    def add_callback_to_sub_component(self, event_type, callback, component, sub_component):
        """ Adds events to a sub-component of a component.

        Parameters
        ----------
        event_type : string
            event code
        callback : function
            callback function
        component : UI
            component
        sub_component: vtkActor
            sub component of a UI
        """
        component.add_callback(event_type, callback, sub_component)

    def build(self):
        """ Builds elements for the file menu.
        """
        self.panel.add_element(element=self.title, relative_position=(0.05, 0.9))
        self.panel.add_element(element=self.file_select, relative_position=(0.5, 0.5))
        self.panel.add_element(element=self.text_box, relative_position=(0.05, 0.075))
        self.panel.add_element(element=self.save_button, relative_position=(0.75, 0.1))
        self.panel.add_element(element=self.cancel_button, relative_position=(0.9, 0.1))

    def handle_folder_change(self):
        """ Effect of folder change on overall FileDialog.
        This function can be called from any of the children.
        """
        self.title.set_message(os.getcwd())

    def handle_file_click(self, file_name):
        """ Effect of file change on overall FileDialog.
        This function can be called from any of the children.
        """
        self.text_box.set_message(file_name)


class FileSelect2D(UI):
    """ A menu to select files in the current folder.
    Can go to new folder, previous folder and select a file and keep in a variable.
    """

    def __init__(self, size, font_size, position, parent):
        """ Conatins some initialisation parameters.
        - n_text_actors: The number of text actors. Calculated dynamically.
        - selected_file: Current selected file.
        - text_actor_list: List of text actors.
        - window: Used for scrolling.

        Parameters
        ----------
        size: (float, float)
        font_size: int
        position: (float, float
        parent: UI
        """
        super(FileSelect2D, self).__init__()
        self.size = size
        self.font_size = font_size
        self.parent_UI = parent

        self.n_text_actors = 0  # Initialisation Value
        self.text_actor_list = []
        self.selected_file = ""
        self.window = 0

        self.menu = self.build_actors(position)

        self.fill_text_actors()

    def add_to_renderer(self, ren):
        """ Add props to renderer

        Parameters
        ----------
        ren : renderer
        """
        self.menu.add_to_renderer(ren)

    def build_actors(self, position):
        """ Builds the number of text actors that will fit in the given size.
        Allots them positions in the panel, which is only there to allot positions,
        otherwise the panel itself is invisible.

        Parameters
        ----------
        position: (float, float)
        """
        line_spacing = 1.4  # Later, we'll have the user set it

        self.n_text_actors = int(self.size[1]/(self.font_size*line_spacing))  # The number of text actors.

        # This panel is just to facilitate the addition of actors at the right positions
        panel = Panel2D(center=position, size=self.size, color=(1, 1, 1))
        self.ui_list.append(panel.panel)

        # Initialisation of empty text actors
        for i in range(self.n_text_actors):

            text = FileSelectText2D(position=(0, 0), font_size=self.font_size, file_select=self)
            text.parent_UI = self.parent_UI
            self.ui_list.append(text)
            self.text_actor_list.append(text)

            panel.add_element(text, (0.1, float(self.n_text_actors-i - 1)/float(self.n_text_actors)))

        up_button = Button2D({"up": read_viz_icons(fname="arrow-up.png")})
        up_button.add_callback("LeftButtonPressEvent", self.up_button_callback)
        panel.add_element(up_button, (0.95, 0.9))

        down_button = Button2D({"down": read_viz_icons(fname="arrow-down.png")})
        down_button.add_callback("LeftButtonPressEvent", self.down_button_callback)
        panel.add_element(down_button, (0.95, 0.1))

        return panel

    def up_button_callback(self, obj, evt):
        """ Pressing up button scrolls up in the menu.
        """
        all_file_names = self.get_all_file_names()
        if self.n_text_actors + self.window <= len(all_file_names):
            if self.window > 0:
                self.window -= 1
                self.fill_text_actors()

        return False

    def down_button_callback(self, obj, evt):
        """ Pressing down button scrolls down in the menu.
        """
        all_file_names = self.get_all_file_names()
        if self.n_text_actors + self.window < len(all_file_names):
            self.window += 1
            self.fill_text_actors()

        return False

    def fill_text_actors(self):
        """ Fills file/folder names to text actors.
        The list is truncated if the number of file/folder names is greater than
        the available number of text actors.
        """
        # Flush all the text actors
        for text_actor in self.text_actor_list:
            text_actor.actor.set_message("")
            text_actor.actor.SetVisibility(False)

        all_file_names = self.get_all_file_names()

        clipped_file_names = all_file_names[self.window:self.n_text_actors+self.window]

        # Allot file names as in the above list
        i = 0
        for file_name in clipped_file_names:
            self.text_actor_list[i].actor.SetVisibility(True)
            self.text_actor_list[i].set_attributes(file_name[0], file_name[1])
            i += 1

    def get_all_file_names(self):
        """Gets file+directory names.
        """
        all_file_names = []

        directory_names = self.get_directory_names()
        for directory_name in directory_names:
            all_file_names.append((directory_name, "directory"))

        file_names = self.get_file_names("png")
        for file_name in file_names:
            all_file_names.append((file_name, "file"))

        return all_file_names

    @staticmethod
    def get_directory_names():
        """ Re-allots file names to the text actors.
        Uses FileSelectText2D and FolderSelectText2D for selecting files and folders.
        """
        # A list of directory names in the current directory
        directory_names = ["../"]
        directory_names += glob.glob("*/")

        return directory_names

    @staticmethod
    def get_file_names(extension):
        """ Re-allots file names to the text actors.
        Uses FileSelectText2D and FolderSelectText2D for selecting files and folders.

        Parameters
        ----------
        extension: string
            Examples: png, jpg, etc.
        """
        # A list of file names with extension in the current directory
        file_names = glob.glob("*." + extension)

        return file_names

    def select_file(self, file_name):
        """ Changes the selected file name for the FileSelect menu.

        Parameters
        ----------
        file_name: string
        """
        self.selected_file = file_name

    def set_center(self, position):
        """ Sets the elements center.

        Parameters
        ----------
        position: (float, float)
        """
        self.menu.set_center(position=position)


class FileSelectText2D(UI):
    """ The text to select folder in a file select menu.
    Provides a callback to change the directory.
    """

    def __init__(self, font_size, position, file_select):
        """

        Parameters
        ----------
        font_size: int
        position: (float, float)
        file_select: FileSelect2D
        """
        super(FileSelectText2D, self).__init__()

        self.file_name = ""
        self.file_type = ""

        self.file_select = file_select
        self.actor = self.build_actor(position=position, font_size=font_size)
        self.add_callback("LeftButtonPressEvent", self.click_callback)

    def build_actor(self, position, text="Text", color=(1, 1, 1), font_family='Arial', justification='left',
                            bold=False, italic=False, shadow=False, font_size='14'):
        """ Builds a text actor.

        Parameters
        ----------
        text : string
            The initial text while building the actor.
        position : (float, float)
        color : (float, float, float)
            Values must be between 0-1.
        font_family : string
            Currently only supports Ariel.
        justification : string
            left, right or center.
        bold : bool
        italic : bool
        shadow : bool
        font_size: int

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
        if vtk.vtkVersion.GetVTKSourceVersion().split(' ')[-1] <= "6.2.0":
            pass
        else:
            text_actor.GetTextProperty().SetBackgroundColor(1, 1, 1)
            text_actor.GetTextProperty().SetBackgroundOpacity(1.0)
        text_actor.GetTextProperty().SetColor(0, 0, 0)
        text_actor.GetTextProperty().SetLineSpacing(1)

        return text_actor

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
        super(FileSelectText2D, self).add_callback(self.actor, event_type, callback)

    def set_attributes(self, file_name, file_type):
        self.file_name = file_name
        self.file_type = file_type
        self.actor.set_message(file_name)
        if vtk.vtkVersion.GetVTKSourceVersion().split(' ')[-1] <= "6.2.0":
            pass
        else:
            if file_type == "file":
                self.actor.GetTextProperty().SetBackgroundColor(0, 0, 0)
                self.actor.GetTextProperty().SetColor(1, 1, 1)
            else:
                self.actor.GetTextProperty().SetBackgroundColor(1, 1, 1)
                self.actor.GetTextProperty().SetColor(0, 0, 0)

    def click_callback(self, obj, evt):
        """ A callback to handle click for this UI element.
        """
        if self.file_type == "directory":
            os.chdir(self.actor.get_message())
            self.file_select.window = 0
            self.file_select.fill_text_actors()
            self.file_select.select_file(file_name="")
            if isinstance(self.parent_UI, FileSaveMenu):
                self.parent_UI.handle_folder_change()
                self.parent_UI.handle_file_click(file_name="FileName")
        else:
            self.file_select.select_file(file_name=self.file_name)
            if isinstance(self.parent_UI, FileSaveMenu):
                self.parent_UI.handle_file_click(file_name=self.file_name)

        return False

    def set_center(self, position):
        """ Sets the text center to position.

        Parameters
        ----------
        position : (float, float)
        """
        self.actor.SetPosition(position)
