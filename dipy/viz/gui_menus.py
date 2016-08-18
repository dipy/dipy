import glob
import os

from dipy.utils.optpkg import optional_package

# Allow import, but disable doctests if we don't have vtk.
from dipy.viz.gui import UI, TextActor2D
from dipy.viz.gui_2d import Panel2D, Text2D, TextBox2D

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
        self.panel = Panel2D(center=self.center, size=self.size, color=(0, 0, 0.9), opacity=0.9)  # type: Panel2D

        self.title = Text2D(message=os.getcwd())
        self.file_select = FileSelect2D(size=(self.size[0] * 0.9, self.size[1] * 0.6), font_size=12, position=(0, 0),
                                        parent=self)
        self.text_box = TextBox2D(width=20, height=1, text="FileName")

        self.build()

    def add_to_renderer(self, ren):
        """ Adds the actor to renderer.

        Parameters
        ----------
        ren : renderer
        """
        self.panel.add_to_renderer(ren)

    def build(self):
        # Needs the following
        # 2 buttons for save and close

        self.panel.add_element(element=self.title, relative_position=(0.05, 0.9))
        self.panel.add_element(element=self.file_select, relative_position=(0.5, 0.5))
        self.panel.add_element(element=self.text_box, relative_position=(0.05, 0.05))

    def handle_folder_change(self):
        self.title.set_message(os.getcwd())

    def handle_file_click(self, file_name):
        self.text_box.set_message(file_name)


class FileSelect2D(UI):
    """ A menu to select files in the current folder.
    Can go to new folder, previous folder and select a file and keep in a variable.
    """

    def __init__(self, size, font_size, position, parent):
        super(FileSelect2D, self).__init__()
        self.size = size
        self.font_size = font_size
        self.parent_UI = parent

        self.n_text_actors = 0  # Initialisation Value
        self.text_actor_list = []
        self.selected_file = ""

        self.menu = self.build_actors(position)

        self.fill_text_actors()

    def add_to_renderer(self, ren):
        """ Add props to renderer

        Parameters
        ----------
        ren : renderer
        """
        # TODO: Fix this, use self.ui_list
        ren.add(self.menu.panel)
        for text_actor in self.text_actor_list:
            ren.add(text_actor.actor)

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
        panel = Panel2D(center=position, size=self.size, color=(1, 1, 1))  # TODO: Somehow add to self.ui_list
        self.ui_list.append(panel.panel)

        # Initialisation of empty text actors
        for i in range(self.n_text_actors):
            text = FolderSelectText2D(position=(0, 0), font_size=self.font_size, file_select=self)
            text.parent_UI = self.parent_UI
            self.ui_list.append(text)
            self.text_actor_list.append(text)
            panel.add_element(text.actor, (0.1, float(self.n_text_actors-i - 1)/float(self.n_text_actors)))

        return panel

    def fill_text_actors(self):
        # Flush all the text actors
        for text_actor in self.text_actor_list:
            text_actor.actor.set_message("")

        all_file_names = []

        directory_names = self.get_directory_names()
        for directory_name in directory_names:
            all_file_names.append((directory_name, "directory"))

        file_names = self.get_file_names("png")
        for file_name in file_names:
            all_file_names.append((file_name, "file"))

        clipped_file_names = all_file_names[:self.n_text_actors]

        # Allot file names as in the above list
        i = 0
        for file_name in clipped_file_names:
            self.text_actor_list[i].set_attributes(file_name[0], file_name[1])
            i += 1

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
        self.selected_file = file_name

    def set_center(self, position):
        self.menu.set_center(position=position)


class FolderSelectText2D(UI):
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
        super(FolderSelectText2D, self).__init__()

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
        super(FolderSelectText2D, self).add_callback(self.actor, event_type, callback)

    def set_attributes(self, file_name, file_type):
        self.file_name = file_name
        self.file_type = file_type
        self.actor.set_message(file_name)
        if file_type == "file":
            self.actor.GetTextProperty().SetBackgroundColor(0, 0, 0)
            self.actor.GetTextProperty().SetColor(1, 1, 1)
        else:
            self.actor.GetTextProperty().SetBackgroundColor(1, 1, 1)
            self.actor.GetTextProperty().SetColor(0, 0, 0)

    def click_callback(self, obj, evt):
        """ A callback to handle click for this UI element.

        Parameters
        ----------
        obj
        evt
        """
        if self.file_type == "directory":
            os.chdir(self.actor.get_message())
            self.file_select.fill_text_actors()
            self.file_select.select_file(file_name="")
            if isinstance(self.parent_UI, FileSaveMenu):
                self.parent_UI.handle_folder_change()
        else:
            self.file_select.select_file(file_name=self.file_name)
            if isinstance(self.parent_UI, FileSaveMenu):
                self.parent_UI.handle_file_click(file_name=self.file_name)

        return False
