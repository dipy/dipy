from dipy.utils.optpkg import optional_package

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
    """ An umbrella class for all UI elements.

    While adding UI elements to the renderer, we need to go over all the
     sub-elements that come with it and add those to the renderer too.
    There are several features that are common to all the UI elements:
    - ui_param : This is an attribute that can be passed to the UI object
                by the interactor. Thanks to Python's dynamic type-setting
                this parameter can be anything.
    - ui_list : This is used when there are more than one UI elements inside
               a UI element. Inside the renderer, they're all iterated and added.

    """
    def __init__(self):
        self.ui_param = None
        self.ui_list = list()

        self.actor = None

    def add_to_renderer(self, ren):
        """ Allows UI objects to add their own props to the renderer.

        Parameters
        ----------
        ren : renderer
        """
        pass

    def set_ui_param(self, ui_param):
        """ Adds a UI Parameter. Can be anything.

        Parameters
        ----------
        ui_param :
        """
        self.ui_param = ui_param

    def add_callback(self, prop, event_type, callback, priority=0):
        """ Adds a callback to a specific event for this UI component.

        Parameters
        ----------
        prop : vtkProp
        event_type : event code
        callback : function
        priority : int
        """
        cmd_id = [None]  # Placeholder needed in the _callback closure.

        def _callback(obj, event_type):
            abort_flag = callback(self, event_type)
            if abort_flag is not None:
                cmd = obj.GetCommand(cmd_id[0])
                cmd.SetAbortFlag(abort_flag)

        cmd_id[0] = prop.AddObserver(event_type, _callback, priority)

    def set_center(self, position):
        """ Sets the center of the UI component

        Parameters
        ----------
        position : (float, float)
        """
        pass


class TextActor2D(vtk.vtkTextActor):
    """ Inherits from the default vtkTextActor and helps setting the text.

    Contains member functions for text formatting.
    """
    def message(self, text):
        """ Set message after initialization.

        Parameters
        ----------
        text : string

        """
        self.SetInput(text)

    def set_message(self, text):
        """ Modify text message.

        Parameters
        ----------
        text : string
        """
        self.SetInput(text)

    def get_message(self):
        """ Gets message from the text.

        Returns
        -------
        message : string

        """
        return self.GetInput()

    def font_size(self, size):
        """ Sets font size.

        Parameters
        ----------
        size : int
        """
        self.GetTextProperty().SetFontSize(size)

    def font_family(self, family='Arial'):
        """ Sets font family.
        Currently defaults to Ariel.
        # TODO: Add other font families.

        Parameters
        ----------
        family : string
        """
        self.GetTextProperty().SetFontFamilyToArial()

    def justification(self, justification):
        """ Justifies text.

        Parameters
        ----------
        justification : string
            Possible values : left, right, center
        """
        tprop = self.GetTextProperty()
        if justification == 'left':
            tprop.SetJustificationToLeft()
        if justification == 'center':
            tprop.SetJustificationToCentered()
        if justification == 'right':
            tprop.SetJustificationToRight()

    def font_style(self, bold=False, italic=False, shadow=False):
        """ Style font.

        Parameters
        ----------
        bold : bool
        italic : bool
        shadow : bool
        """
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
        """ Set text color.

        Parameters
        ----------
        color : (float, float, float)
            Values must be between 0-1
        """
        self.GetTextProperty().SetColor(*color)

    def set_position(self, position):
        """ Set text actor position.

        Parameters
        ----------
        position : (float, float, float)
        """
        self.SetDisplayPosition(*position)

    def get_position(self):
        """ Gets text actor position.

        Returns
        -------
        position : (float, float, float)
        """
        return self.GetDisplayPosition()
