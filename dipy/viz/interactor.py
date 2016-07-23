
# Conditional import machinery for vtk
from dipy.utils.optpkg import optional_package

# Allow import, but disable doctests if we don't have vtk
vtk, have_vtk, setup_module = optional_package('vtk')

if have_vtk:
    vtkInteractorStyleUser = vtk.vtkInteractorStyleUser
    # version = vtk.vtkVersion.GetVTKSourceVersion().split(' ')[-1]
    # major_version = vtk.vtkVersion.GetVTKMajorVersion()
else:
    vtkInteractorStyleUser = object


class CustomInteractorStyle(vtkInteractorStyleUser):
    """ Manipulate the camera and interact with objects in the scene.

    This interactor style allows the user to interactively manipulate (pan,
    rotate and zoom) the camera. It also allows the user to interact (click,
    scroll, etc.) with objects in the scene.

    Several events handling methods from :class:`vtkInteractorStyleUser` have
    been overloaded to allow the propagation of the events to the objects the
    user is interacting with.

    In summary, while interacting with the scene, the mouse events are as
    follows:
    - Left mouse button: rotates the camera
    - Right mouse button: dollys the camera
    - Mouse wheel: dollys the camera
    - Middle mouse button: pans the camera
    """
    def __init__(self):
        # Default interactor is responsible for moving the camera.
        self.default_interactor = vtk.vtkInteractorStyleTrackballCamera()
        # The picker allows us to know which object/actor is under the mouse.
        self.picker = vtk.vtkPropPicker()
        self.chosen_element = None

        # Define some interaction states
        self.left_button_down = False
        self.right_button_down = False
        self.middle_button_down = False
        self.active_props = []

    def get_prop_at_event_position(self):
        """ Returns the prop that lays at the event position. """
        # TODO: return a list of items (i.e. each level of the assembly path).
        event_pos = self.GetInteractor().GetEventPosition()
        self.picker.Pick(event_pos[0], event_pos[1], 0,
                         self.GetCurrentRenderer())

        path = self.picker.GetPath()
        if path is None:
            return None

        node = path.GetLastNode()
        prop = node.GetViewProp()
        return prop

    def add_active_prop(self, prop):
        self.active_props.append(prop)

    def remove_active_prop(self, prop):
        self.active_props.remove(prop)

    def on_left_button_down(self, obj, evt):
        self.active_prop = None
        self.left_button_down = True

        prop = self.get_prop_at_event_position()
        if prop is not None:
            prop.InvokeEvent(evt)  # Propagate event to the prop.

        self.default_interactor.OnLeftButtonDown()

    def on_left_button_up(self, obj, evt):
        self.left_button_down = False
        self.default_interactor.OnLeftButtonUp()

    def on_right_button_down(self, obj, evt):
        self.right_button_down = True

        prop = self.get_prop_at_event_position()
        if prop is not None:
            prop.InvokeEvent(evt)  # Propagate event to the prop.

        self.default_interactor.OnRightButtonDown()

    def on_right_button_up(self, obj, evt):
        self.right_button_down = False
        self.default_interactor.OnRightButtonUp()

    def on_middle_button_down(self, obj, evt):
        self.middle_button_down = True

        prop = self.get_prop_at_event_position()
        if prop is not None:
            prop.InvokeEvent(evt)  # Propagate event to the prop.

        self.default_interactor.OnMiddleButtonDown()

    def on_middle_button_up(self, obj, evt):
        self.middle_button_down = False
        self.default_interactor.OnMiddleButtonUp()

    def on_mouse_moved(self, obj, evt):
        # Propagate event to all active props.
        for prop in self.active_props:
            prop.InvokeEvent(evt)

        self.default_interactor.OnMouseMove()

    def on_mouse_wheel_forward(self, obj, evt):
        # Propagate event to all active props.
        for prop in self.active_props:
            prop.InvokeEvent(evt)

        self.default_interactor.OnMouseWheelForward()

    def on_mouse_wheel_backward(self, obj, evt):
        # Propagate event to all active props.
        for prop in self.active_props:
            prop.InvokeEvent(evt)

        self.default_interactor.OnMouseWheelBackward()

    def on_key_press(self, obj, evt):
        # Propagate event to all active props.
        for prop in self.active_props:
            prop.InvokeEvent(evt)

    def SetInteractor(self, interactor):
        # Internally, `InteractorStyle` objects need a handle to a
        # `vtkWindowInteractor` object and this is done via `SetInteractor`.
        # However, this has the side effect of adding directly all their
        # observers to the `interactor`!
        self.default_interactor.SetInteractor(interactor)

        # Remove all observers previously set. Those were *most likely* set by
        # `vtkInteractorStyleTrackballCamera`, i.e. our `default_interactor`.
        #
        # Note: Be sure that no observer has been manually added to the
        #       `interactor` before setting the InteractorStyle.
        interactor.RemoveAllObservers()

        # This class is a `vtkClass` (instead of `object`), so `super()`
        # cannot be used. Also the method `SetInteractor` is not overridden in
        # `vtkInteractorStyleUser` so we have to call directly the one from
        # `vtkInteractorStyle`. In addition to setting the interactor, the
        # following line adds the necessary hooks to listen to this instance's
        # observers.
        vtk.vtkInteractorStyle.SetInteractor(self, interactor)

        # We do not want `default_interactor` to process `CharEvent`s.
        interactor.RemoveObservers(vtk.vtkCommand.CharEvent)

        self.AddObserver("LeftButtonPressEvent", self.on_left_button_down)
        self.AddObserver("LeftButtonReleaseEvent", self.on_left_button_up)
        self.AddObserver("RightButtonPressEvent", self.on_right_button_down)
        self.AddObserver("RightButtonReleaseEvent", self.on_right_button_up)
        self.AddObserver("MiddleButtonPressEvent", self.on_middle_button_down)
        self.AddObserver("MiddleButtonReleaseEvent", self.on_middle_button_up)
        self.AddObserver("MouseMoveEvent", self.on_mouse_moved)
        self.AddObserver("KeyPressEvent", self.on_key_press)

        # These observers need to be added directly to the interactor because
        # `vtkInteractorStyleUser` does not forward these events.
        interactor.AddObserver("MouseWheelForwardEvent",
                               self.on_mouse_wheel_forward)
        interactor.AddObserver("MouseWheelBackwardEvent",
                               self.on_mouse_wheel_backward)
