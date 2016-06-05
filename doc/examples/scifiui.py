import numpy as np

from dipy.data import read_viz_icons

# Conditional import machinery for vtk.
from dipy.utils.optpkg import optional_package

from dipy.viz import actor, window, gui
from ipdb import set_trace

# Allow import, but disable doctests if we don't have vtk.
vtk, have_vtk, setup_module = optional_package('vtk')

if have_vtk:
    vtkInteractorStyleUser = vtk.vtkInteractorStyleUser
    version = vtk.vtkVersion.GetVTKSourceVersion().split(' ')[-1]
    major_version = vtk.vtkVersion.GetVTKMajorVersion()
else:
    vtkInteractorStyleUser = object

numpy_support, have_ns, _ = optional_package('vtk.util.numpy_support')

class CustomInteractorStyle(vtkInteractorStyleUser):
    """ Interactive manipulation of the camera that can also manipulates
    objects in the scene independent of each other.

    This interactor style allows the user to interactively manipulate (pan,
    rotate and zoom) the camera. It also allows the user to interact (click,
    scroll, etc.) with objects in the scene independent of each other.

    Several events are overloaded from its superclass `vtkInteractorStyle`,
    hence the mouse bindings are different.

    In summary the mouse events for this interaction style are as follows:
    - Left mouse button: rotates the camera
    - Right mouse button: dollys the camera
    - Mouse wheel: dollys the camera
    - Middle mouse button: pans the camera

    """
    def __init__(self, renderer):
        self.renderer = renderer
        self.trackball_interactor_style = vtk.vtkInteractorStyleTrackballCamera()
        # Use a picker to see which actor is under the mouse
        self.picker = vtk.vtkPropPicker()

    def on_left_button_pressed(self, obj, evt):
        click_pos = self.GetInteractor().GetEventPosition()

        self.picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
        # actor = picker.GetActor2D()
        actor_2d = self.picker.GetViewProp()
        if actor_2d is not None:
            actor_2d.InvokeEvent(evt)
        else:
            actor_3d = self.picker.GetProp3D()
            if actor_3d is not None:
                actor_3d.InvokeEvent(evt)
            else:
                pass

        self.trackball_interactor_style.OnLeftButtonDown()

    def on_left_button_released(self, obj, evt):
        self.trackball_interactor_style.OnLeftButtonUp()

    def on_right_button_pressed(self, obj, evt):

        click_pos = self.GetInteractor().GetEventPosition()

        self.picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
        # actor = picker.GetActor2D()
        actor_2d = self.picker.GetViewProp()
        if actor_2d is not None:
            actor_2d.InvokeEvent(evt)
        else:
            actor_3d = self.picker.GetProp3D()
            if actor_3d is not None:
                actor_3d.InvokeEvent(evt)
            else:
                pass

        self.trackball_interactor_style.OnRightButtonDown()

    def on_right_button_released(self, obj, evt):
        self.trackball_interactor_style.OnRightButtonUp()

    def on_middle_button_pressed(self, obj, evt):
        self.trackball_interactor_style.OnMiddleButtonDown()

    def on_middle_button_released(self, obj, evt):
        self.trackball_interactor_style.OnMiddleButtonUp()

    def on_mouse_moved(self, obj, evt):
        self.trackball_interactor_style.OnMouseMove()

    def on_mouse_wheel_forward(self, obj, evt):
        self.trackball_interactor_style.OnMouseWheelForward()

    def on_mouse_wheel_backward(self, obj, evt):
        self.trackball_interactor_style.OnMouseWheelBackward()

    def on_key_press(self, obj, evt):
        # actor = picker.GetActor2D()
        actor_2d = self.picker.GetViewProp()
        if actor_2d is not None:
            ui_list = self.renderer.ui_list
            for ui_item in ui_list:
                if ui_item.actor == actor_2d:
                    ui_item.set_ui_param(obj.GetKeySym())
                    break
            actor_2d.InvokeEvent(evt)
        else:
            actor_3d = self.picker.GetProp3D()
            if actor_3d is not None:
                actor_3d.InvokeEvent(evt)
            else:
                pass

    def SetInteractor(self, interactor):
        # Internally these `InteractorStyle` objects need an handle to a
        # `vtkWindowInteractor` object and this is done via `SetInteractor`.
        # However, this has a the side effect of adding directly their
        # observers to `interactor`!
        self.trackball_interactor_style.SetInteractor(interactor)

        # Remove all observers previously set. Those were *most likely* set by
        # `vtkInteractorStyleTrackballCamera`.
        #
        # Note: Be sure that no observer has been manually added to the
        #       `interactor` before setting the InteractorStyle.
        interactor.RemoveAllObservers()

        # This class is a `vtkClass` (instead of `object`), so `super()` cannot be used.
        # Also the method `SetInteractor` is not overridden by `vtkInteractorStyleUser`
        # so we have to call directly the one from `vtkInteractorStyle`.
        # In addition to setting the interactor, the following line
        # adds the necessary hooks to listen to this instance's observers.
        vtk.vtkInteractorStyle.SetInteractor(self, interactor)

        interactor.RemoveObservers(vtk.vtkCommand.CharEvent)

        self.AddObserver("LeftButtonPressEvent", self.on_left_button_pressed)
        self.AddObserver("LeftButtonReleaseEvent", self.on_left_button_released)
        self.AddObserver("RightButtonPressEvent", self.on_right_button_pressed)
        self.AddObserver("RightButtonReleaseEvent", self.on_right_button_released)
        self.AddObserver("MiddleButtonPressEvent", self.on_middle_button_pressed)
        self.AddObserver("MiddleButtonReleaseEvent", self.on_middle_button_released)
        self.AddObserver("MouseMoveEvent", self.on_mouse_moved)
        self.AddObserver("KeyPressEvent", self.on_key_press)

        # These observers need to be added directly to the interactor because
        # `vtkInteractorStyleUser` does not forward these events.
        interactor.AddObserver("MouseWheelForwardEvent", self.on_mouse_wheel_forward)
        interactor.AddObserver("MouseWheelBackwardEvent", self.on_mouse_wheel_backward)

def cube(color=None, size=(0.2, 0.2, 0.2), center=None):
    cube = vtk.vtkCubeSource()
    cube.SetXLength(size[0])
    cube.SetYLength(size[1])
    cube.SetZLength(size[2])
    if center is not None:
        cube.SetCenter(*center)
    cubeMapper = vtk.vtkPolyDataMapper()
    cubeMapper.SetInputConnection(cube.GetOutputPort())
    cubeActor = vtk.vtkActor()
    cubeActor.SetMapper(cubeMapper)
    if color is not None:
        cubeActor.GetProperty().SetColor(color)
    return cubeActor


cube_actor_1 = cube((1, 0, 0), (50, 50, 50), center=(0, 0, 0))
cube_actor_2 = cube((0, 1, 0), (10, 10, 10), center=(100, 0, 0))

icon_files = {}
icon_files['stop'] = read_viz_icons(fname='stop2.png')
icon_files['play'] = read_viz_icons(fname='play3.png')
icon_files['plus'] = read_viz_icons(fname='plus.png')
icon_files['cross'] = read_viz_icons(fname='cross.png')

button = gui.Button(icon_fnames=icon_files)

def move_button_callback(*args, **kwargs):
    pos_1 = np.array(cube_actor_1.GetPosition())
    pos_1[0] += 2
    cube_actor_1.SetPosition(tuple(pos_1))
    pos_2 = np.array(cube_actor_2.GetPosition())
    pos_2[1] += 2
    cube_actor_2.SetPosition(tuple(pos_2))

def modify_button_callback(*args, **kwargs):
    button.next_icon()

button.add_callback("RightButtonPressEvent", move_button_callback)
button.add_callback("LeftButtonPressEvent", modify_button_callback)


text = gui.TextBox(text="Some Text")

def key_press_callback(*args, **kwargs):
    key = text.ui_param
    if key == "Backspace":
        text.remove_character()
    else:
        text.add_character(key)

text.add_callback("KeyPressEvent", key_press_callback)

renderer = window.ren()
iren_style = CustomInteractorStyle(renderer=renderer)
renderer.add(button)
renderer.add(cube_actor_1)
renderer.add(cube_actor_2)
renderer.add(text)

# set_trace()

showm = window.ShowManager(renderer, interactor_style=iren_style)

showm.initialize()
showm.render()
showm.start()
