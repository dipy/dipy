import numpy as np

# Conditional import machinery for vtk.
from dipy.data import fetch_viz_icons, read_viz_icons
from dipy.utils.optpkg import optional_package

from dipy.viz import actor, window
from dipy.utils.six import string_types
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

    def on_left_button_pressed(self, obj, evt):
        self.trackball_interactor_style.OnLeftButtonDown()

    def on_left_button_released(self, obj, evt):
        self.trackball_interactor_style.OnLeftButtonUp()

    def on_right_button_pressed(self, obj, evt):

        clickPos = self.GetInteractor().GetEventPosition()

        # Use a picker to see which actor is under the mouse
        picker = vtk.vtkPropPicker()
        picker.Pick(clickPos[0], clickPos[1], 0, self.renderer)
        actor = picker.GetProp3D()

        # set_trace()
        # print(actor)
        if actor is not None:
            actor.InvokeEvent(evt)
        else:
            actor = picker.GetViewProp()
            # set_trace()
            if actor is not None:
                actor.InvokeEvent(evt)
            else:
                "No Actor Selected"

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

        self.AddObserver("LeftButtonPressEvent", self.on_left_button_pressed)
        self.AddObserver("LeftButtonReleaseEvent", self.on_left_button_released)
        self.AddObserver("RightButtonPressEvent", self.on_right_button_pressed)
        self.AddObserver("RightButtonReleaseEvent", self.on_right_button_released)
        self.AddObserver("MiddleButtonPressEvent", self.on_middle_button_pressed)
        self.AddObserver("MiddleButtonReleaseEvent", self.on_middle_button_released)
        self.AddObserver("MouseMoveEvent", self.on_mouse_moved)

        # These observers need to be added directly to the interactor because
        # `vtkInteractorStyleUser` does not forward these events.
        interactor.AddObserver("MouseWheelForwardEvent", self.on_mouse_wheel_forward)
        interactor.AddObserver("MouseWheelBackwardEvent", self.on_mouse_wheel_backward)


def figure(pic):
    """ Return a figure as a 2D actor

    Parameters
    ----------
    pic : filename

    Returns
    -------
    image_actor : vtkTexturedActor2D
    """

    png = vtk.vtkPNGReader()
    png.SetFileName(pic)
    png.Update()

    # Convert the image to a polydata
    imageDataGeometryFilter = vtk.vtkImageDataGeometryFilter()
    imageDataGeometryFilter.SetInputConnection(png.GetOutputPort())
    imageDataGeometryFilter.Update()

    mapper = vtk.vtkPolyDataMapper2D()
    mapper.SetInputConnection(imageDataGeometryFilter.GetOutputPort())

    image_actor = vtk.vtkTexturedActor2D()
    image_actor.SetMapper(mapper)

    return image_actor


def create_button(file_name, callback, position=(0, 0), center=None):
    button = figure(file_name)
    button.AddObserver("RightButtonPressEvent", callback)
    if center is not None:
         button.SetCenter(*center)
    button.SetPosition(position[0], position[1])

    return button


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


def button_callback(*args, **kwargs):
    pos = np.array(cube_actor_1.GetPosition())
    print(pos)
    pos[0] += 20
    cube_actor_1.SetPosition(tuple(pos))


cube_actor_1 = cube((1, 0, 0), (50, 50, 50), center=(0, 0, 0))
cube_actor_2 = cube((0, 1, 0), (10, 10, 10), center=(100, 0, 0))

fetch_viz_icons()
filename = read_viz_icons(fname='stop2.png')

button_actor = create_button(file_name=filename, callback=button_callback)

cube_actor_1.AddObserver("RightButtonPressEvent", button_callback)
cube_actor_2.AddObserver("RightButtonPressEvent", button_callback)

renderer = window.ren()
iren_style = CustomInteractorStyle(renderer=renderer)
renderer.add(button_actor)
renderer.add(cube_actor_1)
renderer.add(cube_actor_2)

# set_trace()

showm = window.ShowManager(renderer, interactor_style=iren_style)

showm.initialize()
showm.render()
showm.start()
