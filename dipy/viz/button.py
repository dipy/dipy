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
        actor = picker.GetProp()
        # print(actor)
        if actor is not None:
            actor.InvokeEvent(evt)

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


def figure(pic, interpolation='nearest'):
    """ Return a figure as an image actor

    Parameters
    ----------
    pic : filename or numpy RGBA array

    interpolation : str
        Options are nearest, linear or cubic. Default is nearest.

    Returns
    -------
    image_actor : vtkImageActor
    """

    if isinstance(pic, string_types):
        png = vtk.vtkPNGReader()
        png.SetFileName(pic)
        png.Update()
        vtk_image_data = png.GetOutput()
    else:

        if pic.ndim == 3 and pic.shape[2] == 4:

            vtk_image_data = vtk.vtkImageData()
            if major_version <= 5:
                vtk_image_data.SetScalarTypeToUnsignedChar()

            if major_version <= 5:
                vtk_image_data.AllocateScalars()
                vtk_image_data.SetNumberOfScalarComponents(4)
            else:
                vtk_image_data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 4)

            # width, height
            vtk_image_data.SetDimensions(pic.shape[1], pic.shape[0], 1)
            vtk_image_data.SetExtent(0, pic.shape[1] - 1,
                                     0, pic.shape[0] - 1,
                                     0, 0)
            pic_tmp = np.swapaxes(pic, 0, 1)
            pic_tmp = pic.reshape(pic.shape[1] * pic.shape[0], 4)
            pic_tmp = np.ascontiguousarray(pic_tmp)
            uchar_array = numpy_support.numpy_to_vtk(pic_tmp, deep=True)
            vtk_image_data.GetPointData().SetScalars(uchar_array)

    image_actor = vtk.vtkImageActor()
    image_actor.SetInputData(vtk_image_data)

    if interpolation == 'nearest':
        image_actor.GetProperty().SetInterpolationTypeToNearest()

    if interpolation == 'linear':
        image_actor.GetProperty().SetInterpolationTypeToLinear()

    if interpolation == 'cubic':
        image_actor.GetProperty().SetInterpolationTypeToCubic()

    image_actor.Update()
    return image_actor


def create_button(file_name, callback, position=(0.1, 0.1, 0.1), center=None):
    button = figure(file_name)
    button.AddObserver("RightButtonPressEvent", callback)
    if center is not None:
        button.SetCenter(*center)
    button.SetPosition(position[0], position[1], position[2])

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
    pos = np.array(cube_actor.GetPosition())
    print(pos)
    pos[0] += 2
    cube_actor.SetPosition(tuple(pos))


cube_actor = cube((1, 0, 0), (50, 50, 50), center=(0, 0, 0))

fetch_viz_icons()
filename = read_viz_icons(fname='stop2.png')

button_actor = create_button(file_name=filename, callback=button_callback, position=(100, -100, 10))

renderer = window.ren()
iren_style = CustomInteractorStyle(renderer=renderer)
renderer.add(button_actor)
renderer.add(cube_actor)

# set_trace()

showm = window.ShowManager(renderer, interactor_style=iren_style)

showm.initialize()
showm.render()
showm.start()
