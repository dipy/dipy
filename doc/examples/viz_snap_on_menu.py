import numpy as np

from dipy.viz import actor, window

# Conditional import machinery for vtk.
from dipy.utils.optpkg import optional_package

# Allow import, but disable doctests if we don't have vtk.
vtk, have_vtk, setup_module = optional_package('vtk')

if have_vtk:
    vtkInteractorStyleUser = vtk.vtkInteractorStyleUser
    version = vtk.vtkVersion.GetVTKSourceVersion().split(' ')[-1]
    major_version = vtk.vtkVersion.GetVTKMajorVersion()
else:
    vtkInteractorStyleUser = object


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
        self.chosen_element = None

    def get_ui_item(self, selected_actor):
        ui_list = self.renderer.ui_list
        for ui_item in ui_list:
            if ui_item.actor == selected_actor:
                return ui_item

    def on_left_button_pressed(self, obj, evt):
        click_pos = self.GetInteractor().GetEventPosition()

        self.picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)

        path = self.picker.GetPath()
        node = path.GetLastNode()
        prop = node.GetViewProp()
        prop.VisibilityOff()

        actor_2d = self.picker.GetActor2D()
        if actor_2d is not None:
            self.chosen_element = self.get_ui_item(actor_2d)
            actor_2d.InvokeEvent(evt)
        else:
            actor_3d = self.picker.GetProp3D()
            if actor_3d is not None:
                self.chosen_element = self.get_ui_item(actor_3d)
                actor_3d.InvokeEvent(evt)
            else:
                pass

        self.trackball_interactor_style.OnLeftButtonDown()

    def on_left_button_released(self, obj, evt):
        self.trackball_interactor_style.OnLeftButtonUp()

    def on_right_button_pressed(self, obj, evt):

        click_pos = self.GetInteractor().GetEventPosition()

        self.picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
        actor_2d = self.picker.GetViewProp()
        if actor_2d is not None:
            self.chosen_element = self.get_ui_item(actor_2d)
            actor_2d.InvokeEvent(evt)
        else:
            actor_3d = self.picker.GetProp3D()
            if actor_3d is not None:
                self.chosen_element = self.get_ui_item(actor_3d)
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
        pass

    def add_ui_param(self, class_name, ui_param):
        ui_list = self.renderer.ui_list
        for ui_item in ui_list:
            if ui_item == self.chosen_element:
                if isinstance(ui_item, class_name):
                    ui_item.set_ui_param(ui_param)
                    break

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


def make_assembly_follower(assembly):
    # Usually vtkFollower works by using
    # SetMapper(my_object.GetMapper()) but since our orbital_system
    # is a vtkAssembly there is no Mapper. So, we have to manually
    # update the transformation matrix of our orbital_system according to
    # an empty vtkFollower actor that we explictly add into the assembly.
    # By adding the vtkFollower into the assembly its transformation matrix
    # get automatically updated so it always faces the camera. Using that
    # trasnformation matrix and we can transform our orbital_system
    # accordingly.
    dummy_follower = vtk.vtkFollower()
    orbital_system.AddPart(dummy_follower)
    dummy_follower.SetCamera(ren.GetActiveCamera())

    # Get assembly transformation matrix.
    M = vtk.vtkTransform()
    M.SetMatrix(assembly.GetMatrix())

    # Get the inverse of the assembly transformation matrix.
    M_inv = vtk.vtkTransform()
    M_inv.SetMatrix(assembly.GetMatrix())
    M_inv.Inverse()

    # Create a transform object that gets updated whenever the input matrix
    # is updated, which is whenever the camera moves.
    dummy_follower_transform = vtk.vtkMatrixToLinearTransform()
    dummy_follower_transform.SetInput(dummy_follower.GetMatrix())

    T = vtk.vtkTransform()
    T.PostMultiply()
    # Bring the assembly to the origin.
    T.Concatenate(M_inv)
    # Change orientation of the assembly.
    T.Concatenate(dummy_follower_transform)
    # Bring the assembly back where it was.
    T.Concatenate(M)

    assembly.SetUserTransform(T)
    return assembly


def make_cube(edge=1):
    cube_src = vtk.vtkCubeSource()
    cube_src.SetXLength(edge)
    cube_src.SetYLength(edge)
    cube_src.SetZLength(edge)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(cube_src.GetOutputPort())
    cube = vtk.vtkActor()
    cube.SetMapper(mapper)
    return cube


def send_into_orbit(obj, satellites, r=1., show_orbit=True):
    radius = r * (obj.GetLength() / 2.)
    t = np.linspace(0, 2*np.pi)

    orbit_pts = np.array([np.cos(t) * radius,
                          np.sin(t) * radius,
                          np.zeros_like(t)]).astype(np.float32)
    orbit = actor.streamtube([np.ascontiguousarray(orbit_pts.T)],
                             np.array((0., 0., 1.), dtype=np.float32))

    # Disperse satellites evenly on the orbit.
    # Create an assembly to group multiple actors together.
    # This might cause some issues with the picking though.
    orbital_system = vtk.vtkAssembly()
    if show_orbit:
        orbital_system.AddPart(orbit)

    t = np.linspace(0, 2*np.pi, num=len(satellites), endpoint=False)
    satellites_coords = np.array([np.cos(t) * radius,
                                  np.sin(t) * radius,
                                  np.zeros_like(t)]
                                 ).astype(np.float32)

    for coord, satellite in zip(satellites_coords.T, satellites):
        satellite.SetPosition(coord)
        orbital_system.AddPart(satellite)

    offset = (np.asarray(obj.GetCenter()) - np.asarray(orbital_system.GetCenter()))
    orbital_system.SetPosition(offset)

    return orbital_system


# Create the "Earth" (i.e. object to snap a circular menu onto).
earth = make_cube()

# Create "satellites" (i.e. buttons of the circular menu).
s1 = make_cube(edge=0.25)
s2 = make_cube(edge=0.25)
s3 = make_cube(edge=0.25)

# Position the statellites around the Earth.
orbital_system = send_into_orbit(earth, [s1, s2, s3],  r=1.1)

ren = window.ren()

# Make the orbit always faces the camera.
orbital_system = make_assembly_follower(orbital_system)
ren.add(earth, orbital_system)

# Let's get crazy and add a moon.
moon = make_cube(edge=0.2)
moon.AddPosition(2, 2, 2)
moon.RotateZ(45)

s4 = make_cube(edge=0.05)
s5 = make_cube(edge=0.05)
s6 = make_cube(edge=0.05)
s7 = make_cube(edge=0.05)

orbital_system = send_into_orbit(moon, [s4, s5, s6, s7],  r=1.1)
orbital_system = make_assembly_follower(orbital_system)
ren.add(moon, orbital_system)

iren_style = CustomInteractorStyle(renderer=ren)
show_m = window.ShowManager(ren, interactor_style=iren_style, size=(800, 600))
show_m.initialize()
show_m.render()
show_m.start()
