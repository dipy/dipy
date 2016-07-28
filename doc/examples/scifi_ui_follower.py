import numpy as np

from dipy.data import read_viz_icons

# Conditional import machinery for vtk.
from dipy.utils.optpkg import optional_package

from dipy.viz import actor, window, gui_follower
from ipdb import set_trace

# Allow import, but disable doctests if we don't have vtk.
from dipy.viz.gui_follower import CubeButtonFollower, ButtonFollower, TextFollower

vtk, have_vtk, setup_module = optional_package('vtk')

if have_vtk:
    vtkInteractorStyleUser = vtk.vtkInteractorStyleUser
    version = vtk.vtkVersion.GetVTKSourceVersion().split(' ')[-1]
    major_version = vtk.vtkVersion.GetVTKMajorVersion()
else:
    vtkInteractorStyleUser = object

numpy_support, have_ns, _ = optional_package('vtk.util.numpy_support')

renderer = window.ren()


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


cube_actor_1 = cube((1, 1, 1), (50, 50, 50), center=(0, 0, 0))
cube_actor_2 = cube((1, 0, 0), (25, 25, 25), center=(100, 0, 0))

icon_files = dict()
icon_files['stop'] = read_viz_icons(fname='stop2.png')
icon_files['play'] = read_viz_icons(fname='play3.png')
icon_files['plus'] = read_viz_icons(fname='plus.png')
icon_files['cross'] = read_viz_icons(fname='cross.png')

button_actor_1 = ButtonFollower(icon_fnames=icon_files)
button_actor_2 = ButtonFollower(icon_fnames=icon_files)
# button_actor_3 = ButtonFollower(icon_fnames=icon_files)

# button_actor_1 = CubeButton(size=(10, 10, 10), color=(0, 0, 1))
# button_actor_2 = CubeButton(size=(10, 10, 10), color=(0, 1, 0))
button_actor_3 = CubeButtonFollower(size=(10, 10, 10), color=(1, 0, 0))

text_actor = TextFollower(text="Hello!", color=(0, 1, 0))


def modify_button_callback_1(*args, **kwargs):
    cube_actor_1.GetProperty().SetColor((0, 0, 1))
    button_actor_1.next_icon()


def modify_button_callback_2(*args, **kwargs):
    cube_actor_1.GetProperty().SetColor((0, 1, 0))


def modify_button_callback_3(*args, **kwargs):
    cube_actor_1.GetProperty().SetColor((1, 0, 0))

button_actor_1.add_callback("LeftButtonPressEvent", modify_button_callback_1)
text_actor.add_callback("LeftButtonPressEvent", modify_button_callback_2)
button_actor_3.add_callback("LeftButtonPressEvent", modify_button_callback_3)

slider = gui_follower.LineSliderFollower()


def line_click_callback(obj, evt):
    # obj: LineSlider
    position = showm.iren.GetEventPosition()
    obj.slider_disk.set_position(position)
    showm.iren.GetInteractorStyle().add_active_prop(obj.slider_disk.actor)
    showm.render()
    return True


def disk_press_callback(obj, evt):
    # obj: LineSlider
    showm.iren.GetInteractorStyle().add_active_prop(obj.slider_disk.actor)
    return True


def disk_release_callback(obj, evt):
    # obj: LineSlider
    showm.iren.GetInteractorStyle().remove_active_prop(obj.slider_disk.actor)


def disk_move_callback(obj, evt):
    # obj: LineSlider
    position = showm.iren.GetEventPosition()
    obj.slider_disk.set_position(position)
    showm.render()
    return True

slider.add_callback("LeftButtonPressEvent", line_click_callback, slider.slider_line)
slider.add_callback("MouseMoveEvent", disk_move_callback, slider.slider_disk)
slider.add_callback("LeftButtonPressEvent", disk_press_callback, slider.slider_disk)
slider.add_callback("LeftButtonReleaseEvent", disk_release_callback, slider.slider_disk)

follower_menu = gui_follower.FollowerMenu(position=(0, 0, 0), diameter=87, camera=renderer.GetActiveCamera(),
                                          elements=[button_actor_1, slider, button_actor_3])

renderer.add(follower_menu)
renderer.add(cube_actor_1)
renderer.add(cube_actor_2)


showm = window.ShowManager(renderer, size=(600, 600), title="Sci-Fi UI")
showm.initialize()
showm.render()
showm.start()
