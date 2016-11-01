import numpy as np

from dipy.viz import gui_follower
from dipy.viz import window

from dipy.data import read_viz_icons

# Conditional import machinery for vtk.
from dipy.utils.optpkg import optional_package

# Allow import, but disable doctests if we don't have vtk.
from dipy.viz.gui_follower import ButtonFollower, CubeButtonFollower, TextFollower

vtk, have_vtk, setup_module = optional_package('vtk')

if have_vtk:
    vtkInteractorStyleUser = vtk.vtkInteractorStyleUser
    version = vtk.vtkVersion.GetVTKSourceVersion().split(' ')[-1]
    major_version = vtk.vtkVersion.GetVTKMajorVersion()
else:
    vtkInteractorStyleUser = object

numpy_support, have_ns, _ = optional_package('vtk.util.numpy_support')


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
button_actor_3 = CubeButtonFollower(size=(10, 10, 10), color=(1, 0, 0))

text_actor = TextFollower(text="Hello!", color=(0, 1, 0))


# def modify_button_callback_1(*args, **kwargs):
#     cube_actor_1.GetProperty().SetColor((0, 0, 1))
#     button_actor_1.next_icon()
#
#
# def modify_button_callback_2(*args, **kwargs):
#     cube_actor_1.GetProperty().SetColor((0, 1, 0))
#
#
# def modify_button_callback_3(*args, **kwargs):
#     cube_actor_1.GetProperty().SetColor((1, 0, 0))
#
# button_actor_1.add_callback("LeftButtonPressEvent", modify_button_callback_1)
# text_actor.add_callback("LeftButtonPressEvent", modify_button_callback_2)
# button_actor_3.add_callback("LeftButtonPressEvent", modify_button_callback_3)

renderer = window.ren()

follower_menu = gui_follower.FollowerMenu(position=(0, 0, 0), diameter=87, camera=renderer.GetActiveCamera(),
                                          elements=[button_actor_1, button_actor_3, text_actor])

renderer.add(follower_menu)
renderer.add(cube_actor_1)
renderer.add(cube_actor_2)

# Show Manager
current_size = [600, 600]
showm = window.ShowManager(renderer, size=current_size, title="Sci-Fi UI")
showm.start()
