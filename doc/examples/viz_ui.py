import numpy as np

from dipy.data import read_viz_icons

# Conditional import machinery for vtk.
from dipy.utils.optpkg import optional_package

# Allow import, but disable doctests if we don't have vtk.
from dipy.viz import ui, window

vtk, have_vtk, setup_module = optional_package('vtk')

if have_vtk:
    vtkInteractorStyleUser = vtk.vtkInteractorStyleUser
    version = vtk.vtkVersion.GetVTKSourceVersion().split(' ')[-1]
    major_version = vtk.vtkVersion.GetVTKMajorVersion()
else:
    vtkInteractorStyleUser = object

numpy_support, have_ns, _ = optional_package('vtk.util.numpy_support')


# Cube Actors
def cube_maker(color=None, size=(0.2, 0.2, 0.2), center=None):
    cube = vtk.vtkCubeSource()
    cube.SetXLength(size[0])
    cube.SetYLength(size[1])
    cube.SetZLength(size[2])
    if center is not None:
        cube.SetCenter(*center)
    cube_mapper = vtk.vtkPolyDataMapper()
    cube_mapper.SetInputConnection(cube.GetOutputPort())
    cube_actor = vtk.vtkActor()
    cube_actor.SetMapper(cube_mapper)
    if color is not None:
        cube_actor.GetProperty().SetColor(color)
    return cube_actor

cube_actor_1 = cube_maker((1, 0, 0), (50, 50, 50), center=(0, 0, 0))
cube_actor_2 = cube_maker((0, 1, 0), (10, 10, 10), center=(100, 0, 0))
# /Cube Actors

# Buttons
icon_files = dict()
icon_files['stop'] = read_viz_icons(fname='stop2.png')
icon_files['play'] = read_viz_icons(fname='play3.png')
icon_files['plus'] = read_viz_icons(fname='plus.png')
icon_files['cross'] = read_viz_icons(fname='cross.png')

button_example = ui.Button2D(icon_fnames=icon_files)


def left_mouse_button_click(i_ren, obj, button):
    print("Left Button Clicked")


def left_mouse_button_drag(i_ren, obj, button):
    print ("Left Button Dragged")

button_example.on_left_mouse_button_drag = left_mouse_button_drag
button_example.on_left_mouse_button_pressed = left_mouse_button_click


def right_mouse_button_drag(i_ren, obj, button):
    print("Right Button Dragged")


def right_mouse_button_click(i_ren, obj, button):
    print ("Right Button Clicked")

button_example.on_right_mouse_button_drag = right_mouse_button_drag
button_example.on_right_mouse_button_pressed = right_mouse_button_click


second_button_example = ui.Button2D(icon_fnames=icon_files)


def modify_button_callback(i_ren, obj, button):
    # i_ren: CustomInteractorStyle
    # obj: vtkActor picked
    # button: Button2D
    button.next_icon()
    i_ren.force_render()

second_button_example.on_left_mouse_button_pressed = modify_button_callback

# /Buttons


# Panel
panel = ui.Panel2D(center=(440, 90), size=(300, 150), color=(1, 1, 1), align="right")
panel.add_element(button_example, 'relative', (0.2, 0.2))
panel.add_element(second_button_example, 'absolute', (480, 100))

# /Panel

# TextBox
text = ui.TextBox2D(height=3, width=10)
# /TextBox

# Show Manager
current_size = (600, 600)
show_manager = window.ShowManager(size=current_size, title="DIPY UI Example")

show_manager.ren.add(cube_actor_1)
show_manager.ren.add(cube_actor_2)
show_manager.ren.add(panel)
show_manager.ren.add(text)

show_manager.start()
