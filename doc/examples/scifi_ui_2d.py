import numpy as np

from dipy.data import read_viz_icons

# Conditional import machinery for vtk.
from dipy.utils.optpkg import optional_package

from dipy.viz import actor, window, gui_2d
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

icon_files = dict()
icon_files['stop'] = read_viz_icons(fname='stop2.png')
icon_files['play'] = read_viz_icons(fname='play3.png')
icon_files['plus'] = read_viz_icons(fname='plus.png')
icon_files['cross'] = read_viz_icons(fname='cross.png')

button = gui_2d.Button2D(icon_fnames=icon_files)


def move_button_callback(*args, **kwargs):
    pos_1 = np.array(cube_actor_1.GetPosition())
    pos_1[0] += 2
    cube_actor_1.SetPosition(tuple(pos_1))
    pos_2 = np.array(cube_actor_2.GetPosition())
    pos_2[1] += 2
    cube_actor_2.SetPosition(tuple(pos_2))
    showm.render()
    return True


def modify_button_callback(obj, evt):
    button.next_icon()
    showm.render()
    return True

button.add_callback("RightButtonPressEvent", move_button_callback)
button.add_callback("LeftButtonPressEvent", modify_button_callback)

text = gui_2d.TextBox2D(height=3, width=10)


def key_press_callback(obj, evt):
    # obj: TextBox2D
    key = showm.iren.GetInteractorStyle().GetKeySym()
    is_done = obj.handle_character(key)
    if is_done:
        showm.iren.GetInteractorStyle().remove_active_prop(obj.actor)

    showm.render()


def select_text_callback(obj, evt):
    # obj: TextBox2D
    showm.iren.GetInteractorStyle().add_active_prop(obj.actor)
    obj.edit_mode()
    showm.render()

text.add_callback("KeyPressEvent", key_press_callback)
text.add_callback("LeftButtonPressEvent", select_text_callback)


slider = gui_2d.LineSlider2D()


def line_click_callback(obj, evt):
    # obj: LineSlider2D
    position = showm.iren.GetEventPosition()
    obj.slider_disk.set_position(position)
    obj.text.set_percentage(position[0])
    showm.iren.GetInteractorStyle().add_active_prop(obj.slider_disk.actor)
    showm.render()
    return True


def disk_press_callback(obj, evt):
    # obj: LineSlider2D
    showm.iren.GetInteractorStyle().add_active_prop(obj.slider_disk.actor)
    return True


def disk_release_callback(obj, evt):
    # obj: LineSlider2D
    showm.iren.GetInteractorStyle().remove_active_prop(obj.slider_disk.actor)


def disk_move_callback(obj, evt):
    # obj: LineSlider2D
    position = showm.iren.GetEventPosition()
    obj.slider_disk.set_position(position)
    obj.text.set_percentage(position[0])
    showm.render()
    return True

slider.add_callback("LeftButtonPressEvent", line_click_callback, slider.slider_line)
slider.add_callback("MouseMoveEvent", disk_move_callback, slider.slider_disk)
slider.add_callback("LeftButtonPressEvent", disk_press_callback, slider.slider_disk)
slider.add_callback("LeftButtonReleaseEvent", disk_release_callback, slider.slider_disk)

disk_slider = gui_2d.DiskSlider2D()

# panel = gui_2d.Panel2D(center=(0, 0), size=(1000, 1000))


def outer_disk_click_callback(obj, evt):
    # obj: DiskSlider2D
    click_position = showm.iren.GetEventPosition()
    intersection_coordinate = disk_slider.get_poi(click_position)
    disk_slider.slider_inner_disk.set_position(intersection_coordinate)
    angle = disk_slider.get_angle(intersection_coordinate)
    disk_slider.slider_text.set_percentage(angle)
    showm.iren.GetInteractorStyle().add_active_prop(obj.slider_inner_disk.actor)
    showm.render()
    return True


def inner_disk_move_callback(obj, evt):
    # obj: DiskSlider2D
    click_position = showm.iren.GetEventPosition()
    intersection_coordinate = disk_slider.get_poi(click_position)
    disk_slider.slider_inner_disk.set_position(intersection_coordinate)
    angle = disk_slider.get_angle(intersection_coordinate)
    disk_slider.slider_text.set_percentage(angle)
    showm.render()
    return True


def inner_disk_press_callback(obj, evt):
    # obj: DiskSlider2D
    showm.iren.GetInteractorStyle().add_active_prop(obj.slider_inner_disk.actor)
    return True


def inner_disk_release_callback(obj, evt):
    # obj: DiskSlider2D
    showm.iren.GetInteractorStyle().remove_active_prop(obj.slider_inner_disk.actor)


disk_slider.add_callback("LeftButtonPressEvent", outer_disk_click_callback, disk_slider.slider_outer_disk)
disk_slider.add_callback("MouseMoveEvent", inner_disk_move_callback, disk_slider.slider_inner_disk)
disk_slider.add_callback("LeftButtonPressEvent", inner_disk_press_callback, disk_slider.slider_inner_disk)
disk_slider.add_callback("LeftButtonReleaseEvent", inner_disk_release_callback, disk_slider.slider_inner_disk)

panel = gui_2d.Panel2D(center=(400, 400), size=(200, 100), color=(1, 1, 1))
panel.add_element(button, (0.01, 0.01))
panel.add_element(text, (0.4, 0.01))

renderer = window.ren()
renderer.add(panel)
renderer.add(cube_actor_1)
renderer.add(cube_actor_2)
renderer.add(text)
renderer.add(slider)
renderer.add(disk_slider)

showm = window.ShowManager(renderer, size=(600, 600), title="Sci-Fi UI")
showm.initialize()
showm.render()
showm.start()
