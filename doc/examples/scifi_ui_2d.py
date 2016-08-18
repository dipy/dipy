import numpy as np

from dipy.data import read_viz_icons

# Conditional import machinery for vtk.
from dipy.utils.optpkg import optional_package

from dipy.viz import actor, window, gui_2d
from ipdb import set_trace

# Allow import, but disable doctests if we don't have vtk.
from dipy.viz.gui_2d import FileSelect2D
from dipy.viz.gui_menus import FileSaveMenu

vtk, have_vtk, setup_module = optional_package('vtk')

if have_vtk:
    vtkInteractorStyleUser = vtk.vtkInteractorStyleUser
    version = vtk.vtkVersion.GetVTKSourceVersion().split(' ')[-1]
    major_version = vtk.vtkVersion.GetVTKMajorVersion()
else:
    vtkInteractorStyleUser = object

numpy_support, have_ns, _ = optional_package('vtk.util.numpy_support')


# Cube Actors
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
# /Cube Actors

# Buttons
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
# /Buttons

# Textbox
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
# /Textbox

# Line Slider
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
# /Line Slider

# Disk Slider
disk_slider = gui_2d.DiskSlider2D()


def outer_disk_click_callback(obj, evt):
    # obj: DiskSlider2D
    click_position = showm.iren.GetEventPosition()
    disk_slider.move_disk(click_position=click_position)
    showm.iren.GetInteractorStyle().add_active_prop(obj.slider_inner_disk.actor)
    showm.render()
    return True


def inner_disk_move_callback(obj, evt):
    # obj: DiskSlider2D
    click_position = showm.iren.GetEventPosition()
    disk_slider.move_disk(click_position=click_position)
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
# /Disk Slider

# Panel
panel = gui_2d.Panel2D(center=(440, 90), size=(300, 150), color=(1, 1, 1), align="right")

panel.add_element(button, (0.95, 0.9))
panel.add_element(text, (0.1, 0.2))
panel.add_element(slider, (0.5, 0.9))
panel.add_element(disk_slider, (0.7, 0.3))


def panel_click_callback(obj, evt):
    # obj: Panel2D
    panel.ui_param = (showm.iren.GetEventPosition()[0] - obj.panel.actor.GetPosition()[0] - obj.size[0]/2,
                      showm.iren.GetEventPosition()[1] - obj.panel.actor.GetPosition()[1] - obj.size[1]/2)
    showm.iren.GetInteractorStyle().add_active_prop(obj.panel.actor)
    return True


def panel_release_callback(obj, evt):
    # obj: Panel2D
    showm.iren.GetInteractorStyle().remove_active_prop(obj.panel.actor)


def panel_move_callback(obj, evt):
    # obj: Panel2D
    click_position = showm.iren.GetEventPosition()
    panel.set_center((click_position[0] - panel.ui_param[0], click_position[1] - panel.ui_param[1]))
    showm.render()

panel.add_callback("LeftButtonPressEvent", panel_click_callback, panel.panel)
panel.add_callback("LeftButtonReleaseEvent", panel_release_callback, panel.panel)
panel.add_callback("MouseMoveEvent", panel_move_callback, panel.panel)

# /Panel

# File Dialog

# file_menu = FileSelect2D(size=(200, 300), font_size=12, position=(200, 300))
file_dialog = FileSaveMenu(size=(300, 300), position=(300, 300))

# /File Dialog

# Initialize and add to renderer
renderer = window.ren()

# renderer.add(panel)
renderer.add(cube_actor_1)
renderer.add(cube_actor_2)
renderer.add(file_dialog)
# /Renderer

# Show Manager
current_size = [600, 600]

showm = window.ShowManager(renderer, size=current_size, title="Sci-Fi UI")


def window_callback(obj, evt):
    size_change = (renderer.size()[0] - current_size[0], renderer.size()[1] - current_size[1])
    current_size[0] = renderer.size()[0]
    current_size[1] = renderer.size()[1]
    panel.re_align(size_change)

showm.add_window_callback(window_callback)

showm.initialize()
showm.render()
showm.start()
# /Show Manager
