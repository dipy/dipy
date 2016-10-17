import numpy as np

from dipy.data import read_viz_icons

# Conditional import machinery for vtk.
from dipy.utils.optpkg import optional_package

from dipy.viz import actor, window, gui_2d
from ipdb import set_trace

# Allow import, but disable doctests if we don't have vtk.
from dipy.viz import gui_mod
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


def move_button_callback(iren, obj, button):
    # iren: CustomInteractorStyle
    # obj: vtkActor picked
    # button: Button2D
    pos_1 = np.array(cube_actor_1.GetPosition())
    pos_1[0] += 2
    cube_actor_1.SetPosition(tuple(pos_1))
    pos_2 = np.array(cube_actor_2.GetPosition())
    pos_2[1] += 2
    cube_actor_2.SetPosition(tuple(pos_2))
    iren.force_render()
    iren.event.abort()  # Stop propagating the event.


def modify_button_callback(iren, obj, button):
    # iren: CustomInteractorStyle
    # obj: vtkActor picked
    # button: Button2D
    button.next_icon()
    iren.force_render()
    iren.event.abort()  # Stop propagating the event.

button.add_callback("RightButtonPressEvent", move_button_callback)
button.add_callback("LeftButtonPressEvent", modify_button_callback)
# /Buttons

# Textbox
text = gui_2d.TextBox2D(height=3, width=10)


def key_press_callback(iren, obj, textbox):
    # iren: CustomInteractorStyle
    # obj: vtkActor picked
    # textbox: TextBox2D
    key = iren.event.key
    is_done = textbox.handle_character(key)
    if is_done:
        iren.remove_active_prop(textbox.actor)

    iren.force_render()


def select_text_callback(iren, obj, textbox):
    # iren: CustomInteractorStyle
    # obj: vtkActor picked
    # textbox: TextBox2D
    iren.add_active_prop(textbox.actor)
    textbox.edit_mode()
    iren.force_render()

text.add_callback("KeyPressEvent", key_press_callback)
text.add_callback("LeftButtonPressEvent", select_text_callback)
# /Textbox

# Line Slider
slider = gui_2d.LineSlider2D()


def line_click_callback(iren, obj, slider):
    # iren: CustomInteractorStyle
    # obj: vtkActor picked
    # slider: LineSlider2D
    # Update disk position and grab the focus.
    position = iren.event.position
    slider.set_position(position)
    slider.set_percentage(position[0])
    iren.force_render()
    iren.event.abort()  # Stop propagating the event.


def disk_press_callback(iren, obj, slider):
    # iren: CustomInteractorStyle
    # obj: vtkActor picked
    # slider: LineSlider2D
    # Only need to grab the focus.
    iren.event.abort()  # Stop propagating the event.


def disk_move_callback(iren, obj, slider):
    # iren: CustomInteractorStyle
    # obj: vtkActor picked
    # slider: LineSlider2D
    position = iren.event.position
    slider.set_position(position)
    slider.set_percentage(position[0])
    iren.force_render()
    iren.event.abort()  # Stop propagating the event.

slider.add_callback("LeftButtonPressEvent", line_click_callback, slider.slider_line)
slider.add_callback("LeftButtonPressEvent", disk_press_callback, slider.slider_disk)
slider.add_callback("MouseMoveEvent", disk_move_callback, slider.slider_disk)
slider.add_callback("MouseMoveEvent", disk_move_callback, slider.slider_line)
# /Line Slider

# Disk Slider
disk_slider = gui_mod.DiskSlider2DMod()


def outer_disk_click_callback(iren, obj, disk_slider):
    # iren: CustomInteractorStyle
    # obj: vtkActor picked
    # disk_slider: DiskSlider2D
    # Update disk position and grab the focus.
    click_position = iren.event.position
    disk_slider.move_move_disk(click_position=click_position)
    iren.force_render()
    iren.event.abort()  # Stop propagating the event.


def inner_disk_move_callback(iren, obj, disk_slider):
    # iren: CustomInteractorStyle
    # disk_slider: vtkActor picked
    # ui_component: DiskSlider2D
    click_position = iren.event.position
    disk_slider.move_move_disk(click_position=click_position)
    iren.force_render()
    iren.event.abort()  # Stop propagating the event.


def inner_disk_press_callback(iren, obj, disk_slider):
    # iren: CustomInteractorStyle
    # disk_slider: vtkActor picked
    # ui_component: DiskSlider2D
    # Only need to grab the focus.
    iren.event.abort()  # Stop propagating the event.


disk_slider.add_callback("LeftButtonPressEvent", outer_disk_click_callback, disk_slider.base_disk)
disk_slider.add_callback("LeftButtonPressEvent", inner_disk_press_callback, disk_slider.move_disk)
disk_slider.add_callback("MouseMoveEvent", inner_disk_move_callback, disk_slider.base_disk)
disk_slider.add_callback("MouseMoveEvent", inner_disk_move_callback, disk_slider.move_disk)
# /Disk Slider

# Panel
panel = gui_2d.Panel2D(center=(440, 90), size=(300, 150), color=(1, 1, 1), align="right")

panel.add_element(button, (0.95, 0.9))
panel.add_element(text, (0.1, 0.2))
panel.add_element(slider, (0.5, 0.9))
panel.add_element(disk_slider, (0.7, 0.4))


def panel_click_callback(iren, obj, panel):
    # iren: CustomInteractorStyle
    # obj: vtkActor picked
    # panel: Panel2D
    click_position = iren.event.position
    panel.ui_param = (click_position[0] - panel.panel.actor.GetPosition()[0] - panel.size[0]/2,
                      click_position[1] - panel.panel.actor.GetPosition()[1] - panel.size[1]/2)
    iren.event.abort()  # Stop propagating the event.


def panel_move_callback(iren, obj, panel):
    # iren: CustomInteractorStyle
    # obj: vtkActor picked
    # panel: Panel2D
    click_position = iren.event.position
    panel.set_center((click_position[0] - panel.ui_param[0], click_position[1] - panel.ui_param[1]))
    iren.force_render()

panel.add_callback("LeftButtonPressEvent", panel_click_callback, panel.panel)
panel.add_callback("MouseMoveEvent", panel_move_callback, panel.panel)

# /Panel

# File Dialog

file_dialog = FileSaveMenu(size=(500, 500), position=(300, 300))


def save_callback(iren, obj, file_dialog):
    # iren: CustomInteractorStyle
    # obj: vtkActor picked
    # file_dialog: FileSaveMenu
    print("Saved!")


def cancel_callback(iren, obj, file_dialog):
    # iren: CustomInteractorStyle
    # obj: vtkActor picked
    # file_dialog: FileSaveMenu
    print("Cancelled!")

file_dialog.add_callback("LeftButtonPressEvent", save_callback, file_dialog.save_button)
file_dialog.add_callback("LeftButtonPressEvent", cancel_callback, file_dialog.cancel_button)
file_dialog.text_box.add_callback("KeyPressEvent", key_press_callback)
file_dialog.text_box.add_callback("LeftButtonPressEvent", select_text_callback)

# /File Dialog

# Initialize and add to renderer
renderer = window.ren()

# Show Manager
current_size = [600, 600]
showm = window.ShowManager(renderer, size=current_size, title="Sci-Fi UI")

renderer.add(panel)
renderer.add(cube_actor_1)
renderer.add(cube_actor_2)
# renderer.add(file_dialog)
# renderer.add(disk_slider)

def window_callback(obj, evt):
    size_change = (renderer.size()[0] - current_size[0], renderer.size()[1] - current_size[1])
    current_size[0] = renderer.size()[0]
    current_size[1] = renderer.size()[1]
    panel.re_align(size_change)

showm.add_window_callback(window_callback)
showm.start()
