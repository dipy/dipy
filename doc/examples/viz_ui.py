"""
===============
User Interfaces
===============

This example shows how to use the UI API.
Currently includes button, textbox, panel, and line slider.

First, a bunch of imports.

"""

import os

from dipy.data import read_viz_icons, fetch_viz_icons

from dipy.viz import ui, window

"""
3D Elements
===========

Let's have some cubes in 3D.
"""


def cube_maker(color=None, size=(0.2, 0.2, 0.2), center=None):
    cube = window.vtk.vtkCubeSource()
    cube.SetXLength(size[0])
    cube.SetYLength(size[1])
    cube.SetZLength(size[2])
    if center is not None:
        cube.SetCenter(*center)
    cube_mapper = window.vtk.vtkPolyDataMapper()
    cube_mapper.SetInputConnection(cube.GetOutputPort())
    cube_actor = window.vtk.vtkActor()
    cube_actor.SetMapper(cube_mapper)
    if color is not None:
        cube_actor.GetProperty().SetColor(color)
    return cube_actor


cube_actor_1 = cube_maker((1, 0, 0), (50, 50, 50), center=(0, 0, 0))
cube_actor_2 = cube_maker((0, 1, 0), (10, 10, 10), center=(100, 0, 0))

"""
Buttons
=======

We first fetch the icons required for making the buttons.
"""

fetch_viz_icons()

"""
Add the icon filenames to a dict.
"""

icon_files = dict()
icon_files['stop'] = read_viz_icons(fname='stop2.png')
icon_files['play'] = read_viz_icons(fname='play3.png')
icon_files['plus'] = read_viz_icons(fname='plus.png')
icon_files['cross'] = read_viz_icons(fname='cross.png')

"""
Create a button through our API.
"""

button_example = ui.Button2D(icon_fnames=icon_files)

"""
We now add some click listeners.
"""


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

"""
Let's have another button.
"""

second_button_example = ui.Button2D(icon_fnames=icon_files)

"""
This time, we will call the built in `next_icon` method
via a callback that is triggered on left click.
"""


def modify_button_callback(i_ren, obj, button):
    button.next_icon()
    i_ren.force_render()


second_button_example.on_left_mouse_button_pressed = modify_button_callback

"""
Panels
======

Simply create a panel and add elements to it.
"""

panel = ui.Panel2D(center=(440, 90), size=(300, 150), color=(1, 1, 1),
                   align="right")
panel.add_element(button_example, 'relative', (0.2, 0.2))
panel.add_element(second_button_example, 'absolute', (480, 100))

"""
TextBox
=======
"""

text = ui.TextBox2D(height=3, width=10)

"""
2D Line Slider
==============
"""


def translate_green_cube(i_ren, obj, slider):
    value = slider.value
    cube_actor_2.SetPosition(value, 0, 0)

line_slider = ui.LineSlider2D(initial_value=-2,
                              min_value=-5, max_value=5)

line_slider.add_callback(line_slider.slider_disk,
                         "MouseMoveEvent",
                         translate_green_cube)

line_slider.add_callback(line_slider.slider_line,
                         "LeftButtonPressEvent",
                         translate_green_cube)

"""
2D Disk Slider
==============
"""


def rotate_red_cube(i_ren, obj, slider):
    angle = slider.value
    previous_angle = slider.previous_value
    rotation_angle = angle - previous_angle
    cube_actor_1.RotateY(rotation_angle)


disk_slider = ui.DiskSlider2D()
disk_slider.set_center((200, 200))
disk_slider.add_callback(disk_slider.handle,
                         "MouseMoveEvent",
                         rotate_red_cube)

disk_slider.add_callback(disk_slider.base_disk,
                         "LeftButtonPressEvent",
                         rotate_red_cube)
"""
2D File Select Menu
==============
"""

file_select_menu = ui.FileSelectMenu2D(size=(500, 500),
                                       position=(300, 300),
                                       font_size=16,
                                       extensions=["py", "png"],
                                       directory_path=os.getcwd(),
                                       parent=None)

"""
Adding Elements to the ShowManager
==================================

Once all elements have been initialised, they have
to be added to the show manager in the following manner.
"""

current_size = (600, 600)
show_manager = window.ShowManager(size=current_size, title="DIPY UI Example")

show_manager.ren.add(cube_actor_1)
show_manager.ren.add(cube_actor_2)
show_manager.ren.add(panel)
show_manager.ren.add(text)
show_manager.ren.add(line_slider)
show_manager.ren.add(disk_slider)
show_manager.ren.add(file_select_menu)
show_manager.ren.reset_camera()
show_manager.ren.reset_clipping_range()
show_manager.ren.azimuth(30)

# Uncomment this to start the visualisation
# show_manager.start()

window.record(show_manager.ren, size=current_size, out_path="viz_ui.png")

"""
.. figure:: viz_ui.png
   :align: center

   **User interface example**.
"""
