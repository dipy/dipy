# -*- coding: utf-8 -*-
"""
===============
User Interfaces
===============

This example shows how to use the UI API.

First, a bunch of imports.

"""
import os

from dipy.data import read_viz_icons, fetch_viz_icons

from dipy.viz import ui, window


"""
Buttons
=======

We first fetch the icons required for making the buttons.
"""

fetch_viz_icons()

"""
Add the icon filenames to a dict.
"""

icon_files = []
icon_files.append(('stop', read_viz_icons(fname='stop2.png')))
icon_files.append(('play', read_viz_icons(fname='play3.png')))
icon_files.append(('plus', read_viz_icons(fname='plus.png')))
icon_files.append(('cross', read_viz_icons(fname='cross.png')))

"""
Create buttons
"""

button_example = ui.Button2D(icon_fnames=icon_files)
second_button_example = ui.Button2D(icon_fnames=icon_files)

"""
Call the built in `next_icon` method via a callback that is
triggered on left click.
"""


def modify_button_callback(i_ren, obj, button):
    button.next_icon()
    i_ren.force_render()

second_button_example.on_left_mouse_button_pressed = modify_button_callback


"""
TextBox
=======
"""

text = ui.TextBox2D(height=3, width=10)

"""
Panel
=====

Simply create a panel and add elements to it.
"""

panel = ui.Panel2D(size=(300, 150), color=(1, 1, 1), align="right")
panel.center = (500, 400)
panel.add_element(button_example, (0.2, 0.2))
panel.add_element(second_button_example, (0.8, 0.6))
panel.add_element(text, (150, 50))

"""
Image Container
===============
"""

img = ui.ImageContainer2D(img_path=read_viz_icons(fname='home3.png'),
                          position=(500, 400))

"""
Rectangle2D
==========
"""

rect = ui.Rectangle2D(size=(200, 200), position=(400, 300), color=(1, 0, 1))

"""
Solid Disk
=========
"""

disk = ui.Disk2D(outer_radius=50, center=(500, 500), color=(1, 1, 0))

"""
Ring Disk
=========
"""

ring = ui.Disk2D(outer_radius=50, inner_radius=45, center=(500, 300),
                 color=(0, 1, 1))

"""
Cube actor
==========
"""


def cube_maker(color=(1, 1, 1), size=(0.2, 0.2, 0.2), center=(0, 0, 0)):
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


cube = cube_maker(color=(0, 0, 1), size=(20, 20, 20), center=(20, 0, 0))

"""
Add callbacks for moving the cube

"""


def translate_cube(slider):
    value = slider.value
    cube.SetPosition(value, 0, 0)


def rotate_cube(slider):
    angle = slider.value
    previous_angle = slider.previous_value
    rotation_angle = angle - previous_angle
    cube.RotateX(rotation_angle)


"""
Ring Slider
===========
"""

ring_slider = ui.RingSlider2D(center=(500, 500), initial_value=0,
                              text_template="{angle:5.1f}°")
ring_slider.on_change = rotate_cube

"""
Line Slider
===========
"""

line_slider = ui.LineSlider2D(
    center=(500, 200), initial_value=0, min_value=-10, max_value=10)
line_slider.on_change = translate_cube

"""
Range Slider
============
"""

range_slider = ui.RangeSlider(
    line_width=8, handle_side=25, range_slider_center=(550, 400),
    value_slider_center=(550, 300), length=250, min_value=0,
    max_value=10, font_size=18, range_precision=2, value_precision=4,
    shape="square")


"""
List of all elements used as examples

"""

examples = [[img], [panel], [rect], [disk, ring],
            [ring_slider, line_slider], [range_slider]]

"""
Function to hide all elements
"""


def hide_all_examples():
    for example in examples:
        for element in example:
            element.set_visibility(False)
    cube.SetVisibility(False)


hide_all_examples()

"""
The Menu
========
This is a listbox with each item corresponding to different elements.

"""

values = ["Image", "Panel, Textbox, Buttons", "Rectangle", "Disks",
          "Line and Ring Slider", "Range Slider"]
listbox = ui.ListBox2D(values=values, position=(10, 200), size=(300, 300),
                       multiselection=False)


def display_element():
    hide_all_examples()
    example = examples[values.index(listbox.selected[0])]
    for element in example:
        element.set_visibility(True)
    if values.index(listbox.selected[0]) == 4:
        cube.SetVisibility(True)


listbox.on_change = display_element


"""
Adding Elements to the ShowManager
==================================

Once all elements have been initialised, they have
to be added to the show manager in the following manner.
"""

current_size = (800, 800)
show_manager = window.ShowManager(size=current_size, title="DIPY UI Example")

show_manager.ren.add(listbox)
for example in examples:
    for element in example:
        show_manager.ren.add(element)
show_manager.ren.add(cube)
show_manager.ren.reset_camera()
show_manager.ren.set_camera(position=(0, 0, 200))
show_manager.ren.reset_clipping_range()
show_manager.ren.azimuth(30)

# Uncomment this to start the visualisation
show_manager.start()

window.record(show_manager.ren, size=current_size, out_path="viz_ui.png")

"""
.. figure:: viz_ui.png
   :align: center

   **User interface example**.
"""
