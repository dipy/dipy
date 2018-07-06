# -*- coding: utf-8 -*-
"""
===============
User Interfaces
===============

This example shows how to use the UI API.

"""
import os

from dipy.data import read_viz_icons, fetch_viz_icons

from dipy.viz import ui, window

"""
Range Slider
=======

"""

range_slider_example = ui.RangeSlider(range_precision=2, value_precision=3,
                                      shape="square")

"""
File Menu
=======
"""

menu = ui.FileMenu2D(extensions=["*"], directory_path=os.getcwd(),
                     size=(500, 500))

"""
Adding Elements to the ShowManager
==================================

Once all elements have been initialised, they have
to be added to the show manager in the following manner.
"""

current_size = (600, 600)
show_manager = window.ShowManager(size=current_size, title="DIPY UI Example")

show_manager.ren.add(range_slider_example)
show_manager.ren.add(menu)
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
