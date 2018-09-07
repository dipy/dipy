"""
===============
Using a timer
===============

This example shows how to create a simple animation using a timer callback.

We will use a sphere actor that generates many spheres of different colors,
radii and opacity. Then we will animate this actor by rotating and changing
global opacity levels from inside a user defined callback.

The timer will call this user defined callback every 200 milliseconds. The
application will exit after the callback has been called 100 times.
"""


import numpy as np
from dipy.viz import window, actor, ui
import itertools

xyz = 10 * np.random.rand(100, 3)
colors = np.random.rand(100, 4)
radii = np.random.rand(100) + 0.5

renderer = window.Renderer()

sphere_actor = actor.sphere(centers=xyz,
                            colors=colors,
                            radii=radii)

renderer.add(sphere_actor)

showm = window.ShowManager(renderer,
                           size=(900, 768), reset_camera=False,
                           order_transparent=True)

showm.initialize()

tb = ui.TextBlock2D(bold=True)

# use itertools to avoid global variables
counter = itertools.count()


def timer_callback(obj, event):
    cnt = next(counter)
    tb.message = "Let's count up to 100 and exit :" + str(cnt)
    showm.ren.azimuth(0.05 * cnt)
    sphere_actor.GetProperty().SetOpacity(cnt/100.)
    showm.render()
    if cnt == 100:
        showm.exit()


renderer.add(tb)

# Run every 200 milliseconds
showm.add_timer_callback(True, 200, timer_callback)

showm.start()

window.record(showm.ren, size=(900, 768), out_path="viz_timer.png")

"""
.. figure:: viz_timer.png
   :align: center

   **Showing 100 spheres of random radii and opacity levels**.
"""
