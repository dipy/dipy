"""
=====================================
Create a minimalistic user interface
=====================================

DIPY allows to create a minimalistic interface using widgets.

In this example we will create: a) two parallel steamtubes, b) add some buttons
which will change the opacity of these tubes and c) move the streamtubes using
a slider.
"""

from __future__ import print_function
import numpy as np
from dipy.viz import window, actor, widget
from dipy.data import fetch_viz_icons, read_viz_icons

"""
First, we add a couple of streamtubes to the Renderer
"""

renderer = window.Renderer()

lines = [np.array([[-1, 0, 0.], [1, 0, 0.]]),
         np.array([[-1, 1, 0.], [1, 1, 0.]])]
colors = np.array([[1., 0., 0.], [0., .5, 0.]])
stream_actor = actor.streamtube(lines, colors, linewidth=0.3, lod=False)
renderer.add(stream_actor)

"""
The ``ShowManager`` allows to break the visualization process in steps so that
the widgets can be added and updated properly.
"""

show_manager = window.ShowManager(renderer, size=(800, 800),
                                  order_transparent=True)

"""
Next we add the widgets and their callbacks.
"""


def button_plus_callback(obj, event):
    print('+ pressed')
    opacity = stream_actor.GetProperty().GetOpacity()
    if opacity < 1:
        stream_actor.GetProperty().SetOpacity(opacity + 0.1)


def button_minus_callback(obj, event):
    print('- pressed')

    opacity = stream_actor.GetProperty().GetOpacity()
    if opacity > 0:
        stream_actor.GetProperty().SetOpacity(opacity - 0.1)


"""
We need to download some icons to create a face for our buttons. We provide
some simple icons in this tutorial. But you of course you can use any PNG icon
you may want.
"""

fetch_viz_icons()
button_png_plus = read_viz_icons(fname='plus.png')

button_plus = widget.button(show_manager.iren,
                            show_manager.ren,
                            button_plus_callback,
                            button_png_plus, (.98, .9), (120, 50))

button_png_minus = read_viz_icons(fname='minus.png')
button_minus = widget.button(show_manager.iren,
                             show_manager.ren,
                             button_minus_callback,
                             button_png_minus, (.98, .9), (50, 50))


def move_lines(obj, event):
    stream_actor.SetPosition((obj.get_value(), 0, 0))

"""
Then we create the slider.
"""

slider = widget.slider(show_manager.iren, show_manager.ren,
                       callback=move_lines,
                       min_value=-1,
                       max_value=1,
                       value=0.,
                       label="X",
                       right_normalized_pos=(.98, 0.7),
                       size=(120, 0), label_format="%0.2lf",
                       color=(0.4, 0.4, 0.4),
                       selected_color=(0.2, 0.2, 0.2))

"""
And we add a simple clickable text overlay at the bottom left corner.
"""


def text_clicked(obj, event):
    print("Awesome!")

text = widget.text(show_manager.iren, show_manager.ren,
                   message="Powered by DIPY",
                   callback=text_clicked,
                   color=(1., .5, .0),
                   left_down_pos=(10, 5),
                   right_top_pos=(200, 35))

"""
Position the camera.
"""

renderer.zoom(0.7)
renderer.roll(10.)
renderer.reset_clipping_range()

"""
Uncomment the following lines to start the interaction.
"""

show_manager.initialize()
show_manager.render()
show_manager.start()


window.record(renderer, out_path='mini_ui.png', size=(800, 800),
              reset_camera=False)

del show_manager

"""
.. figure:: mini_ui.png
   :align: center

   **A minimalistic user interface**.
"""
