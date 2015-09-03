"""
=====================================
Create a minimalistic user interface
=====================================

DIPY allows to create a minimalistic interface using widgets.

In this example we will create: a) two parallel steamtubes, b) add some buttons
which will change the opacity of these tubes and c) move the streamtubes using
a slider.
"""

import numpy as np
from dipy.viz import window, actor, widget
from dipy.data import fetch_viz_icons, read_viz_icons

"""
First, we add the streamtubes to the
"""

renderer = window.Renderer()

lines = [np.array([[-1, 0, 0.], [1, 0, 0.]]),
         np.array([[-1, 1, 0.], [1, 1, 0.]])]
colors = np.array([[1., 0., 0.], [0., .5, 0.]])
stream_actor = actor.streamtube(lines, colors, linewidth=0.3)

renderer.add(stream_actor)

"""
The ``ShowManager`` allows to break the visualization process in steps so that
the widgets can be added and updated properly.
"""

show_manager = window.ShowManager(renderer, size=(800, 800),
                                  order_transparent=True)

show_manager.initialize()

"""
Next we add the widgets and their callbacks.
"""

global opacity
opacity = 1.


def button_plus_callback(obj, event):
    print('+ pressed')
    global opacity
    if opacity < 1:
        opacity += 0.1
    stream_actor.GetProperty().SetOpacity(opacity)


def button_minus_callback(obj, event):
    print('- pressed')
    global opacity
    if opacity > 0:
        opacity -= 0.1
    stream_actor.GetProperty().SetOpacity(opacity)


"""
We need to download some icons to create a face for our buttons ...
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
And then we create the slider.
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

global size
size = renderer.size()

"""
This callback is used to update the buttons/sliders' position so they can stay
on the correct side of the window when the window is being resized.
"""


def win_callback(obj, event):
    global size
    if size != obj.GetSize():

        button_plus.place(renderer)
        button_minus.place(renderer)
        slider.place(renderer)
        size = obj.GetSize()

# you can also register any callback in a vtk way like this
# show_manager.window.AddObserver(vtk.vtkCommand.ModifiedEvent,
#                                 win_callback)

renderer.zoom(0.7)
renderer.roll(10.)

"""
Uncomment the following line to start the interaction.
"""

# show_manager.add_window_callback(win_callback)
# show_manager.render()
# show_manager.start()

renderer.reset_clipping_range()

window.record(renderer, out_path='mini_ui.png', size=(800, 800))

del show_manager

"""
.. figure:: mini_ui.png
   :align: center

   **A minimalistic user interface**.
"""
