"""
==================================
Advanced interactive visualization
==================================

In DIPY we created a thin interface to access many of the capabilities
available in the Visualization Toolkit framework (VTK) but tailored to the
needs of structural and diffusion imaging. Initially the 3D visualization
module was named ``fvtk``, meaning functions using vtk. This is still available
for backwards compatibility but now there is a more comprehensive way to access
the main functions using the following modules.
"""

import numpy as np
from dipy.viz import actor, window, widget

"""
In ``window`` we have all the objects that connect what needs to be rendered
to the display or the disk e.g. for saving screenshots. So, there you will find
key objects and functions like the ``Renderer`` class which holds and provides
access to all the actors and the ``show`` function which displays what is
in the renderer on a window. Also, this module provides access to functions
for opening/saving dialogs and printing screenshots (see ``snapshot``).

In the ``actor`` module we can find all the different primitives e.g.
streamtubes, lines, image slices etc.

In the ``widget`` we have some other objects which allow to add buttons
and sliders and these interact both with windows and actors. Because of this
they need input from the operating system so they can process events.

So, let's get started. In this tutorial, we will visualize some bundles
together with FA or T1. We will be able to change the slices using
a ``slider`` widget.

First we need to fetch and load some datasets.
"""

from dipy.data.fetcher import fetch_bundles_2_subjects, read_bundles_2_subjects

fetch_bundles_2_subjects()

"""
The following function outputs a dictionary with the required bundles e.g. af
left and maps, e.g. FA for a specific subject.
"""

res = read_bundles_2_subjects('subj_1', ['t1', 'fa'],
                              ['af.left', 'cst.right', 'cc_1'])

"""
We will use 3 bundles, FA and the affine transformation that brings the voxel
coordinates to world coordinates (RAS 1mm).
"""

af_streamlines = res['af.left']
cst_streamlines = res['cst.right']
cc_streamlines = res['cc_1']
data = res['fa']
shape = data.shape
affine = res['affine']

"""
With our current design it is easy to decide in which space you want the
streamlines and slices to appear. The default we have here is to appear in
world coordinates (RAS 1mm).
"""

world_coords = True

"""
If we want to see the objects in native space we need to make sure that all
objects which are currently in world coordinates are transformed back to
native space using the inverse of the affine.
"""

if not world_coords:
    from dipy.tracking.streamline import transform_streamlines
    af_streamlines = transform_streamlines(af_streamlines,
                                           np.linalg.inv(affine))
    cst_streamlines = transform_streamlines(cst_streamlines,
                                            np.linalg.inv(affine))
    cc_streamlines = transform_streamlines(cc_streamlines,
                                           np.linalg.inv(affine))

"""
Now we create, a ``Renderer`` object and add the streamlines using the ``line``
function and an image plane using the ``slice`` function.
"""

ren = window.Renderer()
af_actor = actor.line(af_streamlines)
cst_actor = actor.line(cst_streamlines)
cc_actor = actor.line(cc_streamlines)

if not world_coords:
    image_actor = actor.slicer(data, affine=np.eye(4))
else:
    image_actor = actor.slicer(data, affine)

"""
For fun let's change also the opacity of the slicer
"""

slicer_opacity = .6
image_actor.opacity(slicer_opacity)

"""
Connect the actors with the Renderer.
"""

ren.add(af_actor)
ren.add(cst_actor)
ren.add(cc_actor)
ren.add(image_actor)

"""
Now we would like to change the position of the ``image_actor`` using a slider.
The sliders are widgets which require access to different areas of the
visualization pipeline and therefore we don't recommend using them with
``show``. The more appropriate way is to use them with the ``ShowManager``
object which allows accessing the pipeline in different areas. Here is how:
"""

show_m = window.ShowManager(ren, size=(1200, 900))
show_m.initialize()

"""
After we have initialized the ``ShowManager`` we can go ahead and create a
callback which will be given to the ``slider`` function.
"""


def change_slice(obj, event):
    z = int(np.round(obj.get_value()))
    image_actor.display_extent(0, shape[0] - 1,
                               0, shape[1] - 1, z, z)

slider = widget.slider(show_m.iren, show_m.ren,
                       callback=change_slice,
                       min_value=0,
                       max_value=shape[2] - 1,
                       value=shape[2] / 2,
                       label="Move slice",
                       right_normalized_pos=(.98, 0.6),
                       size=(120, 0), label_format="%0.lf",
                       color=(1., 1., 1.),
                       selected_color=(0.86, 0.33, 1.))

"""
Then, we can render all the widget and everything else in the screen and
start the interaction using ``show_m.start()``.
"""

# show_m.render()
# show_m.start()

"""
However, if you change the window size, the slider will not update its position
properly. The solution to this issue is to update the position of the slider
using its ``place`` method every time the window size changes.
"""

global size
size = ren.GetSize()


def win_callback(obj, event):
    global size
    if size != obj.GetSize():

        slider.place(ren)
        size = obj.GetSize()

"""
Here, we can add a callback that is triggered everytime we press 'p' on a
slicer and gives us the position and actual value.
"""

resampled = image_actor.get_data()

status = actor.text('', font_size=14, bold=True)
ren.add(status)


def pick_callback(obj, event):

    ijk = obj.GetPointIJK()
    i, j, k = ijk
    v1, v2, v3, v4 = resampled[i, j, k]
    msg = ">>> Position (%d, %d, %d) value %d" % (i, j, k, v1)
    status.message(msg)


show_m.initialize()

ren.zoom(1.5)
ren.reset_clipping_range()

"""
Please uncomment the following lines so that you can interact with
the available 3D and 2D objects.
"""

show_m.render()

bit_rate = 10 * np.prod(show_m.window.GetSize())
print(bit_rate)
bit_rate_tol = 10 * np.prod(show_m.window.GetSize()) * 3
print(bit_rate_tol)
mw = window.MovieWriter('new.avi', show_m.window,
                        bit_rate=bit_rate, bit_rate_tol=bit_rate_tol,
                        frame_rate=None,
                        compression=True, compression_quality=None)
mw.start()

def timer_callback(obj, event):

    ren.azimuth(.1)
    show_m.render()
    mw.write()


show_m.add_window_callback(win_callback)
show_m.add_picker_callback(pick_callback)
show_m.add_timer_callback(True, 20, timer_callback)

show_m.render()
show_m.start()


del mw

# window.record(ren, out_path='bundles_and_a_slice.png', size=(1200, 900))

"""
.. figure:: bundles_and_a_slice.png
   :align: center

   **A few bundles with interactive slicing**.
"""

del show_m
