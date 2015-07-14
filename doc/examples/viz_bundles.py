"""
========================================
Visualize bundles and metrics on bundles
========================================

Frist, let's download some available datasets. Here we are using a dataset
which provides metrics and bundles.
"""

import numpy as np
from dipy.viz import window, actor
from dipy.data import fetch_bundles_2_subjects, read_bundles_2_subjects
from dipy.tracking.streamline import transform_streamlines

fetch_bundles_2_subjects()
dix = read_bundles_2_subjects(subj_id='subj_1', metrics=['fa'],
                              bundles=['cg.left', 'cst.right'])

"""
Store franctional anisotropy.
"""

fa = dix['fa']

"""
Store grid to world transformation matrix.
"""

affine = dix['affine']

"""
Store the cingulum bundle. A bundle is a set of streamlines.
"""

bundle = dix['cg.left']

"""
It happened that this bundle is in world coordinates and therefore we need to
transform in native image coordinates so that it is in the same coordinate
space as the ``fa``.
"""

bundle_img = transform_streamlines(bundle, np.linalg.inv(affine))

"""
Show every streamline with an orientation color
===============================================

This is the default option when you are using ``line`` or ``streamtube``.
"""

renderer = window.Renderer()

stream_actor = actor.line(bundle_img)

renderer.set_camera(position=(-176.42, 118.52, 128.20),
                    focal_point=(113.30, 128.31, 76.56),
                    view_up=(0.18, 0.00, 0.98))

renderer.add(stream_actor)

# window.show(renderer, size=(600, 600), reset_camera=False)
window.snapshot(renderer, 'bundle1.png', size=(600, 600))

"""
.. figure:: bundle1.png
   :align: center

   **One orientation color for every streamline**.

You may wonder how we knew how to set the camera. This is very easy. You just
need to run ``window.show`` once see how you want to see the object and then
close the window and call the ``camera_info`` method which prints the position,
focal point and view up vectors of the camera.
"""

renderer.camera_info()

"""
Show every point with a value from a metric with default colormap
=================================================================

Here we will need to input the ``fa`` map in ``streamtube`` or ``line``.
"""

renderer.clear()
stream_actor2 = actor.line(bundle_img, fa, linewidth=0.1)

"""
We can also show the scalar bar.
"""

bar = actor.scalar_bar()

renderer.add(stream_actor2)
renderer.add(bar)

# window.show(renderer, size=(600, 600), reset_camera=False)
window.snapshot(renderer, 'bundle2.png', size=(600, 600))

"""
.. figure:: bundle2.png
   :align: center

   **Every point with a color from FA**.

Show every point with a value from a metric with your colormap
==============================================================

Here we will need to input the ``fa`` map in ``streamtube`` or ``
"""

renderer.clear()

hue = [0.0, 0.0]  # red only
saturation = [0.0, 1.0]  # white to red

lut_cmap = actor.colormap_lookup_table(hue_range=hue,
                                       saturation_range=saturation)

stream_actor3 = actor.line(bundle_img, fa, linewidth=0.1,
                           lookup_colormap=lut_cmap)
bar2 = actor.scalar_bar(lut_cmap)

renderer.add(stream_actor3)
renderer.add(bar2)

# window.show(renderer, size=(600, 600), reset_camera=False)
window.snapshot(renderer, 'bundle3.png', size=(600, 600))

"""
.. figure:: bundle3.png
   :align: center

   **Every point with a color from FA using a non default colomap**.
"""


