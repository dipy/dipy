"""
=============================
Groupwise Bundle Registration
=============================

This example explains how to coregister a set of bundles to a common space that
is not biased by any specific bundle. This is useful when we want to align all
the bundles but do not have a target reference space defined by an atlas.

How it works
============

The bundle groupwise registration framework in DIPY relies on streamline linear
registration (SLR) and an iterative process.

In each iteration, bundles are shuffled and matched in pairs. Then, each pair
of bundles are simultaneously moved to a common space in between both.

After all pairs have been aligned, a group distance metric is computed as the
mean pairwise distance between all bundle pairs. With each iteration, bundles
get closer to each other and the group distance decreases.

When the reduction in the group distance reaches a tolerance level the process
ends.

To reduce computational time, by default both registration and distance
computation are performed on streamline centroids (obtained with Quickbundles).

Example
=======

We start by importing and creating the necessary functions:
"""

import os
import zipfile
from dipy.align.streamlinear import groupwise_slr
from dipy.data import get_fnames
from dipy.io.streamline import load_tractogram
from dipy.viz import window, actor

import logging
logging.basicConfig(level=logging.INFO)

colors = [[0.91, 0.26, 0.35], [0.99, 0.50, 0.38], [0.99, 0.88, 0.57],
          [0.69, 0.85, 0.64], [0.51, 0.51, 0.63]]


def show_bundles(bundles, colors, show=True, fname=None):

    scene = window.Scene()
    scene.SetBackground(1., 1, 1)
    for (i, bundle) in enumerate(bundles):
        lines_actor = actor.streamtube(bundle, colors[i], linewidth=0.3)
        lines_actor.RotateX(-90)
        lines_actor.RotateZ(90)
        scene.add(lines_actor)
    if show:
        window.show(scene)
    if fname is not None:
        window.record(scene, n_frames=1, out_path=fname, size=(900, 900))


"""
To run groupwise registration we need to have our input bundles stored in a
list.

Here we load 5 left arcuate fasciculi and store them in a list.
"""

example_tracts = get_fnames('minimal_bundles')
subjects = ['sub_1', 'sub_2', 'sub_3', 'sub_4', 'sub_5']
in_dir = os.getcwd()

with zipfile.ZipFile(example_tracts, 'r') as zip_ref:
    zip_ref.extractall(in_dir)

bundles = []
for sub in subjects:
    file = os.path.join(in_dir, sub, 'AF_L.trk')
    bundle_obj = load_tractogram(file, 'same', bbox_valid_check=False)
    bundles.append(bundle_obj.streamlines)

"""
Let's now visualize our input bundles:
"""

show_bundles(bundles, colors, False, 'before_group_registration.png')

"""
.. figure:: before_group_registration.png
   :align: center

   Bundles before registration.

They are in native space and, therefore, not aligned.

Now running groupwise registration is as simple as:
"""

bundles_reg, aff, d = groupwise_slr(bundles, verbose=True)

"""
Finally, we visualize the registered bundles to confirm that they are now in a
common space:
"""

show_bundles(bundles_reg, colors, False, 'after_group_registration.png')

"""
.. figure:: after_group_registration.png
   :align: center

   Bundles after registration.

Extended capabilities
=====================

In addition to the registered bundles, `groupwise_slr` also returns a list with
the individual transformation matrices as well as the pairwise distances
computed in each iteration.

By changing the input arguments the user can modify the transformation (up to
affine), the number of maximum number of streamlines per bundle, the level of
clustering, or the tolerance of the method.
"""
