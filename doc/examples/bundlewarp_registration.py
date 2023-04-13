"""
============================================
Nonrigid Bundle Registration with BundleWarp
============================================

This example explains how you can nonlinearly register two bundles from two
different subjects directly in the space of streamlines [Chandio23]_, [Chandio20]_.

To show the concept, we will use two pre-saved uncinate fasciculus bundles. The
algorithm used here is called BundleWarp, streamline-based nonlinear
registration of white matter tracts [Chandio23]_.

"""

from dipy.viz import window, actor
from dipy.io.streamline import load_trk
from dipy.align.streamwarp import (bundlewarp, bundlewarp_vector_filed)
from dipy.tracking.streamline import (set_number_of_points, unlist_streamlines,
                                      Streamlines)
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
import matplotlib.pyplot as plt
from time import time

"""
Let's download and loaf two uncinate fasciculus bundles in the left hemisphere
of the brain (UF_L) available here:
https://figshare.com/articles/dataset/Test_Bundles_for_DIPY/22557733
"""

uf_subj1 = load_trk("s_UF_L.trk", reference="same",
                    bbox_valid_check=False).streamlines
uf_subj2 = load_trk("m_UF_L.trk", reference="same",
                    bbox_valid_check=False).streamlines

"""
Let's resample the streamlines so that they both have the same number of points
per streamline. Here we will use 20 points.
"""

static = Streamlines(set_number_of_points(uf_subj1, 20))
moving = Streamlines(set_number_of_points(uf_subj2, 20))

"""
We call ``uf_subj2`` a moving bundle as it will be nonlinearly aligned with
``uf_subj1`` (static) bundle. Here is how this is done.
"""

"""
Here we define utility functions for visualization.
"""

def viz_bundles(b1, b2, fname, c1=(1,0,0), c2=(0,1,0), interactive=False):

    ren = window.Scene()
    ren.SetBackground(1, 1, 1)

    actor1 = actor.line(b1, colors=c1)
    actor1.GetProperty().SetEdgeVisibility(1)
    actor1.GetProperty().SetRenderLinesAsTubes(1)
    actor1.GetProperty().SetLineWidth(6)
    actor1.GetProperty().SetOpacity(1)
    actor1.RotateX(-70)
    actor1.RotateZ(90)

    ren.add(actor1)

    actor2 = actor.line(b2, colors=c2)
    actor2.GetProperty().SetEdgeVisibility(1)
    actor2.GetProperty().SetRenderLinesAsTubes(1)
    actor2.GetProperty().SetLineWidth(6)
    actor2.GetProperty().SetOpacity(1)
    actor2.RotateX(-70)
    actor2.RotateZ(90)

    ren.add(actor2)

    if interactive:
        window.show(ren)

    window.record(ren, n_frames=1, out_path=fname, size=(1200, 1200))
    im = plt.imread(fname)
    plt.figure(figsize=(10, 10))
    plt.imshow(im)


def viz_bundle(b1, fname, c1=None, interactive=False):

    ren = window.Scene()
    ren.SetBackground(1, 1, 1)

    actor1 = actor.line(b1, colors=c1)
    actor1.GetProperty().SetEdgeVisibility(1)
    actor1.GetProperty().SetRenderLinesAsTubes(1)
    actor1.GetProperty().SetLineWidth(6)
    actor1.GetProperty().SetOpacity(1)
    actor1.RotateX(-70)
    actor1.RotateZ(90)

    ren.add(actor1)

    if interactive:
        window.show(ren)

    window.record(ren, n_frames=1, out_path=fname, size=(1200, 1200))
    im = plt.imread(fname)
    plt.figure(figsize=(10, 10))
    plt.imshow(im)


def viz_vector_field(points_aligned, directions, colors, offsets, fname,
                     bundle=None, interactive=False):

    scene = window.Scene()
    scene.SetBackground(1., 1, 1)
    arrows = actor.arrow(points_aligned, directions, colors, offsets)
    arrows.RotateX(-70)
    arrows.RotateZ(90)
    scene.add(arrows)

    if bundle:
        actor1 = actor.line(bundle, colors=(0, 0, 1))
        scene.add(actor1)

    if interactive:
        window.show(scene)

    window.record(scene, n_frames=1, out_path=fname, size=(1200, 1200))
    im = plt.imread(fname)
    plt.figure(figsize=(10, 10))
    plt.imshow(im)


def viz_displacement_mag(bundle, offsets, fname, interactive=False):

    scene = window.Scene()
    hue = (0.1, 0.9)
    hue = (0.9, 0.3)
    saturation = (0.5, 1)
    scene.background((1, 1, 1))
    lut_cmap = actor.colormap_lookup_table(
        scale_range=(offsets.min(), offsets.max()),
        hue_range=hue,
        saturation_range=saturation)

    stream_actor = actor.line(bundle, offsets, linewidth=7,
                              lookup_colormap=lut_cmap)

    stream_actor.RotateX(-70)
    stream_actor.RotateZ(90)

    scene.add(stream_actor)
    bar = actor.scalar_bar(lut_cmap)

    scene.add(bar)

    if interactive:
        window.show(scene)

    window.record(scene, n_frames=1, out_path=fname, size=(2000, 1500))
    im = plt.imread(fname)
    plt.figure(figsize=(10, 10))
    plt.imshow(im)


"""
Let's visualize static bundle in red and moving in green before registration.
"""

viz_bundles(static, moving, fname="static_and_moving.png")


"""
BundleWarp method provides a unique ability to either partially or fully deform
a moving bundle by the use of a single regularization parameter alpha.
alpha controls the trade-off between regularizing the deformation and having
points match very closely. The lower the value of alpha, the more closely the
bundles would match.

Let's partially deform bundle by setting alpha=0.5.
"""

start = time()
deformed_bundle, moving_aligned, distances, match_pairs, warp_map = bundlewarp(
                               static, moving, alpha=0.5, beta=20, max_iter=15)
end = time()

print("time taken by BundleWarp registration in seconds = ", end-start)

"""
Let's visualize static bundle in red and moved (warped) in green. Note: You can
set interactive=True in visualization functions throughout this tutorial if you
prefer to get interactive visualization window.
"""

viz_bundles(static, deformed_bundle,
            fname="static_and_partially_deformed.png")

"""
Let's visualize linearly moved bundle in blue and nonlinearly moved bundle in
green to see BundleWarp registration improvement over linear SLR registration.
"""

viz_bundles(moving_aligned, deformed_bundle,
            fname="linearly_and_nonlinearly_moved.png", c1=(0, 0, 1))

"""
Now, let's visualize deformation vector field generated by BundleWarp.
This shows us visually where and how much and in what directions deformations
were added by BundleWarp.
"""

offsets, directions, colors = bundlewarp_vector_filed(moving_aligned,
                                                      deformed_bundle)

points_aligned, _ = unlist_streamlines(moving_aligned)

"""
Visualizing just the vector field.
"""

fname = "partially_vectorfield.png"
viz_vector_field(points_aligned, directions, colors, offsets, fname)

"""
Let's visualize vector field over linearly moved bundle.This will show how much
deformations were introduced after linear registration.
"""

fname = "partially_vectorfield_over_linearly_moved.png"
viz_vector_field(points_aligned, directions, colors, offsets, fname,
                 moving_aligned)

"""
We can also visualize the magnitude of deformations mapped over moved (warped)
bundle. It shows which streamlines were deformed the most.
"""

fname = "partially_deformation_magnitude_over_nonlinearly_moved.png"
viz_displacement_mag(deformed_bundle, offsets, fname, interactive=False)

"""
Saving partially warped bundle.
"""

new_tractogram = StatefulTractogram(deformed_bundle, "m_UF_L.trk", Space.RASMM)
save_tractogram(new_tractogram, "partially_deformed_bundle.trk",
                bbox_valid_check=False)


"""
Let's fully deform the moving bundle by setting alpha <= 0.01

We will use MDF distances computed and returned by previous run of BundleWarp
method. This will save computation time.

"""

start = time()
deformed_bundle2, moving_aligned, distances, match_pairs, warp_map = bundlewarp(
        static, moving, dist=distances, alpha=0.001, beta=20, precomputed=True)
end = time()

print("time taken by BundleWarp registration in seconds = ", end-start)

"""
Let's visualize static bundle in red and moved (completely warped) in green.
"""

viz_bundles(static, deformed_bundle2,
            fname="static_and_fully_deformed.png")

"""
Now, let's visualize the deformation vector field generated by BundleWarp.
This shows us visually where and how much and in what directions deformations
were added by BundleWarp to perfectly warp moving bundle to look like static.
"""

offsets, directions, colors = bundlewarp_vector_filed(moving_aligned,
                                                      deformed_bundle2)

points_aligned, _ = unlist_streamlines(moving_aligned)

"""
Visualizing just the vector field.
"""

fname = "fully_vectorfield.png"
viz_vector_field(points_aligned, directions, colors, offsets, fname)

"""
Let's visualize vector field over linearly moved bundle. This will show how
much deformations were introduced after linear registration by fully deforming
the moving bundle.
"""

fname = "fully_vectorfield_over_linearly_moved.png"
viz_vector_field(points_aligned, directions, colors, offsets, fname,
                 moving_aligned)

"""
Let's visualize the magnitude of deformations mapped over moved
(completely warped) bundle. It shows which streamlines were deformed the most.
"""

fname = "fully_deformation_magnitude_over_nonlinearly_moved.png"
viz_displacement_mag(deformed_bundle2, offsets, fname, interactive=False)

"""
Saving fully warped bundle.
"""

new_tractogram = StatefulTractogram(deformed_bundle2, "m_UF_L.trk",
                                    Space.RASMM)
save_tractogram(new_tractogram, "fully_deformed_bundle.trk",
                bbox_valid_check=False)
