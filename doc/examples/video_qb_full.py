from __future__ import division

import itertools
from dipy.align.streamlinear import StreamlineLinearRegistration
from dipy.tracking.streamline import (set_number_of_points,
                                      select_random_set_of_streamlines,
                                      unlist_streamlines,
                                      transform_streamlines)
from dipy.viz import actor, window, utils, fvtk
from dipy.data.fetcher import fetch_bundles_2_subjects, read_bundles_2_subjects
import numpy as np
from dipy.core.geometry import rodrigues_axis_rotation
from dipy.segment.clustering import QuickBundles, QuickBundlesOnline
from dipy.viz.colormap import distinguishable_colormap
import vtk
import nibabel as nib

qb = QuickBundles(threshold=30)

# Script
fetch_bundles_2_subjects()

bundle_names = ['af.left', 'af.right', 'cc_1', 'cc_2', 'cc_3', 'cc_4', 'cc_5', 'cc_6', 'cc_7', 'cg.left', 'cg.right', 'cst.left', 'cst.right', 'ifof.left', 'ifof.right', 'ilf.left', 'ilf.right', 'mdlf.left', 'mdlf.right', 'slf1.left', 'slf1.right', 'slf2.left', 'slf2.right', 'slf_3.left', 'slf_3.right', 'uf.left', 'uf.right']
subj1 = read_bundles_2_subjects('subj_1', ['fa', 't1'], bundle_names)

rng = np.random.RandomState(42)
tractogram = nib.streamlines.Tractogram(subj1[bundle_names[0]])
for bundle_name in bundle_names[1:]:
    tractogram.streamlines.extend(subj1[bundle_name])

indices = range(len(tractogram))
rng.shuffle(indices)
streamlines = tractogram.streamlines

center = streamlines._data.mean()

# Perform clustering to know home many clusters we will get.
clusters = qb.cluster(streamlines)
# This will be use to display individual cluster on the right panel.
clusters_as_array_sequence = [nib.streamlines.ArraySequence(c)
                              for c in clusters]

bg = (0.2, 0.2, 0.2)
# Create actors
colormap = list(itertools.islice(distinguishable_colormap(bg=bg, exclude=[(1, 1, 1)]), len(clusters)))
# colormap = colormap + [(1, 1, 1), (0, 0, 0)]

global streamlines_color
streamlines_color = (len(colormap)) * np.ones(len(streamlines), dtype="f4")
brain_actor = actor.line(streamlines,
                         colors=streamlines_color,
                         linewidth=1.5)


cluster_actors = []
cluster_centroids = []
for c, cluster_color in zip(clusters, colormap):

    cluster_actor = actor.line(c, colors=cluster_color,
                               linewidth=1.5, lod=False)
    cluster_actors.append(cluster_actor)

    centroid_actor = actor.streamtube([c.centroid], colors=cluster_color,
                                      linewidth=0.2+np.log(len(c))/2)
    cluster_centroids.append(centroid_actor)
# Create renderers
# Main renderder (used for the interaction)

ren_main = window.Renderer()
ren_main.background(bg)
show_m = window.ShowManager(ren_main,
                            size=(1280, 720), interactor_style="trackball")
show_m.initialize()
show_m.window.SetNumberOfLayers(2)
ren_main.SetLayer(1)
ren_main.InteractiveOff()

# Left renderer that contains the brain being clustered.
ren_brain = window.Renderer()
show_m.window.AddRenderer(ren_brain)
ren_brain.background(bg)
ren_brain.SetViewport(0, 0, 0.5, 1)
ren_brain.add(*cluster_actors)
ren_brain.reset_camera_tight()

# Right renderer that contains the centroids.
ren_centroids = window.Renderer()
show_m.window.AddRenderer(ren_centroids)
ren_centroids.background(bg)
ren_centroids.SetViewport(0.5, 0, 1, 1)
ren_centroids.add(*cluster_centroids)
ren_centroids.SetActiveCamera(ren_brain.GetActiveCamera())  # Sync camera with the left one.

# Right renderers: one per cluster.

global cnt, time, stamp_time
repeat_time = 10
cnt = 0
time = 0
stamp_time = 0

global streamline_idx
streamline_idx = 0

position, focal_point, view_up, _, _ = utils.auto_camera(brain_actor,
                                                         10, 'max')
ren_brain.reset_camera_tight()
view_up = (0, 0., 1)
ren_brain.set_camera(position, focal_point, view_up)
ren_brain.zoom(1.5)
ren_brain.reset_clipping_range()


def main_event():
    ren_brain.azimuth(1)




from dipy.viz import timeline

global tm
tm = timeline.TimeLineManager(show_m, [],
                              'video_qb_full.avi')

t = 0
tm.add_sub(t, ['title'], ['Online QuickBundles'])

tm.add_event(
    t, 10,
    [main_event],
    [[]])

def timer_callback(obj, event):
    global tm
    tm.execute()
    #main_event()
    #show_m.render()

show_m.add_timer_callback(True, repeat_time, timer_callback)
show_m.render()
show_m.start()

del tm
del show_m
