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


qb = QuickBundles(threshold=20)
qbo = QuickBundlesOnline(threshold=20)


# Script
fetch_bundles_2_subjects()

bundle_names = ['af.left', 'af.right', 'cc_1', 'cc_2', 'cc_3', 'cc_4', 'cc_5', 'cc_6', 'cc_7', 'cg.left', 'cg.right', 'cst.left', 'cst.right', 'ifof.left', 'ifof.right', 'ilf.left', 'ilf.right', 'mdlf.left', 'mdlf.right', 'slf1.left', 'slf1.right', 'slf2.left', 'slf2.right', 'slf_3.left', 'slf_3.right', 'uf.left', 'uf.right']
subj1 = read_bundles_2_subjects('subj_1', ['fa', 't1'], bundle_names)

rng = np.random.RandomState(42)
tractogram = nib.streamlines.Tractogram(subj1[bundle_names[0]])
for bundle_name in bundle_names[1:]:
    tractogram.streamlines.extend(subj1[bundle_name])

rot_axis = np.array([1, 0, 0], dtype=np.float32)
M_rotation = np.eye(4)
M_rotation[:3, :3] = rodrigues_axis_rotation(rot_axis, 270.).astype(np.float32)
rot_axis = np.array([0, 1, 0], dtype=np.float32)
M_rotation2 = np.eye(4)
M_rotation2[:3, :3] = rodrigues_axis_rotation(rot_axis, 270.).astype(np.float32)
tractogram.apply_affine(np.dot(M_rotation2, M_rotation))

indices = range(len(tractogram))
rng.shuffle(indices)
streamlines = tractogram.streamlines[indices[:1000]].copy()

# Perform clustering to know home many clusters we will get.
clusters = qb.cluster(streamlines)
# This will be use to display individual cluster on the right panel.
clusters_as_array_sequence = [nib.streamlines.ArraySequence(c) for c in clusters]

bg = (0, 0, 0)
# Create actors
colormap = list(itertools.islice(distinguishable_colormap(bg=bg, exclude=[(1, 1, 1)]), len(clusters)))
colormap = colormap + [(1, 1, 1), (0, 0, 0)]

lut = vtk.vtkLookupTable()
lut.SetNumberOfTableValues(len(colormap))
lut.Build()
for i, color in enumerate(colormap):
    lut.SetTableValue(i, tuple(color) + (1,))

lut.SetTableValue(len(colormap)-2, (1, 1, 1, 0.25))  # Semi-invisible
lut.SetTableValue(len(colormap)-1, (0, 0, 0, 0.))  # Invisible
lut.SetTableRange(0, len(colormap)-1)

global streamlines_color
streamlines_color = (len(colormap)-2)*np.ones(len(streamlines), dtype="float32")  # Semi-invisible
brain_actor = actor.line(streamlines, colors=streamlines_color, lookup_colormap=lut, linewidth=1.5)

cluster_actors = []
for c in clusters:
    cluster_color = (len(colormap)-1)*np.ones(len(c), dtype="float32")  # Invisible
    cluster_actor = actor.line(c, colors=cluster_color, lookup_colormap=lut, linewidth=1.5)
    cluster_actors.append(cluster_actor)

# Create renderers
# Main renderder (used for the interaction)
global screen_size
screen_size = (0, 0)
ren_main = window.Renderer()
ren_main.background(bg)
show_m = window.ShowManager(ren_main, size=(1280, 720), interactor_style="trackball")
show_m.initialize()
show_m.window.SetNumberOfLayers(2)
ren_main.SetLayer(1)
ren_main.InteractiveOff()

# Left renderer that contains the brain.
ren_brain = window.Renderer()
show_m.window.AddRenderer(ren_brain)
ren_brain.background(bg)
ren_brain.SetViewport(0, 0, 0.5, 1)
ren_brain.add(brain_actor)
ren_brain.reset_camera_tight()

# Right renderers: one per cluster.
grid_cell_positions = utils.get_grid_cells_position([ren_brain.GetSize()]*len(clusters), aspect_ratio=9/16.)
grid_dim = (len(np.unique(grid_cell_positions[:, 0])), len(np.unique(grid_cell_positions[:, 1])))

cpt = 0
grid_renderers = []
for y in range(grid_dim[1])[::-1]:
    if cpt >= len(clusters):
        break

    for x in range(grid_dim[0]):
        if cpt >= len(clusters):
            break

        ren = window.Renderer()
        show_m.window.AddRenderer(ren)
        ren.background(bg)
        ren.SetViewport(0.5+x/(grid_dim[0]/0.5), y/grid_dim[1], 0.5+(x+1)/(grid_dim[0]/0.5), (y+1)/grid_dim[1])
        ren.add(cluster_actors[cpt])
        ren.SetActiveCamera(ren_brain.GetActiveCamera())  # Sync camera with the left one.
        cpt += 1


global cnt, time, stamp_time
repeat_time = 10
cnt = 0
time = 0
stamp_time = 0

global streamline_idx
streamline_idx = 0

# Prepare placeholders for the centroids' streamtube
centroid_actors = [actor.line(np.array([[0, 0, 0]]), colors=(0, 0, 0))] * len(clusters)


def timer_callback(obj, event):
    global cnt, time, stamp_time, streamline_idx

    print "Frame #{:,}".format(cnt)

    if streamline_idx < len(streamlines):
        cluster_id = qbo.cluster(streamlines[streamline_idx], streamline_idx)

        # Display updated centroid
        ren_brain.RemoveActor(centroid_actors[cluster_id])
        centroid_actors[cluster_id] = actor.streamtube([qbo.clusters[cluster_id].centroid], colormap[cluster_id], linewidth=0.2+np.log(len(qbo.clusters[cluster_id]))/2)
        ren_brain.add(centroid_actors[cluster_id])

        # Replace color
        scalars = brain_actor.GetMapper().GetInput().GetPointData().GetScalars()
        start = streamlines._offsets[streamline_idx]
        end = start + streamlines._lengths[streamline_idx]
        for i in range(start, end):
            scalars.SetValue(i, len(colormap)-1)  # Make streamlines invisible.

        scalars.Modified()

        # Show streamlines in the clusters view.
        scalars = cluster_actors[cluster_id].GetMapper().GetInput().GetPointData().GetScalars()
        start = clusters_as_array_sequence[cluster_id]._offsets[len(qbo.clusters[cluster_id])-1]
        end = start + clusters_as_array_sequence[cluster_id]._lengths[len(qbo.clusters[cluster_id])-1]
        for i in range(start, end):
            scalars.SetValue(i, cluster_id)  # Make streamlines visible.

        scalars.Modified()

        show_m.render()

        streamline_idx += 1

    cnt += 1

show_m.add_timer_callback(True, repeat_time, timer_callback)
show_m.render()
show_m.start()
