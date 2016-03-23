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
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import dist
from dipy.viz.colormap import distinguishable_colormap
import vtk
import nibabel as nib


qb = QuickBundles(threshold=20)


def replay_clustering(clusters):
    Ns = np.zeros(len(clusters), dtype=int)
    nb_points = 50
    centroids = [np.zeros((nb_points, 3), dtype=np.float32) for c in clusters]

    cluster_iters = map(lambda c: iter(c.indices), clusters)
    nexts = [next(cluster_iter) for cluster_iter in cluster_iters]

    nb_iters_done = 0
    while nb_iters_done < len(clusters):
        cluster_id = np.argmin(nexts)
        streamlines_id = nexts[cluster_id]

        # Check if we need to flip the streamlines first
        s = clusters.refdata[streamlines_id]
        if dist(qb.metric, clusters.refdata[streamlines_id][::-1], centroids[cluster_id]) < dist(qb.metric, clusters.refdata[streamlines_id], centroids[cluster_id]):
            s = clusters.refdata[streamlines_id][::-1]

        s = set_number_of_points(s, nb_points)
        centroids[cluster_id] = ((centroids[cluster_id]*Ns[cluster_id]) + s) / (Ns[cluster_id]+1)

        id_in_cluster = Ns[cluster_id]
        Ns[cluster_id] += 1

        yield streamlines_id, cluster_id, id_in_cluster, centroids[cluster_id], Ns[cluster_id]

        try:
            nexts[cluster_id] = next(cluster_iters[cluster_id])
        except StopIteration:
            # print "Cluster #{} exhausted".format(cluster_id)
            nb_iters_done += 1
            nexts[cluster_id] = np.inf


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

# Perform clustering
clusters = qb.cluster(streamlines)
clusters_as_array_sequence = [nib.streamlines.ArraySequence(c) for c in clusters]

bg = (0, 0, 0)
# Create actors
colormap = list(itertools.islice(distinguishable_colormap(bg=bg, exclude=[(1, 1, 1)]), len(clusters)))
colormap = [(1, 1, 1)] + colormap + [(0, 0, 0)]

lut = vtk.vtkLookupTable()
lut.SetNumberOfTableValues(len(colormap))
lut.Build()
for i, color in enumerate(colormap):
    lut.SetTableValue(i, tuple(color) + (1,))

lut.SetTableValue(0, (1, 1, 1, 0.25))
lut.SetTableValue(len(colormap)-1, (0, 0, 0, 0.))  # Invisible
lut.SetTableRange(0, len(colormap)-1)

global streamlines_color
streamlines_color = np.zeros(len(streamlines), dtype="float32")
brain_actor = actor.line(streamlines, colors=streamlines_color, lookup_colormap=lut, linewidth=1.5)

# Create an actor for each cluster
lut2 = vtk.vtkLookupTable()
lut2.SetNumberOfTableValues(len(colormap))
lut2.Build()
for i, color in enumerate(colormap):
    lut2.SetTableValue(i, tuple(color) + (1,))

lut2.SetTableValue(0, (1, 1, 1, 0))
lut2.SetTableRange(0, len(colormap))

cluster_actors = []
for c in clusters:
    cluster_color = np.zeros(len(c), dtype="float32")
    cluster_actor = actor.line(c, colors=cluster_color, lookup_colormap=lut2, linewidth=1.5)
    cluster_actors.append(cluster_actor)

# Create renderers
# Main renderder
global screen_size
screen_size = (0, 0)
ren_main = window.Renderer()
ren_main.background(bg)
show_m = window.ShowManager(ren_main, size=(1280, 720), interactor_style="trackball")
show_m.initialize()
show_m.window.SetNumberOfLayers(2)
ren_main.SetLayer(1)
ren_main.InteractiveOff()

# Outlierness renderer
ren_brain = window.Renderer()
show_m.window.AddRenderer(ren_brain)
ren_brain.background(bg)
ren_brain.SetViewport(0, 0, 0.5, 1)
ren_brain.add(brain_actor)
ren_brain.reset_camera_tight()

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
        ren.SetActiveCamera(ren_brain.GetActiveCamera())
        cpt += 1

global cnt, time, stamp_time
repeat_time = 10
cnt = 0
time = 0
stamp_time = 0

global replay, stream_actor
replay = replay_clustering(clusters)

centroid_actors = [None] * len(clusters)


def timer_callback(obj, event):
    global cnt, time, stamp_time

    print "Frame #{:,}".format(cnt)

    try:
        streamlines_idx, cluster_id, id_in_cluster, centroid, N = next(replay)

        # Display updated centroid
        if centroid_actors[cluster_id] is None:
            centroid_actors[cluster_id] = actor.streamtube([centroid], colormap[cluster_id+1], linewidth=0.2+np.log(N)/2)
        else:
            ren_brain.RemoveActor(centroid_actors[cluster_id])
            centroid_actors[cluster_id] = actor.streamtube([centroid], colormap[cluster_id+1], linewidth=0.2+np.log(N)/2)

        ren_brain.add(centroid_actors[cluster_id])

        # Replace color
        scalars = brain_actor.GetMapper().GetInput().GetPointData().GetScalars()
        start = streamlines._offsets[streamlines_idx]
        end = start + streamlines._lengths[streamlines_idx]
        for i in range(start, end):
            scalars.SetValue(i, len(colormap)-1)

        scalars.Modified()

        # Show streamlines in the clusters view.
        scalars = cluster_actors[cluster_id].GetMapper().GetInput().GetPointData().GetScalars()
        start = clusters_as_array_sequence[cluster_id]._offsets[id_in_cluster]
        end = start + clusters_as_array_sequence[cluster_id]._lengths[id_in_cluster]
        for i in range(start, end):
            scalars.SetValue(i, cluster_id+1)

        scalars.Modified()

        show_m.render()
    except StopIteration:
        pass

    cnt += 1

show_m.add_timer_callback(True, repeat_time, timer_callback)
show_m.render()
show_m.start()
