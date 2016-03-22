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
from dipy.viz.colormap import distinguishable_colormap
import vtk
import nibabel as nib


def replay_clustering(clusters):
    Ns = np.zeros(len(clusters), dtype=int)
    #centroids = [np.zeros_like(c.centroid) for c in clusters]

    cluster_iters = map(lambda c: iter(c.indices), clusters)
    nexts = [next(cluster_iter) for cluster_iter in cluster_iters]

    nb_iters_done = 0
    while nb_iters_done < len(clusters):
        cluster_id = np.argmin(nexts)
        streamlines_id = nexts[cluster_id]
        #centroids[cluster_id] = ((centroids[cluster_id]*Ns[cluster_id]) + clusters.refdata[streamlines_id]) / (Ns[cluster_id]+1)
        id_in_cluster = Ns[cluster_id]
        Ns[cluster_id] += 1

        yield streamlines_id, cluster_id, id_in_cluster#, centroids[cluster_id]

        try:
            nexts[cluster_id] = next(cluster_iters[cluster_id])
        except StopIteration:
            # print "Cluster #{} exhausted".format(cluster_id)
            nb_iters_done += 1
            nexts[cluster_id] = np.inf


# Script
fetch_bundles_2_subjects()

subj1 = read_bundles_2_subjects('subj_1', ['fa', 't1'],
                                ['af.left', 'cst.left', 'cst.right', 'cc_1', 'cc_2'])

tractogram = nib.streamlines.Tractogram(subj1['af.left'])
streamlines = tractogram.streamlines

rot_axis = np.array([1, 0, 0], dtype=np.float32)
M_rotation = np.eye(4)
M_rotation[:3, :3] = rodrigues_axis_rotation(rot_axis, 90.).astype(np.float32)
rot_axis = np.array([0, 1, 0], dtype=np.float32)
M_rotation2 = np.eye(4)
M_rotation2[:3, :3] = rodrigues_axis_rotation(rot_axis, 90.).astype(np.float32)
tractogram.apply_affine(np.dot(M_rotation2, M_rotation))

qb = QuickBundles(threshold=12)
clusters = qb.cluster(streamlines)
clusters_as_array_sequence = [nib.streamlines.ArraySequence(c) for c in clusters]

bg = (0, 0, 0)
# Create actors
colormap = list(itertools.islice(distinguishable_colormap(bg=bg, exclude=[(1, 1, 1)]), len(clusters)))
colormap = [(1, 1, 1)] + colormap

lut = vtk.vtkLookupTable()
lut.SetNumberOfTableValues(len(clusters))
lut.Build()
for i, color in enumerate(colormap):
    lut.SetTableValue(i, tuple(color) + (1,))

lut.SetTableValue(0, (1, 1, 1, 0.5))
lut.SetTableRange(0, len(clusters))

global streamlines_color
streamlines_color = np.zeros(len(streamlines), dtype="float32")
brain_actor = actor.line(streamlines, colors=streamlines_color, lookup_colormap=lut, linewidth=1.5)

# Create an actor for each cluster
lut2 = vtk.vtkLookupTable()
lut2.SetNumberOfTableValues(len(clusters))
lut2.Build()
for i, color in enumerate(colormap):
    lut2.SetTableValue(i, tuple(color) + (1,))

lut2.SetTableValue(0, (1, 1, 1, 0))
lut2.SetTableRange(0, len(clusters))

cluster_actors = []
for c in clusters:
    cluster_color = np.zeros(len(c), dtype="float32")
    cluster_actor = actor.line(c, colors=cluster_color, lookup_colormap=lut2, linewidth=1.5)
    cluster_actors.append(cluster_actor)

grid_actor = actor.grid(cluster_actors, aspect_ratio=9/16.)

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

ren_clusters = window.Renderer()
show_m.window.AddRenderer(ren_clusters)
ren_clusters.background(bg)
ren_clusters.SetViewport(0.5, 0, 1, 1)
ren_clusters.add(grid_actor)
ren_clusters.reset_camera_tight()
# ren_clusters.SetActiveCamera(ren_brain.GetActiveCamera())


global cnt, time, stamp_time
repeat_time = 10
cnt = 0
time = 0
stamp_time = 0

global replay, stream_actor
replay = replay_clustering(clusters)


def timer_callback(obj, event):
    global cnt, time, stamp_time
    global streamlines_color

    print "Frame #{:,}".format(cnt)

    try:
        streamlines_idx, cluster_id, id_in_cluster = next(replay)
        print "Streamline #{:,} -> cluster #{}".format(streamlines_idx, cluster_id)
        streamlines_color[streamlines_idx] = cluster_id+1

        # Color each point
        colors = []
        for color, streamline in zip(streamlines_color, streamlines):
            colors += [color] * len(streamline)

        # Replace color
        scalars = brain_actor.GetMapper().GetInput().GetPointData().GetScalars()
        for i, c in enumerate(colors):
            scalars.SetValue(i, c)

        scalars.Modified()

        # Show streamlines in the clusters view.
        scalars = grid_actor.items[cluster_id].GetMapper().GetInput().GetPointData().GetScalars()
        start = clusters_as_array_sequence[cluster_id]._offsets[id_in_cluster]
        end = start + clusters_as_array_sequence[cluster_id]._lengths[id_in_cluster]
        for i in range(start, end):
            scalars.SetValue(i, cluster_id+1)

        scalars.Modified()

        show_m.render()
    except StopIteration:
        pass

    cnt += 1


#     print(time)
#     if time == 0:
#         middle_message.VisibilityOn()
#         static_actor.VisibilityOff()
#         moving_actor.VisibilityOff()
#         ref_message.VisibilityOff()
#         iteration_message.VisibilityOff()
#         description_message.VisibilityOff()

#     if time == 2000:
#         middle_message.VisibilityOff()
#         static_actor.VisibilityOn()
#         moving_actor.VisibilityOn()
#         description_message.VisibilityOn()
#         msg = 'Two bundles in their native space'
#         description_message.set_message(msg)
#         description_message.Modified()

#     if time == 3000:
#         middle_message.VisibilityOff()
#         msg = 'Orange bundle will register \n to the red bundle'
#         description_message.set_message(msg)
#         description_message.Modified()
#         iteration_message.VisibilityOn()

#     if time > 3000 and cnt % 10 == 0:
#         apply_transformation()

#     if cnt_trans == len(transforms) and stamp_time == 0:

#         print('Show description')
#         msg = 'Registration finished! \n Visualizing only the points to highlight the overlap.'
#         description_message.set_message(msg)
#         description_message.Modified()

#         stamp_time = time
#         ren.rm(static_actor)
#         ren.rm(moving_actor)
#         ren.add(static_dots)
#         ren.add(moved_dots)
#         static_dots.GetProperty().SetOpacity(.5)
#         moved_dots.GetProperty().SetOpacity(.5)

#     if time == 8500:

#         ren.add(static_actor)
#         ren.add(moving_actor)
#         ren.rm(static_dots)
#         ren.rm(moved_dots)
#         msg = 'Showing final registration.'
#         description_message.set_message(msg)
#         description_message.Modified()
#         ref_message.VisibilityOn()

#     if time == 10000:

#         middle_message.VisibilityOn()

#     ren.reset_clipping_range()
#     show_m.ren.azimuth(.4)
#     show_m.render()
#     mw.write()

#     # print('Time %d' % (time,))
#     # print('Cnt %d' % (cnt,))
#     # print('Len transforms %d' % (cnt_trans,))
#     time = repeat_time * cnt
#     cnt += 1

show_m.add_timer_callback(True, repeat_time, timer_callback)
show_m.render()
show_m.start()
