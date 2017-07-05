from __future__ import division

import sys
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
from dipy.segment.clustering import QuickBundlesX, QuickBundlesXOnline
from dipy.viz.colormap import distinguishable_colormap
import vtk
import nibabel as nib
from dipy.viz.fvtk import colors


# Some settings for the video
bg = (1, 1, 1)  # White
qb_thresholds = [40, 30, 20]
rng = np.random.RandomState(42)


def load_streamlines(nb_streamlines=1000):
    fetch_bundles_2_subjects()
    bundle_names = ['af.left', 'af.right', 'cc_1', 'cc_2', 'cc_3', 'cc_4',
                    'cc_5', 'cc_6', 'cc_7', 'cg.left', 'cg.right', 'cst.left',
                    'cst.right', 'ifof.left', 'ifof.right', 'ilf.left', 'ilf.right',
                    'mdlf.left', 'mdlf.right', 'slf1.left', 'slf1.right', 'slf2.left',
                    'slf2.right', 'slf_3.left', 'slf_3.right', 'uf.left', 'uf.right']
    subj1 = read_bundles_2_subjects('subj_1', ['fa', 't1'], bundle_names)

    tractogram = nib.streamlines.Tractogram(subj1[bundle_names[0]])
    for bundle_name in bundle_names[1:]:
        tractogram.streamlines.extend(subj1[bundle_name])

    # Rotate streamlines so we have a sagittal view of them at the beginning.
    rot_axis = np.array([1, 0, 0], dtype=np.float32)
    M_rotation = np.eye(4)
    M_rotation[:3, :3] = rodrigues_axis_rotation(rot_axis, 270.).astype(np.float32)
    rot_axis = np.array([0, 1, 0], dtype=np.float32)
    M_rotation2 = np.eye(4)
    M_rotation2[:3, :3] = rodrigues_axis_rotation(rot_axis, 270.).astype(np.float32)
    tractogram.apply_affine(np.dot(M_rotation2, M_rotation))

    # Keep a random subsample of the streamlines.
    indices = range(len(tractogram))
    rng.shuffle(indices)
    streamlines = tractogram.streamlines[indices[:nb_streamlines]].copy()
    return streamlines


def create_colors_lookup_table(colormap):
    """ Create a lookup table from a colormap.

    The last two additionals colors are respectively: semi-invisible (gray-ish) and invisible.
    """
    # Create lookup table
    colormap = [tuple(c) + (1,) for c in colormap]
    colormap += [(0, 0, 0, 0.25)]  # Semi-invisible
    colormap += [(0, 0, 0, 0)]  # Invisible

    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(len(colormap))
    lut.Build()
    for i, color in enumerate(colormap):
        lut.SetTableValue(i, color)

    lut.SetTableRange(0, len(colormap)-1)
    return lut


def color_tree(tree, bg=(1, 1, 1), min_level=0):
    import colorsys
    from dipy.viz.colormap import distinguishable_colormap
    global colormap
    colormap = iter(distinguishable_colormap(bg=bg, exclude=[(0, 0, 0), (1., 1., 0.93103448)]))

    def _color_subtree(node, color=None, level=0,  max_lum=0.9, min_lum=0.1):
        global colormap
        node.color = color

        #max_lum = 1
        if color is not None:
            hls = np.asarray(colorsys.rgb_to_hls(*color))
            max_lum = min(max_lum, hls[1] + 0.2)
            min_lum = max(min_lum, hls[1] - 0.2)

        children_sizes = map(len, node.children)
        indices = np.argsort(children_sizes)[::-1]
        luminosities = np.linspace(max_lum, min_lum, len(node.children)+1)
        #for child, luminosity, offset in zip(node.children, luminosities, offsets):
        for i, idx in enumerate(indices):
            child = node.children[idx]
            if level <= min_level:
                color = next(colormap)
                _color_subtree(child, color, level+1)
            else:
                hls = np.asarray(colorsys.rgb_to_hls(*color))
                #rbg = colorsys.hls_to_rgb(hls[0], (luminosities[i]+luminosities[i+1])/2, hls[2])
                rbg = colorsys.hls_to_rgb(hls[0], luminosities[i+1], hls[2])
                _color_subtree(child, np.asarray(rbg), level+1, luminosities[i], luminosities[i+1])

    _color_subtree(tree.root)


def buld_clusters_tree(tree, show_id=False, hide_smaller_than=1):
    import networkx as nx
    from dipy.viz.fvtk import colors

    G = nx.Graph()
    cpt = [0]

    def _tag_node(node):
        node.id = cpt[0]
        cpt[0] += 1

        indices = np.argsort(map(len, node.children))[::-1]
        for idx in indices:
            child = node.children[idx]
            if len(child) < hide_smaller_than:
                continue

            _tag_node(child)

    _tag_node(tree.root)

    def _build_graph(node):
        for child in node.children:
            if len(child) < hide_smaller_than:
                continue

            #G.add_edge(node, child)
            G.add_edge(node.id, child.id)
            _build_graph(child)

    _build_graph(tree.root)
    positions = nx.graphviz_layout(G, prog='twopi', args='')

    scaling = 4
    lines = [[]]

    text_actors = []

    def _draw_subtree(node, level=0):
        # Create node
        node_pos = np.hstack([positions[node.id], 0]) * scaling

        if node.color is not None:
            mean = np.mean([np.mean(s, axis=0) for s in node], axis=0)
            stream_actor = actor.line([s - mean + node_pos for s in node], [tuple(node.color) + (1,)]*len(node), linewidth=2)
            # stream_actor = actor.streamtube([s - mean + node_pos for s in node], [node.color]*len(node), linewidth=1)
            #stream_actor = auto_orient(stream_actor, direction=(-1, 0, 0), bbox_type="AABB")
            #stream_actor.SetPosition(node_pos - stream_actor.GetCenter())
            node.actor = stream_actor

        if node.parent is not None:
            parent_pos = np.hstack([positions[node.parent.id], 0]) * scaling
            lines[0].append(np.array([parent_pos, node_pos]))

            if show_id:
                #fvtk.label(renderer, text=str(node.id), pos=node_pos+np.array([10, 10, 0]), scale=50, color=(0, 0, 0))
                text_actor = actor.text_3d(text=str(node.id),
                                           position=node_pos+(parent_pos-node_pos)/2.,
                                           color=(0, 0, 0))
                text_actors.append(text_actor)

        for child in node.children:
            _draw_subtree(child, level+1)

    _draw_subtree(tree.root)

    line_actor = actor.streamtube(lines[0], colors.grey, linewidth=1, opacity=0.6)

    return text_actors, line_actor


streamlines = load_streamlines()
# Perform clustering to know home many clusters we will get.
qb = QuickBundlesX(thresholds=qb_thresholds)
qbo = QuickBundlesXOnline(thresholds=qb_thresholds)
qb_results = qb.cluster(streamlines)
clusters = qb_results.get_clusters(-1)
tree = qb_results.get_tree_cluster_map()
tree.refdata = streamlines

def _make_array_sequence(node):
    node.arr_seq = nib.streamlines.ArraySequence(node)

tree.traverse_postorder(tree.root, _make_array_sequence)

colormap = list(itertools.islice(distinguishable_colormap(bg=bg, exclude=[(0, 0, 0)]), len(clusters)))
lut = create_colors_lookup_table(colormap)
semi_visible_color = (0, 0, 0, 0.25)

streamlines_color = semi_visible_color * np.ones((len(streamlines), 4), dtype="float32")
brain_actor = actor.line(streamlines, colors=streamlines_color, linewidth=1.5)

color_tree(tree, bg)
text_actors, line_actor = buld_clusters_tree(tree)

# Make the streamlines of the tree clusters invisible.
def _make_node_actor_invisible(node):
    if hasattr(node, 'actor'):
        scalars = node.actor.GetMapper().GetInput().GetPointData().GetScalars()
        for i in range(scalars.GetNumberOfTuples()):
            scalars.SetTuple4(i, *(scalars.GetTuple4(i)[:3] + (0,)))

tree.traverse_postorder(tree.root, _make_node_actor_invisible)

# Color streamlines of the final clustering.
streamlines_color = np.zeros((len(streamlines), 3), dtype="float32")
for i, n in enumerate(tree.get_clusters(len(qb_thresholds))):
    scalars = n.actor.GetMapper().GetInput().GetPointData().GetScalars()
    color = np.array(scalars.GetTuple4(0)[:3])
    streamlines_color[n.indices] = color/255.

clustered_brain_actor = actor.line(streamlines, colors=streamlines_color, linewidth=1.5)


# Create renderers
# Main renderder (used for the interaction and textoverlay)
ren_main = window.Renderer()
ren_main.background(bg)
show_m = window.ShowManager(ren_main, size=(1280, 720), interactor_style="trackball")
show_m.initialize()
show_m.window.SetNumberOfLayers(2)
ren_main.SetLayer(1)
ren_main.InteractiveOff()

# Left renderer that contains the brain being clustered.
ren_brain = window.Renderer()
show_m.window.AddRenderer(ren_brain)
ren_brain.background(bg)
ren_brain.SetViewport(0, 0, 0.499, 1)
ren_brain.add(brain_actor)
ren_brain.add(clustered_brain_actor)
ren_brain.reset_camera_tight()

# Middle renderer that contains the centroids.
ren_tree = window.Renderer()
show_m.window.AddRenderer(ren_tree)
ren_tree.background(bg)
ren_tree.SetViewport(0.5, 0, 1, 1)

def _add_node_actor_to_renderer(node):
    if hasattr(node, 'actor'):
        ren_tree.add(node.actor)

tree.traverse_postorder(tree.root, _add_node_actor_to_renderer)

ren_tree.add(*text_actors)
# ren_tree.add(line_actor)
ren_tree.reset_camera_tight()


# Add captions
text_picking = actor.text_overlay('Next streamline',
                                  position=(ren_brain.size()[0]//2, 25),
                                  color=(0, 0, 0),
                                  font_size=34,
                                  justification='center',
                                  bold=True)
ren_main.add(text_picking)

text_clusters = actor.text_overlay('Hierarchical clusters ',
                                   position=(ren_brain.size()[0]+ren_tree.size()[0]//2, 25),
                                   color=(0, 0, 0),
                                   font_size=34,
                                   justification='center',
                                   bold=True)
ren_main.add(text_clusters)


global cnt, time, stamp_time
repeat_time = 10
cnt = 0
time = 0
stamp_time = 0

global frame, last_frame_with_updates, streamline_idx
frame = 0
last_frame_with_updates = 0
streamline_idx = 0

highlighted_streamlines_actor = [actor.streamtube(np.array([[0, 0, 0]]), colors=(0, 0, 0), linewidth=0.5)]
lines_actor = [actor.streamtube(np.array([[0, 0, 0]]), colors=(0, 0, 0), linewidth=0.5)]
lines = []

def main_event(speed=1):
    global last_frame_with_updates, frame, streamline_idx, lines

    if streamline_idx >= len(streamlines):
        return

    nb_streamlines_to_cluster = int((frame-last_frame_with_updates)*speed)
    frame += 1

    if nb_streamlines_to_cluster < 1:
        return


    last_frame_with_updates = frame
    for i in range(nb_streamlines_to_cluster):
        path = qbo.cluster(streamlines[streamline_idx], streamline_idx)
        tree2 = qbo.tree_cluster_map

        # Show streamlines in the clusters tree view.
        node = tree.root
        node2 = tree2.root
        line = []
        for level, cluster_id in enumerate(path):
            if hasattr(node, 'actor'):
                line.append(np.array([node.actor.GetCenter(), node.children[cluster_id].actor.GetCenter()]))
            else:
                line.append(np.array([line_actor.GetCenter(), node.children[cluster_id].actor.GetCenter()]))

            node = node.children[cluster_id]
            node2 = node2.children[cluster_id]

            id_streamline_in_cluster = len(node2)-1
            scalars = node.actor.GetMapper().GetInput().GetPointData().GetScalars()
            start = node.arr_seq._offsets[id_streamline_in_cluster]
            end = start + node.arr_seq._lengths[id_streamline_in_cluster]

            for i in range(start, end):
                color = scalars.GetTuple4(i)[:3] + (255,)
                scalars.SetTuple4(i, *color)  # Make streamline visible.

            scalars.Modified()

        lines.extend(line)

        # Highligth streamline being clustered.
        ren_brain.RemoveActor(highlighted_streamlines_actor[0])
        ren_brain.RemoveActor(brain_actor)
        highlighted_streamlines_actor[0] = actor.line([streamlines[streamline_idx]], colors=[np.array(color)/255.], linewidth=5)
        ren_brain.add(highlighted_streamlines_actor[0])
        ren_brain.add(brain_actor)

        # Draw tree lines
        ren_tree.RemoveActor(lines_actor[0])
        lines_actor[0] = actor.streamtube(lines, colors.grey, linewidth=1, opacity=0.6)
        ren_tree.add(lines_actor[0])

        streamline_idx += 1

    show_m.render()

def main_event_step_by_step(speed=0.05, nb_streamlines=10):
    global last_frame_with_updates, frame, streamline_idx, lines

    while True:
        if streamline_idx >= nb_streamlines:
            print "Step by step done"
            yield
            continue

        nb_streamlines_to_cluster = int((frame-last_frame_with_updates)*speed)
        frame += 1

        if nb_streamlines_to_cluster < 1:
            yield
            continue

        last_frame_with_updates = frame
        for i in range(nb_streamlines_to_cluster):
            path = qbo.cluster(streamlines[streamline_idx], streamline_idx)
            tree2 = qbo.tree_cluster_map

            # Show streamlines in the clusters tree view.
            node = tree.root
            node2 = tree2.root
            for level, cluster_id in enumerate(path):
                if hasattr(node, 'actor'):
                    line = np.array([node.actor.GetCenter(), node.children[cluster_id].actor.GetCenter()])
                else:
                    line = np.array([line_actor.GetCenter(), node.children[cluster_id].actor.GetCenter()])

                node = node.children[cluster_id]
                node2 = node2.children[cluster_id]

                id_streamline_in_cluster = len(node2)-1
                scalars = node.actor.GetMapper().GetInput().GetPointData().GetScalars()
                start = node.arr_seq._offsets[id_streamline_in_cluster]
                end = start + node.arr_seq._lengths[id_streamline_in_cluster]

                for i in range(start, end):
                    color = scalars.GetTuple4(i)[:3] + (255,)
                    scalars.SetTuple4(i, *color)  # Make streamline visible.

                scalars.Modified()
                lines.append(line)

                # Highligth streamline being clustered.
                ren_brain.RemoveActor(highlighted_streamlines_actor[0])
                ren_brain.RemoveActor(brain_actor)
                highlighted_streamlines_actor[0] = actor.line([streamlines[streamline_idx]], colors=[np.array(color)/255.], linewidth=5)
                ren_brain.add(highlighted_streamlines_actor[0])
                ren_brain.add(brain_actor)

                # Draw tree lines
                ren_tree.RemoveActor(lines_actor[0])
                lines_actor[0] = actor.streamtube(lines, colors.grey, linewidth=1, opacity=0.6)
                ren_tree.add(lines_actor[0])

                show_m.render()

                # Slow down the process
                for _ in range(10):
                    yield

            streamline_idx += 1


def show_clustered_brain(already_done=[False]):
    if already_done[0]:
        return

    # Highligth streamline being clustered.
    scalars = brain_actor.GetMapper().GetInput().GetPointData().GetScalars()
    start = streamlines._offsets[streamline_idx]
    end = start + streamlines._lengths[streamline_idx]
    for i in range(start, end):
        scalars.SetValue(i, cluster_id)  # Highlight streamline

    already_done[0] = True

def show_clusters_of_level(level, memoized=[None], existing_actors=[None, None]):
    if memoized[0] != level:  # Avoid unnecessary computations
        memoized[0] = level

        # Remove old lines
        ren_tree.RemoveActor(lines_actor[0])

        ren_tree.RemoveActor(line_actor)
        ren_tree.add(line_actor)

        for lvl in range(1, len(qb_thresholds)+1):

            for i, n in enumerate(tree.get_clusters(lvl)):
                scalars = n.actor.GetMapper().GetInput().GetPointData().GetScalars()

                color = scalars.GetTuple4(0)[:3] + (5,)
                for j in range(scalars.GetNumberOfTuples()):
                    scalars.SetTuple4(j, *color)  # Make streamline visible.

                scalars.Modified()

        streamlines_color = np.zeros((len(streamlines), 3), dtype="float32")
        for i, n in enumerate(tree.get_clusters(level)):
            scalars = n.actor.GetMapper().GetInput().GetPointData().GetScalars()
            streamlines_color[n.indices] = np.array(scalars.GetTuple4(0)[:3])/255.

            color = scalars.GetTuple4(0)[:3] + (255,)
            for j in range(scalars.GetNumberOfTuples()):
                scalars.SetTuple4(j, *color)  # Make streamline visible.

            scalars.Modified()

        if existing_actors[0] is not None:
            ren_brain.RemoveActor(existing_actors[0])

        # final_clusters_actor
        existing_actors[0] = actor.line(streamlines, colors=streamlines_color, linewidth=1.5)
        ren_brain.add(existing_actors[0])
        show_m.render()


def rotate_camera(angle=0.8):
    ren_brain.azimuth(angle)


def change_msg(text_actor, msg):
    text_actor.set_message(msg)
    text_actor.Modified()


from dipy.viz import timeline

global tm
tm = timeline.TimeLineManager(show_m, [],
                              'video_qbx.avi')

title = 'QuickBundlesX \n'
title += 'Garyfallidis et al. ISMRM 2016'

t = 0
tm.add_state(t, [text_picking, text_clusters], ['off', 'off'])
tm.add_state(t, [brain_actor, clustered_brain_actor], ['off', 'off'])
tm.add_sub(t, ['title'], [title])

t += 5
tm.add_sub(t, ['title'], [' '])

tm.add_state(t, [brain_actor, clustered_brain_actor], ['on', 'off'])

tm.add_event(
    t, 3,
    [rotate_camera],
    [[5]])

t += 3
tm.add_state(t, [text_picking, text_clusters], ['on', 'off'])
tm.add_state(t+1, [text_picking, text_clusters], ['on', 'on'])
tm.add_event(
    t, 7,
    [next],
    [[main_event_step_by_step(speed=0.1, nb_streamlines=4)]])

t += 7
tm.add_event(
    t, 5,
    [main_event],
    [[0.1]])

t += 5
tm.add_event(
    t, 5,
    [main_event],
    [[1]])

tm.add_event(
    t, np.inf,
    [rotate_camera],
    [[2]])

t += 5
tm.add_event(
    t, 20,
    [main_event],
    [[4]])

t += 20
tm.add_state(t, [brain_actor], ['off'])
tm.add_state(t, [text_picking, text_clusters], ['on', 'off'])
tm.add_event(t, 4,
             [show_clusters_of_level],
             [[1]])

tm.add_event(
    t, 4,
    [change_msg],
    [[text_picking, 'Final clusters at layer 1']])

t+=4
tm.add_event(t, 4,
             [show_clusters_of_level],
             [[2]])

tm.add_event(
    t, 4,
    [change_msg],
    [[text_picking, 'Final clusters at layer 2']])

t+=4
tm.add_event(t, 4,
             [show_clusters_of_level],
             [[3]])

tm.add_event(
    t, 4,
    [change_msg],
    [[text_picking, 'Final clusters at layer 3']])

t+=4
tm.add_event(
    t, np.inf,
    [sys.exit],
    [[]])

def timer_callback(obj, event):
    global tm
    tm.execute()

show_m.add_timer_callback(True, repeat_time, timer_callback)
show_m.render()
show_m.start()

del tm
del show_m
