from __future__ import division

import numpy as np
from itertools import izip

from dipy.viz import fvtk
from dipy.viz import window, actor
from dipy.viz.axycolor import distinguishable_colormap

from dipy.tracking.streamline import get_bounding_box_streamlines
from itertools import chain

from dipy.viz import interactor
from dipy.viz.utils import get_grid_cells_position, get_bounding_box_sizes, auto_orient, shallow_copy
from dipy.tracking.metrics import principal_components

import vtk


# With autoresize
def show_grid(ren, actors, texts=None, title="Grid view", size=(800, 600)):
    ren.projection('parallel')
    show_m = window.ShowManager(ren, title=title, size=size,
                                #interactor_style=InteractorStyleImageAndTrackballActor())
                                interactor_style=InteractorStyleBundlesGrid(actors))
                                #interactor_style="trackball")

    # Size of every cell corresponds to the diagonal of the largest bounding box.
    longest_diagonal = np.max([a.GetLength() for a in actors])
    shapes = [(longest_diagonal, longest_diagonal)] * len(actors)
    positions = get_grid_cells_position(shapes, aspect_ratio=size[0]/size[1])

    for a, pos in zip(actors, positions):
        a.SetPosition(pos - a.GetCenter())
        ren.add(a)

    last_ren_size = [size]

    def resize_grid(obj, ev):
        ren_size = ren.GetSize()
        if last_ren_size[0] != ren_size:
            last_ren_size[0] = ren_size

            print "Resizing..."
            ren.ComputeAspect()
            positions = get_grid_cells_position(shapes, aspect_ratio=ren.GetAspect()[0])

            for a, pos in zip(actors, positions):
                a.SetPosition(pos - a.GetCenter())

            ren.reset_camera_tight()
            show_m.render()

    show_m.add_window_callback(resize_grid)

    ren.reset_camera_tight()
    show_m.initialize()
    show_m.render()
    show_m.start()


def cluster_and_interactive_show(streamlines):
    from dipy.segment.clustering import QuickBundles

    if streamlines is None:
        import nibabel as nib
        streamlines = nib.streamlines.load("/home/marc/research/dat/streamlines/ismrm/bundles_af.left.trk", ref=None)

    qb = QuickBundles(threshold=12.)
    clusters = qb.cluster(streamlines.points)

    bg = (0, 0, 0)
    colormap = distinguishable_colormap(bg=bg)

    ren = window.Renderer()

    actors = []
    texts = []
    for cluster, color in izip(clusters, colormap):
        stream_actor = actor.line(cluster, [color]*len(cluster), linewidth=1)
        actors.append(stream_actor)

        text = actor.text_3d(str(len(cluster)), font_size=32, justification="center", vertical_justification="top")
        texts.append(text)

    brain = actor.Container()
    brain.add(*actors, borrow=False)
    grid = actor.grid(actors, texts, cell_padding=(50, 100), cell_shape="rect")
    grid.SetVisibility(False)

    # Grid renderer
    ren.background(bg)
    ren.projection("perspective")
    ren.add(brain)
    ren.add(grid)
    #ren.add(actor.axes((50, 50, 50)))
    ren.reset_camera()

    show_m = window.ShowManager(ren, interactor_style="trackball")

    brain_interactor_style = vtk.vtkInteractorStyleTrackballCamera()
    grid_interactor_style = interactor.InteractorStyleBundlesGrid(actors)

    def toggle_grid_view(obj, event):
        if obj.GetKeySym() == "g":
            grid.SetVisibility(not grid.GetVisibility())
            brain.SetVisibility(not brain.GetVisibility())

            if grid.GetVisibility():
                ren.projection("parallel")
                grid_interactor_style.SetInteractor(show_m.iren)
                show_m.iren.SetInteractorStyle(grid_interactor_style)
                ren.reset_camera_tight()
            else:
                ren.projection("perspective")
                brain_interactor_style.SetInteractor(show_m.iren)
                show_m.iren.SetInteractorStyle(brain_interactor_style)
                ren.reset_camera()

            # We have to reset the callback since InteractorStyleBundlesGrid erase them :/
            show_m.iren.AddObserver("KeyPressEvent", toggle_grid_view)
            show_m.iren.Render()

    show_m.iren.AddObserver("KeyPressEvent", toggle_grid_view)
    show_m.start()


def auto_orient_example(streamlines=None):
    from dipy.segment.clustering import QuickBundles

    if streamlines is None:
        import nibabel as nib
        #streamlines = nib.streamlines.load("/home/marc/research/dat/streamlines/ismrm/bundles_af.left.trk", ref=None)
        #streamlines2 = nib.streamlines.load("/home/marc/research/dat/streamlines/ismrm/bundles_cst.right.trk", ref=None)
        streamlines = nib.streamlines.load("/home/marc/research/dat/streamlines/ismrm/bundles_cc_mohawk.trk", ref=None)

    qb = QuickBundles(threshold=16.)
    #clusters = qb.cluster(streamlines.points + streamlines2.points)
    clusters = qb.cluster(streamlines.points[::100])

    bg = (0, 0, 0)
    colormap = distinguishable_colormap(bg=bg)

    ren = window.Renderer()
    ren.background(bg)
    ren.projection("parallel")

    actors = []
    texts = []
    for cluster, color in izip(clusters[:15], colormap):
        stream_actor = actor.line(cluster, [color]*len(cluster), linewidth=1)
        pretty_actor = auto_orient(stream_actor, ren.camera_direction(), data_up=(0, 0, 1), show_bounds=True)
        pretty_actor_aabb = auto_orient(stream_actor, ren.camera_direction(), bbox_type="AABB", show_bounds=True)

        actors.append(stream_actor)
        actors.append(pretty_actor_aabb)
        actors.append(pretty_actor)

        text = actor.text_3d(str(len(cluster)), font_size=32, justification="center", vertical_justification="top")
        texts.append(text)

        text = actor.text_3d("AABB", font_size=32, justification="center", vertical_justification="top")
        texts.append(text)

        text = actor.text_3d("OBB", font_size=32, justification="center", vertical_justification="top")
        texts.append(text)

    grid = actor.grid(actors, texts, cell_padding=(50, 100), cell_shape="rect")
    ren.add(grid)

    ren.reset_camera_tight()
    show_m = window.ShowManager(ren, interactor_style=interactor.InteractorStyleBundlesGrid(actor))
    show_m.start()


# def show_hierarchical_clusters(tree, theta_range=(0, np.pi), show_circles=False, size=(900, 900)):
#     bg = (1, 1, 1)
#     ren = fvtk.ren()
#     fvtk.clear(ren)
#     ren.SetBackground(*bg)

#     box_min, box_max = get_bounding_box_streamlines(tree.root)
#     width, height, depth = box_max - box_min
#     box_size = max(width, height, depth)

#     thresholds = set()
#     max_threshold = tree.root.threshold
#     box_size *= len(tree.root.children) * (theta_range[1]-theta_range[0]) / (2*np.pi)

#     def _draw_subtree(node, color=fvtk.colors.orange_red, theta_range=theta_range, parent_pos=(0, 0, 0)):
#         print np.array(theta_range) / np.pi * 360

#         # Draw node
#         offset = np.zeros(3)
#         theta = theta_range[0] + (theta_range[1] - theta_range[0]) / 2.

#         radius = max_threshold - node.threshold
#         thresholds.add(node.threshold)

#         offset[0] += radius*box_size * np.cos(theta)
#         offset[1] -= radius*box_size * np.sin(theta)
#         fvtk.add(ren, fvtk.line([s + offset for s in node], [color]*len(node), linewidth=2))
#         fvtk.add(ren, fvtk.line(np.array([parent_pos, offset]), fvtk.colors.black, linewidth=1))

#         if len(node.children) == 0:
#             return

#         children = sorted(node.children, key=lambda c: len(c))
#         ratios = np.maximum([len(c) / len(node) for c in children], 0.1)
#         ratios = ratios / np.sum(ratios)  # Renormalize
#         sections = theta_range[0] + np.cumsum([0] + ratios.tolist()) * (theta_range[1] - theta_range[0])

#         colormap = distinguishable_colormap(bg=bg)
#         for i, (node, color) in enumerate(izip(children, colormap)):
#             _draw_subtree(node, color, (sections[i], sections[i+1]), offset)

#     _draw_subtree(tree.root)

#     # Draw circles for the different radius
#     if show_circles:
#         for threshold in sorted(thresholds)[:-1]:
#             radius = max_threshold - threshold
#             theta = -np.linspace(*theta_range, num=200)
#             X = radius*box_size * np.cos(theta)
#             Y = radius*box_size * np.sin(theta)
#             Z = np.zeros_like(X)
#             dashed_line = zip(np.array([X, Y, Z]).T[::4], np.array([X, Y, Z]).T[1::4])
#             fvtk.add(ren, fvtk.line(dashed_line, fvtk.colors.black, linewidth=1))

#             scale = box_size/8.
#             text = "{:.1f}mm".format(threshold)
#             pos = np.array([X[0], Y[0], Z[0]]) + np.array([-len(text)/2.*scale, scale/2., 0])
#             fvtk.label(ren, text=text, pos=pos, scale=scale, color=(0, 0, 0))

#             pos = np.array([X[-1], Y[-1], Z[-1]]) + np.array([-len(text)/2.*scale, scale/2., 0])
#             fvtk.label(ren, text=text, pos=pos, scale=scale, color=(0, 0, 0))

#     fvtk.show(ren, size=size)
