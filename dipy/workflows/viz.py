import numpy as np
import nibabel as nib
from nibabel import trackvis as tv
from dipy.segment.clustering import QuickBundles
from dipy.tracking.streamline import transform_streamlines
from dipy.viz import actor, window, widget
from dipy.viz import fvtk
from copy import copy, deepcopy
import os.path as path
from glob import glob


def load_trk(fname):
    streams, hdr = tv.read(fname, points_space='rasmm')
    return [i[0] for i in streams], hdr


def save_trk(fname, streamlines, hdr=None):
    streams = ((s, None, None) for s in streamlines)
    if hdr is not None:
        hdr_dict = {key: hdr[key] for key in hdr.dtype.names}
        tv.write(fname, streams, hdr_mapping=hdr_dict, points_space='rasmm')
    else:
        tv.write(fname, streams, points_space='rasmm')


def horizon(tractograms, data, affine, qb_thr=30, random_colors=True):

    slicer_opacity = .8

    ren = window.Renderer()
    global centroid_actors
    centroid_actors = []
    for streamlines in tractograms:

        if len(streamlines) > 50000:
            qb = QuickBundles(qb_thr)
            clusters = qb.cluster(streamlines)
            streamlines = clusters.centroids
            sizes = np.array([len(c) for c in clusters])
            sizes = np.interp(sizes, [sizes.min(), sizes.max()], [0.1, 2.])

            for (i, s) in enumerate(streamlines):
                act = actor.streamtube([s], linewidth=sizes[i], lod=False)
                #if len(clusters[i]) > 1000:
                #   act = actor.line([s], linewidth=10, lod=False)
                #elif len(clusters[i]) <= 1000 and len(clusters[i]) > 50:
                #   act = actor.line([s], linewidth=1, lod=False)

                centroid_actors.append(act)
                ren.add(act)
            # ren.add(actor.line(clusters[10], linewidth=2, lod=True))
        else:
            if not random_colors:
                colors = fvtk.colors.white
            else:
                colors = np.random.rand(3)
            ren.add(actor.line(streamlines, colors,
                               opacity=1., lod_points=10 ** 5))

    class SimpleTrackBallNoBB(window.vtk.vtkInteractorStyleTrackballCamera):
        def HighlightProp(self, p):
            pass

    style = SimpleTrackBallNoBB()
    # very hackish way
    style.SetPickColor(0, 0, 0)
    # style.HighlightProp(None)
    show_m = window.ShowManager(ren, size=(1200, 900), interactor_style=style)
    show_m.initialize()

    if data is not None:
        image_actor = actor.slicer(data, affine)
        image_actor.opacity(slicer_opacity)
        ren.add(image_actor)

        ren.add(fvtk.axes((10, 10, 10)))

        def change_slice(obj, event):
            z = int(np.round(obj.get_value()))
            image_actor.display(None, None, z)

        slider = widget.slider(show_m.iren, show_m.ren,
                               callback=change_slice,
                               min_value=0,
                               max_value=image_actor.shape[1] - 1,
                               value=image_actor.shape[1] / 2,
                               label="Move slice",
                               right_normalized_pos=(.98, 0.6),
                               size=(120, 0), label_format="%0.lf",
                               color=(1., 1., 1.),
                               selected_color=(0.86, 0.33, 1.))

    global size
    size = ren.GetSize()
    global picked_actors
    picked_actors = {}

    def pick_callback(obj, event):
        global centroid_actors
        global picked_actors
        # from ipdb import set_trace
        # set_trace()
        prop = obj.GetProp3D()
        # print('prop')
        # print(prop)

        ac = np.array(centroid_actors)
        index = np.where(ac == prop)[0]

        if len(index) > 0:
            try:
                bundle = picked_actors[prop]
                ren.rm(bundle)
                del picked_actors[prop]
            except:
                bundle = actor.line(clusters[index])
                # print('bundle')
                # print(bundle)
                picked_actors[prop] = bundle
                ren.add(bundle)

        if prop in picked_actors.values():
            ren.rm(prop)

    def win_callback(obj, event):
        global size
        if size != obj.GetSize():

            if data is not None:
                slider.place(ren)
            size = obj.GetSize()

    show_m.initialize()
    show_m.add_window_callback(win_callback)
    show_m.add_picker_callback(pick_callback)
    show_m.render()
    show_m.start()


def horizon_flow(input_files, qb_thr=30, random_colors=True, verbose=True):
    """ Horizon

    Parameters
    ----------
    input_files : string
    qb_thr : float, optional
    random_colors : bool, optional
    verbose : bool, optional
    """

    filenames = input_files
    # glob(input_files)
    tractograms = []

    data = None
    affine = None
    for f in filenames:
        print(f)
        sp = path.splitext(f)[1]

        if sp == '.trk':

            streamlines, hdr = load_trk(f)
            tractograms.append(streamlines)

        if sp == '.nii.gz' or sp == '.nii':

            img = nib.load(f)
            data = img.get_data()
            affine = img.get_affine()
            print(affine)

    horizon(tractograms, data, affine, qb_thr, random_colors)
