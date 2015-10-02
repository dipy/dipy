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


def viz(tractograms, data, affine):

    slicer_opacity = .8

    ren = window.Renderer()

    for streamlines in tractograms:

        if len(streamlines) > 50000:
            qb = QuickBundles(30)
            clusters = qb.cluster(streamlines)
            streamlines = clusters.centroids

        ren.add(actor.line(streamlines, lod_points=10 ** 5))

    image_actor = actor.slicer(data, affine)

    image_actor.opacity(slicer_opacity)

    ren.add(image_actor)

    ren.add(fvtk.axes((10, 10, 10)))

    show_m = window.ShowManager(ren, size=(1200, 900))
    show_m.initialize()

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

    def win_callback(obj, event):
        global size
        if size != obj.GetSize():

            slider.place(ren)
            size = obj.GetSize()

    show_m.initialize()
    show_m.add_window_callback(win_callback)
    show_m.render()
    show_m.start()


def horizon_flow(input_files, verbose=True):
    """ Horizon

    Parameters
    ----------
    input_files : string
    verbose : bool, optional
    """

    #input_files = [input_file] + files
    print(input_files)

    filenames = input_files
    # glob(input_files)
    tractograms = []
    print(filenames)

    for f in filenames:

        sp = path.splitext(f)[1]

        if sp == '.trk':

            streamlines, hdr = load_trk(f)
            tractograms.append(streamlines)

        if sp == '.nii.gz' or sp == '.nii':

            img = nib.load(f)
            data = img.get_data()
            affine = img.get_affine()
            print(affine)

    viz(tractograms, data, affine)
