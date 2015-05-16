"""
===============================================
Advanced interactive visualization capabilities
===============================================

In DIPY we created a thin interface to access many of the capabilities
available in the Visualization Toolkit framework (VTK) but tailored to the
needs of structural and diffusion imaging. Initially the 3D visualization
module was named ``fvtk``, meaning functions using vtk. This is still available
for backwards compatibility but now there is a more comprehensive way to access
the main functions using the following modules.
"""

from dipy.viz import actor, window, widget

"""
In ``window`` we have all the objects that connect what needs to be rendered
to the display or the disk e.g. for saving screenshots. So, there you will find
key objects and functions like the ``Renderer`` class which holds and provides
access to all the actors and the ``show`` function which displays what is
in the renderer on a window. Also, this module provides access to functions
for opening/saving dialogs and printing screenshots (see ``snapshot``).

In the ``actor`` module we can find all the different primitives e.g.
streamtubes, lines, image slices etc.

In the ``widget`` we have some other objects which allow to add buttons
and sliders and these interact both with windows and actors. Because of this
they need input from the operating system so they can process events.

So, let's get started. In this tutorial we will visualize some bundles and a
slicer. We will be able to change the slices using a ``slider`` widget.

"""

def read_bundles_2_subjects(subj_id='subj_1', metrics=['fa'],
                            bundles=['af.left', 'cst.right', 'cc_1']):


    dname = '/home/eleftherios/Data/bundles_2_subjects'

    import nibabel as nib
    from nibabel import trackvis as tv
    from os.path import join as pjoin

    res = {}

    if 't1' in metrics:
        img = nib.load(pjoin(dname, subj_id, 't1_warped.nii.gz'))
        data = img.get_data()
        affine = img.get_affine()
        res['t1'] = data

    if 'fa' in metrics:
        img_fa = nib.load(pjoin(dname, subj_id, 'fa_1x1x1.nii.gz'))
        fa = img_fa.get_data()
        affine = img_fa.get_affine()
        res['fa'] = fa


    res['affine'] = affine

    for bun in bundles:

        streams, hdr = tv.read(pjoin(dname, subj_id,
                                     'bundles', 'bundles_' + bun + '.trk'),
                               points_space="rasmm")
        streamlines = [s[0] for s in streams]
        res[bun] = streamlines

    return res


world_coords = True
streamline_opacity = 1.
slicer_opacity = 1.
depth_peeling = False

res = read_bundles_2_subjects('subj_1', ['t1', 'fa'],
                              ['af.left', 'cst.right', 'cc_1'])


streamlines = res['af.left'] + res['cst.right'] + res['cc_1']
data = res['fa']
shape = data.shape
affine = res['affine']

if not world_coords:
    from dipy.tracking.streamline import transform_streamlines
    streamlines = transform_streamlines(streamlines, np.linalg.inv(affine))

ren = window.Renderer()

stream_actor = actor.line(streamlines)

if not world_coords:
    image = actor.slice(data, affine=np.eye(4))
else:
    image = actor.slice(data, affine)

# slicer.GetProperty().SetOpacity(slicer_opacity)
# stream_actor.GetProperty().SetOpacity(streamline_opacity)

ren.add(stream_actor)
ren.add(image)

show_m = window.ShowManager(ren, size=(1200, 900))
show_m.initialize()

def change_slice(obj, event):
    z = int(np.round(obj.GetSliderRepresentation().GetValue()))
    print(z)

    image.SetDisplayExtent(0, shape[0] - 1,
                           0, shape[1] - 1, z, z)
    image.Update()

slider = widget.slider(show_m.iren, show_m.ren,
                       callback=change_slice,
                       min_value=0,
                       max_value=shape[2] - 1,
                       value=shape[2] / 2,
                       label="Z-axis",
                       right_normalized_pos=(.98, 0.6),
                       size=(120, 0), label_format="%0.lf")

show_m.render()
show_m.start()





# if depth_peeling:
#    # http://www.vtk.org/Wiki/VTK/Depth_Peeling
#    ren_win.SetAlphaBitPlanes(1)
#    ren_win.SetMultiSamples(0)
#    renderer.SetUseDepthPeeling(1)
#    renderer.SetMaximumNumberOfPeels(10)
#    renderer.SetOcclusionRatio(0.1)

#if depth_peeling:
#    dp_bool = str(bool(renderer.GetLastRenderingUsedDepthPeeling()))
#    print('Depth peeling used? ' + dp_bool)




