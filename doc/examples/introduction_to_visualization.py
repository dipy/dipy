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

So, let's get started. In this tutorial we will create a

"""


import numpy as np

# Change with Stanford data
#dname = '/home/eleftherios/Data/Cunnane_Elef/08-111-609-AC15/work/'
dname = '/home/eleftherios/Data/fancy_data/2013_02_08_Gabriel_Girard/'

import nibabel as nib
from nibabel import trackvis as tv

world_coords = False
streamline_opacity = 1.
slicer_opacity = 1.
depth_peeling = False


img = nib.load(dname + 't1_warped.nii.gz')
data = img.get_data()
affine = img.get_affine()


img_fa = nib.load(dname + 'fa_1x1x1.nii.gz')
fa = img_fa.get_data()
affine_fa = img_fa.get_affine()


streams, hdr = tv.read(dname + 'TRK_files/bundles_cst.right.trk',
                       points_space="rasmm")
streamlines = [s[0] for s in streams]

streams, hdr = tv.read(dname + 'TRK_files/bundles_af.left.trk',
                       points_space="rasmm")
streamlines += [s[0] for s in streams]

streams, hdr = tv.read(dname + 'TRK_files/bundles_cc_1.trk',
                       points_space="rasmm")
streamlines += [s[0] for s in streams]

if not world_coords:
    from dipy.tracking.streamline import transform_streamlines
    streamlines = transform_streamlines(streamlines, np.linalg.inv(affine))

ren = window.Renderer()

stream_actor = actor.streamtube(streamlines, fa)

if not world_coords:
    slicer = actor.slice(data, affine=np.eye(4))
else:
    slicer = actor.slice(data, affine)

slicer.GetProperty().SetOpacity(slicer_opacity)
stream_actor.GetProperty().SetOpacity(streamline_opacity)

ren.add(stream_actor)
ren.add(slicer)

def change_slice(obj, event):
    global slicer
    z = int(np.round(obj.GetSliderRepresentation().GetValue()))

    print(obj)
    print(event)
    print(z)
    slicer.SetDisplayExtent(0, 255, 0, 255, z, z)
    slicer.Update()

import vtk

ren_win = vtk.vtkRenderWindow()
ren_win.AddRenderer(renderer)

if depth_peeling:
    # http://www.vtk.org/Wiki/VTK/Depth_Peeling
    ren_win.SetAlphaBitPlanes(1)
    ren_win.SetMultiSamples(0)
    renderer.SetUseDepthPeeling(1)
    renderer.SetMaximumNumberOfPeels(10)
    renderer.SetOcclusionRatio(0.1)


iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(ren_win)

slider = widget.slider(iren=iren, ren=renderer, callback=change_slice)

iren.Initialize()

ren_win.Render()

if depth_peeling:
    dp_bool = str(bool(renderer.GetLastRenderingUsedDepthPeeling()))
    print('Depth peeling used? ' + dp_bool)

iren.Start()


# ren_win.RemoveRenderer(renderer)
# renderer.SetRenderWindow(None)








