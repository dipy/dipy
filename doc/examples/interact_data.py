import numpy as np
from dipy.viz import actor, window, widget


# Change with Stanford data
dname = '/home/eleftherios/Data/Cunnane_Elef/08-111-609-AC15/work/'
import nibabel as nib
from nibabel import trackvis as tv

world_coords = False
streamline_opacity = .5
slicer_opacity = .5
depth_peeling = False


img = nib.load(dname + 't1_brain_warp.nii.gz')
data = img.get_data()
affine = img.get_affine()


img_fa = nib.load(dname + 'results/metrics/fa.nii')
fa = img_fa.get_data()
affine_fa = img_fa.get_affine()


streams, hdr = tv.read(dname + 'results/bundles/cst.right.trk',
                       points_space="rasmm")
streamlines = [s[0] for s in streams]

streams, hdr = tv.read(dname + 'results/bundles/af.left.trk',
                       points_space="rasmm")
streamlines += [s[0] for s in streams]

streams, hdr = tv.read(dname + 'results/bundles/cc_1.trk',
                       points_space="rasmm")
streamlines += [s[0] for s in streams]

if not world_coords:
    from dipy.tracking.streamline import transform_streamlines
    streamlines = transform_streamlines(streamlines, np.linalg.inv(affine))

renderer = window.renderer()

stream_actor = actor.streamtube(streamlines, fa)

if not world_coords:
    slicer = actor.butcher(data, affine=np.eye(4))
else:
    slicer = actor.butcher(data, affine)

slicer.GetProperty().SetOpacity(slicer_opacity)
stream_actor.GetProperty().SetOpacity(streamline_opacity)

window.add(renderer, stream_actor)
window.add(renderer, slicer)


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

slider = widget.slider(iren=iren, callback=change_slice)

iren.Initialize()

ren_win.Render()

if depth_peeling:
    dp_bool = str(bool(renderer.GetLastRenderingUsedDepthPeeling()))
    print('Depth peeling used? ' + dp_bool)

iren.Start()


# ren_win.RemoveRenderer(renderer)
# renderer.SetRenderWindow(None)
