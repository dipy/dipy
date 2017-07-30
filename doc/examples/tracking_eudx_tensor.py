"""

=================================================
Deterministic Tracking with EuDX on Tensor Fields
=================================================

In this example we do deterministic fiber tracking on Tensor fields with EuDX
[Garyfallidis12]_.

This example requires to import example `reconst_dti.py` to run. EuDX was
primarily made with cpu efficiency in mind. Therefore, it should be useful to
give you a quick overview of your reconstruction results with the help of
tracking.

"""

import os
import sys
import numpy as np
import nibabel as nib

if not os.path.exists('tensor_fa.nii.gz'):
    import reconst_dti

"""
EuDX will use the directions (eigen vectors) of the Tensors to propagate
streamlines from voxel to voxel and fractional anisotropy to stop tracking.
"""

fa_img = nib.load('tensor_fa.nii.gz')
FA = fa_img.get_data()
evecs_img = nib.load('tensor_evecs.nii.gz')
evecs = evecs_img.get_data()

"""
In the background of the image the fitting will not be accurate because there all
measured signal is mostly noise and possibly we will find FA values with nans
(not a number). We can easily remove these in the following way.
"""

FA[np.isnan(FA)] = 0

"""
EuDX takes as input discretized voxel directions on a unit sphere. Therefore,
it is necessary to discretize the eigen vectors before feeding them in EuDX.

For the discretization procedure we use an evenly distributed sphere of 724
points which we can access using the get_sphere function.
"""

from dipy.data import get_sphere

sphere = get_sphere('symmetric724')

"""
We use quantize_evecs (evecs here stands for eigen vectors) to apply the
discretization.
"""

from dipy.reconst.dti import quantize_evecs

peak_indices = quantize_evecs(evecs, sphere.vertices)

"""
EuDX is the fiber tracking algorithm that we use in this example.
The most important parameters are the first one which represents the
magnitude of the peak of a scalar anisotropic function, the
second which represents the indices of the discretized directions of
the peaks and odf_vertices are the vertices of the input sphere.
"""

from dipy.tracking.eudx import EuDX

eu = EuDX(FA.astype('f8'), peak_indices, seeds=50000, odf_vertices = sphere.vertices, a_low=0.2)

tensor_streamlines = [streamline for streamline in eu]

"""
We can now save the results in the disk. For this purpose we can use the
TrackVis format (``*.trk``). First, we need to create a header.
"""

hdr = nib.trackvis.empty_header()
hdr['voxel_size'] = fa_img.header.get_zooms()[:3]
hdr['voxel_order'] = 'LAS'
hdr['dim'] = FA.shape

"""
Then we need to input the streamlines in the way that Trackvis format expects them.
"""

tensor_streamlines_trk = ((sl, None, None) for sl in tensor_streamlines)

ten_sl_fname = 'tensor_streamlines.trk'

"""
Save the streamlines.
"""

nib.trackvis.write(ten_sl_fname, tensor_streamlines_trk, hdr, points_space='voxel')

"""
If you don't want to use Trackvis to visualize the file you can use our
lightweight `fvtk` module.
"""

try:
    from dipy.viz import fvtk
except ImportError:
    raise ImportError('Python vtk module is not installed')
    sys.exit()

"""
Create a scene.
"""

ren = fvtk.ren()

"""
Every streamline will be coloured according to its orientation
"""

from dipy.viz.colormap import line_colors

"""
fvtk.line adds a streamline actor for streamline visualization
and fvtk.add adds this actor in the scene
"""

fvtk.add(ren, fvtk.streamtube(tensor_streamlines, line_colors(tensor_streamlines)))

print('Saving illustration as tensor_tracks.png')

ren.SetBackground(1, 1, 1)
fvtk.record(ren, n_frames=1, out_path='tensor_tracks.png', size=(600, 600))

"""
.. figure:: tensor_tracks.png
   :align: center

   Deterministic streamlines with EuDX on a Tensor Field.

References
----------

.. [Garyfallidis12] Garyfallidis E., "Towards an accurate brain tractography",
   PhD thesis, University of Cambridge, 2012.

.. include:: ../links_names.inc

"""
