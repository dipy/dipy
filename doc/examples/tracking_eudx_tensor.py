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
from dipy.tracking.streamline import Streamlines

eu = EuDX(FA.astype('f8'), peak_indices, seeds=50000,
          odf_vertices=sphere.vertices, a_low=0.2)

tensor_streamlines = Streamlines(eu)

"""
We can now save the results in the disk. For this purpose we can use the
TrackVis format (``*.trk``). First, we need to import ``save_trk`` function.
"""

from dipy.io.streamline import save_trk

"""
Save the streamlines.
"""

ten_sl_fname = 'tensor_streamlines.trk'

save_trk(ten_sl_fname, tensor_streamlines,
         affine=np.eye(4),
         vox_size=fa_img.header.get_zooms()[:3],
         shape=FA.shape)

"""
If you don't want to use Trackvis to visualize the file you can use our
lightweight `dipy.viz` module.
"""

try:
    from dipy.viz import window, actor
except ImportError:
    raise ImportError('Python fury module is not installed')
    import sys
    sys.exit()

"""
Create a scene.
"""

ren = window.Renderer()

"""
Every streamline will be coloured according to its orientation
"""

from dipy.viz import colormap as cmap

"""
`actor.line` creates a streamline actor for streamline visualization
and `ren.add` adds this actor to the scene
"""

ren.add(actor.streamtube(tensor_streamlines,
                         cmap.line_colors(tensor_streamlines)))

print('Saving illustration as tensor_tracks.png')

ren.SetBackground(1, 1, 1)
window.record(ren, out_path='tensor_tracks.png', size=(600, 600))
# Enables/disables interactive visualization
interactive = False
if interactive:
    window.show(ren)

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
