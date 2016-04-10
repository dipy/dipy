"""

=============================================
Deterministic Tracking with EuDX on ODF Peaks
=============================================

.. NOTE::
    Dipy has updated tools for fiber tracking. Our new machinery for fiber
    tracking is featured in the example titled Introduction to Basic Tracking.
    The tools demonstrated in this example are no longer actively being
    maintained and will likely be deprecated at some point.

In this example we do deterministic fiber tracking on fields of ODF peaks. EuDX
[Garyfallidis12]_ will be used for this.

This example requires importing example `reconst_csa.py` in order to run. EuDX was
primarily made with cpu efficiency in mind. The main idea can be used with any
model that is a child of OdfModel.

"""

from reconst_csa import csapeaks, sphere

"""
This time we will not use FA as input to EuDX but we will use GFA (generalized FA),
which is more suited for ODF functions. Tracking will stop when GFA is less
than 0.2.
"""

from dipy.tracking.eudx import EuDX

eu = EuDX(csapeaks.gfa,
          csapeaks.peak_indices[..., 0],
          seeds=10000,
          odf_vertices=sphere.vertices,
          a_low=0.2)

csa_streamlines = [streamline for streamline in eu]

"""
Now that we have our streamlines in memory we can save the results on the disk.
For this purpose we can use the TrackVis format (``*.trk``). First, we need to
create a header.
"""

import nibabel as nib

hdr = nib.trackvis.empty_header()
hdr['voxel_size'] = (2., 2., 2.)
hdr['voxel_order'] = 'LAS'
hdr['dim'] = csapeaks.gfa.shape[:3]

"""
Save the streamlines.
"""

csa_streamlines_trk = ((sl, None, None) for sl in csa_streamlines)

csa_sl_fname = 'csa_streamline.trk'

nib.trackvis.write(csa_sl_fname, csa_streamlines_trk, hdr, points_space='voxel')

"""
Visualize the streamlines with fvtk (python vtk is required).
"""

from dipy.viz import fvtk
from dipy.viz.colormap import line_colors

r = fvtk.ren()

fvtk.add(r, fvtk.line(csa_streamlines, line_colors(csa_streamlines)))

print('Saving illustration as tensor_tracks.png')

fvtk.record(r, n_frames=1, out_path='csa_tracking.png', size=(600, 600))

"""
.. figure:: csa_tracking.png
   :align: center

   **Deterministic streamlines with EuDX on ODF peaks field modulated by GFA**.

It is also possible to use EuDX with multiple ODF peaks, which is very helpful when
tracking in crossing areas.
"""

eu = EuDX(csapeaks.peak_values,
          csapeaks.peak_indices,
          seeds=10000,
          odf_vertices=sphere.vertices,
          ang_thr=20.,
          a_low=0.6)

csa_streamlines_mult_peaks = [streamline for streamline in eu]

fvtk.clear(r)

fvtk.add(r, fvtk.line(csa_streamlines_mult_peaks, line_colors(csa_streamlines_mult_peaks)))

print('Saving illustration as csa_tracking_mpeaks.png')

fvtk.record(r, n_frames=1, out_path='csa_tracking_mpeaks.png', size=(600, 600))

"""
.. figure:: csa_tracking_mpeaks.png
   :align: center

   **Deterministic streamlines with EuDX on multiple ODF peaks**.

.. [Garyfallidis12] Garyfallidis E., "Towards an accurate brain tractography", PhD thesis, University of Cambridge, 2012.

.. include:: ../links_names.inc
"""
