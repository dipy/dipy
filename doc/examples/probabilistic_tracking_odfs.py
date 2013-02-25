"""

==============================================
Probabilistic Tracking with EuDX on ODF fields
==============================================

In this example we perform probabilistic fiber tracking on fields of ODF peaks.

This example requires importing example `reconst_csa.py`.

"""

import numpy as np
from reconst_csa import *
from dipy.reconst.interpolate import NearestNeighborInterpolator

from dipy.tracking.markov import (BoundaryStepper,
                                  FixedSizeStepper,
                                  ProbabilisticOdfWeightedTracker)

from dipy.tracking.utils import seeds_from_mask


mask[[0, -1]] = 0
mask[:, [0, -1]] = 0
mask[:, :, [0, -1]] = 0

stepper = FixedSizeStepper(1)

seeds = seeds_from_mask(mask, [1, 1, 1], new_zooms)
seeds = seeds[:2000]

interpolator = NearestNeighborInterpolator(data2, new_zooms)

pwt = ProbabilisticOdfWeightedTracker(csamodel, interpolator, mask,
                                      stepper, 90, seeds, sphere)
csa_streamlines = [streamline for streamline in pwt]

"""
Now that we have our streamlines in memory we cab save the results in the disk.
For this purpose we can use the TrackVis format (*.trk). First, we need to create a
header.
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

csa_sl_fname = 'csa_prob_streamline.trk'

nib.trackvis.write(csa_sl_fname, csa_streamlines_trk, hdr, points_space='voxel')

"""
Visualize the streamlines with fvtk (python vtk is required).
"""

from dipy.viz import fvtk
from dipy.viz.colormap import line_colors

r = fvtk.ren()

fvtk.add(r, fvtk.line(csa_streamlines, line_colors(csa_streamlines)))

print('Saving illustration as csa_prob_tracks.png')

fvtk.record(r, n_frames=1, out_path='csa_prob_tracks.png', size=(600, 600))

"""
.. figure:: csa_prob_tracks.png
   :align: center

   **Probabilistic streamlines applied on an ODF field modulated by GFA**.
"""
