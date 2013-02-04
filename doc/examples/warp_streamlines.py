"""

====================================
Warp FA and streamlines to MNI space
====================================

In this example we show how to apply FSL deformations to FAs and streamlines
created by Dipy_. This is an example of non-linear registration.

This example requires to have FSL installed and added to your path. In addition
it is requires to import example `tracking_eudx_tensor.py`.

"""

import tracking_eudx_tensor
from dipy.external.fsl import (create_displacements,
                               warp_displacements,
                               warp_displacements_tracks)

ffa = 'tensor_fa.nii.gz'

fmat = 'flirt.mat'
fnon = 'fnirt.nii.gz'
finv = 'invw.nii.gz'
fdis = 'dis.nii.gz'
fdisa = 'disa.nii.gz'

create_displacements(ffa, fmat, fnon, finv, fdis, fdisa)

warp_displacements(ffa, fmat, fdis, fref, ffaw2, order=1)

from nibabel import trackvis

ftrk = 'tensor_streamlines.trk'

streams, hdr = trackvis.read(fname)
streamlines = [s[0] for s in streams]

from dipy.io.dpy import Dpy

fdpy = 'tensor_streamlines.dpy'

pw = Dpy(fdpy, 'w')

"""
Write many streamlines at once.
"""

dpw.write_tracks(streamlines)

dpw.close()

warp_displacements_tracks(fdpy, ffa, fmat, finv, fdis, fdisa, fref, fdpyw)
