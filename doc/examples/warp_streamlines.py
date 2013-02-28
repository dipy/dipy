"""

====================================
Warp FA and streamlines to MNI space
====================================

In this example we show how to apply FSL deformations to FAs and streamlines
created by Dipy_. This is an example of non-linear registration.

This example requires to have FSL installed and added to your environment. In
addition it requires two source files that we created from example
`tracking_eudx_tensor.py`: 'tensor_fa.nii.gz', and 'tensor_streamlines.trk' and
a reference file which should be the FA template image in MNI space.

"""
import os
from os.path import exists, join

if not (exists('tensor_fa.nii.gz') and exists('tensor_streamlines.trk')):
    import tracking_eudx_tensor

from dipy.external.fsl import (create_displacements,
                               warp_displacements,
                               warp_displacements_tracks)

"""
`ffa` holds the file path for our initial source FA image
"""

ffa = 'tensor_fa.nii.gz'

"""
We also specify the filenames of the FSL's command which will be called by
Dipy_'s wrapper functions.
"""

fmat = 'flirt.mat'
fnon = 'fnirt.nii.gz'
finv = 'invw.nii.gz'
fdis = 'dis.nii.gz'
fdisa = 'disa.nii.gz'

"""
Our reference image is the FA 1mm MNI template.
"""

try:
    fref = join(os.environ['FSLDIR'], 'data', 'standard', 'FMRIB58_FA_1mm.nii.gz')
except KeyError, IOError:
    print "Cannot find FSL data files."
    sys.exit()

"""
We create the linear and non-linear maps to warp FA from native to FA_1mm.
"""

create_displacements(ffa, fmat, fnon, finv, fdis, fdisa, fref)

"""
We apply those displacement to the initial FA image.
"""

ffaw = 'tensor_fa_warped.nii.gz'

warp_displacements(ffa, fmat, fdis, fref, ffaw, order=1)

"""
Now we will try to apply the streamlines in MNI using the previous created
displacements. For this purpose we will use the function
`warp_displacements_tracks`. However, this expects input in .dpy format
therefore we need to export them from .trk to .dpy. We do this here.
"""

from nibabel import trackvis

ftrk = 'tensor_streamlines.trk'

streams, hdr = trackvis.read(ftrk, points_space='voxel')
streamlines = [s[0] for s in streams]

from dipy.io.dpy import Dpy

fdpy = 'tensor_streamlines.dpy'

dpw = Dpy(fdpy, 'w')

"""
Write all streamlines at once.
"""

dpw.write_tracks(streamlines)

dpw.close()

"""
Warp the streamlines to MNI space.
"""

fdpyw = 'tensor_streamlines_warped.dpy'

warp_displacements_tracks(fdpy, ffa, fmat, finv, fdis, fdisa, fref, fdpyw)

"""
.. include:: ../links_names.inc

"""
