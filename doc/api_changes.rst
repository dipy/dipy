============
API changes
============

Here we provide information about functions or classes that have been removed,
renamed or are deprecated (not recommended) during different release circles.

Dipy 0.8 Changes
----------------

**Peaks**

The module ``peaks`` is now available from ``dipy.direction`` and it can still
be accessed from ``dipy.reconst`` but it will be completelly removed in version
0.10.

**Resample**

The function ``resample`` from ``dipy.align.aniso2iso`` is deprecated. Please,
use instead ``reslice`` from ``dipy.align.reslice``. The module ``aniso2iso``
will be completely removed in version 0.10.


Changes between 0.7.1 and 0.6
------------------------------

**Peaks_from_model**

The function ``peaks_from_model`` is now available from ``dipy.reconst.peaks``
. Please replace all imports like ::

    from dipy.reconst.odf import peaks_from_model

with ::

    from dipy.reconst.peaks import peaks_from_model

**Target**

The function ``target`` from ``dipy.tracking.utils`` now takes an affine
transform instead of a voxel sizes array. Please update all code using
``target`` in a way similar to this ::

    img = nib.load(anat)
    voxel_dim = img.get_header()['pixdim'][1:4]
    streamlines = utils.target(streamlines, img.get_data(), voxel_dim)

to something similar to ::

    img = nib.load(anat)
    streamlines = utils.target(streamlines, img.get_data(), img.get_affine())
