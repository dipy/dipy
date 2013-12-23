"""Various tools related to creating and working with streamlines

This module provides tools for targeting streamlines using ROIs, for making
connectivity matrices from whole brain fiber tracking and some other tools that
allow streamlines to interact with image data.

Important Note:
---------------
Some functions in this module use an affine matrix to represent the coordinate
system associated with the points of a streamline. Dipy uses a similar
convention to nifti files when interpreting this affine matrix. This convention
is that the point at the center of voxel ``[i, j, k`]` is represented by the
point ``[x, y, z]`` where ``[x, y, z, 1] = affine * [i, j, k, 1]``.
Also when the phrase "voxel coordinates" is used, it is understood to be the
same as ``affine = eye(4)``.

As an example, lets take a 2d image where the affine is
``[[1., 0., 0.],
   [0., 2., 0.],
   [0., 0., 1.]]``:

A------------
|   |   |   |
| C |   |   |
|   |   |   |
----B--------
|   |   |   |
|   |   |   |
|   |   |   |
-------------
|   |   |   |
|   |   |   |
|   |   |   |
------------D

A = [-.5, -1.]
B = [ .5,  1.]
C = [ 0.,  0.]
D = [ 2.5,  5.]
"""

# In order to avoid circular imports, this module was split into two parts.
# The python part is implemented in _utils.py, the cython part is implemented
# in vox2track.pyx (which imports from _utils.py).
from ._utils import *
from .vox2track import *


