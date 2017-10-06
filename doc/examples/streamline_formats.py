"""

===========================
Read/Write streamline files
===========================

Overview
========

dipy_ can read and write many different file formats. In this example
we give a short introduction on how to use it for loading or saving streamlines.

Read :ref:`faq`

"""

import numpy as np
from dipy.data import get_data
from nibabel import trackvis

"""
1. Read/write trackvis streamline files with nibabel.
"""

fname = get_data('fornix')
print(fname)

streams, hdr = trackvis.read(fname)
streamlines = [s[0] for s in streams]

"""
Similarly you can use ``trackvis.write`` to save the streamlines.

2. Read/write streamlines with numpy.
"""

streamlines_np = np.array(streamlines, dtype=np.object)
np.save('fornix.npy', streamlines_np)

streamlines2 = list(np.load('fornix.npy'))

"""
3. We also work on our HDF5 based file format which can read/write massive datasets
(as big as the size of you free disk space). With `Dpy` we can support

  * direct indexing from the disk
  * memory usage always low
  * extensions to include different arrays in the same file

Here is a simple example.
"""

from dipy.io.dpy import Dpy
dpw = Dpy('fornix.dpy', 'w')

"""
Write many streamlines at once.
"""

dpw.write_tracks(streamlines2)

"""
Write one track
"""

dpw.write_track(streamlines2[0])

"""
or one track each time.
"""

for t in streamlines:
    dpw.write_track(t)

dpw.close()

"""
Read streamlines directly from the disk using their indices

.. include:: ../links_names.inc
"""

dpr = Dpy('fornix.dpy', 'r')
some_streamlines = dpr.read_tracksi([0, 10, 20, 30, 100])
dpr.close()


print(len(streamlines))
print(len(some_streamlines))
