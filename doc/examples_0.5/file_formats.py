""" 

=====================
File Format Friendly 
=====================

Overview
========

Read :ref:`faq`

"""

import numpy as np
from dipy.data import get_data
from nibabel import trackvis

"""
read trackvis
"""

fname=get_data('fornix')
print(fname)

streams,hdr=trackvis.read(fname)
tracks=[s[0] for s in streams]

"""
quick way use numpy.save
"""

tracks_np=np.array(tracks,dtype=np.object)
np.save('fornix.npy',tracks_np)

"""
it is good practice to remove what is not necessary any more
"""

del tracks_np

tracks2=list(np.load('fornix.npy'))

"""
huge datasets use dipy.io.dpy

* direct indexing from the disk
* memory usage always low
* extendable

"""

from dipy.io.dpy import Dpy
dpw=Dpy('fornix.dpy','w')

"""
write many tracks at once
"""

dpw.write_tracks(tracks2)

"""
write one track
"""

dpw.write_track(tracks2[0]*6)

"""
or one track each time
"""

for t in tracks:
    dpw.write_track(t*3)

dpw.close()

"""
read tracks directly from the disk using their indices
"""

dpr=Dpy('fornix.dpy','r')
some_tracks=dpr.read_tracksi([0,10,20,30,100])
dpr.close()


"""
Number of tracks in before and after
"""

print(len(tracks))
print(len(some_tracks))






