"""
==========================
Direct Bundle Registration
==========================

This example explains how you can register two bundles from two different
subjects directly in native space [Garyfallidis14]_.  

To show the concept we will use two pre-saved cingulum bundles.
"""

from dipy.viz import fvtk
from dipy.io.pickles import load_pickle
from dipy.data import get_data

fname = get_data('cb_2')
cingulum_bundles = load_pickle(fname)

cb_subj1 = cingulum_bundles[0]
cb_subj2 = cingulum_bundles[1]


from dipy.align.streamwarp import (StreamlineRigidRegistration, 
                                   vectorize_streamlines)

"""
An important step before running the registration is to resample the streamlines
so that they both have the same number of points per streamline. Here we will
use 20 points.
"""

cb_subj1 = vectorize_streamlines(cb_subj1, 20)
cb_subj2 = vectorize_streamlines(cb_subj2, 20)

"""
Let's say now that we want to move the ``cb_subj2`` (moving) so that it can be 
aligned with ``cb_subj1`` (static). Here is how this is done.
"""

srr = StreamlineRigidRegistration()

srm = srr.optimize(static=cb_subj1, moving=cb_subj2)

"""
After the optimization is finished we can apply the learned transformation to 
``cb_subj2``.
"""

cb_subj2_aligned = srm.transform(cb_subj2)


def show_both_bundles(bundles, colors=None, show=False, fname=None):

    ren = fvtk.ren()
    ren.SetBackground(1., 1, 1)
    for (i, bundle) in enumerate(bundles):
        color = colors[i]
        lines = fvtk.streamtube(bundle, color, linewidth=0.3)
        lines.RotateX(-90)
        lines.RotateZ(90)
        fvtk.add(ren, lines)
    if show:
        fvtk.show(ren)
    if fname is not None:
        from time import sleep
        sleep(1)
        fvtk.record(ren, n_frames=1, out_path=fname, size=(900, 900))


show_both_bundles([cb_subj1, cb_subj2], 
                  colors=[fvtk.colors.orange, fvtk.colors.red],
                  fname='before_registration.png')

"""
.. figure:: before_registration.png
   :align: center

   **Before bundle registration**.
"""

show_both_bundles([cb_subj1, cb_subj2_aligned], 
                  colors=[fvtk.colors.orange, fvtk.colors.red],
                  fname='after_registration.png')

"""
.. figure:: after_registration.png
   :align: center

   **After bundle registration**.

.. [Garyfallidis14] Garyfallidis et. al, "Direct native-space fiber bundle 
                    alignment for group comparisons", ISMRM, 2014.

"""
