"""
==========================
Direct Bundle Registration
==========================

This example explains how you can register two bundles from two different
subjects directly in the space of streamlines [Garyfallidis15]_, [Garyfallidis14]_.

To show the concept we will use two pre-saved cingulum bundles. The algorithm
used here is called Streamline-based Linear Registration (SLR) [Garyfallidis15]_.
"""

from dipy.viz import window, actor
from time import sleep
from dipy.data import two_cingulum_bundles

cb_subj1, cb_subj2 = two_cingulum_bundles()

from dipy.align.streamlinear import StreamlineLinearRegistration
from dipy.tracking.streamline import set_number_of_points


"""
An important step before running the registration is to resample the
streamlines so that they both have the same number of points per streamline.
Here we will use 20 points. This step is not optional. Inputting streamlines
with different number of points will break the theoretical advantages of using
the SLR as explained in [Garyfallidis15]_.
"""

cb_subj1 = set_number_of_points(cb_subj1, 20)
cb_subj2 = set_number_of_points(cb_subj2, 20)

"""
Let's say now that we want to move the ``cb_subj2`` (moving) so that it can be
aligned with ``cb_subj1`` (static). Here is how this is done.
"""

srr = StreamlineLinearRegistration()

srm = srr.optimize(static=cb_subj1, moving=cb_subj2)

"""
After the optimization is finished we can apply the transformation to
``cb_subj2``.
"""

cb_subj2_aligned = srm.transform(cb_subj2)


def show_both_bundles(bundles, colors=None, show=True, fname=None):

    ren = window.Renderer()
    ren.SetBackground(1., 1, 1)
    for (i, bundle) in enumerate(bundles):
        color = colors[i]
        lines_actor = actor.streamtube(bundle, color, linewidth=0.3)
        lines_actor.RotateX(-90)
        lines_actor.RotateZ(90)
        ren.add(lines_actor)
    if show:
        window.show(ren)
    if fname is not None:
        sleep(1)
        window.record(ren, n_frames=1, out_path=fname, size=(900, 900))


show_both_bundles([cb_subj1, cb_subj2],
                  colors=[window.colors.orange, window.colors.red],
                  show=False,
                  fname='before_registration.png')

"""
.. figure:: before_registration.png
   :align: center

   Before bundle registration.
"""

show_both_bundles([cb_subj1, cb_subj2_aligned],
                  colors=[window.colors.orange, window.colors.red],
                  show=False,
                  fname='after_registration.png')

"""
.. figure:: after_registration.png
   :align: center

   After bundle registration.

As you can see the two cingulum bundles are well aligned although they contain
many streamlines of different length and shape.

.. [Garyfallidis15] Garyfallidis et al., "Robust and efficient linear
                    registration of white-matter fascicles in the space
                    of streamlines", Neuroimage, 117:124-140, 2015.
.. [Garyfallidis14] Garyfallidis et al., "Direct native-space fiber bundle
                    alignment for group comparisons", ISMRM, 2014.

"""
