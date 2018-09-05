"""
==========================
Bundle Atlas Generation
==========================

This example demonstrates how to create a streamline bundle atlas from a set of
segmented bundles. This atlas can be used with Recobundles to segment a
desired bundle. See this preprint _[Jordan_2018_bundle_templates] for details.

In this example, we use two pre-saved cingulum bundles from two different
subjects, but ideally this procedure would be applied to a much larger dataset.

"""


from dipy.viz import window, actor
from dipy.data import two_cingulum_bundles
from dipy.tracking.templates import make_bundle_atlas


cb_subj1, cb_subj2 = two_cingulum_bundles()

"""
First, we load the cingulum bundles from two patients that will make our
demo template.
"""


bundle_list = [cb_subj1, cb_subj2]
colors = [window.colors.purple, window.colors.yellow]

show = False
ren = window.Renderer()
for (i, bundle) in enumerate(bundle_list):
    color = colors[i]
    lines_actor = actor.streamtube(bundle, color, linewidth=0.3)
    lines_actor.RotateX(-90)
    lines_actor.RotateZ(90)
    ren.add(lines_actor)
if show:
    window.show(ren)
window.record(ren, n_frames=1, out_path='bundles_for_template.png',
              size=(900, 900))


"""
.. figure:: bundles_for_template.png
   :align: center

   These are the bundles we are using to demonstrate how to run the atlas
   generation code.
"""

"""
The keystone subject is the subject that all of the other bundles will be
registered to. If you want to export bundles in MNI space, an affine
transform between MNI space and the keystone subject space should be
calculated. One way to do this is to perform whole brain SLR between the MNI
streamline template by Yeh et. al. and the keystone subject's whole brain
tractography dataset.
"""

keystone_boi = cb_subj1

"""
Next, we apply the function make_bundle_template, in which we resample the
streamlines so that they both have the same number of
points per streamline, cluster all of the templates, and put the resulting
centroids in the same space. For more information on how choice of quickbundles
parameters influence the template, see _[Jordan_2018_bundle_templates].

Notes on parameters:
- Nsubsamp will determine the resolution of your bundle atlas.
- qb_thresh and clsz_thresh are related. If you have a fine parcellation
(low qb_thresh) then the clsz_threshold should be quite low since clusters
will be small.
- If you want to export your bundles in MNI space, input the MNI-to-keystone
space affine transformation into the variable keystone2MNI_xfm.
NOTE: set this variable if (and only if) you want the result to be in
MNI space. Otherwise it will be in keystone space (keystone patient's
diffusion space)
"""

rb_template = make_bundle_atlas(bundle_list, keystone_boi, qb_thresh=5.,
                                Nsubsamp=20, clsz_thresh=5,
                                keystone2MNI_xfm=None, verbose=False)

show = False
ren = window.Renderer()
lines_actor = actor.streamtube(bundle, linewidth=0.3)
lines_actor.RotateX(-90)
lines_actor.RotateZ(90)
ren.add(lines_actor)
if show:
    window.show(ren)
window.record(ren, n_frames=1, out_path='bundle_template.png',
              size=(900, 900))

"""
.. figure:: bundle_template.png
   :align: center

   Final bundle template.


.. [Jordan_2018_bundle_templates] Jordan et al., "Generating Bundle Atlases
from a group of Segmented Bundles", PREPRINT (biorxiv), 2018.

"""
