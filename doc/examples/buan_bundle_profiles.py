"""
================================
BUAN bundle profiles 
================================

This example shows how to compute weighted mean bundle profiles
using the BUAN tractometry :footcite:p:`Chandio2020a`, :footcite:p:`chandio2024bundle`.

The bundle is first divided into a set of along-tract segments by assigning each point 
on every streamline to its nearest model centroid. For each segment, points from all 
streamlines contribute with a weight based on their distance to the corresponding centroid.
A weighted mean is then computed across these contributions, producing a smooth along-tract 
profile that summarizes the bundle.

First import the necessary modules.

"""

import numpy as np
import nibabel as nib
from dipy.stats.analysis import assignment_map
from dipy.io.streamline import load_tractogram
from dipy.stats.analysis import buan_profile
from dipy.viz.plotting import bundle_profile_plot
from fury import actor, window

###############################################################################
# We need to download tutorial data from:
#    `<https://nih.figshare.com/account/projects/270472/articles/31080034>`_


###############################################################################
# We use example data from a single subject selected from the PPMI dataset, consisting of:

# - ``AF_L_recognized_orig``: left arcuate fasciculus in native diffusion space,
# - ``AF_L_recognized``: the same bundle transformed to MNI space,
# - ``AF_L``: a model (template) arcuate fasciculus bundle in MNI space,
# - ``fa.nii``: the subject’s fractional anisotropy (FA) map in native space.

# Together, these files mirror a BUAN tractometry workflow: a subject-specific
# bundle, its aligned representation in common space, a model bundle defining
# the centroid for along-tract segmentation, and an along-tract scalar metric.
# We use these to compute a weighted mean FA profile along the arcuate fasciculus.

###############################################################################
# Load example data

af_orig_file = "AF_L_recognized_orig.trk"
af_mni_file = "AF_L_recognized.trk"
af_model_file = "AF_L.trk"
fa_file = "fa.nii.gz"

# Load bundles
sft_orig = load_tractogram(af_orig_file, reference="same", bbox_valid_check=False)
orig_bundle = sft_orig.streamlines

sft_mni = load_tractogram(af_mni_file, reference="same", bbox_valid_check=False)
bundle_mni = sft_mni.streamlines

sft_model = load_tractogram(af_model_file, reference="same", bbox_valid_check=False)
model_bundle = sft_model.streamlines

# Load FA volume (native space)
fa_img = nib.load(fa_file)
fa = fa_img.get_fdata()
affine = fa_img.affine

###############################################################################
# Visualize BUAN bundle segments along the length of the tract

interactive = False

n_segments = 100

_, indx = assignment_map(model_bundle, model_bundle, n_segments)
indx = np.array(indx)

rng = np.random.default_rng()

colors = [rng.random(3) for si in range(n_segments)]

disks_color = []
for i in range(len(indx)):
    disks_color.append(tuple(colors[indx[i]]))

scene = window.Scene()
scene.SetBackground(1, 1, 1)
scene.add(actor.line(model_bundle, colors=disks_color, fake_tube=True, linewidth=5))
scene.set_camera(
    focal_point=(-18.17281532, -19.55606842, 6.92485857),
    position=(-360.11, -30.46, -40.44),
    view_up=(0.25, 0.10, 0.95),
)
window.record(scene=scene, out_path="af_l_segments.png", size=(600, 600))
if interactive:
    window.show(scene)
    
###############################################################################
# Compute BUAN weighted mean bundle profile

profile = buan_profile(
    model_bundle=model_bundle,
    bundle=bundle_mni,
    orig_bundle=orig_bundle,
    metric=fa,
    affine=affine,
    no_disks=n_segments,
)

###############################################################################
# Plot the along-tract FA profile

x = np.arange(n_segments)
bundle_profile_plot(x, profile, ylabel='Fractional Anisotropy (FA)',
                    title='BUAN Weighted Mean FA Profile (AF_L)',
                    save_path='af_l_buan_profile.png')

###############################################################################
# References
# ----------
#
# .. footbibliography::
#
