"""
===============================
Parallel Transport Tractography
===============================
Parallel Transport Tractography (PTT) :footcite:p:`Aydogan2021`

Let's start by importing the necessary modules.
"""

from dipy.core.gradients import gradient_table
from dipy.data import default_sphere, get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, load_nifti_data
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response_ssst
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
from dipy.tracking.streamline import Streamlines
from dipy.tracking.tracker import ptt_tracking
from dipy.tracking.utils import seeds_from_mask
from dipy.viz import actor, colormap, has_fury, window

# Enables/disables interactive visualization
interactive = False

hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames(name="stanford_hardi")
label_fname = get_fnames(name="stanford_labels")

data, affine, hardi_img = load_nifti(hardi_fname, return_img=True)
labels = load_nifti_data(label_fname)
bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs=bvecs)

seed_mask = labels == 2
seeds = seeds_from_mask(seed_mask, affine, density=2)

white_matter = (labels == 1) | (labels == 2)
sc = BinaryStoppingCriterion(white_matter)

response, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)
csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order_max=6)
csd_fit = csd_model.fit(data, mask=white_matter)


###############################################################################
# Prepare the Parallel Transport Tractography using the fiber ODF (FOD)
# obtained with CSD.
# Start the local tractography using ``ptt_tracking``.

fod = csd_fit.odf(default_sphere)

streamline_generator = ptt_tracking(
    seeds,
    sc,
    affine,
    sf=fod,
    random_seed=1,
    sphere=default_sphere,
    max_angle=20,
    step_size=0.5,
)

streamlines = Streamlines(streamline_generator)
sft = StatefulTractogram(streamlines, hardi_img, Space.RASMM)
save_trk(sft, "tractogram_ptt.trk")

if has_fury:
    scene = window.Scene()
    scene.add(actor.line(streamlines, colors=colormap.line_colors(streamlines)))
    window.record(scene=scene, out_path="tractogram_ptt.png", size=(800, 800))
    if interactive:
        window.show(scene)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Corpus Callosum using ptt direction tracker from PMF
#
#
#
# References
# ----------
#
# .. footbibliography::
#
