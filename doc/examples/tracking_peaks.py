"""
==============================
Peak Direction Getters Example
==============================

... this example ...

This example is an extension of the :ref:`example_tracking_introduction_eudx`
example. Let's start by loading the necessary modules for executing this
tutorial.
"""

from dipy.core.gradients import gradient_table
from dipy.data import get_fnames, small_sphere
from dipy.direction import (PeakDirectionGetter,  peaks_from_model)
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, load_nifti_data
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response)
from dipy.reconst.shm import CsaOdfModel
from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.viz import window, actor, colormap, has_fury

# Enables/disables interactive visualization
interactive = False

hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi')
label_fname = get_fnames('stanford_labels')

data, affine, hardi_img = load_nifti(hardi_fname, return_img=True)
labels = load_nifti_data(label_fname)
bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs)


seed_mask = (labels == 2)
white_matter = (labels == 1) | (labels == 2)
seeds = utils.seeds_from_mask(seed_mask, affine, density=1)

"""
Next, we fit the CSD model and extract the peaks
"""

response, ratio = auto_response(gtab, data, roi_radius=10, fa_thr=0.7)
csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=6)
csd_fit = csd_model.fit(data, mask=white_matter)

pam = peaks_from_model(csd_model, data, small_sphere,
                       relative_peak_threshold=.8,
                       min_separation_angle=45,
                       mask=white_matter)
peaks = pam.peak_dirs

"""
we use the CSA fit to calculate GFA, which will serve as our stopping
criterion.
"""

csa_model = CsaOdfModel(gtab, sh_order=6)
gfa = csa_model.fit(data, mask=white_matter).gfa
stopping_criterion = ThresholdStoppingCriterion(gfa, .25)

"""
Next, we need to set up our two direction getters
"""


peak_dg = PeakDirectionGetter.from_peaks(peaks, max_angle=30.)
peak_streamline_generator = LocalTracking(peak_dg, stopping_criterion,
                                          seeds, affine, step_size=.5)
streamlines = Streamlines(peak_streamline_generator)
sft = StatefulTractogram(streamlines, hardi_img, Space.RASMM)
save_trk(sft, "tractogram_bootstrap_dg.trk")

if has_fury:
    r = window.Renderer()
    r.add(actor.line(streamlines, colormap.line_colors(streamlines)))
    window.record(r, out_path='tractogram_peak_dg.png', size=(800, 800))
    if interactive:
        window.show(r)

"""
.. figure:: tractogram_peak_dg.png
   :align: center

   **Corpus Callosum Peak Direction Getter**

"""
