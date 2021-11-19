"""
=====================================================
An introduction to the PTT
=====================================================

"""

from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, load_nifti_data
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response_ssst)
from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.viz import window, actor, colormap, has_fury


# Enables/disables interactive visualization
interactive = True

hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi')
label_fname = get_fnames('stanford_labels')

data, affine, hardi_img = load_nifti(hardi_fname, return_img=True)
labels = load_nifti_data(label_fname)
bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs)

seed_mask = (labels == 2)
white_matter = (labels == 1) | (labels == 2)
seeds = utils.seeds_from_mask(seed_mask, affine, density=1)

response, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)
csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=6)
csd_fit = csd_model.fit(data, mask=white_matter)

"""
We use the GFA of the CSA model to build a stopping criterion.
"""

from dipy.reconst.shm import CsaOdfModel

csa_model = CsaOdfModel(gtab, sh_order=6)
gfa = csa_model.fit(data, mask=white_matter).gfa
stopping_criterion = ThresholdStoppingCriterion(gfa, .25)

"""

"""

from dipy.direction import PTTDirectionGetter
from dipy.data import small_sphere
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk

fod = csd_fit.odf(small_sphere)
pmf = fod.clip(min=0)
prob_dg = PTTDirectionGetter.from_pmf(pmf, max_angle=1,
                                      sphere=small_sphere)

streamline_generator = LocalTracking(direction_getter=prob_dg,
                                     stopping_criterion=stopping_criterion,
                                     seeds=seeds,
                                     affine=affine,
                                     step_size=1/20)
streamlines = Streamlines(streamline_generator)
sft = StatefulTractogram(streamlines, hardi_img, Space.RASMM)
save_trk(sft, "tractogram_ptt_dg_pmf.trk")

if has_fury:
    scene = window.Scene()
    scene.add(actor.line(streamlines, colormap.line_colors(streamlines)))
    window.record(scene, out_path='tractogram_ptt_dg_pmf.png',
                  size=(800, 800))
    if interactive:
        window.show(scene)
"""
.. figure:: tractogram_ptt_dg_pmf.png
   :align: center

   **Corpus Callosum using ptt direction getter from PMF**
"""
"""
One disadvantage of using a discrete PMF to represent possible tracking
directions is that it tends to take up a lot of memory (RAM). The size of the
PMF, the FOD in this case, must be equal to the number of possible tracking
directions on the hemisphere, and every voxel has a unique PMF. In this case
the data is ``(81, 106, 76)`` and ``small_sphere`` has 181 directions so the
FOD is ``(81, 106, 76, 181)``. One way to avoid sampling the PMF and holding it
in memory is to build the direction getter directly from the spherical harmonic
(SH) representation of the FOD. By using this approach, we can also use a
larger sphere, like ``default_sphere`` which has 362 directions on the
hemisphere, without having to worry about memory limitations.
"""

from dipy.data import default_sphere

prob_dg = PTTDirectionGetter.from_shcoeff(csd_fit.shm_coeff,
                                          max_angle=1.,
                                          sphere=default_sphere)
streamline_generator = LocalTracking(prob_dg, stopping_criterion, seeds,
                                     affine, step_size=1/20)
streamlines = Streamlines(streamline_generator)
sft = StatefulTractogram(streamlines, hardi_img, Space.RASMM)
save_trk(sft, "tractogram_ptt_dg_sh.trk")

if has_fury:
    scene = window.Scene()
    scene.add(actor.line(streamlines, colormap.line_colors(streamlines)))
    window.record(scene, out_path='tractogram_ptt_dg_sh.png',
                  size=(800, 800))
    if interactive:
        window.show(scene)
"""
.. figure:: tractogram_ptt_dg_sh.png
   :align: center

   **Corpus Callosum using ptt direction getter from SH**
"""
"""
Not all model fits have the ``shm_coeff`` attribute because not all models use
this basis to represent the data internally. However we can fit the ODF of any
model to the spherical harmonic basis using the ``peaks_from_model`` function.
"""

from dipy.direction import peaks_from_model

peaks = peaks_from_model(csd_model, data, default_sphere, .5, 25,
                         mask=white_matter, return_sh=True, parallel=True)
fod_coeff = peaks.shm_coeff

prob_dg = PTTDirectionGetter.from_shcoeff(fod_coeff, max_angle=1.,
                                          sphere=default_sphere)
streamline_generator = LocalTracking(prob_dg, stopping_criterion, seeds,
                                     affine, step_size=1/20)
streamlines = Streamlines(streamline_generator)
sft = StatefulTractogram(streamlines, hardi_img, Space.RASMM)
save_trk(sft, "tractogram_ptt_dg_sh_pfm.trk")

if has_fury:
    scene = window.Scene()
    scene.add(actor.line(streamlines, colormap.line_colors(streamlines)))
    window.record(scene, out_path='tractogram_ptt_dg_sh_pfm.png',
                  size=(800, 800))
    if interactive:
        window.show(scene)
"""
.. figure:: tractogram_ptt_dg_sh_pfm.png
   :align: center

   **Corpus Callosum using ptt direction getter from SH (
   peaks_from_model)**
"""
"""
.. include:: ../links_names.inc

"""
