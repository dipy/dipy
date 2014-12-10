"""
=============================================================
An introduction to the Deterministic Maximum Direction Getter
=============================================================

Deterministic maximum fiber tracking is an alternative to EuDX deterministic
tractography. Unlike EuDx, which follows the peaks of the local models, 
deterministic maximum fiber tracking follows the trajectory of the most
probable pathway with the tracking constraint (e.g. max angle). It follows the 
direction corresponding to the highest value of a distribution. The distribution 
at each point is different and depends on the observed diffusion data at that 
point. The maximum deterministic direction getter is equivalent to the 
probabilistic direction getter returning always the maximum value of the 
distribution.

This example is an extension of the "probabilistic fiber tracking" example.
We'll begin by loading the data and fitting a constrained spherical 
deconvolution (CSD) model.
"""

from dipy.data import read_stanford_labels
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.tracking import utils
from dipy.tracking.local import (ThresholdTissueClassifier, LocalTracking)

hardi_img, gtab, labels_img = read_stanford_labels()
data = hardi_img.get_data()
labels = labels_img.get_data()
affine = hardi_img.get_affine()

seed_mask = labels == 2
white_matter = (labels == 1) | (labels == 2)
seeds = utils.seeds_from_mask(seed_mask, density=1, affine=affine)

csd_model = ConstrainedSphericalDeconvModel(gtab, None, sh_order=6)
csd_fit = csd_model.fit(data, mask=white_matter)

"""
We use the GFA of the CSA model to build a tissue classifier.
"""

from dipy.reconst.shm import CsaOdfModel

csa_model = CsaOdfModel(gtab, sh_order=6)
gfa = csa_model.fit(data, mask=white_matter).gfa
classifier = ThresholdTissueClassifier(gfa, .25)

"""
The fiber orientation distribution (FOD) of the CSD model estimates the
distribution of small fiber bundles within each voxel. This distribution 
can be use for deterministic fiber tracking. As for probabilistic tracking, 
there are many ways to provide the deterministic maximum direction getter 
those distributions. Here, the spherical harmonic represnetation of the FOD 
is used.
"""

from dipy.data import default_sphere
from dipy.direction import DeterministicMaximumDirectionGetter
from dipy.io.trackvis import save_trk

detmax_dg = DeterministicMaximumDirectionGetter.from_shcoeff(csd_fit.shm_coeff,
                                                    max_angle=30.,
                                                    sphere=default_sphere)
streamlines = LocalTracking(detmax_dg, classifier, seeds, affine, step_size=.5)

save_trk("deterministic_maximum_shm_coeff.trk", streamlines, affine, labels.shape)



