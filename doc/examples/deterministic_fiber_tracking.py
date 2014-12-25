"""
=============================================================
An introduction to the Deterministic Maximum Direction Getter
=============================================================

Deterministic maximum direction getter is the deterministic version of the
probabilistic direction getter. It can be used with the same local models
and has the same parameters. Deterministic maximum fiber tracking follows
the trajectory of the most probable pathway within the tracking constraint
(e.g. max angle). In other words, it follows the direction with the highest
probability from a distribution, as opposed to the probabilistic direction
getter which draws the direction from the distribution. Therefore, the maximum
deterministic direction getter is equivalent to the probabilistic direction
getter returning always the maximum value of the distribution.

Deterministic maximum fiber tracking is an alternative to EuDX deterministic
tractography and unlike EuDX does not follow the peaks of the local models but
uses the entire orientation distributions.

This example is an extension of the
:ref:``example_probabilistic_fiber_tracking`` example. We begin by loading the
data and fitting a constrained spherical deconvolution (CSD) reconstruction
model.
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
We use the fractional anisotropy (FA) of the DTI model to build a tissue
classifier.
"""

import dipy.reconst.dti as dti
from dipy.reconst.dti import fractional_anisotropy

tensor_model = dti.TensorModel(gtab)
tenfit = tensor_model.fit(data, mask=white_matter)

FA = fractional_anisotropy(tenfit.evals)
classifier = ThresholdTissueClassifier(FA, .2)

"""
The fiber orientation distribution (FOD) of the CSD model estimates the
distribution of small fiber bundles within each voxel. This distribution
can be used for deterministic fiber tracking. As for probabilistic tracking,
there are many ways to provide those distributions to the deterministic maximum
direction getter. Here, the spherical harmonic representation of the FOD
is used.
"""

from dipy.data import default_sphere
from dipy.direction import DeterministicMaximumDirectionGetter
from dipy.io.trackvis import save_trk

detmax_dg = DeterministicMaximumDirectionGetter.from_shcoeff(csd_fit.shm_coeff,
                                                             max_angle=30.,
                                                             sphere=default_sphere)
streamlines = LocalTracking(detmax_dg, classifier, seeds, affine, step_size=.5)

save_trk("deterministic_maximum_shm_coeff.trk", streamlines, affine,
         labels.shape)
