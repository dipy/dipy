"""

Probabilistic fiber tracking is a way of reconstructing white matter
connections using diffusion MR imaging. Like deterministic fiber tracking, the
probabilistic approach follows the trajectory of a possible pathway step by
step starting at a seed, however, unlike deterministic tracking, the tracking
direction at each point along the path is chosen at random from a distribution.
The distribution at each point is different and depends on the observed
diffusion data at that point. The distribution of tracking directions at each
point can be represented as a PMF (probability mass function) if the possible
tracking directions are restricted to discrete number of well distributed
points on a sphere.

This example picks up where "introduction to basic tracking" leaves off.
We'll begin by repeating a few steps from that example, loading the data and
fitting a CSD model.
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
classifier = ThresholdTissueClassifier(csd_fit.gfa, .25)

"""
The FOD, fiber orientation distribution, of the CSD model estimates the
distribution of small fiber bundles within each voxel. We can use this
distribution for probabilistic fiber tracking. One way to do this is to
represent the FOD using a discrete sphere. This discrete FOD can be used
by the Probabilistic Direction Getter as a PMF to pick tracking directions.
"""

from dipy.tracking.local import ProbabilisticDirectionGetter
from dipy.data import small_sphere

fod = csd_fit.odf(small_sphere)
fod.clip(0, out=fod)
prob_dg = ProbabilisticDirectionGetter.from_pmf(fod, max_angle=30.,
                                                sphere=small_sphere)
streamlines = LocalTracking(prob_dg, classifier, seeds, affine, step_size=.5)

"""
One disadvantage of using a discrete PMF to represent possible tracking
directions is that it tends to take up a lot of memory (RAM). The size of the
PMF, the FOD in this case, must be equal to the number of possible tracking
directions on the hemisphere, and every voxel has a unique PMF. In this case
the data is ``(81, 106, 76)`` and ``small_sphere`` has 181 directions so the
FOD is ``(81, 106, 76, 181)``. One way to avoid sampling the PMF and holding it
in memory is to build the direction getter directly from the spherical harmonic
representation of the FOD. By using this approach, we can also use a larger
sphere, like ``default_sphere`` which has 362 directions on the hemisphere,
without having to worry about memory limitations.
"""

from dipy.data import default_sphere

prob_dg = ProbabilisticDirectionGetter.from_shcoeff(csd_fit.shm_coeff,
                                                    max_angle=30.,
                                                    sphere=default_sphere)
streamlines = LocalTracking(prob_dg, classifier, seeds, affine, step_size=.5)

"""
Not all model fits have the ``shm_coeff`` attribute because not all models use
this basis to represent the data internally. However we can fit the ODF of any
model to the spherical harmonic basis using the ``peaks_from_model`` function.
"""

from dipy.reconst.peaks import peaks_from_model

peaks = peaks_from_model(csd_model, data, default_sphere, .5, 25,
                         mask=white_matter, return_sh=True, parallel=True)
fod_coeff = peaks.shm_coeff
prob_dg = ProbabilisticDirectionGetter.from_shcoeff(fod_coeff, max_angle=30.,
                                                    sphere=default_sphere)
streamlines = LocalTracking(prob_dg, classifier, seeds, affine, step_size=.5)

