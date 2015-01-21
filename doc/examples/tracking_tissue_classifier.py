"""
================================================
Using Various Tissue Classifier for Tractography
================================================

...

This example is an extension of the
:ref:``example_probabilistic_fiber_tracking`` example. We begin by loading the
data and fitting a constrained spherical deconvolution (CSD) reconstruction
model.
"""

from dipy.data import read_stanford_labels
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.tracking import utils
from dipy.tracking.local import LocalTracking

hardi_img, gtab, labels_img = read_stanford_labels()
data = hardi_img.get_data()
labels = labels_img.get_data()
print labels
affine = hardi_img.get_affine()

seed_mask = labels == 2
white_matter = (labels == 1) | (labels == 2)
seeds = utils.seeds_from_mask(seed_mask, density=1, affine=affine)

csd_model = ConstrainedSphericalDeconvModel(gtab, None, sh_order=6)
csd_fit = csd_model.fit(data, mask=white_matter)

"""
The fractional anisotropy (FA) of the DTI model can be used to build a tissue
classifier by thresholding the FA to a fixed value.
"""

import dipy.reconst.dti as dti
from dipy.reconst.dti import fractional_anisotropy
from dipy.tracking.local import ThresholdTissueClassifier

tensor_model = dti.TensorModel(gtab)
tenfit = tensor_model.fit(data, mask=white_matter)
FA = fractional_anisotropy(tenfit.evals)
threshold_classifier = ThresholdTissueClassifier(FA, .2)


"""
Alternativly, a binary mask can be used as tissue classifier for the
tractography. Here, we show how to obtain the binary tissue classifier from
the white matter mask defined above.
"""
from dipy.tracking.local import BinaryTissueClassifier

binary_classifier = BinaryTissueClassifier(white_matter)

"""
We used the DeterministicMaximumDirectionGetter for tractography as in the
example ``deterministic_fiber_tracking``. We used various tissue classifier
to determine when the tractography stops.
"""

from dipy.data import default_sphere
from dipy.direction import DeterministicMaximumDirectionGetter
from dipy.io.trackvis import save_trk

detmax_dg = DeterministicMaximumDirectionGetter.from_shcoeff(csd_fit.shm_coeff,
                                                             max_angle=30.,
                                                             sphere=default_sphere)

streamlines = LocalTracking(detmax_dg, threshold_classifier, seeds, affine, step_size=.5)

save_trk("deterministic_threshold_classifier.trk", streamlines, affine,
         labels.shape)

streamlines = LocalTracking(detmax_dg, binary_classifier, seeds, affine, step_size=.5)

save_trk("deterministic_binary_classifier.trk", streamlines, affine,
         labels.shape)
