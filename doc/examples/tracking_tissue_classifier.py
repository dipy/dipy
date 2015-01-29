"""
================================================
Using Various Tissue Classifier for Tractography
================================================
The tissue classifier determines if the tracking stops or continues at each
tracking position. If the tracking stops, it can either be because it reaches
an ending region (e.g. low FA, gray matter or corticospinal fluid regions),
because it exit the image boundary or because the direction getter has no
direction to follow. Each tissue classifier determines if the stoping is 'valid'
or 'invalid'. A streamline is 'valid' when the tissue classifier determines the
streamline stops in a position classified as 'ENDPOINT' or 'OUTSIDEIMAGE'. A
streamline is 'invalid' when it stops in a position classified as 'TRACKPOINT'
or 'INVALIDPOINT'. These conditions are described below. The 'LocalTracking'
generator can be set to output all generated streamlines or only the 'valid'
ones.

This example is an extension of the
:ref:``example_probabilistic_fiber_tracking`` example. We begin by loading the
data and fitting a constrained spherical deconvolution (CSD) reconstruction
model.
"""
import numpy as np

from dipy.data import read_stanford_labels
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.tracking import utils

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
-- ThresholdTissueClassifier --
A scalar map can be used to build a tissue classifier by thresholding it to a
fixed value. Here, we show an example using the fractional anisotropy (FA) map
of the DTI model. The threshold tissue classifier uses a trilinear
interpolation method at the tracking position.

Parameters:
    metric_map: numpy array [:,:,:],
    threshold: float.

'ENDPOINT': metric_map < threshold,
'OUTSIDEIMAGE': tracking point outside of metric_map,
'TRACKPOINT': stopped because no direction are available,
'INVALIDPOINT': N/A.
"""

import dipy.reconst.dti as dti
from dipy.reconst.dti import fractional_anisotropy
from dipy.tracking.local import ThresholdTissueClassifier

tensor_model = dti.TensorModel(gtab)
tenfit = tensor_model.fit(data, mask=white_matter)
FA = fractional_anisotropy(tenfit.evals)

threshold_classifier = ThresholdTissueClassifier(FA, .2)

"""
-- BinaryTissueClassifier --
A binary mask can be used as tissue classifier for the tractography.
Here, we show how to obtain the binary tissue classifier from
the white matter mask defined above. The binary tissue classifier uses a
nearest-neighbourhood interpolation method at the tracking position.

Parameters:
    mask: numpy array [:,:,:],

'ENDPOINT': mask = 0,
'OUTSIDEIMAGE': tracking point outside of mask,
'TRACKPOINT': stopped because no direction are available,
'INVALIDPOINT': N/A.
"""
from dipy.tracking.local import BinaryTissueClassifier

binary_classifier = BinaryTissueClassifier(white_matter)

"""
-- ActTissueClassifier --
Anatomically-constrained tractography (ACT) uses information from
anatomical images to determine when the tractography stops. The 'include_map'
defines when the streamline reached a 'valid' stopping region (e.g. gray
matter partial volume estimation (PVE) map) and the 'exclude_map' defines when
the streamline reached an 'invalid' stopping region (e.g. corticospinal fluid
(PVE) map). The background of the anatomical image should be added to the
'include_map' to keep streamline exiting the brain (e.g. throught the
brain stem). The act tissue classifier uses a trilinear interpolation method
at the tracking position.

Parameters:
    include_map: numpy array [:,:,:],
    exclude_map: numpy array [:,:,:],

'ENDPOINT': include_map > 0.5,
'OUTSIDEIMAGE': tracking point outside of metric_map,
'TRACKPOINT': stopped because no direction are available,
'INVALIDPOINT': exclude_map > 0.5.
"""
from dipy.data import read_stanford_pve_maps
from dipy.tracking.local import ActTissueClassifier

img_pve_csf, img_pve_gm, img_pve_wm = read_stanford_pve_maps()

background = np.ones(img_pve_gm.shape)
background[(img_pve_gm.get_data() +
            img_pve_wm.get_data() +
            img_pve_csf.get_data()) > 0] = 0

include_map = img_pve_gm.get_data()
include_map[background > 0] = 1
exclude_map = img_pve_csf.get_data()

act_classifier = ActTissueClassifier(include_map, exclude_map)

"""
We used the DeterministicMaximumDirectionGetter for tractography as in the
example ``deterministic_fiber_tracking``. Various tissue classifier are used
to determine when the tractography stops.
"""

from dipy.data import default_sphere
from dipy.direction import DeterministicMaximumDirectionGetter
from dipy.io.trackvis import save_trk
from dipy.tracking.local import LocalTracking

detmax_dg = DeterministicMaximumDirectionGetter.from_shcoeff(csd_fit.shm_coeff,
                                                             max_angle=30.,
                                                             sphere=default_sphere)

all_streamlines_threshold_classifier = LocalTracking(detmax_dg,
                                                     threshold_classifier,
                                                     seeds,
                                                     affine,
                                                     step_size=.5,
                                                     return_all=True)

save_trk("deterministic_threshold_classifier_all.trk",
         all_streamlines_threshold_classifier,
         affine,
         labels.shape)

all_streamlines_binary_classifier = LocalTracking(detmax_dg,
                                                  binary_classifier,
                                                  seeds,
                                                  affine,
                                                  step_size=.5,
                                                  return_all=True)

save_trk("deterministic_binary_classifier_all.trk",
         all_streamlines_binary_classifier,
         affine,
         labels.shape)

all_streamlines_act_classifier = LocalTracking(detmax_dg,
                                               act_classifier,
                                               seeds,
                                               affine,
                                               step_size=.5,
                                               return_all=True)

save_trk("deterministic_act_classifier_all.trk",
         all_streamlines_act_classifier,
         affine,
         labels.shape)

valid_streamlines_act_classifier = LocalTracking(detmax_dg,
                                                 act_classifier,
                                                 seeds,
                                                 affine,
                                                 step_size=.5,
                                                 return_all=False)

save_trk("deterministic_act_classifier_valid.trk",
         valid_streamlines_act_classifier,
         affine,
         labels.shape)
