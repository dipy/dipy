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
:ref:``example_deterministic_fiber_tracking`` example. We begin by loading the
data, fitting a constrained spherical deconvolution (CSD) reconstruction
model and creating the maximum determnistic direction getter.
"""

import numpy as np

from dipy.data import read_stanford_labels, default_sphere
from dipy.direction import DeterministicMaximumDirectionGetter
from dipy.io.trackvis import save_trk
from dipy.reconst.shm import CsaOdfModel
from dipy.tracking.local import LocalTracking
from dipy.tracking import utils
from dipy.viz import fvtk
from dipy.viz.colormap import line_colors

ren = fvtk.ren()
hardi_img, gtab, labels_img = read_stanford_labels()
data = hardi_img.get_data()
labels = labels_img.get_data()
affine = hardi_img.get_affine()
sli = data.shape[2] / 2

seed_mask = labels == 2
white_matter = (labels == 1) | (labels == 2)
seeds = utils.seeds_from_mask(seed_mask, density=1, affine=affine)


csa_model = CsaOdfModel(gtab, 4)
csa_fit = csa_model.fit(data, mask=white_matter)

dg = DeterministicMaximumDirectionGetter.from_shcoeff(csa_fit.shm_coeff,
                                                      max_angle=30.,
                                                      sphere=default_sphere)

"""
Threshold Tissue Classifier
---------------------------
A scalar map can be used to build a tissue classifier by thresholding it to a
fixed value. Here, we show an example using the fractional anisotropy (FA) map
of the DTI model. The threshold tissue classifier uses a trilinear
interpolation method at the tracking position.

**Parameters**

- metric_map: numpy array [:, :, :]
- threshold: float

**Stopping criterion**

- 'ENDPOINT': metric_map < threshold,
- 'OUTSIDEIMAGE': tracking point outside of metric_map,
- 'TRACKPOINT': stopped because no direction are available,
- 'INVALIDPOINT': N/A.
"""

import matplotlib.pyplot as plt
import dipy.reconst.dti as dti
from dipy.reconst.dti import fractional_anisotropy
from dipy.tracking.local import ThresholdTissueClassifier

tensor_model = dti.TensorModel(gtab)
tenfit = tensor_model.fit(data, mask=labels > 0)
FA = fractional_anisotropy(tenfit.evals)

threshold_classifier = ThresholdTissueClassifier(FA, .2)

mask_fa = FA.copy()
mask_fa[mask_fa < 0.2] = 0
plt.xticks([])
plt.yticks([])
plt.imshow(mask_fa[:, :, sli].T, cmap='gray', origin='lower')
plt.savefig('threshold_fa.png')

"""
.. figure:: threshold_fa.png
 :align: center

 **Thresholded fractional anysotropy map**
"""

all_streamlines_threshold_classifier = LocalTracking(dg,
                                                     threshold_classifier,
                                                     seeds,
                                                     affine,
                                                     step_size=.5,
                                                     return_all=True)

save_trk("deterministic_threshold_classifier_all.trk",
         all_streamlines_threshold_classifier,
         affine,
         labels.shape)

streamlines = [sl for sl in all_streamlines_threshold_classifier]

fvtk.clear(ren)
fvtk.add(ren, fvtk.line(streamlines, line_colors(streamlines)))
fvtk.record(ren, out_path='all_streamlines_threshold_classifier.png',
            size=(400, 400))

"""
.. figure:: all_streamlines_threshold_classifier.png
 :align: center

 **Deterministic tractography using a thresholded fractional anisotropy map.**
"""


"""
Binary Tissue Classifier
------------------------
A binary mask can be used as tissue classifier for the tractography.
Here, we show how to obtain the binary tissue classifier from
the white matter mask defined above. The binary tissue classifier uses a
nearest-neighbourhood interpolation method at the tracking position.

**Parameters**

- mask: numpy array [:, :, :]

**Stopping criterion**

- 'ENDPOINT': mask = 0
- 'OUTSIDEIMAGE': tracking point outside of mask
- 'TRACKPOINT': stopped because no direction are available
- 'INVALIDPOINT': N/A
"""

from dipy.tracking.local import BinaryTissueClassifier

binary_classifier = BinaryTissueClassifier(white_matter)


plt.xticks([])
plt.yticks([])
plt.imshow(white_matter[:, :, sli].T, cmap='gray', origin='lower')
plt.savefig('white_matter_mask.png')

"""
.. figure:: white_matter_mask.png
 :align: center

 **White matter binary mask**
"""

all_streamlines_binary_classifier = LocalTracking(dg,
                                                  binary_classifier,
                                                  seeds,
                                                  affine,
                                                  step_size=.5,
                                                  return_all=True)

save_trk("deterministic_binary_classifier_all.trk",
         all_streamlines_binary_classifier,
         affine,
         labels.shape)

streamlines = [sl for sl in all_streamlines_binary_classifier]
fvtk.clear(ren)
fvtk.add(ren, fvtk.line(streamlines, line_colors(streamlines)))
fvtk.record(ren, out_path='all_streamlines_binary_classifier.png',
            size=(400, 400))

"""
.. figure:: all_streamlines_binary_classifier.png
 :align: center

 **Deterministic tractography using a binary white matter mask.**
"""

"""
Act Tissue Classifier
---------------------
Anatomically-constrained tractography (ACT) [Smith2012]_ uses information from
anatomical images to determine when the tractography stops. The 'include_map'
defines when the streamline reached a 'valid' stopping region (e.g. gray
matter partial volume estimation (PVE) map) and the 'exclude_map' defines when
the streamline reached an 'invalid' stopping region (e.g. corticospinal fluid
(PVE) map). The background of the anatomical image should be added to the
'include_map' to keep streamlines exiting the brain (e.g. through the
brain stem). The ACT tissue classifier uses a trilinear interpolation method
at the tracking position. The proposed method that cuts streamlines going
through subcortical gray matter regions is not implemented.
The backtracking technique for streamlines reaching INVALIDPOINT is not
implemented either.

**Parameters**

- include_map: numpy array [:, :, :],
- exclude_map: numpy array [:, :, :],

**Stopping criterions**

- 'ENDPOINT': include_map > 0.5,
- 'OUTSIDEIMAGE': tracking point outside of include_map or exclude_map,
- 'TRACKPOINT': stopped because no direction are available,
- 'INVALIDPOINT': exclude_map > 0.5.
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
plt.figure()
plt.subplot(121)
plt.xticks([])
plt.yticks([])
plt.imshow(include_map[:, :, sli].T, cmap='gray', origin='lower')
plt.subplot(122)
plt.xticks([])
plt.yticks([])
plt.imshow(exclude_map[:, :, sli].T, cmap='gray', origin='lower')
plt.savefig('act_maps.png')

"""
.. figure:: act_maps.png
 :align: center

 **Include (left) and exclude (right) maps for ACT.**
"""

all_streamlines_act_classifier = LocalTracking(dg,
                                               act_classifier,
                                               seeds,
                                               affine,
                                               step_size=.5,
                                               return_all=True)

save_trk("deterministic_act_classifier_all.trk",
         all_streamlines_act_classifier,
         affine,
         labels.shape)

streamlines = [sl for sl in all_streamlines_act_classifier]

fvtk.clear(ren)
fvtk.add(ren, fvtk.line(streamlines, line_colors(streamlines)))
fvtk.record(ren, out_path='all_streamlines_act_classifier.png',
            size=(400, 400))

"""
.. figure:: all_streamlines_act_classifier.png
 :align: center

 **Deterministic tractography using a anatomically-constrained tractography
 stopping criterion.**
"""

valid_streamlines_act_classifier = LocalTracking(dg,
                                                 act_classifier,
                                                 seeds,
                                                 affine,
                                                 step_size=.5,
                                                 return_all=False)

save_trk("deterministic_act_classifier_valid.trk",
         valid_streamlines_act_classifier,
         affine,
         labels.shape)

streamlines = [sl for sl in valid_streamlines_act_classifier]

fvtk.clear(ren)
fvtk.add(ren, fvtk.line(streamlines, line_colors(streamlines)))
fvtk.record(ren, out_path='valid_streamlines_act_classifier.png',
            size=(400, 400))

"""
.. figure:: valid_streamlines_act_classifier.png
 :align: center

 **Deterministic tractography using a anatomically-constrained tractography
 stopping criterion. Streamlines ending in gray matter region only.**
"""

"""
References
----------

.. [Smith2012] Smith, R. E., Tournier, J.-D., Calamante, F., & Connelly, A.
    Anatomically-constrained tractography: Improved diffusion MRI
    streamlines tractography through effective use of anatomical
    information. NeuroImage, 63(3), 1924-1938, 2012.
"""
