"""
=================================================
Particle Filtering Tractography
=================================================
Particle Filtering Tractography (PFT) [Girard2014]_ uses tissue partial
volume estimation (PVE) to reconstruct trajectories connecting the gray matter,
and not incorrectly stopping in the white matter or in the corticospinal fluid.
It relies on a tissue classifier that identifies the tissue where the
streamline stopped. If the streamline correctly stopped in the gray matter, the
trajectory is kept. If the streamline incorrecly stopped in the white matter or
in the corticospinal fluid, PFT uses anatomical information to find an
alternative streamline segment to extend the trajectory. When this segment is
found, the tractography continues until the streamline correctly stops in the
gray matter.

PFT finds an alternative streamline segment whenever the tissue classifier
returns a position classified as 'INVALIDPOINT'.

This example is an extension of the
:ref:`probabilistic_fiber_tracking` example. We begin by loading the
data, fitting a Constrained Spherical Deconvolution (CSD) reconstruction
model and creating the probabilistic direction getter.
"""

import numpy as np

from dipy.data import (read_stanford_labels, default_sphere,
                       read_stanford_pve_maps)
from dipy.direction import ProbabilisticDirectionGetter
from dipy.io.trackvis import save_trk
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response)
from dipy.tracking.local import LocalTracking, ParticleFilteringTracking
from dipy.tracking import utils
from dipy.viz import window, actor
from dipy.viz.colormap import line_colors


renderer = window.Renderer()

img_pve_csf, img_pve_gm, img_pve_wm = read_stanford_pve_maps()
hardi_img, gtab, labels_img = read_stanford_labels()

data = hardi_img.get_data()
labels = labels_img.get_data()
affine = hardi_img.get_affine()
shape = labels.shape

response, ratio = auto_response(gtab, data, roi_radius=10, fa_thr=0.7)
csd_model = ConstrainedSphericalDeconvModel(gtab, response)
csd_fit = csd_model.fit(data, mask=img_pve_wm.get_data())

dg = ProbabilisticDirectionGetter.from_shcoeff(csd_fit.shm_coeff,
                                               max_angle=20.,
                                               sphere=default_sphere)


"""
CMC/ACT Tissue Classifiers
---------------------
Continuous map criterion (CMC) [Girard2014]_ and Anatomically-constrained
tractography (ACT) [Smith2012]_ both uses PVEs information from
anatomical images to determine when the tractography stops.
Both tissue classifiers use a trilinear interpolation
at the tracking position. CMC tissue classifier uses a probability derived from
the PVE maps to determine if the streamline reaches a 'valid' or 'invalid'
region. ACT uses a fixed threshold on the PVE maps. Both tissue classifiers can
be used in conjunction with PFT. In this example, we used CMC.
"""

from dipy.tracking.local import CmcTissueClassifier
from dipy.tracking.streamline import Streamlines

voxel_size = np.average(img_pve_wm.get_header()['pixdim'][1:4])
step_size = 0.2

cmc_classifier = CmcTissueClassifier.from_pve(img_pve_wm.get_data(),
                                              img_pve_gm.get_data(),
                                              img_pve_csf.get_data(),
                                              step_size=step_size,
                                              average_voxel_size=voxel_size)

# seeds are place in voxel of the corpus callosum containing only white matter
seed_mask = labels == 2
seed_mask[img_pve_wm.get_data() < 0.5] = 0
seeds = utils.seeds_from_mask(seed_mask, density=2, affine=affine)

# Particle Filtering Tractography
pft_streamline_generator = ParticleFilteringTracking(dg,
                                                     cmc_classifier,
                                                     seeds,
                                                     affine,
                                                     max_cross=1,
                                                     step_size=step_size,
                                                     maxlen=1000,
                                                     pft_back_tracking_dist=2,
                                                     pft_front_tracking_dist=1,
                                                     particle_count=15,
                                                     return_all=False)

#streamlines = list(pft_streamline_generator)                                                     
streamlines = Streamlines(pft_streamline_generator)
save_trk("pft_streamline.trk", streamlines, affine, shape)


renderer.clear()
renderer.add(actor.line(streamlines, line_colors(streamlines)))
window.record(renderer, out_path='pft_streamlines.png', size=(600, 600))

"""
.. figure:: pft_streamlines.png
 :align: center

 **Particle Filtering Tractography**
"""

# Local Probabilistic Tractography
prob_streamline_generator = LocalTracking(dg,
                                          cmc_classifier,
                                          seeds,
                                          affine,
                                          max_cross=1,
                                          step_size=step_size,
                                          maxlen=1000,
                                          return_all=False)
#streamlines = list(pro)
streamlines = Streamlines(prob_streamline_generator)
save_trk("probabilistic_streamlines.trk", streamlines, affine, shape)

renderer.clear()
renderer.add(actor.line(streamlines, line_colors(streamlines)))
window.record(renderer, out_path='probabilistic_streamlines.png',
              size=(600, 600))

"""
.. figure:: probabilistic_streamlines.png
 :align: center

 **Probabilistic Tractography**
"""

"""
References
----------
.. [Girard2014] Girard, G., Whittingstall, K., Deriche, R., & Descoteaux, M.
    Towards quantitative connectivity analysis: reducing tractography biases.
    NeuroImage, 98, 266-278, 2014.

.. [Smith2012] Smith, R. E., Tournier, J.-D., Calamante, F., & Connelly, A.
    Anatomically-constrained tractography: Improved diffusion MRI
    streamlines tractography through effective use of anatomical
    information. NeuroImage, 63(3), 1924-1938, 2012.
"""
