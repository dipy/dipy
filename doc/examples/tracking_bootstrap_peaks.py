"""
=================================================
Bootstrap and Closest Peak Direction Getters Example
=================================================

This example shows how choices in direction-getter impact fiber
tracking results by demonstrating the bootstrap direction getter (a type of
probabilistic tracking) and the closest peak direction getter (a type of
deterministic tracking).

Let's load the necessary modules for executing this tutorial.
"""

from dipy.data import read_stanford_labels
from dipy.tracking import utils
from dipy.tracking.local import (ThresholdTissueClassifier, LocalTracking)
from dipy.io.trackvis import save_trk
from dipy.viz import window, actor
from dipy.viz.colormap import line_colors

renderer = window.Renderer()

"""
Now we import the CSD model
"""

from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel

"""
First we load our images and establish seeds. See the Introduction to Basic
Tracking tutorial for more background on these steps.
"""

hardi_img, gtab, labels_img = read_stanford_labels()
data = hardi_img.get_data()
labels = labels_img.get_data()
affine = hardi_img.get_affine()

seed_mask = labels == 2
white_matter = (labels == 1) | (labels == 2)
seeds = utils.seeds_from_mask(seed_mask, density=1, affine=affine)

"""
Next, we fit the CSD model
"""

csd_model = ConstrainedSphericalDeconvModel(gtab, None, sh_order=6)
csd_fit = csd_model.fit(data, mask=white_matter)


"""
we use the CSA fit to calculate GFA, which will serve as our tissue
classifier
"""

from dipy.reconst.shm import CsaOdfModel
csa_model = CsaOdfModel(gtab, sh_order=6)
gfa = csa_model.fit(data, mask=white_matter).gfa
classifier = ThresholdTissueClassifier(gfa, .25)

"""
Next, we need to set up our two direction getters
"""

"""
Example #1: Bootstrap direction getter with CSD Model
"""

from dipy.direction import BootDirectionGetter
from dipy.tracking.streamline import Streamlines
from dipy.data import small_sphere

boot_dg_csd = BootDirectionGetter.from_data(data, csd_model, max_angle=30.,
                                            sphere=small_sphere)
boot_streamline_generator = LocalTracking(boot_dg_csd, classifier, seeds,
                                          affine, step_size=.5)
streamlines = Streamlines(boot_streamline_generator)

renderer.clear()
renderer.add(actor.line(streamlines, line_colors(streamlines)))
window.record(renderer, out_path='bootstrap_dg_CSD.png', size=(600, 600))

"""
.. figure:: bootstrap_dg_CSD.png
   :align: center

   **Corpus Callosum Bootstrap Probabilistic Direction Getter**

We have created a bootstrapped probabilistic set of streamlines. If you repeat
the fiber tracking (keeping all inputs the same) you will NOT get exactly the
same set of streamlines. We can save the streamlines as a Trackvis file so it
can be loaded into other software for visualization or further analysis.
"""

save_trk("bootstrap_dg_CSD.trk", streamlines, affine, labels.shape)

"""
Example #2: Closest peak direction getter with CSD Model
"""

from dipy.direction import ClosestPeakDirectionGetter

pmf = csd_fit.odf(small_sphere).clip(min=0)
peak_dg = ClosestPeakDirectionGetter.from_pmf(pmf, max_angle=30.,
                                              sphere=small_sphere)
peak_streamline_generator = LocalTracking(peak_dg, classifier, seeds, affine,
                                          step_size=.5)
streamlines = Streamlines(peak_streamline_generator)

renderer.clear()
renderer.add(actor.line(streamlines, line_colors(streamlines)))
window.record(renderer, out_path='closest_peak_dg_CSD.png', size=(600, 600))

"""
.. figure:: closest_peak_dg_CSD.png
   :align: center

   **Corpus Callosum Closest Peak Deterministic Direction Getter**

We have created a set of streamlines using the closest peak direction getter,
which is a type of deterministic tracking. If you repeat the fiber tracking
(keeping all inputs the same) you will get exactly the same set of streamlines.
We can save the streamlines as a Trackvis file so it can be loaded into other
software for visualization or further analysis.
"""

save_trk("closest_peak_dg_CSD.trk", streamlines, affine, labels.shape)

"""
.. [Berman_boot] Berman, J. et al. Probabilistic streamline q-ball
tractography using the residual bootstrap

.. [Jeurissen_boot] Jeurissen, B. et al. Probabilistic fiber tracking
using the residual bootstrap with constrained spherical deconvolution.


"""
