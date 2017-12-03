"""
====================
Tracking Compare Options
====================

This example shows how choices in model and direction-getter impact fiber
tracking results.

We will use two models: Constrained Spherical Deconvolution (CSD) and
Constant Solid Angle (CSA), also called Q-Ball, for local reconstructions.
These models will be used to generate streamlines with both a
probabilistic direction getter and a bootstrap direciton getter.

Let's load the necessary modules for executing this tutorial.
"""

from dipy.data import read_stanford_labels
from dipy.tracking import utils
from dipy.tracking.local import (ThresholdTissueClassifier, LocalTracking)
from dipy.io.trackvis import save_trk

from dipy.viz import fvtk
from dipy.viz.colormap import line_colors

"""
Now we import both the CSD and CSA models
"""

from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.reconst.shm import CsaOdfModel


"""
First we load our images, like in previous tutorials, and establish seeds
"""

hardi_img, gtab, labels_img = read_stanford_labels()
data = hardi_img.get_data()
labels = labels_img.get_data()
affine = hardi_img.get_affine()

seed_mask = labels == 2
white_matter = (labels == 1) | (labels == 2)
seeds = utils.seeds_from_mask(seed_mask, density=1, affine=affine)

"""
Next, we fit the CSD and CSA models with sh_order 6 (see bago's dissertation)
"""

csd_model = ConstrainedSphericalDeconvModel(gtab, None, sh_order=6)
csd_fit = csd_model.fit(data, mask=white_matter)

csa_model = CsaOdfModel(gtab, sh_order=6)

"""
we use the CSA fit to calculate the GFA, which will serve as our tissue
classifier
"""

gfa = csa_model.fit(data, mask=white_matter).gfa
classifier = ThresholdTissueClassifier(gfa, .25)

"""
Next, we need to set up our two direction getters
"""

"""
Example #1: Bootstrap direction getter with CSD Model
"""

from dipy.direction import bootstrap_direction_getter
from dipy.data import small_sphere


fod = csd_fit.odf(small_sphere)
boot_dg_csd = bootstrap_direction_getter.BootDirectionGetter.from_data(
            data, csd_model, max_angle=30., sphere=small_sphere)
streamlines_boot_csd = LocalTracking(boot_dg_csd, classifier, seeds, affine, step_size=.5)
#save_trk("bootstrapped_small_sphere_CSD.trk", streamlines_boot_csd, affine, labels.shape)



if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()

"""
.. [Berman_boot] Berman, J. et al. Probabilistic streamline q-ball
tractography using the residual bootstrap

.. [Jeurissen_boot] Jeurissen, B. et al. Probabilistic fiber tracking
using the residual bootstrap with constrained spherical deconvolution.


"""
