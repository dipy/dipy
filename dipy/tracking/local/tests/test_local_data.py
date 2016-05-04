import numpy as np

from dipy.data import default_sphere
from dipy.direction import DeterministicMaximumDirectionGetter
from dipy.io.trackvis import save_trk

from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.tracking import utils
from dipy.tracking.local import (ThresholdTissueClassifier, LocalTracking)
import dipy.data as dpd
import dipy.core.gradients as dpg
import nibabel as nib

import dipy.reconst.dti as dti
from dipy.reconst.dti import fractional_anisotropy


fdata, fbval, fbvec = dpd.get_data('small_101D')
img = nib.load(fdata)
gtab = dpg.gradient_table(fbval, fbvec)
data = img.get_data()
affine = img.get_affine()

seed_mask = np.ones(data.shape[:3])
seeds = utils.seeds_from_mask(seed_mask, density=1, affine=affine)
csd_model = ConstrainedSphericalDeconvModel(gtab, None, sh_order=6)
csd_fit = csd_model.fit(data, mask=seed_mask)


tensor_model = dti.TensorModel(gtab)
tenfit = tensor_model.fit(data, mask=seed_mask)

FA = fractional_anisotropy(tenfit.evals)
classifier = ThresholdTissueClassifier(FA, .2)


detmax_dg = DeterministicMaximumDirectionGetter.from_shcoeff(
                                                    csd_fit.shm_coeff,
                                                    max_angle=30.,
                                                    sphere=default_sphere)

streamlines = LocalTracking(detmax_dg, classifier, seeds, affine, step_size=.5)
