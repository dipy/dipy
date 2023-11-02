"""
=================================
Parallel reconstruction using CSD
=================================

This example shows how to use parallelism (multiprocessing) using
``peaks_from_model`` in order to speedup the signal reconstruction
process. For this example will we use the same initial steps
as we used in :ref:`sphx_glr_examples_built_reconstruction_reconst_csd.py`.

Import modules, fetch and read data, apply the mask and calculate the response
function.
"""

import multiprocessing
import time

from dipy.core.gradients import gradient_table
from dipy.data import get_fnames, default_sphere
from dipy.direction import peaks_from_model
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.reconst.csdeconv import auto_response_ssst
from dipy.segment.mask import median_otsu


hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi')

data, affine = load_nifti(hardi_fname)

bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs)

maskdata, mask = median_otsu(data, vol_idx=range(10, 50), median_radius=3,
                             numpass=1, autocrop=False, dilate=2)

response, ratio = auto_response_ssst(gtab, maskdata, roi_radii=10, fa_thr=0.7)

data = maskdata[:, :, 33:37]
mask = mask[:, :, 33:37]

###############################################################################
# Now we are ready to import the CSD model and fit the datasets.

csd_model = ConstrainedSphericalDeconvModel(gtab, response)

###############################################################################
# Compute the CSD-based ODFs using ``peaks_from_model``. This function has a
# parameter called ``parallel`` which allows for the voxels to be processed in
# parallel. If ``num_processes`` is None it will figure out automatically the
# number of CPUs available in your system. Alternatively, you can set
# ``num_processes`` manually. Here, we show an example where we compare the
# duration of execution with or without parallelism.

start_time = time.time()
csd_peaks_parallel = peaks_from_model(model=csd_model,
                                      data=data,
                                      sphere=default_sphere,
                                      relative_peak_threshold=.5,
                                      min_separation_angle=25,
                                      mask=mask,
                                      return_sh=True,
                                      return_odf=False,
                                      normalize_peaks=True,
                                      npeaks=5,
                                      parallel=True,
                                      num_processes=2)

time_parallel = time.time() - start_time
print(f"peaks_from_model using 2 processes ran in : {time_parallel} seconds")

start_time = time.time()
csd_peaks = peaks_from_model(model=csd_model,
                             data=data,
                             sphere=default_sphere,
                             relative_peak_threshold=.5,
                             min_separation_angle=25,
                             mask=mask,
                             return_sh=True,
                             return_odf=False,
                             normalize_peaks=True,
                             npeaks=5,
                             parallel=False,
                             num_processes=None)

time_single = time.time() - start_time
print("peaks_from_model ran in :" + str(time_single) + " seconds")

print("Speedup factor : " + str(time_single / time_parallel))

###############################################################################
# In Windows if you get a runtime error about frozen executable please start
# your script by adding your code above in a ``main`` function and use::
#
#    if __name__ == '__main__':
#        import multiprocessing
#        multiprocessing.freeze_support()
#        main()
