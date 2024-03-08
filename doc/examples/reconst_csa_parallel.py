"""
====================================
Parallel reconstruction using Q-Ball
====================================

We show an example of parallel reconstruction using a Q-Ball Constant Solid
Angle model (see Aganj et al. (MRM 2010)) and `peaks_from_model`.

Import modules, fetch and read data, and compute the mask.
"""

import time
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames, get_sphere
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.reconst.shm import CsaOdfModel
from dipy.direction import peaks_from_model
from dipy.segment.mask import median_otsu


hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi')

data, affine = load_nifti(hardi_fname)

bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs)

maskdata, mask = median_otsu(data, vol_idx=range(10, 50), median_radius=3,
                             numpass=1, autocrop=True, dilate=2)

###############################################################################
# We instantiate our CSA model with spherical harmonic order (l) of 4

csamodel = CsaOdfModel(gtab, 4)

###############################################################################
# `Peaks_from_model` is used to calculate properties of the ODFs (Orientation
# Distribution Function) and return for
# example the peaks and their indices, or GFA which is similar to FA but for
# ODF based models. This function mainly needs a reconstruction model, the
# data and a sphere as input. The sphere is an object that represents the
# spherical discrete grid where the ODF values will be evaluated.

sphere = get_sphere('repulsion724')

start_time = time.time()

###############################################################################
# We will first run `peaks_from_model` using parallelism with 2 processes. If
# `num_processes` is None (default option) then this function will find the
# total number of processors from the operating system and use this number as
# `num_processes`. Sometimes it makes sense to use only a few of the processes
# in order to allow resources for other applications. However, most of the
# times using the default option will be sufficient.

csapeaks_parallel = peaks_from_model(model=csamodel,
                                     data=maskdata,
                                     sphere=sphere,
                                     relative_peak_threshold=.5,
                                     min_separation_angle=25,
                                     mask=mask,
                                     return_odf=False,
                                     normalize_peaks=True,
                                     npeaks=5,
                                     parallel=True,
                                     num_processes=2)

time_parallel = time.time() - start_time
print("peaks_from_model using 2 processes ran in : " +
      str(time_parallel) + " seconds")

###############################################################################
# If we don't use parallelism then we need to set `parallel=False`:

start_time = time.time()
csapeaks = peaks_from_model(model=csamodel,
                            data=maskdata,
                            sphere=sphere,
                            relative_peak_threshold=.5,
                            min_separation_angle=25,
                            mask=mask,
                            return_odf=False,
                            normalize_peaks=True,
                            npeaks=5,
                            parallel=False,
                            num_processes=None)

time_single = time.time() - start_time
print("peaks_from_model ran in : " + str(time_single) + " seconds")

print("Speedup factor : " + str(time_single / time_parallel))

###############################################################################
# In Windows if you get a runtime error about frozen executable please start
# your script by adding your code above in a ``main`` function and use::
#
#    if __name__ == '__main__':
#        import multiprocessing
#        multiprocessing.freeze_support()
#        main()
