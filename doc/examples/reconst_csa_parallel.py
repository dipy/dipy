"""
====================================
Parallel reconstruction using Q-Ball
====================================

We show an example of parallel reconstruction using a Q-Ball Constant Solid
Angle model (see Aganj et al. (MRM 2010)) and `peaks_from_model`.

Import modules, fetch and read data, and compute the mask.
"""

import time
from dipy.data import fetch_stanford_hardi, read_stanford_hardi, get_sphere
from dipy.reconst.shm import CsaOdfModel
from dipy.direction import peaks_from_model
from dipy.segment.mask import median_otsu

fetch_stanford_hardi()
img, gtab = read_stanford_hardi()

data = img.get_data()

maskdata, mask = median_otsu(data, 3, 1, True,
                             vol_idx=range(10, 50), dilate=2)

"""
We instantiate our CSA model with spherical harmonic order of 4
"""

csamodel = CsaOdfModel(gtab, 4)

"""
`Peaks_from_model` is used to calculate properties of the ODFs (Orientation
Distribution Function) and return for
example the peaks and their indices, or GFA which is similar to FA but for ODF
based models. This function mainly needs a reconstruction model, the data and a
sphere as input. The sphere is an object that represents the spherical discrete
grid where the ODF values will be evaluated.
"""

sphere = get_sphere('symmetric724')

start_time = time.time()

"""
We will first run `peaks_from_model` using parallelism with 2 processes. If
`nbr_processes` is None (default option) then this function will find the total
number of processors from the operating system and use this number as
`nbr_processes`. Sometimes it makes sense to use only a few of the processes in
order to allow resources for other applications. However, most of the times
using the default option will be sufficient.
"""

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
                                     nbr_processes=2)

time_parallel = time.time() - start_time
print("peaks_from_model using 2 processes ran in : " +
      str(time_parallel) + " seconds")

"""
peaks_from_model using 2 process ran in  : 114.333221912 seconds, using 2
process

If we don't use parallelism then we need to set `parallel=False`:
"""

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
                            nbr_processes=None)

time_single = time.time() - start_time
print("peaks_from_model ran in : " + str(time_single) + " seconds")

"""
peaks_from_model ran in : 196.872478008 seconds
"""

print("Speedup factor : " + str(time_single / time_parallel))

"""
Speedup factor : 1.72191839533

In Windows if you get a runtime error about frozen executable please start
your script by adding your code above in a ``main`` function and use::

    if __name__ == '__main__':
        import multiprocessing
        multiprocessing.freeze_support()
        main()

"""
