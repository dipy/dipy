"""
=================================
Parallel reconstruction using CSD
=================================

This example shows how to use parallelism (multiprocessing) using
``peaks_from_model`` in order to speedup the signal reconstruction
process. For this example will we use the same initial steps
as we used in :ref:`example_reconst_csd`.

Import modules, fetch and read data, apply the mask and calculate the response
function.
"""

import multiprocessing

from dipy.data import fetch_stanford_hardi, read_stanford_hardi

fetch_stanford_hardi()
img, gtab = read_stanford_hardi()
data = img.get_data()

from dipy.segment.mask import median_otsu

maskdata, mask = median_otsu(data, 3, 1, False,
                             vol_idx=range(10, 50), dilate=2)

from dipy.reconst.csdeconv import auto_response

response, ratio = auto_response(gtab, maskdata, roi_radius=10, fa_thr=0.7)

data = maskdata[:, :, 33:37]
mask = mask[:, :, 33:37]

"""
Now we are ready to import the CSD model and fit the datasets.
"""

from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel

csd_model = ConstrainedSphericalDeconvModel(gtab, response)

from dipy.data import get_sphere

sphere = get_sphere('symmetric724')

"""
Compute the CSD-based ODFs using ``peaks_from_model``. This function has a
parameter called ``parallel`` which allows for the voxels to be processed in
parallel. If ``nbr_processes`` is None it will figure out automatically the
number of CPUs available in your system. Alternatively, you can set
``nbr_processes`` manually. Here, we show an example where we compare the
duration of execution with or without parallelism.
"""

import time
from dipy.direction import peaks_from_model

start_time = time.time()
csd_peaks_parallel = peaks_from_model(model=csd_model,
                                      data=data,
                                      sphere=sphere,
                                      relative_peak_threshold=.5,
                                      min_separation_angle=25,
                                      mask=mask,
                                      return_sh=True,
                                      return_odf=False,
                                      normalize_peaks=True,
                                      npeaks=5,
                                      parallel=True,
                                      nbr_processes=None)

time_parallel = time.time() - start_time
print("peaks_from_model using " + str(multiprocessing.cpu_count())
      + " process ran in :" + str(time_parallel) + " seconds")

"""
``peaks_from_model`` using 8 processes ran in :114.425682068 seconds
"""

start_time = time.time()
csd_peaks = peaks_from_model(model=csd_model,
                             data=data,
                             sphere=sphere,
                             relative_peak_threshold=.5,
                             min_separation_angle=25,
                             mask=mask,
                             return_sh=True,
                             return_odf=False,
                             normalize_peaks=True,
                             npeaks=5,
                             parallel=False,
                             nbr_processes=None)

time_single = time.time() - start_time
print("peaks_from_model ran in :" + str(time_single) + " seconds")

"""
``peaks_from_model`` ran in :242.772505999 seconds
"""

print("Speedup factor : " + str(time_single / time_parallel))

"""
Speedup factor : 2.12166099088

In Windows if you get a runtime error about frozen executable please start
your script by adding your code above in a ``main`` function and use:

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
"""
