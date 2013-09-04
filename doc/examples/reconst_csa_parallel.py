"""
=============================================================================
Reconstruction with Constrained Spherical Deconvolution using multiprocessing
=============================================================================

See "Reconstruction with Constrained Spherical Deconvolution" example for intermediate details

First import the necessary modules:
"""

import numpy as np
import nibabel as nib
import time
from dipy.data import fetch_stanford_hardi, read_stanford_hardi, get_sphere
from dipy.reconst.shm import CsaOdfModel, normalize_data
from dipy.reconst.odf import peaks_from_model, peaks_from_model_parallel

"""
Download and read the data for this tutorial.
"""

fetch_stanford_hardi()
img, gtab = read_stanford_hardi()

"""
img contains a nibabel Nifti1Image object (data) and gtab contains a GradientTable
object (gradient information e.g. b-values). For example to read the b-values
it is possible to write print(gtab.bvals).

Load the raw diffusion data and the affine.
"""

data = img.get_data()
print('data.shape (%d, %d, %d, %d)' % data.shape)

"""
data.shape ``(81, 106, 76, 160)``

Remove most of the background using dipy's mask module.
"""

from dipy.segment.mask import median_otsu

maskdata, mask = median_otsu(data, 3, 2, True, range(0,10))

"""
We instantiate our CSA model with sperical harmonic order of 4
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
csapeaks_parallel = peaks_from_model_parallel(model=csamodel,
                            data=maskdata,
                            sphere=sphere,
                            relative_peak_threshold=.8,
                            min_separation_angle=45,
                            mask=mask,
                            return_odf=False,
                            normalize_peaks=True,
                            nbr_process=None)
end_time = time.time()
print("peaks_from_model_parallel ran in :" + str(end_time - start_time) + " seconds")

start_time = time.time()
csapeaks = peaks_from_model(model=csamodel,
                            data=maskdata,
                            sphere=sphere,
                            relative_peak_threshold=.8,
                            min_separation_angle=45,
                            mask=mask,
                            return_odf=False,
                            normalize_peaks=True)
end_time = time.time()
print("peaks_from_model ran in :" + str(end_time - start_time) + " seconds")
