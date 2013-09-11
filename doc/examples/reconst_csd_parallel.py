"""
=============================================================================
Reconstruction with Constrained Spherical Deconvolution using multiprocessing
=============================================================================

This example shows how to use Constrained Spherical Deconvolution (CSD)
introduced by Tournier et al. [Tournier2007]_.

This method is mainly useful with datasets with gradient directions acquired on
a spherical grid.

The basic idea with this method is that if we could estimate the response function of a
single fiber then we could deconvolve the measured signal and obtain the underlying
fiber distribution.

Lets first load the data. We will use a dataset with 10 b0s and 150 non-b0s with b-value 2000.
"""

import numpy as np

from dipy.data import fetch_stanford_hardi, read_stanford_hardi

fetch_stanford_hardi()
img, gtab = read_stanford_hardi()

"""
You can verify the b-values of the datasets by looking at the attribute `gtab.bvals`.

In CSD there is an important pre-processing step: the estimation of the fiber response function. In order to
do this we look for voxel with very anisotropic configurations. For example here we use an ROI (20x20x20) at the center
of the volume and store the signal values for the voxels with FA values higher than 0.7. Of course, if we haven't
precalculated FA we need to fit a Tensor model to the datasets. Which is what we do here.
"""

from dipy.reconst.dti import TensorModel

data = img.get_data()

data = data[:, :, 37:39]
print('data.shape (%d, %d, %d, %d)' % data.shape)

affine = img.get_affine()
zooms = img.get_header().get_zooms()[:3]

mask = data[..., 0] > 50

tenmodel = TensorModel(gtab)

ci, cj, ck = np.array(data.shape[:3]) / 2

w = 10

roi = data[ci - w: ci + w,
           cj - w: cj + w,
           ck - w: ck + w]

tenfit = tenmodel.fit(roi)

from dipy.reconst.dti import fractional_anisotropy

FA = fractional_anisotropy(tenfit.evals)
FA[np.isnan(FA)] = 0

indices = np.where(FA > 0.7)

lambdas = tenfit.evals[indices][:, :2]

"""
Using `gtab.b0s_mask()` we can find all the S0 volumes (which correspond to b-values equal 0) in the dataset.
"""

S0s = roi[indices][:, np.nonzero(gtab.b0s_mask)[0]]

"""
The response function in this example consists of a prolate tensor created
by averaging the highest and second highest eigenvalues. We also include the
average S0s.
"""

S0 = np.mean(S0s)

l01 = np.mean(lambdas, axis=0)

evals = np.array([l01[0], l01[1], l01[1]])

response = (evals, S0)

"""
Now we are ready to import the CSD model and fit the datasets.
"""

from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel

csd_model = ConstrainedSphericalDeconvModel(gtab, response)


"""
Compute the CSD-based ODFs, peaks and other metrics
"""

from dipy.data import get_sphere

sphere = get_sphere('symmetric724')

import time
from dipy.reconst.odf import peaks_from_model, peaks_from_model_parallel


start_time = time.time()
csd_peaks_parallel = peaks_from_model_parallel(model=csd_model,
                            data=data,
                            sphere=sphere,
                            relative_peak_threshold=.8,
                            min_separation_angle=45,
                            mask=mask,
                            return_sh=True,
                            return_odf=False,
                            normalize_peaks=True,
                            nbr_process=4)
end_time = time.time()
print("peaks_from_model_parallel ran in :" + str(end_time - start_time) + " seconds")

"""
peaks_from_model_parallel ran in: 69.6995100975 seconds
"""

start_time = time.time()
csd_peaks = peaks_from_model(model=csd_model,
                            data=data,
                            sphere=sphere,
                            relative_peak_threshold=.8,
                            min_separation_angle=45,
                            mask=mask,
                            return_sh=True,
                            return_odf=False,
                            normalize_peaks=True)
end_time = time.time()
print("peaks_from_model ran in :" + str(end_time - start_time) + " seconds")

"""
peaks_from_model ran in: 185.693586111 seconds
"""