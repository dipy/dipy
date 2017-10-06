"""

.. _reconst-csd:

=======================================================
Reconstruction with Constrained Spherical Deconvolution
=======================================================

This example shows how to use Constrained Spherical Deconvolution (CSD)
introduced by Tournier et al. [Tournier2007]_.

This method is mainly useful with datasets with gradient directions acquired on
a spherical grid.

The basic idea with this method is that if we could estimate the response
function of a single fiber then we could deconvolve the measured signal and
obtain the underlying fiber distribution.

Let's first load the data. We will use a dataset with 10 b0s and 150 non-b0s
with b-value 2000.
"""

import numpy as np

from dipy.data import fetch_stanford_hardi, read_stanford_hardi

fetch_stanford_hardi()
img, gtab = read_stanford_hardi()

data = img.get_data()

"""
You can verify the b-values of the datasets by looking at the attribute
``gtab.bvals``.

In CSD there is an important pre-processing step: the estimation of the fiber
response function. In order to do this we look for regions of the brain where
it is known that there are single coherent fiber populations. For example if we
use an ROI at the center of the brain, we will find single fibers from the
corpus callosum. The ``auto_response`` function will calculate FA for an ROI of
radius equal to ``roi_radius`` in the center of the volume and return the
response function estimated in that region for the voxels with FA higher than
0.7.
"""

from dipy.reconst.csdeconv import auto_response

response, ratio = auto_response(gtab, data, roi_radius=10, fa_thr=0.7)

"""
The ``response`` tuple contains two elements. The first is an array with
the eigenvalues of the response function and the second is the average S0 for
this response.

It is good practice to always validate the result of auto_response. For
this purpose we can print the elements of ``response`` and have a look at their
values.
"""

print(response)

"""
(array([ 0.0014,  0.00029,  0.00029]), 416.206)

The tensor generated from the response must be prolate (two smaller eigenvalues
should be equal) and look anisotropic with a ratio of second to first eigenvalue
of about 0.2. Or in other words, the axial diffusivity of this tensor should
be around 5 times larger than the radial diffusivity.
"""

print(ratio)

"""
0.21197

We can double-check that we have a good response function by visualizing the
response function's ODF. Here is how you would do that:
"""

from dipy.viz import fvtk
ren = fvtk.ren()
evals = response[0]
evecs = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).T
from dipy.data import get_sphere
sphere = get_sphere('symmetric724')
from dipy.sims.voxel import single_tensor_odf
response_odf = single_tensor_odf(sphere.vertices, evals, evecs)
response_actor = fvtk.sphere_funcs(response_odf, sphere)
fvtk.add(ren, response_actor)
print('Saving illustration as csd_response.png')
fvtk.record(ren, out_path='csd_response.png', size=(200, 200))

"""
.. figure:: csd_response.png
   :align: center

   Estimated response function.

"""

fvtk.rm(ren, response_actor)

"""
Depending on the dataset, FA threshold may not be the best way to find the
best possible response function. For one, it depends on the diffusion tensor
(FA and first eigenvector), which has lower accuracy at high
b-values. Alternatively, the response function can be calibrated in a
data-driven manner [Tax2014]_.

First, the data is deconvolved with a 'fat' response function. All voxels that
are considered to contain only one peak in this deconvolution (as determined by
the peak threshold which gives an upper limit of the ratio of the second peak
to the first peak) are maintained, and from these voxels a new response
function is determined. This process is repeated until convergence is
reached. Here we calibrate the response function on a small part of the data.
"""

from dipy.reconst.csdeconv import recursive_response

"""
A WM mask can shorten computation time for the whole dataset. Here it is created
based on the DTI fit.
"""

import dipy.reconst.dti as dti
tenmodel = dti.TensorModel(gtab)
tenfit = tenmodel.fit(data, mask=data[..., 0] > 200)

from dipy.reconst.dti import fractional_anisotropy
FA = fractional_anisotropy(tenfit.evals)
MD = dti.mean_diffusivity(tenfit.evals)
wm_mask = (np.logical_or(FA >= 0.4, (np.logical_and(FA >= 0.15, MD >= 0.0011))))

response = recursive_response(gtab, data, mask=wm_mask, sh_order=8,
                              peak_thr=0.01, init_fa=0.08,
                              init_trace=0.0021, iter=8, convergence=0.001,
                              parallel=True)


"""
We can check the shape of the signal of the response function, which should be
like  a pancake:
"""

response_signal = response.on_sphere(sphere)
response_actor = fvtk.sphere_funcs(response_signal, sphere)

ren = fvtk.ren()

fvtk.add(ren, response_actor)
print('Saving illustration as csd_recursive_response.png')
fvtk.record(ren, out_path='csd_recursive_response.png', size=(200, 200))

"""
.. figure:: csd_recursive_response.png
   :align: center

   Estimated response function using recursive calibration.

"""

fvtk.rm(ren, response_actor)

"""
Now, that we have the response function, we are ready to start the deconvolution
process. Let's import the CSD model and fit the datasets.
"""

from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
csd_model = ConstrainedSphericalDeconvModel(gtab, response)

"""
For illustration purposes we will fit only a small portion of the data.
"""

data_small = data[20:50, 55:85, 38:39]
csd_fit = csd_model.fit(data_small)

"""
Show the CSD-based ODFs also known as FODFs (fiber ODFs).
"""

csd_odf = csd_fit.odf(sphere)

"""
Here we visualize only a 30x30 region.
"""

fodf_spheres = fvtk.sphere_funcs(csd_odf, sphere, scale=1.3, norm=False)

fvtk.add(ren, fodf_spheres)

print('Saving illustration as csd_odfs.png')
fvtk.record(ren, out_path='csd_odfs.png', size=(600, 600))

"""
.. figure:: csd_odfs.png
   :align: center

   CSD ODFs.

In Dipy we also provide tools for finding the peak directions (maxima) of the
ODFs. For this purpose we recommend using ``peaks_from_model``.
"""

from dipy.direction import peaks_from_model

csd_peaks = peaks_from_model(model=csd_model,
                             data=data_small,
                             sphere=sphere,
                             relative_peak_threshold=.5,
                             min_separation_angle=25,
                             parallel=True)

fvtk.clear(ren)
fodf_peaks = fvtk.peaks(csd_peaks.peak_dirs, csd_peaks.peak_values, scale=1.3)
fvtk.add(ren, fodf_peaks)

print('Saving illustration as csd_peaks.png')
fvtk.record(ren, out_path='csd_peaks.png', size=(600, 600))

"""
.. figure:: csd_peaks.png
   :align: center

   CSD Peaks.

We can finally visualize both the ODFs and peaks in the same space.
"""

fodf_spheres.GetProperty().SetOpacity(0.4)

fvtk.add(ren, fodf_spheres)

print('Saving illustration as csd_both.png')
fvtk.record(ren, out_path='csd_both.png', size=(600, 600))


"""
.. figure:: csd_both.png
   :align: center

   CSD Peaks and ODFs.

References
----------

.. [Tournier2007] J-D. Tournier, F. Calamante and A. Connelly, "Robust
   determination of the fibre orientation distribution in diffusion MRI:
   Non-negativity constrained super-resolved spherical deconvolution",
   Neuroimage, vol. 35, no. 4, pp. 1459-1472, 2007.

.. [Tax2014] C.M.W. Tax, B. Jeurissen, S.B. Vos, M.A. Viergever, A. Leemans,
   "Recursive calibration of the fiber response function for spherical
   deconvolution of diffusion MRI data", Neuroimage, vol. 86, pp. 67-80, 2014.

.. include:: ../links_names.inc

"""
