"""
=======================================================
Reconstruction with Constrained Spherical Deconvolution
=======================================================

This example shows how to use Constrained Spherical Deconvolution (CSD)
introduced by Tournier et al. [Tournier2007]_.

This method is mainly useful with datasets with gradient directions acquired on
a spherical grid.

The basic idea with this method is that if we could estimate the response function of a
single fiber then we could deconvolve the measured signal and obtain the underlying
fiber distribution.

Let's first load the data. We will use a dataset with 10 b0s and 150 non-b0s
with b-value 2000.
"""

import numpy as np

from dipy.data import fetch_stanford_hardi, read_stanford_hardi

fetch_stanford_hardi()
img, gtab = read_stanford_hardi()

data = img.get_data()

"""
You can verify the b-values of the datasets by looking at the attribute `gtab.bvals`.

In CSD there is an important pre-processing step: the estimation of the fiber
response function. In order to do this we look for regions of the brain where
it is known to have single fibers. For example if we use an ROI at the center of
the brain, we will find single fibers from the corpus callosum. The
``auto_response`` function will calculate FA for an ROI of radius equal to
``roi_radius`` in the center of the volume and return the response function
estimated in that region for the voxels with FA higher than 0.7.
"""

from dipy.reconst.csdeconv import auto_response

response, ratio = auto_response(gtab, data, roi_radius=10, fa_thr=0.7)

"""
The ``response`` parameter contains two parameters. The first is an array with
the eigenvalues of the response function and the second is the average S0 for
this response.

It is a very good practice to always validate the result of auto_response. For,
this purpose we can print it and have a look at its values.
"""

print(response)

"""
(array([ 0.0014,  0.00029,  0.00029]), 416.206)

The tensor generated from the response must be prolate (two smaller eigenvalues
should be equal) and look anisotropic with ratio of second to first eigenvalue
of about 0.2. Or in other words the maximum eigenvalue must be around 5 times
larger than the second maximum eigenvalue.
"""

print(ratio)

"""
0.21197

We can double check that we have a good response function by visualizing the
response's function's ODF. Here is how:
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

   **Estimated response function**.

"""

fvtk.rm(ren, response_actor)

"""
Now, that we have the response function, we are ready to start the deconvolution
process. Let's import the CSD model and fit the datasets.
"""

from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel

csd_model = ConstrainedSphericalDeconvModel(gtab, response)

"""
For illustration purposes we will fit only a slice of the datasets.
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

   **CSD ODFs**.

In Dipy we also provide tools for finding the peak directions (maxima) of the
ODFs. For this purpose we strongly recommend using ``peaks_from_model``.
"""

from dipy.reconst.peaks import peaks_from_model

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

   **CSD Peaks**.

We can finally visualize both the ODFs and peaks in the same space.
"""

fodf_spheres.GetProperty().SetOpacity(0.4)

fvtk.add(ren, fodf_spheres)

print('Saving illustration as csd_both.png')
fvtk.record(ren, out_path='csd_both.png', size=(600, 600))


"""
.. figure:: csd_both.png
   :align: center

   **CSD Peaks and ODFs**.

.. [Tournier2007] J-D. Tournier, F. Calamante and A. Connelly, "Robust determination of the fibre orientation distribution in diffusion MRI: Non-negativity constrained super-resolved spherical deconvolution", Neuroimage, vol. 35, no. 4, pp. 1459-1472, 2007.

.. include:: ../links_names.inc

"""
