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

Lets first load the data. We will use a dataset with 10 b0s and 150 non-b0s with b-value 2000.
"""

import numpy as np

from dipy.data import fetch_stanford_hardi, read_stanford_hardi

fetch_stanford_hardi()
img, gtab = read_stanford_hardi()

data = img.get_data()

"""
You can verify the b-values of the datasets by looking at the attribute `gtab.bvals`.

In CSD there is an important pre-processing step: the estimation of the fiber response 
function. In order to do this we look for voxels with very anisotropic configurations. 
For example here we use an ROI (20x20x20) at the center of the volume hoping to find 
an area of corpus callosum and store the signal values for the voxels with FA values 
higher than 0.7. Of course, if we haven't precalculated FA we need to fit a Tensor 
model to the datasets. Which is what we do with the `auto_response` function here.
"""

from dipy.reconst.csdeconv import auto_response

response, ratio = auto_response(gtab, data, w=10, fa_thr=0.7)

"""
Now we are ready to import the CSD model and fit the datasets.
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

from dipy.data import get_sphere

sphere = get_sphere('symmetric724')

csd_odf = csd_fit.odf(sphere)

from dipy.viz import fvtk

ren = fvtk.ren()

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

.. [Tournier2007] J-D. Tournier, F. Calamante and A. Connelly, "Robust determination of the fibre orientation distribution in diffusion MRI: Non-negativity constrained super-resolved spherical deconvolution", Neuroimage, vol. 35, no. 4, pp. 1459-1472, 2007.

.. include:: ../links_names.inc

"""
