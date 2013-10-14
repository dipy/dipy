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

"""
You can verify the b-values of the datasets by looking at the attribute `gtab.bvals`.

In CSD there is an important pre-processing step: the estimation of the fiber response function. In order to
do this we look for voxel with very anisotropic configurations. For example here we use an ROI (20x20x20) at the center
of the volume and store the signal values for the voxels with FA values higher than 0.7. Of course, if we haven't
precalculated FA we need to fit a Tensor model to the datasets. Which is what we do here.
"""

from dipy.reconst.dti import TensorModel

data = img.get_data()

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
