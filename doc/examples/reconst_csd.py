"""

=======================================================
Reconstruction with Constrained Spherical Deconvolution
=======================================================

This example shows how to use Constrained Spherical Deconvolution 
introduced by Tournier et al. [Tournier2007]_.

This method is mainly useful with datasets with gradient directions acquired on 
a spherical grid.

The basic idea here is that if we could estimate the response function of a 
single fiber then we could deconvolve the measured signal and obtain the underlying
fiber distribution.

Load data.
"""

import numpy as np

from dipy.data import fetch_stanford_hardi, read_stanford_hardi

fetch_stanford_hardi()
img, gtab = read_stanford_hardi()


"""
Estimate single fiber response function using an ROI at the center of the volume 
and FA values higher 0.7
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

data2 = data[ci - w : ci + w, 
             cj - w : cj + w,
             ck - w : ck + w]

tenfit = tenmodel.fit(data2)

from dipy.reconst.dti import fractional_anisotropy

FA = fractional_anisotropy(tenfit.evals)
FA[np.isnan(FA)] = 0

indices = np.where(FA > 0.7)

lambdas = tenfit.evals[indices][:, :2]

S0s = data2[indices][:, 0]

S0 = np.mean(S0s)

l01 = np.mean(lambdas, axis = 0) 

evals = np.array([l01[0], l01[1], l01[1]])

"""
Import ther CSD model and fit a slice of the data.
"""

from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel

csd_model = ConstrainedSphericalDeconvModel(gtab, (evals, S0))

csd_fit = csd_model.fit(data[:, :, 30], mask[:, :, 30])

"""
Visualize the CSD-based ODFs also known as FODFs.
"""

from dipy.data import get_sphere

sphere = get_sphere('symmetric724')

csd_odf = csd_fit.odf(sphere)

from dipy.viz import fvtk

r = fvtk.ren()
fvtk.add(r, fvtk.sphere_funcs(csd_odf[30:60, 40:70, None], sphere))

print('Saving illustration as csd_odfs.png')
fvtk.record(r, n_frames=1, out_path='csd_odfs.png', size=(600, 600))

"""
.. figure:: csd_odfs.png
   :align: center

   **CSD ODFs**.

.. [Tournier2007] J-D. Tournier, F. Calamante and A. Connelly, "Robust determination of the fibre orientation distribution in diffusion MRI: Non-negativity constrained super-resolved spherical deconvolution", Neuroimage, vol. 35, no. 4, pp. 1459-1472, 2007.

.. include:: ../links_names.inc

"""
