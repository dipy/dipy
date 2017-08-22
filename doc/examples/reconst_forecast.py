"""
==============================================================
Crossing invariant fiber response function with FORECAST model
==============================================================

We show how to obtain a voxel specific response function in the form of
axially symmetric tensor and the fODF using the FORECAST model from
[Anderson2005]_ , [Kaden2016]_ and [Zucchelli2017]_.

First import the necessary modules:
"""
import numpy as np
import matplotlib.pyplot as plt

from dipy.reconst.forecast import ForecastModel
from dipy.reconst.shm import sh_to_sf
from dipy.viz import fvtk
from dipy.data import fetch_sherbrooke_3shell, read_sherbrooke_3shell, get_sphere
from dipy.core.gradients import gradient_table

"""
Download and read the data for this tutorial.

fetch_sherbrooke_3shell() provides data acquired using three shells at
b-values 1000, 2000, and 3500.
"""

fetch_sherbrooke_3shell()
img, gtab = read_sherbrooke_3shell()
data = img.get_data()

print('data.shape (%d, %d, %d, %d)' % data.shape)

"""
sherbrooke_3shell has the x axis of the gradients flipped with respect to
Dipy convention. We fix this by flipping the bvecs x axis 
"""

bvecs_corrected = gtab.bvecs * np.array([-1,1,1])
gtab_corrected = gradient_table(gtab.bvals, bvecs_corrected)

"""
First, let us mask the data
"""

from dipy.segment.mask import median_otsu
data_masked, mask = median_otsu(data, 2, 1)

"""
sherbrooke_3shell contains only one b0 image, in these cases it is always better
to denoise the b0 image before processing the data.
For more information about nlmeans denoising check the relative example in the
gallery.
"""

from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma

data_b0 = data[..., 0]

sigma = estimate_sigma(data_b0, N=4)

denoised_b0 = nlmeans(data_b0, sigma=sigma, mask=mask, patch_radius=1, block_radius=3, rician=False)

data[...,0] = denoised_b0

"""
Let us consider only a single slice for the FORECAST fitting	
"""
# axial_middle = data.shape[2] // 2
# data_small = data[:, :, axial_middle]
# mask_small = mask[:, :, axial_middle]

data_small = data[26:100, 64:65, :]
mask_small = mask[26:100, 64:65, :] 

"""
Instantiate the FORECAST Model.

sh_order is the spherical harmonics order used for the fODF.

optimizer is the algorithm used for the FORECAST basis fitting, in this case
we used the Constrained Spherical Deconvolution (CSD) algorithm.

"""
fm = ForecastModel(gtab_corrected, sh_order=6, optimizer='csd')

"""
Fit the FORECAST to the data
"""

f_fit = fm.fit(data_small, mask_small)

"""
Calculate the crossing invariant tensor indices [Kaden2016]_ : the parallel diffusivity,
the perpendicular diffusivity, the fractional anisotropy and the mean
diffusivity.
"""

d_par = f_fit.dpar
d_perp = f_fit.dperp
fa = f_fit.fractional_anisotropy()
md = f_fit.mean_diffusivity()

"""
Show the indices and save them in FORECAST_indices.png.
"""

fig = plt.figure(figsize=(6, 6))
ax1 = fig.add_subplot(2, 2, 1, title='parallel diffusivity')
ax1.set_axis_off()
ind = ax1.imshow(d_par[:,0,:].T, interpolation='nearest', origin='lower', cmap = plt.cm.gray)
plt.colorbar(ind, shrink=0.6)
ax2 = fig.add_subplot(2, 2, 2, title='perpendicular diffusivity')
ax2.set_axis_off()
ind = ax2.imshow(d_perp[:,0,:].T, interpolation='nearest', origin='lower', cmap = plt.cm.gray)
plt.colorbar(ind, shrink=0.6)
ax3 = fig.add_subplot(2, 2, 3, title='fractional anisotropy')
ax3.set_axis_off()
ind = ax3.imshow(fa[:,0,:].T, interpolation='nearest', origin='lower', cmap = plt.cm.gray)
plt.colorbar(ind, shrink=0.6)
ax4 = fig.add_subplot(2, 2, 4, title='mean diffusivity')
ax4.set_axis_off()
ind = ax4.imshow(md[:,0,:].T, interpolation='nearest', origin='lower', cmap = plt.cm.gray)
plt.colorbar(ind, shrink=0.6)
plt.savefig('FORECAST_indices.png')

"""
.. figure:: FORECAST_indices.png
   :align: center

   **FORECAST scalar indices**.

"""


"""
Load an odf reconstruction sphere
"""

sphere = get_sphere('symmetric724')

"""
Compute the fODFs. 
"""

odf = f_fit.odf(sphere)
print('fODF.shape (%d, %d, %d, %d)' % odf.shape)

"""
Display a part of the fODFs
"""

from dipy.viz import fvtk
r = fvtk.ren()
sfu = fvtk.sphere_funcs(odf[40:60,:,30:45], sphere, colormap='jet')
sfu.RotateX(-90)
fvtk.add(r, sfu)
fvtk.show(r)
fvtk.record(r, n_frames=1, out_path='fODFs.png', size=(600, 600), 
			magnification=4)

"""
.. figure:: fODFs.png
   :align: center

   **Fiber Orientation Distribution Functions, in a small ROI of the brain**.
   
.. [Anderson2005] Anderson A. W., "Measurement of Fiber Orientation Distributions
       Using High Angular Resolution Diffusion Imaging", Magnetic
       Resonance in Medicine, 2005.

.. [Kaden2016] Kaden E. et. al, "Quantitative Mapping of the Per-Axon Diffusion 
       Coefficients in Brain White Matter", Magnetic Resonance in 
       Medicine, 2016.

.. [Zucchelli2017] Zucchelli E. et. al, "A generalized SMT-based framework for
       Diffusion MRI microstructural model estimation", MICCAI Workshop
       on Computational DIFFUSION MRI (CDMRI), 2017.
.. include:: ../links_names.inc

"""
