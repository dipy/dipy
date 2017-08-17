"""
==============================================================
Crossing invariant fiber response function with FORECAST model
==============================================================

We show how to obtain a voxel specific response function in the form of
axially symmetric tensor and the fODF using the FORECAST model.

First import the necessary modules:
"""

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

data_small = data[26:100, 64, :]

print('data.shape (%d, %d, %d, %d)' % data.shape)

"""
Instantiate the FORECAST Model.

sh_order is the spherical harmonics order used for the fODF.

optimizer is the algorithm used for the FORECAST basis fitting, in this case
we used the Constrained Spherical Deconvolution (CSD) algorithm.

"""
fm = ForecastModel(gtab, sh_order=8, optimizer='csd')

"""
Fit the FORECAST to the data
"""

f_fit = fm.fit(data_small)

"""
Calculate the crossing invariant tensor indices: the parallel diffusivity,
the perpendicular diffusivity, the fractional anisotropy and the mean
diffusivity.
"""

d_par = f_fit.dpar
d_perp = f_fit.dperp
fa = f_fit.fractional_anisotropy()
md = f_fit.mean_diffusivity()

"""
Show the calculated indices
"""


"""
Show the indices and save them in FORECAST_indices.png.
"""

fig = plt.figure(figsize=(6, 6))
ax1 = fig.add_subplot(2, 2, 1, title='parallel diffusivity')
ax1.set_axis_off()
ind = ax1.imshow(d_par.T, interpolation='nearest', origin='lower', cmap = plt.cm.gray)
plt.colorbar(ind)
ax2 = fig.add_subplot(2, 2, 2, title='perpendicular diffusivity')
ax2.set_axis_off()
ind = ax2.imshow(d_perp.T, interpolation='nearest', origin='lower', cmap = plt.cm.gray)
plt.colorbar(ind)
ax3 = fig.add_subplot(2, 2, 3, title='fractional anisotropy')
ax3.set_axis_off()
ind = ax3.imshow(fa.T, interpolation='nearest', origin='lower', cmap = plt.cm.gray)
plt.colorbar(ind)
ax4 = fig.add_subplot(2, 2, 4, title='mean diffusivity')
ax4.set_axis_off()
ind = ax4.imshow(md.T, interpolation='nearest', origin='lower', cmap = plt.cm.gray)
plt.colorbar(ind)
plt.savefig('FORECAST_indices.png')


"""
.. figure:: FORECAST_indices.png
   :align: center

   **FORECAST scalar indices**.

"""


# """
# Load an odf reconstruction sphere
# """

# sphere = get_sphere('symmetric724')

# """
# Compute the ODFs
# """

# odf = asmfit.odf(sphere)
# print('odf.shape (%d, %d, %d)' % odf.shape)

# """
# Display the ODFs
# """

# r = fvtk.ren()
# sfu = fvtk.sphere_funcs(odf[:, None, :], sphere, colormap='jet')
# sfu.RotateX(-90)
# fvtk.add(r, sfu)
# fvtk.record(r, n_frames=1, out_path='odfs.png', size=(600, 600))

"""
.. figure:: odfs.png
   :align: center

   **Orientation distribution functions**.
   
.. [Merlet2013] Merlet S. et. al, "Continuous diffusion signal, EAP and ODF
				estimation via Compressive Sensing in diffusion MRI", Medical
				Image Analysis, 2013.

.. [Cheng2011] Cheng J. et. al, "Theoretical Analysis and Pratical Insights
			   on EAP Estimation via Unified HARDI Framework", MICCAI
			   workshop on Computational Diffusion MRI, 2011.

.. include:: ../links_names.inc

"""
