# -*- coding: utf-8 -*-
"""
================================================================
Bayesian uncertainty quantification in linear dMRI models
================================================================

We show how to quantify the uncertainty of properties
estimated with popular dMRI models that use linear least-squares
[Sjolund2018]_.

We consider examples using mean diffusivity (MD) and fractional
anistropy (FA) in DTI, and Return-To-Origin-Probability (RTOP)
in MAPMRI [Ozarslan2013]_ [Fick2016]_.

First import the necessary modules:
"""

import numpy as np
import matplotlib.pyplot as plt

from dipy.data import fetch_cfin_multib, read_cfin_dwi, get_sphere
from dipy.core.gradients import gradient_table

import dipy.reconst.mapmri as mapmri
import dipy.reconst.dti as dti

"""
Download and read the data for this tutorial.

MAPMRI requires multi-shell data, to properly fit the radial part of the basis.
The total size of the downloaded data is 187.66 MBytes, however you only need
to fetch it once.
"""

fetch_cfin_multib()

"""
``data`` contains the voxel data and ``gtab`` contains a ``GradientTable``
object (gradient information e.g. b-values). For example, to show the b-values
it is possible to write::

   print(gtab.bvals)

For the values of the q-space indices to make sense it is necessary to
explicitly state the ``big_delta`` and ``small_delta`` parameters in the
gradient table.
"""

img, gtab = read_cfin_dwi()
big_delta = 0.0365  # seconds
small_delta = 0.0157  # seconds
gtab = gradient_table(bvals=gtab.bvals, bvecs=gtab.bvecs,
                      big_delta=big_delta,
                      small_delta=small_delta)
data = img.get_data()
data_small = data[40:65, 50:51]

print('data.shape (%d, %d, %d, %d)' % data.shape)

"""
First, lets consider DTI and fit a model using weighted least-squares.
"""

tenmodel = dti.TensorModel(gtab, fit_method='WLS', return_S0_hat=True)

tenfit = tenmodel.fit(data_small)

"""
Mean diffusivity (MD) is linear in the DTI coefficients, so we can
evaluate the posterior in closed form and e.g. compute confidence intervals.
"""

md_mean = tenfit.md
md_95_width = tenfit.md_interval_width(confidence=0.95)
confidence_interval = np.stack(
    [md_mean - md_95_width/2, md_mean + md_95_width/2], axis=-1)

# Plot
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 3, 1, title=r'MD - Mean')
ax1.set_axis_off()

ind = ax1.imshow(md_mean[:, 0, :].T,
                 interpolation='nearest', origin='lower', cmap=plt.cm.gray)

ax2 = fig.add_subplot(1, 3, 2, title=r'MD - 2.5%')
ax2.set_axis_off()

vmin = np.min(confidence_interval[:, 0, :, 0])
vmax = np.max(confidence_interval[:, 0, :, 1])
ind = ax2.imshow(confidence_interval[:, 0, :, 0].T,
                 interpolation='nearest', origin='lower', cmap=plt.cm.gray,
                 vmin=vmin, vmax=vmax)

ax3 = fig.add_subplot(1, 3, 3, title=r'MD - 97.5%')
ax3.set_axis_off()
ind = ax3.imshow(confidence_interval[:, 0, :, 1].T,
                 interpolation='nearest', origin='lower', cmap=plt.cm.gray,
                 vmin=vmin, vmax=vmax)

plt.savefig('md.png')

"""
.. figure:: md.png
   :align: center

Fractional anistropy (FA) on the other hand, is
nonlinear in the DTI coefficients. This means
we have to resort to sampling.
Here, we'll compare two samples.
"""

np.random.seed(0)
fa_samples = tenfit.fa_samples(n_samples=2)

# Plot
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 3, 1, title=r'FA - Sample 1')
ax1.set_axis_off()

ind = ax1.imshow(fa_samples[:, 0, :, 0].T, interpolation='nearest',
                 origin='lower', cmap=plt.cm.gray)

ax2 = fig.add_subplot(1, 3, 2, title=r'FA - Sample 2')
ax2.set_axis_off()

ind = ax2.imshow(fa_samples[:, 0, :, 1].T, interpolation='nearest',
                 origin='lower', cmap=plt.cm.gray)

ax3 = fig.add_subplot(1, 3, 3, title=r'FA - Diff 2-1')
ax3.set_axis_off()
ind = ax3.imshow(fa_samples[:, 0, :, 1].T-fa_samples[:, 0, :, 0].T,
                 interpolation='nearest', origin='lower', cmap=plt.cm.gray)

plt.savefig('fa_samples.png')

"""
.. figure:: fa_samples.png
   :align: center

Now, we'll consider RTOP estimation for MAP-MRI and fit a model
with Laplacian regularization [Fick2016]_.
"""
radial_order = 6
map_model_laplacian_aniso = mapmri.MapmriModel(
    gtab, radial_order=radial_order,
    laplacian_regularization=True, laplacian_weighting=.2,
    anisotropic_scaling=True, positivity_constraint=False)

mapfit = map_model_laplacian_aniso.fit(data_small)

"""
The conventional RTOP estimate is merely
the mean of a t-distribution [Sjolund2018]_.
"""

rtop_mean = mapfit.rtop()

"""
RTOP is linear in the MAPMRI coefficients,
so we can evaluate the posterior in closed form.
"""

rtop_percentiles = mapfit.rtop_percentile(np.array([0.25, 0.75]))
rtop_iqr = rtop_percentiles[..., 1] - rtop_percentiles[..., 0]

# For convenience, there's also an equivalent property:
# mapfit.rtop_interquartile_range

"""
But, we could also estimate it by sampling. However, for multi-voxel fits
it doesn't work to define external functions like this:
"""


def rtop_func(coeff):
    return np.dot(mapfit.rtop_matrix, coeff.T)

np.random.seed(0)
rtop_sampled_percentiles = mapfit.percentiles(
    rtop_func, np.array([0.25, 0.75]), n_samples=1000)
rtop_iqr_sampled = (rtop_sampled_percentiles[..., 1]
                    - rtop_sampled_percentiles[..., 0])

plt.hist((rtop_iqr.flatten() - rtop_iqr_sampled.flatten())
         / rtop_iqr.flatten(), bins=25)  # Relative errors

"""
This seems to have something to do with references to self,
because it works if you define essentially the same function
as an instance method:
"""

"""
def rtop_percentile_sampled(self, probabilities): # Define this in MapmriFit

    def rtop_func(coeff):
        return np.dot(self.rtop_matrix, coeff.T)

    return self.percentiles(rtop_func, probabilities)

np.random.seed(0)
rtop_sampled_percentiles = mapfit.rtop_percentile_sampled(
    np.array([0.25, 0.75]))
rtop_iqr_sampled = (rtop_sampled_percentiles[..., 1]
                    - rtop_sampled_percentiles[..., 0])

plt.hist((rtop_iqr.flatten() - rtop_iqr_sampled.flatten())
         / rtop_iqr.flatten(), bins=25);  # Relative errors
"""
"""
Plot a comparison of the analytical and stochastic estimates:
"""
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 3, 1, title=r'RTOP - Mean')
ax1.set_axis_off()

ind = ax1.imshow(rtop_mean[:, 0, :].T, interpolation='nearest',
                 origin='lower', cmap=plt.cm.gray)

ax2 = fig.add_subplot(1, 3, 2, title=r'RTOP IQR - Analytical')
ax2.set_axis_off()

vmin = np.min(rtop_iqr[:, 0, :].T)
vmax = np.max(rtop_iqr[:, 0, :].T)
ind = ax2.imshow(rtop_iqr[:, 0, :].T, interpolation='nearest',
                 origin='lower', cmap=plt.cm.gray, vmin=vmin, vmax=vmax)

ax3 = fig.add_subplot(1, 3, 3, title=r'RTOP IQR - Stochastic')
ax3.set_axis_off()
ind = ax3.imshow(rtop_iqr_sampled[:, 0, :].T,
                 interpolation='nearest', origin='lower', cmap=plt.cm.gray,
                 vmin=vmin, vmax=vmax)

plt.savefig('mapmri_rtop.png')

"""
.. figure:: mapmri_rtop.png
   :align: center

TODO: perhaps it would be clearner to refactor such that there's just two
quantile functions (percentiles are a special case of quantiles):
one analytical for the linear case (that takes a matrix as input) and one
sampling based for the nonlinear case (that takes a function as input).
Among other things, this would mean replacing all uses of interval_width
with quantile functions.

TODO: add exemple on crossing angle estimation with CSD

References
----------

.. [Ozarslan2013] Ozarslan E. et al., "Mean apparent propagator (MAP) MRI: A
   novel diffusion imaging method for mapping tissue microstructure",
   NeuroImage, 2013.

.. [Fick2016] Fick, Rutger HJ, et al. "MAPL: Tissue microstructure estimation
   using Laplacian-regularized MAP-MRI and its application to HCP data."
   NeuroImage (2016).

.. [Sjolund2018] Sjölund, Jens, et al. "Bayesian uncertainty quantification
   in linear models for diffusion MRI", NeuroImage, 2018.

.. include:: ../links_names.inc
"""