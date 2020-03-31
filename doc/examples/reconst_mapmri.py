# -*- coding: utf-8 -*-
"""
================================================================
Continuous and analytical diffusion signal modelling with MAPMRI
================================================================

We show how to model the diffusion signal as a linear combination
of continuous functions from the MAPMRI basis [Ozarslan2013]_.
This continuous representation allows for the computation of many
properties of both the signal and diffusion propagator.

We show how to estimate the analytical Orientation Distribution
Function (ODF) and a variety of scalar indices. These include rotationally
invariant quantities such as the Mean Squared Displacement (MSD), Q-space
Inverse Variance (QIV), Return-To-Origin Probability (RTOP) and
Non-Gaussianity (NG). Interestingly, the MAP-MRI basis also allows for
the computation of directional indices, such as the Return To the Axis
Probability (RTAP), the Return To the Plane Probability (RTPP), and
the parallel and perpendicular Non-Gaussianity.

The estimation of these properties from noisy DWIs requires that the
fitting of the MAPMRI basis is regularized. We will show that this can
be done using both constraining the diffusion propagator to positive
values [Ozarslan2013]_ and analytic Laplacian Regularization (MAPL)
[Fick2016a]_.

First import the necessary modules:
"""

from dipy.reconst import mapmri
from dipy.viz import window, actor
from dipy.data import get_fnames, get_sphere
from dipy.core.gradients import gradient_table
from dipy.io.image import load_nifti
from dipy.io.gradients import read_bvals_bvecs
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

"""
Download and read the data for this tutorial.

MAPMRI requires multi-shell data, to properly fit the radial part of the basis.
to ``False`` to only download eddy-current/motion corrected data.
The total size of the downloaded data is 187.66 MBytes, however you only need
to fetch it once.
"""

fraw, fbval, fbvec, t1_fname = get_fnames('cfin_multib')

"""
``data`` contains the voxel data and ``gtab`` contains a ``GradientTable``
object (gradient information e.g. b-values). For example, to show the b-values
it is possible to write::

   print(gtab.bvals)

For the values of the q-space indices to make sense it is necessary to
explicitly state the ``big_delta`` and ``small_delta`` parameters in the
gradient table.
"""

data, affine = load_nifti(fraw)
bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
gtab = gradient_table(bvals, bvecs)

big_delta = 0.0365  # seconds
small_delta = 0.0157  # seconds
gtab = gradient_table(bvals=gtab.bvals, bvecs=gtab.bvecs,
                      big_delta=big_delta,
                      small_delta=small_delta)

data_small = data[40:65, 50:51]

print('data.shape (%d, %d, %d, %d)' % data.shape)

"""
The MAPMRI Model can now be instantiated. The ``radial_order`` determines the
expansion order of the basis, i.e., how many basis functions are used to
approximate the signal.

First, we must decide to use the anisotropic or isotropic MAPMRI basis. As was
shown in [Fick2016a]_, the isotropic basis is best used for tractography
purposes, as the anisotropic basis has a bias towards smaller crossing angles
in the ODF. For signal fitting and estimation of scalar quantities the
anisotropic basis is suggested. The choice can be made by setting
``anisotropic_scaling=True`` or ``anisotropic_scaling=False``.

First, we must select the method of regularization and/or constraining the
basis fitting.

- ``laplacian_regularization=True`` makes it use Laplacian regularization
  [Fick2016a]_. This method essentially reduces spurious oscillations in the
  reconstruction by minimizing the Laplacian of the fitted signal.
  Several options can be given to select the regularization weight:

    - ``regularization_weighting=GCV`` uses generalized cross-validation
      [Craven1978]_ to find an optimal weight.
    - ``regularization_weighting=0.2`` for example would omit the GCV and
      just set it to 0.2 (found to be reasonable in HCP data [Fick2016a]_).
    - ``regularization_weighting=np.array(weights)`` would make the GCV use
      a custom range to find an optimal weight.

- ``positivity_constraint=True`` makes it use the positivity constraint on the
  diffusion propagator [Ozarslan2013]_. This method constrains the final
  solution of the diffusion propagator to be positive at a set of discrete
  points, since negative values should not exist.

    - The ``pos_grid`` and ``pos_radius`` affect the location and number of
      constraint points in the diffusion propagator.

Both methods do a good job of producing viable solutions to the signal fitting
in practice. The difference is that the Laplacian regularization imposes
smoothness over the entire signal, including the extrapolation beyond the
measured signal. In practice this results in, but does not guarantee positive
solutions of the diffusion propagator. The positivity constraint guarantees a
positive solution in a set of discrete points, which in general results in
smooth solutions, but does not guarantee it.

A suggested strategy is to use a low Laplacian weight together with the
positivity constraint. In this way both desired properties are guaranteed in
the final solution.

We use package CVXPY (http://www.cvxpy.org/) to solve convex optimization
problems when "positivity_constraint=True", so we need to first install
CVXPY.

For now we will generate the anisotropic models for all combinations.
"""

radial_order = 6
map_model_laplacian_aniso = mapmri.MapmriModel(gtab, radial_order=radial_order,
                                               laplacian_regularization=True,
                                               laplacian_weighting=.2)

map_model_positivity_aniso = mapmri.MapmriModel(gtab,
                                                radial_order=radial_order,
                                                laplacian_regularization=False,
                                                positivity_constraint=True)

map_model_both_aniso = mapmri.MapmriModel(gtab, radial_order=radial_order,
                                          laplacian_regularization=True,
                                          laplacian_weighting=.05,
                                          positivity_constraint=True)

"""
Note that when we use only Laplacian regularization, the ``GCV`` option may
select very low regularization weights in very anisotropic white matter such
as the corpus callosum, resulting in corrupted scalar indices. In this example
we therefore choose a preset value of 0.2, which has shown to be quite robust
and also faster in practice [Fick2016a]_.

We can then fit the MAPMRI model to the data:
"""

mapfit_laplacian_aniso = map_model_laplacian_aniso.fit(data_small)
mapfit_positivity_aniso = map_model_positivity_aniso.fit(data_small)
mapfit_both_aniso = map_model_both_aniso.fit(data_small)

"""
From the fitted models we will first illustrate the estimation of q-space
indices. For completeness, we will compare the estimation using only Laplacian
regularization, positivity constraint or both. We first show the RTOP
[Ozarslan2013]_.
"""

# generating RTOP plots
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 3, 1, title=r'RTOP - Laplacian')
ax1.set_axis_off()
rtop_laplacian = np.array(mapfit_laplacian_aniso.rtop()[:, 0, :].T,
                          dtype=float)
ind = ax1.imshow(rtop_laplacian, interpolation='nearest',
                 origin='lower', cmap=plt.cm.gray)

ax2 = fig.add_subplot(1, 3, 2, title=r'RTOP - Positivity')
ax2.set_axis_off()
rtop_positivity = np.array(mapfit_positivity_aniso.rtop()[:, 0, :].T,
                           dtype=float)
ind = ax2.imshow(rtop_positivity, interpolation='nearest',
                 origin='lower', cmap=plt.cm.gray)

ax3 = fig.add_subplot(1, 3, 3, title=r'RTOP - Both')
ax3.set_axis_off()
rtop_both = np.array(mapfit_both_aniso.rtop()[:, 0, :].T, dtype=float)
ind = ax3.imshow(rtop_both, interpolation='nearest', origin='lower',
                 cmap=plt.cm.gray)
divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(ind, cax=cax)

plt.savefig('MAPMRI_maps_regularization.png')

"""
.. figure:: MAPMRI_maps_regularization.png
   :align: center

It can be seen that all maps appear quite smooth and similar. Though, it is
possible to see some subtle differences near the corpus callosum. The
similarity and differences in reconstruction can be further illustrated by
visualizing the analytic norm of the Laplacian of the fitted signal.
"""

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 3, 1, title=r'Laplacian norm - Laplacian')
ax1.set_axis_off()
laplacian_norm_laplacian = np.array(mapfit_laplacian_aniso.norm_of_laplacian_signal()[:, 0, :].T,
                dtype=float)
ind = ax1.imshow(laplacian_norm_laplacian, interpolation='nearest',
                 origin='lower', cmap=plt.cm.gray, vmin=0, vmax=3)

ax2 = fig.add_subplot(1, 3, 2, title=r'Laplacian norm - Positivity')
ax2.set_axis_off()
laplacian_norm_positivity = np.array(mapfit_positivity_aniso.norm_of_laplacian_signal()[:, 0, :].T,
                dtype=float)
ind = ax2.imshow(laplacian_norm_positivity, interpolation='nearest',
                 origin='lower', cmap=plt.cm.gray, vmin=0, vmax=3)

ax3 = fig.add_subplot(1, 3, 3, title=r'Laplacian norm - Both')
ax3.set_axis_off()
laplacian_norm_both = np.array(mapfit_both_aniso.norm_of_laplacian_signal()[:, 0, :].T,
                dtype=float)
ind = ax3.imshow(laplacian_norm_both, interpolation='nearest', origin='lower',
                 cmap=plt.cm.gray, vmin=0, vmax=3)
divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(ind, cax=cax)
plt.savefig('MAPMRI_norm_laplacian.png')

"""
.. figure:: MAPMRI_norm_laplacian.png
   :align: center

A high Laplacian norm indicates that the gradient in the three-dimensional
signal reconstruction changes a lot - something that may indicate spurious
oscillations. In the Laplacian reconstruction (left) we see that there are some
isolated voxels that have a higher norm than the rest. In the
positivity constraint reconstruction the norm is already smoother. When both
methods are used together the overall norm gets smoother still, since both
smoothness of the signal and positivity of the propagator are imposed.

From now on we just use the combined approach, show all maps we can generate
and explain their significance.
"""

fig = plt.figure(figsize=(15, 6))
ax1 = fig.add_subplot(1, 5, 1, title=r'MSD')
ax1.set_axis_off()
msd = np.array(mapfit_both_aniso.msd()[:, 0, :].T, dtype=float)
ind = ax1.imshow(msd, interpolation='nearest', origin='lower',
                 cmap=plt.cm.gray)

ax2 = fig.add_subplot(1, 5, 2, title=r'QIV')
ax2.set_axis_off()
qiv = np.array(mapfit_both_aniso.qiv()[:, 0, :].T, dtype=float)
ind = ax2.imshow(qiv, interpolation='nearest', origin='lower',
                 cmap=plt.cm.gray)

ax3 = fig.add_subplot(1, 5, 3, title=r'RTOP')
ax3.set_axis_off()
rtop = np.array((mapfit_both_aniso.rtop()[:, 0, :]).T, dtype=float)
ind = ax3.imshow(rtop, interpolation='nearest', origin='lower',
                 cmap=plt.cm.gray)

ax4 = fig.add_subplot(1, 5, 4, title=r'RTAP')
ax4.set_axis_off()
rtap = np.array((mapfit_both_aniso.rtap()[:, 0, :]).T, dtype=float)
ind = ax4.imshow(rtap, interpolation='nearest', origin='lower',
                 cmap=plt.cm.gray)

ax5 = fig.add_subplot(1, 5, 5, title=r'RTPP')
ax5.set_axis_off()
rtpp = np.array(mapfit_both_aniso.rtpp()[:, 0, :].T, dtype=float)
ind = ax5.imshow(rtpp, interpolation='nearest', origin='lower',
                 cmap=plt.cm.gray)
plt.savefig('MAPMRI_maps.png')

"""
.. figure:: MAPMRI_maps.png
   :align: center

From left to right:

- Mean Squared Displacement (MSD) is a measure for how far protons are able to
  diffuse. a decrease in MSD indicates protons are hindered/restricted more,
  as can be seen by the high MSD in the CSF, but low in the white matter.
- Q-space Inverse Variance (QIV) is a measure of variance in the signal, which
  is said to have higher contrast to white matter than the MSD
  [Hosseinbor2013]_. We also showed that QIV has high sensitivity to tissue
  composition change in a simulation study [Fick2016b]_.
- Return to origin probability (RTOP) quantifies the probability that a proton
  will be at the same position at the first and second diffusion gradient
  pulse. A higher RTOP indicates that the volume a spin is inside of is
  smaller, meaning more overall restriction. This is illustrated by the low
  values in CSF and high values in white matter.
- Return to axis probability (RTAP) is a directional index that quantifies
  the probability that a proton will be along the axis of the main eigenvector
  of a diffusion tensor during both diffusion gradient pulses. RTAP has been
  related to the apparent axon diameter [Ozarslan2013]_ [Fick2016a]_ under
  several strong assumptions on the tissue composition and acquisition
  protocol.
- Return to plane probability (RTPP) is a directional index that quantifies the
  probability that a proton will be on the plane perpendicular to the main
  eigenvector of a diffusion tensor during both gradient pulses. RTPP is
  related to the length of a pore [Ozarslan2013]_ but in practice should be
  similar to that of Gaussian diffusion.

It is also possible to estimate the amount of non-Gaussian diffusion in every
voxel [Ozarslan2013]_. This quantity is estimated through the ratio between
Gaussian volume (MAPMRI's first basis function) and the non-Gaussian volume
(all other basis functions) of the fitted signal. For this value to be
physically meaningful we must use a b-value threshold in the MAPMRI model. This
threshold makes the scale estimation in MAPMRI only use samples that
realistically describe Gaussian diffusion, i.e., at low b-values.
"""

map_model_both_ng = mapmri.MapmriModel(gtab, radial_order=radial_order,
                                       laplacian_regularization=True,
                                       laplacian_weighting=.05,
                                       positivity_constraint=True,
                                       bval_threshold=2000)

mapfit_both_ng = map_model_both_ng.fit(data_small)

fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(1, 3, 1, title=r'NG')
ax1.set_axis_off()
ng = np.array(mapfit_both_ng.ng()[:, 0, :].T, dtype=float)
ind = ax1.imshow(ng, interpolation='nearest', origin='lower',
                 cmap=plt.cm.gray)
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(ind, cax=cax)


ax2 = fig.add_subplot(1, 3, 2, title=r'NG perpendicular')
ax2.set_axis_off()
ng_perpendicular = np.array(mapfit_both_ng.ng_perpendicular()[:, 0, :].T,
                            dtype=float)
ind = ax2.imshow(ng_perpendicular, interpolation='nearest', origin='lower',
                 cmap=plt.cm.gray)
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(ind, cax=cax)

ax3 = fig.add_subplot(1, 3, 3, title=r'NG parallel')
ax3.set_axis_off()
ng_parallel = np.array(mapfit_both_ng.ng_parallel()[:, 0, :].T, dtype=float)
ind = ax3.imshow(ng_parallel, interpolation='nearest', origin='lower',
                 cmap=plt.cm.gray)
divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(ind, cax=cax)
plt.savefig('MAPMRI_ng.png')

"""
.. figure:: MAPMRI_ng.png
   :align: center

On the left we see the overall NG and on the right the directional
perpendicular NG and parallel NG. The NG ranges from 1 (completely
non-Gaussian) to 0 (completely Gaussian). The overall NG of a voxel is always
higher or equal than each of its components. It can be seen that NG has low
values in the CSF and higher in the white matter.

Increases or decreases in these values do not point to a specific
microstructural change, but can indicate clues as to what is happening, similar
to Fractional Anisotropy. An initial simulation study that quantifies the added
value of q-space indices over DTI indices is given in [Fick2016b]_.

The MAPMRI framework also allows for the estimation of Orientation Distribution
Functions (ODFs). We recommend to use the isotropic implementation for this
purpose, as the anisotropic implementation has a bias towards smaller crossing
angles.

For the isotropic basis we recommend to use a ``radial_order`` of 8, as the
basis needs more generic and needs more basis functions to approximate the
signal.
"""

radial_order = 8
map_model_both_iso = mapmri.MapmriModel(gtab, radial_order=radial_order,
                                        laplacian_regularization=True,
                                        laplacian_weighting=.1,
                                        positivity_constraint=True,
                                        anisotropic_scaling=False)

mapfit_both_iso = map_model_both_iso.fit(data_small)

"""
Load an ODF reconstruction sphere
"""

sphere = get_sphere('repulsion724')

"""
Compute the ODFs.

The radial order ``s`` can be increased to sharpen the results, but it might
also make the ODFs noisier. Always check the results visually.
"""

odf = mapfit_both_iso.odf(sphere, s=2)
print('odf.shape (%d, %d, %d, %d)' % odf.shape)

"""
Display the ODFs.
"""

# Enables/disables interactive visualization
interactive = False

r = window.Renderer()
sfu = actor.odf_slicer(odf, sphere=sphere, colormap='plasma', scale=0.5)
sfu.display(y=0)
sfu.RotateX(-90)
r.add(sfu)
window.record(r, out_path='odfs.png', size=(600, 600))
if interactive:
    window.show(r)

"""
.. figure:: odfs.png
   :align: center

   Orientation distribution functions (ODFs).

References
----------

.. [Ozarslan2013] Ozarslan E. et al., "Mean apparent propagator (MAP) MRI: A
   novel diffusion imaging method for mapping tissue microstructure",
   NeuroImage, 2013.

.. [Fick2016a] Fick, Rutger HJ, et al. "MAPL: Tissue microstructure estimation
   using Laplacian-regularized MAP-MRI and its application to HCP data."
   NeuroImage (2016).

.. [Craven1978] Craven et al. "Smoothing Noisy Data with Spline Functions."
   NUMER MATH 31.4 (1978): 377-403.

.. [Hosseinbor2013] Hosseinbor et al. "Bessel fourier orientation
   reconstruction (bfor): an analytical diffusion propagator reconstruction
   for hybrid diffusion imaging and computation of q-space indices. NeuroImage
   64, 650-670.

.. [Fick2016b] Fick et al. "A sensitivity analysis of Q-space indices with
   respect to changes in axonal diameter, dispersion and tissue composition.
   ISBI 2016.

.. include:: ../links_names.inc

"""
