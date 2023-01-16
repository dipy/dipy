# -*- coding: utf-8 -*-
"""
=================================================================
Continuous and analytical diffusion signal modelling with MAP-MRI
=================================================================

We show how to model the diffusion signal as a linear combination of continuous
functions from the MAP-MRI basis [Ozarslan2013]_. This continuous representation
allows for the computation of many properties of both the signal and diffusion
propagator.

We show how to estimate the analytical orientation distribution function (ODF)
and a variety of scalar indices. These include rotationally invariant quantities
such as the mean squared displacement (MSD), Q-space inverse variance (QIV),
teturn-to-origin probability (RTOP) and non-Gaussianity (NG). Interestingly, the
MAP-MRI basis also allows for the computation of directional indices, such as
the return-to-axis probability (RTAP), the return-to-plane probability
(RTPP), and the parallel and perpendicular non-Gaussianity.

The estimation of these properties from noisy and sparse DWIs requires that the
fitting of the MAP-MRI basis is constrained and/or regularized. This can be done
by constraining the diffusion propagator to positive values [Ozarslan2013]_
[DelaHaije2020]_, and through analytic Laplacian regularization (MAPL)
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

MAP-MRI requires multi-shell data, to properly fit the radial part of the basis.
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
The MAP-MRI Model can now be instantiated. The ``radial_order`` determines the
expansion order of the basis, i.e., how many basis functions are used to
approximate the signal.

First, we must decide to use the anisotropic or isotropic MAP-MRI basis. As was
shown in [Fick2016a]_, the isotropic basis is best used for tractography
purposes, as the anisotropic basis has a bias towards smaller crossing angles
in the ODF. For signal fitting and estimation of scalar quantities the
anisotropic basis is suggested. The choice can be made by setting
``anisotropic_scaling=True`` or ``anisotropic_scaling=False``.

First, we must select the method of regularization and/or constraining the
basis fitting.

- ``laplacian_regularization=True`` makes it use Laplacian regularization
  [Fick2016a]_. This method essentially reduces spurious oscillations in the
  reconstruction by minimizing the Laplacian of the fitted signal. Several
  options can be given to select the regularization weight:

    - ``regularization_weighting=GCV`` uses generalized cross-validation
      [Craven1978]_ to find an optimal weight.
    - ``regularization_weighting=0.2`` for example would omit the GCV and
      just set it to 0.2 (found to be reasonable in HCP data [Fick2016a]_).
    - ``regularization_weighting=np.array(weights)`` would make the GCV use
      a custom range to find an optimal weight.

- ``positivity_constraint=True`` makes it use a positivity constraint on the
  diffusion propagator. This method constrains the final solution of the
  diffusion propagator to be positive either globally [DelaHaije2020] or at a
  set of discrete points [Ozarslan2013]_, since negative values should not
  exist.

    - Setting ``constraint_type='global'`` enforces positivity everywhere.
      With the setting ``constraint_type='local'`` positivity is enforced on
      a grid determined by ``pos_grid`` and ``pos_radius``.

Both methods do a good job of producing viable solutions to the signal fitting
in practice. The difference is that the Laplacian regularization imposes
smoothness over the entire signal, including the extrapolation beyond the
measured signal. In practice this may result in, but does not guarantee,
positive solutions of the diffusion propagator. The positivity constraint
guarantees a positive solution which in general results in smooth solutions, but
does not guarantee it.

A suggested strategy is to use a low Laplacian weight together with a positivity
constraint. In this way both desired properties are guaranteed in final
solution. Higher Laplacian weights are recommended when the data is very noisy.

We use the package CVXPY (https://www.cvxpy.org/) to solve convex optimization
problems when ``positivity_constraint=True``, so we need to first install CVXPY.
When using ``constraint_type='global'`` to ensure global positivity, it is
recommended to use the MOSEK solver (https://www.mosek.com/) together with CVXPY
by setting ``cvxpy_solver='MOSEK'``.

For now we will generate the anisotropic models for different combinations.
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

map_model_plus_aniso = mapmri.MapmriModel(gtab,
                                          radial_order=radial_order,
                                          laplacian_regularization=False,
                                          positivity_constraint=True,
                                          constraint_type='global')

"""
Note that when we use only Laplacian regularization, the ``GCV`` option may
select very low regularization weights in very anisotropic white matter such
as the corpus callosum, resulting in corrupted scalar indices. In this example
we therefore choose a preset value of 0.2, which was shown to be quite robust
and also faster in practice [Fick2016a]_.

We can then fit the MAP-MRI model to the data:
"""

mapfit_laplacian_aniso = map_model_laplacian_aniso.fit(data_small)
mapfit_positivity_aniso = map_model_positivity_aniso.fit(data_small)
mapfit_both_aniso = map_model_both_aniso.fit(data_small)
mapfit_plus_aniso = map_model_plus_aniso.fit(data_small)

"""
From the fitted models we will first illustrate the estimation of q-space
indices. We will compare the estimation using only Laplacian regularization,
using only the global positivity constraint, using only positivity in discrete
points, or using both Laplacian regularization and positivity in discrete
points. We first show the RTOP [Ozarslan2013]_.
"""

fig = plt.figure(figsize=(18, 3))

ax1 = fig.add_subplot(1, 5, 1, title=r'RTOP - Laplacian')
ax1.set_axis_off()
rtop_laplacian = np.array(mapfit_laplacian_aniso.rtop()[:, 0, :].T, dtype=float)
ind = ax1.imshow(rtop_laplacian, interpolation='nearest', origin='lower',
                 cmap=plt.cm.gray)

ax2 = fig.add_subplot(1, 5, 2, title=r'RTOP - Discrete positivity')
ax2.set_axis_off()
rtop_positivity = np.array(mapfit_positivity_aniso.rtop()[:, 0, :].T,
                           dtype=float)
ind = ax2.imshow(rtop_positivity, interpolation='nearest', origin='lower',
                 cmap=plt.cm.gray)

ax3 = fig.add_subplot(1, 5, 3, title=r'RTOP - Both')
ax3.set_axis_off()
rtop_both = np.array(mapfit_both_aniso.rtop()[:, 0, :].T, dtype=float)
ind = ax3.imshow(rtop_both, interpolation='nearest', origin='lower',
                 cmap=plt.cm.gray)

ax4 = fig.add_subplot(1, 5, 4, title=r'RTOP - Global positivity')
ax4.set_axis_off()
rtop_plus = np.array(mapfit_plus_aniso.rtop()[:, 0, :].T, dtype=float)
ind = ax4.imshow(rtop_plus, interpolation='nearest', origin='lower',
                 cmap=plt.cm.gray)

ax5 = fig.add_subplot(1, 5, 5)
ax5.set_axis_off()
divider = make_axes_locatable(ax5)
cax = divider.append_axes("left", size="10%")
plt.colorbar(ind, cax=cax)

plt.savefig('MAPMRI_rtop.png')

"""
.. figure:: MAPMRI_rtop.png
   :align: center

It can be seen that all maps appear quite smooth and similar, although it is
clear that the global positivity constraints provide smoother maps than the
discrete constraints. Higher Laplacian weights also produce smoother maps, but
tend to suppress the estimated RTOP values. The similarity and differences in
reconstruction can be further illustrated by visualizing the analytic norm of
the Laplacian of the fitted signal.
"""

fig = plt.figure(figsize=(18, 3))

ax1 = fig.add_subplot(1, 5, 1, title=r'Laplacian norm - Laplacian')
ax1.set_axis_off()
laplacian_norm_laplacian = np.array(
    mapfit_laplacian_aniso.norm_of_laplacian_signal()[:, 0, :].T, dtype=float)
ind = ax1.imshow(laplacian_norm_laplacian, interpolation='nearest',
                 origin='lower', cmap=plt.cm.gray, vmin=0, vmax=3)

ax2 = fig.add_subplot(1, 5, 2, title=r'Laplacian norm - Discrete positivity')
ax2.set_axis_off()
laplacian_norm_positivity = np.array(
    mapfit_positivity_aniso.norm_of_laplacian_signal()[:, 0, :].T, dtype=float)
ind = ax2.imshow(laplacian_norm_positivity, interpolation='nearest',
                 origin='lower', cmap=plt.cm.gray, vmin=0, vmax=3)

ax3 = fig.add_subplot(1, 5, 3, title=r'Laplacian norm - Both')
ax3.set_axis_off()
laplacian_norm_both = np.array(
    mapfit_both_aniso.norm_of_laplacian_signal()[:, 0, :].T, dtype=float)
ind = ax3.imshow(laplacian_norm_both, interpolation='nearest', origin='lower',
                 cmap=plt.cm.gray, vmin=0, vmax=3)

ax4 = fig.add_subplot(1, 5, 4, title=r'Laplacian norm - Global positivity')
ax4.set_axis_off()
laplacian_norm_plus = np.array(
    mapfit_plus_aniso.norm_of_laplacian_signal()[:, 0, :].T, dtype=float)
ind = ax4.imshow(laplacian_norm_plus, interpolation='nearest', origin='lower',
                 cmap=plt.cm.gray, vmin=0, vmax=3)

ax5 = fig.add_subplot(1, 5, 5)
ax5.set_axis_off()
divider = make_axes_locatable(ax5)
cax = divider.append_axes("left", size="10%")
plt.colorbar(ind, cax=cax)

plt.savefig('MAPMRI_norm_laplacian.png')

"""
.. figure:: MAPMRI_norm_laplacian.png
   :align: center

A high Laplacian norm indicates that the gradient in the three-dimensional
signal reconstruction changes a lot -- something that may indicate spurious
oscillations. In the Laplacian reconstruction (left) we see that there are some
isolated voxels that have a higher norm than the rest. In the positivity
constrained reconstruction the norm is already smoother. When both methods are
used together the overall norm gets smoother still, since both smoothness of the
signal and positivity of the propagator are imposed.

From now on we will just use the combined approach and the globally constrained
approach, show all maps we can generate, and explain their significance.
"""

fig = plt.figure(figsize=(18, 6))

ax1 = fig.add_subplot(2, 5, 1, title=r'MSD - Both')
ax1.set_axis_off()
msd = np.array(mapfit_both_aniso.msd()[:, 0, :].T, dtype=float)
ind = ax1.imshow(msd, interpolation='nearest', origin='lower',
                 cmap=plt.cm.gray)

ax2 = fig.add_subplot(2, 5, 2, title=r'QIV - Both')
ax2.set_axis_off()
qiv = np.array(mapfit_both_aniso.qiv()[:, 0, :].T, dtype=float)
ind = ax2.imshow(qiv, interpolation='nearest', origin='lower',
                 cmap=plt.cm.gray)

ax3 = fig.add_subplot(2, 5, 3, title=r'RTOP - Both')
ax3.set_axis_off()
rtop = np.array((mapfit_both_aniso.rtop()[:, 0, :]).T, dtype=float)
ind = ax3.imshow(rtop, interpolation='nearest', origin='lower',
                 cmap=plt.cm.gray)

ax4 = fig.add_subplot(2, 5, 4, title=r'RTAP - Both')
ax4.set_axis_off()
rtap = np.array((mapfit_both_aniso.rtap()[:, 0, :]).T, dtype=float)
ind = ax4.imshow(rtap, interpolation='nearest', origin='lower',
                 cmap=plt.cm.gray)

ax5 = fig.add_subplot(2, 5, 5, title=r'RTPP - Both')
ax5.set_axis_off()
rtpp = np.array(mapfit_both_aniso.rtpp()[:, 0, :].T, dtype=float)
ind = ax5.imshow(rtpp, interpolation='nearest', origin='lower',
                 cmap=plt.cm.gray)

ax6 = fig.add_subplot(2, 5, 6, title=r'MSD - Global positivity')
ax6.set_axis_off()
msd2 = np.array(mapfit_plus_aniso.msd()[:, 0, :].T, dtype=float)
ind = ax6.imshow(msd2, interpolation='nearest', origin='lower',
                 cmap=plt.cm.gray)

ax7 = fig.add_subplot(2, 5, 7, title=r'QIV - Global positivity')
ax7.set_axis_off()
qiv2 = np.array(mapfit_plus_aniso.qiv()[:, 0, :].T, dtype=float)
ind = ax7.imshow(qiv2, interpolation='nearest', origin='lower',
                 cmap=plt.cm.gray)

ax8 = fig.add_subplot(2, 5, 8, title=r'RTOP - Global positivity')
ax8.set_axis_off()
rtop2 = np.array((mapfit_plus_aniso.rtop()[:, 0, :]).T, dtype=float)
ind = ax8.imshow(rtop2, interpolation='nearest', origin='lower',
                 cmap=plt.cm.gray)

ax9 = fig.add_subplot(2, 5, 9, title=r'RTAP - Global positivity')
ax9.set_axis_off()
rtap2 = np.array((mapfit_plus_aniso.rtap()[:, 0, :]).T, dtype=float)
ind = ax9.imshow(rtap2, interpolation='nearest', origin='lower',
                 cmap=plt.cm.gray)

ax10 = fig.add_subplot(2, 5, 10, title=r'RTPP - Global positivity')
ax10.set_axis_off()
rtpp2 = np.array(mapfit_plus_aniso.rtpp()[:, 0, :].T, dtype=float)
ind = ax10.imshow(rtpp2, interpolation='nearest', origin='lower',
                  cmap=plt.cm.gray)

plt.savefig('MAPMRI_maps.png')

"""
.. figure:: MAPMRI_maps.png
   :align: center

From left to right:

- Mean squared displacement (MSD) is a measure for how far protons are able to
  diffuse. a decrease in MSD indicates protons are hindered/restricted more,
  as can be seen by the high MSD in the CSF, but low in the white matter.
- Q-space Inverse Variance (QIV) is a measure of variance in the signal, which
  is said to have higher contrast to white matter than the MSD
  [Hosseinbor2013]_. QIV has also been shown to have high sensitivity to tissue
  composition change in a simulation study [Fick2016b]_.
- Return-to-origin probability (RTOP) quantifies the probability that a proton
  will be at the same position at the first and second diffusion gradient
  pulse. A higher RTOP indicates that the volume a spin is inside of is
  smaller, meaning more overall restriction. This is illustrated by the low
  values in CSF and high values in white matter.
- Return-to-axis probability (RTAP) is a directional index that quantifies
  the probability that a proton will be along the axis of the main eigenvector
  of a diffusion tensor during both diffusion gradient pulses. RTAP has been
  related to the apparent axon diameter [Ozarslan2013]_ [Fick2016a]_ under
  several strong assumptions on the tissue composition and acquisition
  protocol.
- Return-to-plane probability (RTPP) is a directional index that quantifies the
  probability that a proton will be on the plane perpendicular to the main
  eigenvector of a diffusion tensor during both gradient pulses. RTPP is
  related to the length of a pore [Ozarslan2013]_ but in practice should be
  similar to that of Gaussian diffusion.

It is also possible to estimate the amount of non-Gaussian diffusion in every
voxel [Ozarslan2013]_. This quantity is estimated through the ratio between
Gaussian volume (MAP-MRI's first basis function) and the non-Gaussian volume
(all other basis functions) of the fitted signal. For this value to be
physically meaningful we must use a b-value threshold in the MAP-MRI model. This
threshold makes the scale estimation in MAP-MRI use only samples that
realistically describe Gaussian diffusion, i.e., at low b-values.
"""

map_model_both_ng = mapmri.MapmriModel(gtab, radial_order=radial_order,
                                       laplacian_regularization=True,
                                       laplacian_weighting=.05,
                                       positivity_constraint=True,
                                       bval_threshold=2000)

map_model_plus_ng = mapmri.MapmriModel(gtab, radial_order=radial_order,
                                       positivity_constraint=True,
                                       constraint_type='global',
                                       bval_threshold=2000)

mapfit_both_ng = map_model_both_ng.fit(data_small)
mapfit_plus_ng = map_model_plus_ng.fit(data_small)

fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(2, 3, 1, title=r'NG - Both')
ax1.set_axis_off()
ng = np.array(mapfit_both_ng.ng()[:, 0, :].T, dtype=float)
ind = ax1.imshow(ng, interpolation='nearest', origin='lower',
                 cmap=plt.cm.gray)
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(ind, cax=cax)

ax2 = fig.add_subplot(2, 3, 2, title=r'NG perpendicular - Both')
ax2.set_axis_off()
ng_perpendicular = np.array(mapfit_both_ng.ng_perpendicular()[:, 0, :].T,
                            dtype=float)
ind = ax2.imshow(ng_perpendicular, interpolation='nearest', origin='lower',
                 cmap=plt.cm.gray)
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(ind, cax=cax)

ax3 = fig.add_subplot(2, 3, 3, title=r'NG parallel - Both')
ax3.set_axis_off()
ng_parallel = np.array(mapfit_both_ng.ng_parallel()[:, 0, :].T, dtype=float)
ind = ax3.imshow(ng_parallel, interpolation='nearest', origin='lower',
                 cmap=plt.cm.gray)
divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(ind, cax=cax)

ax4 = fig.add_subplot(2, 3, 4, title=r'NG - Global positivity')
ax4.set_axis_off()
ng2 = np.array(mapfit_plus_ng.ng()[:, 0, :].T, dtype=float)
ind = ax4.imshow(ng2, interpolation='nearest', origin='lower',
                 cmap=plt.cm.gray)
divider = make_axes_locatable(ax4)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(ind, cax=cax)

ax5 = fig.add_subplot(2, 3, 5, title=r'NG perpendicular - Global positivity')
ax5.set_axis_off()
ng_perpendicular2 = np.array(mapfit_plus_ng.ng_perpendicular()[:, 0, :].T,
                             dtype=float)
ind = ax5.imshow(ng_perpendicular2, interpolation='nearest', origin='lower',
                 cmap=plt.cm.gray)
divider = make_axes_locatable(ax5)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(ind, cax=cax)

ax6 = fig.add_subplot(2, 3, 6, title=r'NG parallel - Global positivity')
ax6.set_axis_off()
ng_parallel2 = np.array(mapfit_plus_ng.ng_parallel()[:, 0, :].T, dtype=float)
ind = ax6.imshow(ng_parallel2, interpolation='nearest', origin='lower',
                 cmap=plt.cm.gray)
divider = make_axes_locatable(ax6)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(ind, cax=cax)

plt.savefig('MAPMRI_ng.png')

"""
.. figure:: MAPMRI_ng.png
   :align: center

On the left we see the overall NG and on the right the directional perpendicular
NG and parallel NG. The NG ranges from 1 (completely non-Gaussian) to 0
(completely Gaussian). The overall NG of a voxel is always higher or equal than
each of its components. It can be seen that NG has low values in the CSF and
higher in the white matter.

Increases or decreases in these values do not point to a specific
microstructural change, but can indicate clues as to what is happening, similar
to fractional anisotropy. An initial simulation study that quantifies the added
value of q-space indices over DTI indices is given in [Fick2016b]_.

The MAP-MRI framework also allows for the estimation of orientation distribution
functions (ODFs). We recommend to use the isotropic implementation for this
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
Load an ODF reconstruction sphere.
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

scene = window.Scene()
sfu = actor.odf_slicer(odf, sphere=sphere, colormap='plasma', scale=0.5)
sfu.display(y=0)
sfu.RotateX(-90)
scene.add(sfu)
window.record(scene, out_path='odfs.png', size=(600, 600))
if interactive:
    window.show(scene)

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

.. [DelaHaije2020] Dela Haije et al. "Enforcing necessary non-negativity
   constraints for common diffusion MRI models using sum of squares
   programming". NeuroImage 209, 2020, 116405.

.. include:: ../links_names.inc

"""
