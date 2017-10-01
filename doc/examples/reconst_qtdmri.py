# -*- coding: utf-8 -*-
"""
================================================================
Estimating diffusion time dependent q-space indices using qt-dMRI
================================================================
Effective representation of the four-dimensional diffusion MRI signal --
varying over three-dimensional q-space and diffusion time -- is a sought-after
and still unsolved challenge in diffusion MRI (dMRI). We propose a functional
basis approach that is specifically designed to represent the dMRI signal in
this qtau-space [Fick2017]_.  Following recent terminology, we refer to our
qtau-functional basis as ``q$\tau$-dMRI''. We use GraphNet regularization --
imposing both signal smoothness and sparsity -- to drastically reduce the
number of diffusion-weighted images (DWIs) that is needed to represent the dMRI
signal in the qtau-space. As the main contribution, q$\tau$-dMRI provides the
framework to -- without making biophysical assumptions -- represent the
q$\tau$-space signal and estimate time-dependent q-space indices
(q$\tau$-indices), providing a new means for studying diffusion in nervous
tissue. qtau-dMRI is the first of its kind in being specifically designed to
provide open interpretation of the qtau-diffusion signal.

q$\tau$-dMRI can be seen as a time-dependent extension of the MAP-MRI
functional basis [Ozarslan2013]_, and all the previously proposed q-space
can be estimated for any diffusion time. These include rotationally
invariant quantities such as the Mean Squared Displacement (MSD), Q-space
Inverse Variance (QIV) and Return-To-Origin Probability (RTOP). Also
directional indices such as the Return To the Axis Probability (RTAP) and
Return To the Plane Probability (RTPP) are available, as well as the
Orientation Distribution Function (ODF).

In this example we illustrate how to use the qtau-dMRI to estimate
time-dependent q-space indices from a qtau-acquisition of a mouse.

First import the necessary modules:
"""

from dipy.data.fetcher import (fetch_qtdMRI_test_retest_2subjects,
                               read_qtdMRI_test_retest_2subjects)
from dipy.reconst import qtdmri, dti
import matplotlib.pyplot as plt
import numpy as np

"""
Download and read the data for this tutorial.

qt-dMRI requires data with multiple gradient directions, gradient strength and
diffusion times. We will use the test acquisition of one of the mice that was
used in the test-retest study by [Fick2017]_.
"""

fetch_qtdMRI_test_retest_2subjects()
data, cc_masks, gtabs = read_qtdMRI_test_retest_2subjects()
print 'data read'

"""
data contains the voxel data and gtab contains a GradientTable
object (gradient information e.g. b-values). For example, to show the b-values
it is possible to write print(gtab.bvals).

For the values of the q-space
indices to make sense it is necessary to explicitly state the big_delta and
small_delta parameters in the gradient table.
"""

subplot_titles = ["Subject1 Test", "Subject1 Retest",
                  "Subject2 Test", "Subject2 Tetest"]
fig = plt.figure()
plt.subplots(nrows=2, ncols=2)
for i, (data_, mask_, gtab_) in enumerate(zip(data, cc_masks, gtabs)):
    # take the middle slice
    data_middle_slice = data_[:, :, 2]
    mask_middle_slice = mask_[:, :, 2]

    # estimate fractional anisotropy (FA) for this slice
    tenmod = dti.TensorModel(gtab_)
    tenfit = tenmod.fit(data_middle_slice, data_middle_slice[..., 0] > 0)
    fa = tenfit.fa

    # set mask color to green with 0.5 opacity as overlay
    mask_template = np.zeros(np.r_[mask_middle_slice.shape, 4])
    mask_template[mask_middle_slice == 1] = np.r_[0., 1., 0., .5]

    # produce the FA images with corpus callosum masks.
    plt.subplot(2, 2, 1 + i)
    plt.title(subplot_titles[i], fontsize=15)
    plt.imshow(fa, cmap='Greys_r', origin=True, interpolation='nearest')
    plt.imshow(mask_template, origin=True, interpolation='nearest')
    plt.axis('off')
plt.tight_layout()
plt.savefig('qt-dMRI_datasets_fa_with_ccmasks.png')

print 'fa images made'
"""
.. figure:: qt-dMRI_datasets_fa_with_ccmasks.png
   : align: center
"""


plt.figure()
qtdmri.visualise_gradient_table_G_Delta_rainbow(gtabs[0])
plt.savefig('qt-dMRI_acquisition_scheme.png')

print 'scheme image made'
"""
.. figure:: qt-dMRI_acquisition_scheme.png
   :align: center
"""

tau_min = gtabs[0].tau.min()
tau_max = gtabs[0].tau.max()
taus = np.linspace(tau_min, tau_max, 5)

qtdmri_fits = []
msds = []
rtops = []
rtaps = []
rtpps = []
for i, (data_, mask_, gtab_) in enumerate(zip(data, cc_masks, gtabs)):
    cc_voxels = data_[mask_ == 1]
    qtdmri_mod = qtdmri.QtdmriModel(
        gtab_, radial_order=6, time_order=2,
        laplacian_regularization=True, laplacian_weighting='GCV',
        l1_regularization=True, l1_weighting='CV'
    )
    qtdmri_fit = qtdmri_mod.fit(cc_voxels[::30])
    qtdmri_fits.append(qtdmri_fit)
    msds.append(np.array(list(map(qtdmri_fit.msd, taus))))
    rtops.append(np.array(list(map(qtdmri_fit.rtop, taus))))
    rtaps.append(np.array(list(map(qtdmri_fit.rtap, taus))))
    rtpps.append(np.array(list(map(qtdmri_fit.rtpp, taus))))

print 'data fitted'


def plot_mean_with_std(ax, time, ind1, plotcolor, ls='-', std_mult=1,
                       label=''):
    means = np.mean(ind1, axis=1)
    stds = np.std(ind1, axis=1)
    ax.plot(time, means, c=plotcolor, lw=3, label=label, ls=ls)
    ax.fill_between(time,
                    means + std_mult * stds,
                    means - std_mult * stds,
                    alpha=0.15, color=plotcolor)
    ax.plot(time, means + std_mult * stds, alpha=0.25, color=plotcolor)
    ax.plot(time, means - std_mult * stds, alpha=0.25, color=plotcolor)


std_mult = .75
fig = plt.figure(figsize=(10, 3))
ax = plt.subplot(1, 2, 1)
Delta_ = np.linspace(0.005, 0.02, 100)
MSD_ = np.linspace(4e-5, 10e-5, 100)
Delta_grid, MSD_grid = np.meshgrid(Delta_, MSD_)
D_grid = MSD_grid / (6 * Delta_grid)
plt.contourf(Delta_ * 1e3, 1e5 * MSD_, D_grid,
             levels=np.r_[1, 5, 7, 10, 14, 23, 30] * 1e-4,
             cmap='Greys', alpha=.5)

plot_mean_with_std(ax, taus * 1e3, 1e5 * msds[0], 'red', 'dashdot',
                   std_mult=std_mult, label='MSD Test')
plot_mean_with_std(ax, taus * 1e3, 1e5 * msds[1], 'green', 'dashdot',
                   std_mult=std_mult, label='MSD Retest')
ax.legend(fontsize=13)
ax.text(.0091 * 1e3, 6.33, 'D=14e-4', fontsize=12, rotation=35)
ax.text(.0091 * 1e3, 4.55, 'D=10e-4', fontsize=12, rotation=25)
ax.set_ylim(4, 9.5)
ax.set_xlim(.009 * 1e3, 0.0185 * 1e3)
ax.set_title(r'Test-Retest MSD($\tau$) Subject 1', fontsize=15)
ax.set_xlabel('Diffusion Time (ms)', fontsize=17)
ax.set_ylabel('MSD ($10^{-5}mm^2$)', fontsize=17)

ax = plt.subplot(1, 2, 2)
plt.contourf(Delta_ * 1e3, 1e5 * MSD_, D_grid,
             levels=np.r_[1, 5, 7, 10, 14, 23, 30] * 1e-4,
             cmap='Greys', alpha=.5)
cb = plt.colorbar()
cb.set_label('Free Diffusivity ($mm^2/s$)', fontsize=18)

plot_mean_with_std(ax, taus * 1e3, 1e5 * msds[2], 'red', 'dashdot',
                   std_mult=std_mult, label='MSD Test')
plot_mean_with_std(ax, taus * 1e3, 1e5 * msds[3], 'green', 'dashdot',
                   std_mult=std_mult, label='MSD Retest')
ax.set_ylim(4, 9.5)
ax.set_xlim(.009 * 1e3, 0.0185 * 1e3)
ax.set_xlabel('Diffusion Time (ms)', fontsize=17)
ax.set_title(r'Test-Retest MSD($\tau$) Subject 2', fontsize=15)
plt.savefig('qt_indices_msd.png')

print 'msd images made'

# rtap

std_mult = .75
fig = plt.figure(figsize=(10, 3))
ax = plt.subplot(1, 2, 1)
Delta_ = np.linspace(0.005, 0.02, 100)
RTXP_ = np.linspace(1, 200, 100)
Delta_grid, RTXP_grid = np.meshgrid(Delta_, RTXP_)

D_grid = 1 / (4 * RTXP_grid ** 2 * np.pi * Delta_grid)
plt.contourf(Delta_ * 1e3, RTXP_, D_grid,
             colors=np.tile(np.linspace(.8, 0, 7), (3, 1)).T,
             levels=np.r_[1, 2, 3, 4, 6, 9, 15, 30] * 1e-4,
             alpha=.5)

plot_mean_with_std(ax, taus * 1e3, np.sqrt(rtaps[0]), 'r', '-',
                   std_mult=std_mult, label='RTAP$^{1/2}$ Test')
plot_mean_with_std(ax, taus * 1e3, np.sqrt(rtaps[1]), 'g', '-',
                   std_mult=std_mult, label='RTAP$^{1/2}$ Retest')
ax.legend(fontsize=13)
ax.text(.0091 * 1e3, 162, 'D=3e-4', fontsize=12, rotation=-22)
ax.text(.0091 * 1e3, 140, 'D=4e-4', fontsize=12, rotation=-20)
ax.text(.0091 * 1e3, 113, 'D=6e-4', fontsize=12, rotation=-16)
ax.set_ylim(54, 170)
ax.set_xlim(.009 * 1e3, 0.0185 * 1e3)
ax.set_title(r'Test-Retest RTAP($\tau$) Subject 1', fontsize=15)
ax.set_xlabel('Diffusion Time (ms)', fontsize=17)
ax.set_ylabel('RTAP$^{1/2}$ (1/mm)', fontsize=17)

ax = plt.subplot(1, 2, 2)
plt.contourf(Delta_ * 1e3, RTXP_, D_grid,
             colors=np.tile(np.linspace(.8, 0, 7), (3, 1)).T,
             levels=np.r_[1, 2, 3, 4, 6, 9, 15, 30] * 1e-4,
             alpha=.5)
cb = plt.colorbar()
cb.set_label('Free Diffusivity ($mm^2/s$)', fontsize=18)

plot_mean_with_std(ax, taus * 1e3, np.sqrt(rtaps[2]), 'r', '-',
                   std_mult=std_mult)
plot_mean_with_std(ax, taus * 1e3, np.sqrt(rtaps[3]), 'g', '-',
                   std_mult=std_mult)
ax.set_ylim(54, 170)
ax.set_xlim(.009 * 1e3, 0.0185 * 1e3)
ax.set_xlabel('Diffusion Time (ms)', fontsize=17)
ax.set_title(r'Test-Retest RTAP($\tau$) Subject 2', fontsize=15)
plt.savefig('qt_indices_rtap.png')

print 'rtap images made'

# rtop

std_mult = .75
fig = plt.figure(figsize=(10, 3))
ax = plt.subplot(1, 2, 1)
Delta_ = np.linspace(0.005, 0.02, 100)
RTXP_ = np.linspace(1, 200, 100)
Delta_grid, RTXP_grid = np.meshgrid(Delta_, RTXP_)

D_grid = 1 / (4 * RTXP_grid ** 2 * np.pi * Delta_grid)
plt.contourf(Delta_ * 1e3, RTXP_, D_grid,
             colors=np.tile(np.linspace(.8, 0, 7), (3, 1)).T,
             levels=np.r_[1, 2, 3, 4, 6, 9, 15, 30] * 1e-4,
             alpha=.5)

plot_mean_with_std(ax, taus * 1e3, rtops[0] ** (1/3.), 'r', '--',
                   std_mult=std_mult, label='RTOP$^{1/3}$ Test')
plot_mean_with_std(ax, taus * 1e3, rtops[1] ** (1/3.), 'g', '--',
                   std_mult=std_mult, label='RTOP$^{1/3}$ Retest')
ax.legend(fontsize=13)
ax.text(.0091 * 1e3, 162, 'D=3e-4', fontsize=12, rotation=-22)
ax.text(.0091 * 1e3, 140, 'D=4e-4', fontsize=12, rotation=-20)
ax.text(.0091 * 1e3, 113, 'D=6e-4', fontsize=12, rotation=-16)
ax.set_ylim(54, 170)
ax.set_xlim(.009 * 1e3, 0.0185 * 1e3)
ax.set_title(r'Test-Retest RTOP($\tau$) Subject 1', fontsize=15)
ax.set_xlabel('Diffusion Time (ms)', fontsize=17)
ax.set_ylabel('RTOP$^{1/3}$ (1/mm)', fontsize=17)

ax = plt.subplot(1, 2, 2)
plt.contourf(Delta_ * 1e3, RTXP_, D_grid,
             colors=np.tile(np.linspace(.8, 0, 7), (3, 1)).T,
             levels=np.r_[1, 2, 3, 4, 6, 9, 15, 30] * 1e-4,
             alpha=.5)
cb = plt.colorbar()
cb.set_label('Free Diffusivity ($mm^2/s$)', fontsize=18)

plot_mean_with_std(ax, taus * 1e3, rtops[2] ** (1/3.), 'r', '--',
                   std_mult=std_mult)
plot_mean_with_std(ax, taus * 1e3, rtops[3] ** (1/3.), 'g', '--',
                   std_mult=std_mult)
ax.set_ylim(54, 170)
ax.set_xlim(.009 * 1e3, 0.0185 * 1e3)
ax.set_xlabel('Diffusion Time (ms)', fontsize=17)
ax.set_title(r'Test-Retest RTOP($\tau$) Subject 2', fontsize=15)
plt.savefig('qt_indices_rtop.png')

print 'rtop images made'

# rtpp
std_mult = .75
fig = plt.figure(figsize=(10, 3))
ax = plt.subplot(1, 2, 1)
Delta_ = np.linspace(0.005, 0.02, 100)
RTXP_ = np.linspace(1, 200, 100)
Delta_grid, RTXP_grid = np.meshgrid(Delta_, RTXP_)

D_grid = 1 / (4 * RTXP_grid ** 2 * np.pi * Delta_grid)
plt.contourf(Delta_ * 1e3, RTXP_, D_grid,
             colors=np.tile(np.linspace(.8, 0, 7), (3, 1)).T,
             levels=np.r_[1, 2, 3, 4, 6, 9, 15, 30] * 1e-4,
             alpha=.5)

plot_mean_with_std(ax, taus * 1e3, rtpps[0], 'r', ':', std_mult=std_mult,
                   label='RTPP Test')
plot_mean_with_std(ax, taus * 1e3, rtpps[1], 'g', ':', std_mult=std_mult,
                   label='RTPP Retest')
ax.legend(fontsize=13)
ax.text(.0091 * 1e3, 113, 'D=6e-4', fontsize=12, rotation=-16)
ax.text(.0091 * 1e3, 91, 'D=9e-4', fontsize=12, rotation=-13)
ax.text(.0091 * 1e3, 69, 'D=15e-4', fontsize=12, rotation=-10)
ax.set_ylim(54, 170)
ax.set_xlim(.009 * 1e3, 0.0185 * 1e3)
ax.set_title(r'Test-Retest RTPP($\tau$) Subject 1', fontsize=15)
ax.set_xlabel('Diffusion Time (ms)', fontsize=17)
ax.set_ylabel('RTPP (1/mm)', fontsize=17)

ax = plt.subplot(1, 2, 2)
plt.contourf(Delta_ * 1e3, RTXP_, D_grid,
             colors=np.tile(np.linspace(.8, 0, 7), (3, 1)).T,
             levels=np.r_[1, 2, 3, 4, 6, 9, 15, 30] * 1e-4,
             alpha=.5)
cb = plt.colorbar()
cb.set_label('Free Diffusivity ($mm^2/s$)', fontsize=18)

plot_mean_with_std(ax, taus * 1e3, rtpps[2], 'r', ':', std_mult=std_mult)
plot_mean_with_std(ax, taus * 1e3, rtpps[3], 'g', ':', std_mult=std_mult)
ax.set_ylim(54, 170)
ax.set_xlim(.009 * 1e3, 0.0185 * 1e3)
ax.set_xlabel('Diffusion Time (ms)', fontsize=17)
ax.set_title(r'Test-Retest RTPP($\tau$) Subject 2', fontsize=15)
plt.savefig('qt_indices_rtpp.png')

print 'rtpp images made'

"""
- show mask over FA overlay.
- fit qt-dMRI
- estimate qt-space indices
- show test-retest reproducibility
"""

"""
.. [Fick2017]_ Fick, Rutger HJ, et al. "Non-Parametric GraphNet-Regularized
            Representation of dMRI in Space and Time", Medical Image Analysis,
            2017.
"""
