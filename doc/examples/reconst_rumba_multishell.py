"""
========================================
Multi-Shell Reconstruction with RUMBA-SD
========================================

This example shows how to use Robust and Unbiased Model-BAsed Spherical
Deconvolution (RUMBA-SD) with multi-shell data. This model was introduced
by Canales-Rodriguez et al [CanalesRodriguez2015]_. For a general introduction
to the method using single-shell data, see :ref:`example_reconst_rumba`.

For multi-shell data, a modified white matter response function can be used
with shell-specific diffusivities.
"""

import numpy as np

from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.segment.mask import median_otsu
from dipy.reconst.rumba import RumbaSD, global_fit
from dipy.reconst.mcsd import auto_response_msmt
from dipy.viz import window, actor

from dipy.data import get_sphere, get_fnames
sphere = get_sphere('symmetric724')

"""
We begin by loading multi-shell data provided by Hansen and Jespersen
[Hansen2016]_. From this dataset, we extract only two non-zero b-values.
"""

fraw, fbval, fbvec, t1_fname = get_fnames('cfin_multib')

data, affine = load_nifti(fraw)
bvals, bvecs = read_bvals_bvecs(fbval, fbvec)

sel_b = np.logical_or(np.logical_or(bvals == 0, bvals == 1000), bvals == 2000)
data = data[..., sel_b]

gtab = gradient_table(bvals[sel_b], bvecs[sel_b])

"""
To compute the response function, we first need to find segment the data into
white matter, grey matter, and cerebrospinal fluid voxels. This is done using
`mask_for_response_msmt`, which applies thresholds on mean diffusivity and
fractional anisotropy to make this distinction. An optimal fit only uses
b-values less than 1200, so a warning will be generated.

Then, `response_from_mask_msmt` computes the multi-shell response function
using these masks. These functions can be combined using `auto_response_msmt`,
shown below.
"""

wm_response, gm_response, csf_response = \
    auto_response_msmt(gtab, data, roi_radii=10)

"""
This white matter response can now be used in RUMBA-SD as follows. Note that
RUMBA-SD requires three values per shell (the diffusion tensor eigenvalues).
There is no way to pass a multi-shell GM or CSF response.
"""

rumba = RumbaSD(gtab, wm_response=wm_response[:, :-1], voxelwise=False)

"""
Now we can compute our fODFs. For efficiency, we are using the global fit
without total variation regularization. We will also first fit a mask using
`median_otsu`. The fit will take about 10 minutes.
"""

b0_mask, mask = median_otsu(data, median_radius=2, numpass=1, vol_idx=[0, 1])
rumba_fit = rumba.fit(data, mask=mask)

odf = rumba_fit.odf()
f_gm = rumba_fit.f_gm()
f_csf = rumba_fit.f_csf()
f_wm = rumba_fit.f_wm()
combined = rumba_fit.combined_odf_iso()

"""
We can now visualize these fODFs
"""

fodf_spheres = actor.odf_slicer(combined, sphere=sphere, scale=0.9,
                                norm=True, colormap=None)

interactive = True
scene = window.Scene()
scene.add(fodf_spheres)
scene.reset_camera_tight()

print('Saving illustration as rumba_multishell_odf.png')
window.record(scene, out_path='rumba_multishell_odf.png', size=(600, 600))

if interactive:
    window.show(scene)

"""
.. figure:: rumba_multishell_odf.png
   :align: center

   RUMBA-SD multi-shell fODFs
"""

scene.rm(fodf_spheres)

"""
RUMBA-SD also has compartments for cerebrospinal fluid (CSF) and grey matter
(GM). The GM compartment can only be estimated with at least 3 shells of data.
In cases with less shells, we recommend only computing the CSF compartment.
Since this data has many shells, GM can be computed and the the volume
fractions of white matter, CSF, and GM are shown below. They are normalized to
add up to 1 for each voxel.
"""

import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 3, figsize=(12, 3))

ax0 = axs[0].imshow(f_wm[..., 0].T, origin='lower')
axs[0].set_title("Voxelwise WM Volume Fraction")

ax1 = axs[1].imshow(f_csf[..., 0].T, origin='lower')
axs[1].set_title("Voxelwise CSF Volume Fraction")

ax2 = axs[2].imshow(f_gm[..., 0].T, origin='lower')
axs[2].set_title("Voxelwise GM Volume Fraction")

plt.colorbar(ax0, ax=axs[0])
plt.colorbar(ax1, ax=axs[1])
plt.colorbar(ax2, ax=axs[2])

plt.savefig('wm_gm_csf_partition.png')

"""
We can now zoom into a small patch of this image to take a closer look at the
fODFs and visualize the peaks.
"""
combined_patch = combined[35:60, 50:70]
odf_patch = odf[35:60, 50:70]

fodf_spheres = actor.odf_slicer(combined_patch, sphere=sphere, scale=0.9,
                                norm=True, colormap=None)

interactive = True
scene = window.Scene()
scene.add(fodf_spheres)
scene.reset_camera_tight()

print('Saving illustration as rumba_multishell_patch.png')
window.record(scene, out_path='rumba_multishell_patch.png', size=(600, 600))

if interactive:
    window.show(scene)

"""
.. figure:: rumba_multishell_patch.png
   :align: center

   RUMBA-SD multi-shell fODFs patch
"""

scene.rm(fodf_spheres)

"""
We can now extract the peaks from this patch using a for loop.
"""

from dipy.direction.peaks import peak_directions
shape = odf_patch.shape[:3]
npeaks = 5  # maximum number of peaks returned for a given voxel
peak_dirs = np.zeros((shape + (npeaks, 3)))
peak_values = np.zeros((shape + (npeaks,)))

for idx in np.ndindex(shape):  # iterate through each voxel
    # Get peaks of odf
    direction, pk, _ = peak_directions(odf_patch[idx], sphere,
                                       relative_peak_threshold=0.5,
                                       min_separation_angle=25)

    # Calculate peak metrics
    if pk.shape[0] != 0:
        n = min(npeaks, pk.shape[0])
        peak_dirs[idx][:n] = direction[:n]
        peak_values[idx][:n] = pk[:n]

peak_values = np.clip(peak_values * 30, 0, 1)  # scale up for visualization

fodf_peaks = actor.peak_slicer(
    peak_dirs[:, :, 0:1, :], peak_values[:, :, 0:1, :])
scene.add(fodf_peaks)

print('Saving illustration as rumba_multishell_peaks.png')
window.record(scene, out_path='rumba_multishell_peaks.png', size=(600, 600))
if interactive:
    window.show(scene)


"""
.. figure:: rumba_multishell_peaks.png
   :align: center

   RUMBA-SD multishell peaks
"""

scene.rm(fodf_peaks)


"""
References
----------

.. [CanalesRodriguez2015] Canales-Rodríguez, E. J., Daducci, A., Sotiropoulos,
   S. N., Caruyer, E., Aja-Fernández, S., Radua, J., Mendizabal, J. M. Y.,
   Iturria-Medina, Y., Melie-García, L., Alemán-Gómez, Y., Thiran, J.-P.,
   Sarró, S., Pomarol-Clotet, E., & Salvador, R. (2015). Spherical
   Deconvolution of Multichannel Diffusion MRI Data with Non-Gaussian Noise
   Models and Spatial Regularization. PLOS ONE, 10(10), e0138910.
   https://doi.org/10.1371/journal.pone.0138910

.. [Hansen2016] Hansen, B., & Jespersen, S. N. (2016). Data for evaluation of
   fast kurtosis strategies, b-value optimization and exploration
   of diffusion MRI contrast. Scientific Data, 3(1), 160072.
   https://doi.org/10.1038/sdata.2016.72

.. include:: ../links_names.inc

"""
