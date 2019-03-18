# -*- coding: utf-8 -*-
"""
===============================================================================
Suppress gibbs artefact in DWI
===============================================================================

Magnectic Resonance (MR) images are reconstructed from the Fourier coefficients
of acquired k-space images. Since only a finite number of Fourier coefficients
can be acquired in practice, reconstructed MR images can be corrupted by Gibbs
artefacts, which manifest by intensity oscilations adjacent to edges of
different tissues types [1]_. In the context of diffusion-weighted
imaging, these oscilations can significantly corrupt derived estimates
[1]_, [2]_.

In the following example, we show how to suppress Gibbs artefacts of MRI images
in dipy. This algorithm is based on an adapted version of the sub-voxel
Gibbs suppersion procedure [3]_. Full details of the implemented algorithm
can be found in the Chapter 3 of [4]_  (please cite [3]_, [4]_ if you are using
this code).

For this example, we dowload a diffusion-weighted dataset:
"""

from dipy.data import read_cenir_multib

bvals = [200, 400, 1000, 2000]

img, gtab = read_cenir_multib(bvals)

data = img.get_data()

""" For illustration proposes, we select two slices of this data """

data_slices = data[:, :, 40:42, :]

""" The algorithm to suppress Gibbs oscilations can be imported from the
denoise module of dipy:
"""

from dipy.denoise.gibbs import gibbs_removal

""" Gibbs oscilation suppression can be performed by running the following
command:
"""

data_corrected = gibbs_removal(data_slices, slice_axis=2)

""" In the above step, we recommend you to specify which is the axis of data
matrix that corresponds to different slices using the optional parameter
'slice_axis'. Below we plot the raw and gibbs suppressed data: """

import matplotlib.pyplot as plt

fig1, ax = plt.subplots(1, 2, figsize=(12, 6),
                        subplot_kw={'xticks': [], 'yticks': []})

fig1.subplots_adjust(hspace=0.3, wspace=0.05)

ax.flat[0].imshow(data_slices[:, :, 0, 0].T, cmap='gray', origin='lower',
                  vmin=0, vmax=10000)
ax.flat[0].set_title('Raw')
ax.flat[1].imshow(data_corrected[:, :, 0, 0].T, cmap='gray',
                  origin='lower', vmin=0, vmax=10000)
ax.flat[1].set_title('corrupted')

plt.show()
fig1.savefig('Gibbs_suppresion_b0.png')

"""
.. figure:: Gibbs_suppresion_b0.png
   :align: center

   Uncorrected (left panel) and corrected (right panel) b-value=0 images.

For a better visualization of the benifit of Gibbs artefact suppression, we
process some diffusion derived metrics for both uncorrected and corrected
version of the data. Below, we show the results for the mean signal kurtosis
of the mean signal diffusion image technique (:ref:`example_reconst_msdki`).
"""

# Create a brain mask
from dipy.segment.mask import median_otsu

maskdata, mask = median_otsu(data_slices, 4, 2, False, vol_idx=[0, 1],
                             dilate=1)

# Define mean signal diffusion kurtosis model
import dipy.reconst.msdki as msdki

dki_model = msdki.MeanDiffusionKurtosisModel(gtab)

# Fit the uncorrected data
dki_fit = dki_model.fit(data_slices, mask=mask)
MSKini = dki_fit.msk

# Fit the corrected data
dki_fit = dki_model.fit(data_corrected, mask=mask)
MSKgib = dki_fit.msk

"""
Let's plot the results
"""

fig2, ax = plt.subplots(1, 3, figsize=(12, 12),
                        subplot_kw={'xticks': [], 'yticks': []})

fig2.subplots_adjust(hspace=0.3, wspace=0.05)

ax.flat[0].imshow(MSKini[:, :, 0].T, cmap='gray', origin='lower',
                  vmin=0, vmax=1.5)
ax.flat[0].set_title('MSK (uncorrected)')
ax.flat[1].imshow(MSKgib[:, :, 0].T, cmap='gray', origin='lower',
                  vmin=0, vmax=1.5)
ax.flat[1].set_title('MSK (corrected)')
ax.flat[2].imshow(MSKgib[:, :, 0].T - MSKini[:, :, 0].T, cmap='gray',
                  origin='lower', vmin=-0.2, vmax=0.2)
ax.flat[2].set_title('MSK (uncorrected - corrected')

"""
.. figure:: Gibbs_suppresion_msdki.png
   :align: center

   Uncorrected and corrected mean signal kurtosis images are shown in the right
   and middle panel. The difference between uncorrected and corrected images
   are show in the right panel.

In the left panel of the figure above, Gibbs artefacts can be appriciated by
the negative values of mean signal kurtosis (black voxels) adjacent to the
brain ventricle. These negative values seem to be suppressed after the
`gibbs_removal` funtion is applied. For a better visualization, of gibbs
oscilations the difference between corrected and uncorrected images are shown
in the right panel.


References
----------
.. [1] Veraart, J., Fieremans, E., Jelescu, I.O., Knoll, F., Novikov, D.S.,
       2015. Gibbs Ringing in Diffusion MRI. Magn Reson Med 76(1): 301-314.
       doi: 10.1002/mrm.25866
.. [2] Perrone, D., Aelterman, J., Pižurica, A., Jeurissen, B., Philips, W.,
       Leemans A., 2015. The effect of Gibbs ringing artifacts on measures
       derived from diffusion MRI. Neuroimage 120, 441-455.
       doi: 10.1016/j.neuroimage.2015.06.068.
.. [3] Kellner, E., Dhital, B., Kiselev, V.G, Reisert, M., 2016. Gibbs‐ringing
       artifact removal based on local subvoxel‐shifts. Magn Reson Med
       76:1574–1581. doi:10.1002/mrm.26054.
.. [4] Neto Henriques, R., 2018. Advanced Methods for Diffusion MRI Data
       Analysis and their Application to the Healthy Ageing Brain
       (Doctoral thesis). https://doi.org/10.17863/CAM.29356

.. include:: ../links_names.inc
"""
