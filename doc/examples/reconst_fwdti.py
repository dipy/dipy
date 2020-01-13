"""
=============================================================================
Using the free water elimination model to remove DTI free water contamination
=============================================================================

As shown previously (see :ref:`example_reconst_dti`), the diffusion tensor
model is a simple way to characterize diffusion anisotropy. However, in regions
near the cerebral ventricle and parenchyma can be underestimated by partial
volume effects of the cerebral spinal fluid (CSF). This free water
contamination can particularly corrupt Diffusion Tensor Imaging analysis of
microstructural changes when different groups of subjects show different brain
morphology (e.g. brain ventricle enlargement associated with brain tissue
atrophy that occurs in several brain pathologies and ageing).

A way to remove this free water influences is to expand the DTI model to take
into account an extra compartment representing the contributions of free water
diffusion [Pasternak2009]_. The expression of the expanded DTI model is shown
below:

.. math::

    S(\mathbf{g}, b) = S_0(1-f)e^{-b\mathbf{g}^T \mathbf{D}
    \mathbf{g}}+S_0fe^{-b D_{iso}}

where $\mathbf{g}$ and $b$ are diffusion gradient direction and weighted
(more information see :ref:`example_reconst_dti`), $S(\mathbf{g}, b)$ is the
diffusion-weighted signal measured, $S_0$ is the signal in a measurement with
no diffusion weighting, $\mathbf{D}$ is the diffusion tensor, $f$ the volume
fraction of the free water component, and $D_{iso}$ is the isotropic value of
the free water diffusion (normally set to $3.0 \times 10^{-3} mm^{2}s^{-1}$).

In this example, we show how to process a diffusion weighting dataset using
an adapted version of the free water elimination proposed by [Hoy2014]_.

The full details of Dipy's free water DTI implementation was published in
[Henriques2017]_. Please cite this work if you use this algorithm.

Let's start by importing the relevant modules:
"""

import numpy as np
import dipy.reconst.fwdti as fwdti
import dipy.reconst.dti as dti
import matplotlib.pyplot as plt
from dipy.data import fetch_cenir_multib
from dipy.data import read_cenir_multib
from dipy.segment.mask import median_otsu

"""
Without spatial constrains the free water elimination model cannot be solved
in data acquired from one non-zero b-value [Hoy2014]_. Therefore, here we
download a dataset that was required from multiple b-values.
"""

fetch_cenir_multib(with_raw=False)

"""
From the downloaded data, we read only the data acquired with b-values up to
2000 $s/mm^2$ to decrease the influence of non-Gaussian diffusion
effects of the tissue which are not taken into account by the free water
elimination model [Hoy2014]_.
"""

bvals = [200, 400, 1000, 2000]

img, gtab = read_cenir_multib(bvals)

data = np.asarray(img.dataobj)

affine = img.affine

"""
The free water DTI model can take some minutes to process the full data set.
Thus, we remove the background of the image to avoid unnecessary calculations.
"""

maskdata, mask = median_otsu(data, vol_idx=[0, 1], median_radius=4, numpass=2,
                             autocrop=False, dilate=1)

"""
Moreover, for illustration purposes we process only an axial slice of the
data.
"""

axial_slice = 40

mask_roi = np.zeros(data.shape[:-1], dtype=bool)
mask_roi[:, :, axial_slice] = mask[:, :, axial_slice]

"""
The free water elimination model fit can then be initialized by instantiating
a FreeWaterTensorModel class object:
"""

fwdtimodel = fwdti.FreeWaterTensorModel(gtab)

"""
The data can then be fitted using the ``fit`` function of the defined model
object:
"""

fwdtifit = fwdtimodel.fit(data, mask=mask_roi)


"""
This 2-steps procedure will create a FreeWaterTensorFit object which contains
all the diffusion tensor statistics free for free water contaminations. Below
we extract the fractional anisotropy (FA) and the mean diffusivity (MD) of the
free water diffusion tensor."""

FA = fwdtifit.fa
MD = fwdtifit.md

"""
For comparison we also compute the same standard measures processed by the
standard DTI model
"""

dtimodel = dti.TensorModel(gtab)

dtifit = dtimodel.fit(data, mask=mask_roi)

dti_FA = dtifit.fa
dti_MD = dtifit.md

"""
Below the FA values for both free water elimination DTI model and standard DTI
model are plotted in panels A and B, while the repective MD values are ploted
in panels D and E. For a better visualization of the effect of the free water
correction, the differences between these two metrics are shown in panels C and
E. In addition to the standard diffusion statistics, the estimated volume
fraction of the free water contamination is shown on panel G.
"""

fig1, ax = plt.subplots(2, 4, figsize=(12, 6),
                        subplot_kw={'xticks': [], 'yticks': []})

fig1.subplots_adjust(hspace=0.3, wspace=0.05)
ax.flat[0].imshow(FA[:, :, axial_slice].T, origin='lower',
                  cmap='gray', vmin=0, vmax=1)
ax.flat[0].set_title('A) fwDTI FA')
ax.flat[1].imshow(dti_FA[:, :, axial_slice].T, origin='lower',
                  cmap='gray', vmin=0, vmax=1)
ax.flat[1].set_title('B) standard DTI FA')

FAdiff = abs(FA[:, :, axial_slice] - dti_FA[:, :, axial_slice])
ax.flat[2].imshow(FAdiff.T, cmap='gray', origin='lower', vmin=0, vmax=1)
ax.flat[2].set_title('C) FA difference')

ax.flat[3].axis('off')

ax.flat[4].imshow(MD[:, :, axial_slice].T, origin='lower',
                  cmap='gray', vmin=0, vmax=2.5e-3)
ax.flat[4].set_title('D) fwDTI MD')
ax.flat[5].imshow(dti_MD[:, :, axial_slice].T, origin='lower',
                  cmap='gray', vmin=0, vmax=2.5e-3)
ax.flat[5].set_title('E) standard DTI MD')

MDdiff = abs(MD[:, :, axial_slice] - dti_MD[:, :, axial_slice])
ax.flat[6].imshow(MDdiff.T, origin='lower', cmap='gray', vmin=0, vmax=2.5e-3)
ax.flat[6].set_title('F) MD difference')

F = fwdtifit.f

ax.flat[7].imshow(F[:, :, axial_slice].T, origin='lower',
                  cmap='gray', vmin=0, vmax=1)
ax.flat[7].set_title('G) free water volume')

plt.show()
fig1.savefig('In_vivo_free_water_DTI_and_standard_DTI_measures.png')

"""

.. figure:: In_vivo_free_water_DTI_and_standard_DTI_measures.png
   :align: center

   In vivo diffusion measures obtain from the free water DTI and standard
   DTI. The values of Fractional Anisotropy for the free water DTI model and
   standard DTI model and their difference are shown in the upper panels (A-C),
   while respective MD values are shown in the lower panels (D-F). In addition
   the free water volume fraction estimated from the fwDTI model is shown in
   panel G.

From the figure, one can observe that the free water elimination model
produces in general higher values of FA and lower values of MD than the
standard DTI model. These differences in FA and MD estimation are expected
due to the suppression of the free water isotropic diffusion components.
Unexpected high amplitudes of FA are however observed in the periventricular
gray matter. This is a known artefact of regions associated to voxels with high
water volume fraction (i.e. voxels containing basically CSF). We are able to
remove this problematic voxels by excluding all FA values associated with
measured volume fractions above a reasonable threshold of 0.7:
"""

FA[F > 0.7] = 0
dti_FA[F > 0.7] = 0

"""
Above we reproduce the plots of the in vivo FA from the two DTI fits and where
we can see that the inflated FA values were practically removed:
"""

fig1, ax = plt.subplots(1, 3, figsize=(9, 3),
                        subplot_kw={'xticks': [], 'yticks': []})

fig1.subplots_adjust(hspace=0.3, wspace=0.05)
ax.flat[0].imshow(FA[:, :, axial_slice].T, origin='lower',
                  cmap='gray', vmin=0, vmax=1)
ax.flat[0].set_title('A) fwDTI FA')
ax.flat[1].imshow(dti_FA[:, :, axial_slice].T, origin='lower',
                  cmap='gray', vmin=0, vmax=1)
ax.flat[1].set_title('B) standard DTI FA')

FAdiff = abs(FA[:, :, axial_slice] - dti_FA[:, :, axial_slice])
ax.flat[2].imshow(FAdiff.T, cmap='gray', origin='lower', vmin=0, vmax=1)
ax.flat[2].set_title('C) FA difference')

plt.show()
fig1.savefig('In_vivo_free_water_DTI_and_standard_DTI_corrected.png')

"""

.. figure:: In_vivo_free_water_DTI_and_standard_DTI_corrected.png
   :align: center

   In vivo FA measures obtain from the free water DTI (A) and standard
   DTI (B) and their difference (C). Problematic inflated FA values of the
   images were removed by dismissing voxels above a volume fraction threshold
   of 0.7.

References
----------
.. [Pasternak2009] Pasternak, O., Sochen, N., Gur, Y., Intrator, N., Assaf, Y.,
   2009. Free water elimination and mapping from diffusion MRI. Magn. Reson.
   Med. 62(3): 717-30. doi: 10.1002/mrm.22055.
.. [Hoy2014] Hoy, A.R., Koay, C.G., Kecskemeti, S.R., Alexander, A.L., 2014.
   Optimization of a free water elimination two-compartmental model for
   diffusion tensor imaging. NeuroImage 103, 323-333. doi:
   10.1016/j.neuroimage.2014.09.053
.. [Henriques2017] Henriques, R.N., Rokem, A., Garyfallidis, E., St-Jean, S.,
   Peterson E.T., Correia, M.M., 2017. [Re] Optimization of a free water
   elimination two-compartment model for diffusion tensor imaging.
   ReScience volume 3, issue 1, article number 2

"""
