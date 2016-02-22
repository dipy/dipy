"""
=====================================================================
Title
=====================================================================

Some background

"""

import numpy as np
import dipy.reconst.fwdti as fwdti
import dipy.reconst.dti as dti
import matplotlib.pyplot as plt
from dipy.data import fetch_cenir_multib
from dipy.data import read_cenir_multib
from dipy.segment.mask import median_otsu

"""
Free water elimination DTI model requires multi-shell data, i.e. data acquired
from more than one non-zero b-value. Here, we use fetch to download a
multi-shell dataset with parameters.
"""

fetch_cenir_multib(with_raw=False)

"""
Next, we read the saved dataset. To decrease the influence of non-Gaussain
diffusion signal components, we only select the b-values up to 2000 $s.mm^{-2}$
(_[Hoy2014]):
"""

bvals = [200, 400, 1000, 2000]

img, gtab = read_cenir_multib(bvals)

data = img.get_data()

affine = img.get_affine()

"""
Before fitting the data, we preform some data pre-processing. We first compute
a brain mask to avoid unnecessary calculations on the background of the image.
"""

maskdata, mask = median_otsu(data, 4, 2, False, vol_idx=[0, 1], dilate=1)

"""
Test only an axial slice
"""

axial_slice = 40

mask_roi = np.zeros(data.shape[:-1], dtype=bool)
mask_roi[:, :, axial_slice] = mask[:, :, axial_slice]

"""
Now that we have loaded and prepared the voxels to process we can go forward
with the voxel reconstruction. This can be done by first instantiating the
FreeWaterTensorModel in the following way:
"""

fwdtimodel = fwdti.FreeWaterTensorModel(gtab, 'NLS', cholesky=False)

"""
To fit the data using the defined model object, we call the ``fit`` function of
this object:
"""

from time import time
t0 = time()

fwdtifit = fwdtimodel.fit(data, mask=mask_roi)

print (time() - t0), ' sec'

"""
The fit method creates a FreeWaterTensorFit object which contains the diffusion
tensor free of the water contamination. Below we extract the fractional
anisotropy (FA), the mean diffusivity (MD), the axial diffusivity (AD) and 
the radial diffusivity (RD) of the free water diffusion tensor."""

FA = fwdtifit.fa
MD = fwdtifit.md
AD = fwdtifit.ad
RD = fwdtifit.rd

"""
From comparison we also compute the standard measures using the standard DTI
model
"""
dtimodel = dti.TensorModel(gtab)

dtifit = dtimodel.fit(data, mask=mask_roi)

dti_FA = dtifit.fa
dti_MD = dtifit.md
dti_AD = dtifit.ad
dti_RD = dtifit.rd

"""
The DT based measures can be easly visualized using matplotlib. For example,
the FA, MD, AD, and RD obtain from the diffusion kurtosis model (upper panels)
and the tensor model (lower panels) are plotted for the selected axial slice.
"""

fig1, ax = plt.subplots(2, 4, figsize=(12, 6),
                        subplot_kw={'xticks': [], 'yticks': []})

fig1.subplots_adjust(hspace=0.3, wspace=0.05)

ax.flat[0].imshow(FA[:, :, axial_slice], cmap='gray', vmin=0, vmax=1)
ax.flat[0].set_title('FA (fwDTI)')
ax.flat[1].imshow(MD[:, :, axial_slice], cmap='gray', vmin=0, vmax=2.5e-3)
ax.flat[1].set_title('MD (fwDTI)')
ax.flat[2].imshow(AD[:, :, axial_slice], cmap='gray', vmin=0, vmax=2.5e-3)
ax.flat[2].set_title('AD (fwDTI)')
ax.flat[3].imshow(RD[:, :, axial_slice], cmap='gray', vmin=0, vmax=2.5e-3)
ax.flat[3].set_title('RD (fwDTI)')

ax.flat[4].imshow(dti_FA[:, :, axial_slice], cmap='gray', vmin=0, vmax=1)
ax.flat[4].set_title('FA (DTI)')
ax.flat[5].imshow(dti_MD[:, :, axial_slice], cmap='gray', vmin=0, vmax=2.5e-3)
ax.flat[5].set_title('MD (DTI)')
ax.flat[6].imshow(dti_AD[:, :, axial_slice], cmap='gray', vmin=0, vmax=2.5e-3)
ax.flat[6].set_title('AD (DTI)')
ax.flat[7].imshow(dti_RD[:, :, axial_slice], cmap='gray', vmin=0, vmax=2.5e-3)
ax.flat[7].set_title('RD (DTI)')

plt.show()
fig1.savefig('Diffusion_measures_from_free_water_DTI_and_standard_DTI.png')

"""
.. figure:: Diffusion_measures_from_free_water_DTI_and_standard_DTI.png
   :align: center
   **Diffusion tensor measures obtain from the diffusion tensor estimated from
   the free water DTI (upper panels) and standard DTI (lower panels).**.

From the figure, we can see that ...

In addition to the standard diffusion statistics, the FreeWaterDiffusionFit
instance can be used to display the volume fraction of the free water diffusion
componet
"""

F = fwdtifit.f
S0 = fwdtifit.S0

fig2, ax = plt.subplots(1, 2, figsize=(12, 6),
                        subplot_kw={'xticks': [], 'yticks': []})

fig2.subplots_adjust(hspace=0.3, wspace=0.05)

ax.flat[0].imshow(F[:, :, axial_slice], cmap='gray', vmin=0, vmax=1)
ax.flat[0].set_title('F')
ax.flat[1].imshow(S0[:, :, axial_slice], cmap='gray')
ax.flat[1].set_title('S0')

plt.show()
fig2.savefig('Volume_fraction_of_the_water_diffusion_component.png')

"""
.. figure:: Volume_fraction_of_the_water_diffusion_component.png
   :align: center
   **Volume fraction of the water diffusion component.**.

[Insert Figure explanation]

References:

.. [Hoy2014] Hoy, A.R., Koay, C.G., Kecskemeti, S.R., Alexander, A.L., 2014.
             Optimization of a free water elimination two-compartmental model
             for diffusion tensor imaging. NeuroImage 103, 323-333.
             doi: 10.1016/j.neuroimage.2014.09.053
"""
