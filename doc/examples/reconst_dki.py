"""
=====================================================================
Reconstruction of the diffusion signal with the kurtosis tensor model
=====================================================================

The diffusion kurtois model is an expansion of the diffusion tensor model
(see :ref:`example_reconst_dti`). In addition to the diffusion tensor (DT), the
diffusion kurtosis model quantifies the degree to which water diffusion in
biologic tissues is non-Gaussian using the kurtosis tensor (KT) [Jensen2005]_.

Measurements of non-Gaussian diffusion from the diffusion kurtosis model are of
interest because they can be used to charaterize tissue microstructural
heterogeneity [Jensen2010]_ and to derive concrete biophysical parameters, such
as the density of axonal fibres and diffusion tortuosity [Fierem2011]_.
Moreover, DKI can be used to resolve crossing fibers in tractography and to
obtain invariant rotational measures not limited to well aligned fiber
populations [NetoHe2015]_.

The diffusion kurtosis model expresses the diffusion-weighted signal as:

.. math::

    S(n,b)=S_{0}e^{-bD(n)+\frac{1}{6}b^{2}D(n)^{2}K(n)}

where $\mathbf{b}$ is the applied diffusion weighting (which is dependent on
the measuremenent parameters), $S_0$ the signal in the absence of diffusion
gradient sensitization, $\mathbf{D(n)}$ the value of diffusion along direction
$\mathbf{n}$, and $\mathbf{K(n)}$ the value of kurtosis along direction
$\mathbf{n}$. The directional diffusion $\mathbf{D(n)}$ and kurtosis
$\mathbf{K(n)}$ can be related to the diffusion tensor (DT) and kurtosis tensor
(KT) using the following equations:

.. math::
     D(n)=\sum_{i=1}^{3}\sum_{j=1}^{3}n_{i}n_{j}D_{ij}

and

.. math::
     K(n)=\frac{MD^{2}}{D(n)^{2}}\sum_{i=1}^{3}\sum_{j=1}^{3}\sum_{k=1}^{3}
     \sum_{l=1}^{3}n_{i}n_{j}n_{k}n_{l}W_{ijkl}

where $D_{ij}$ are the elements of the second-order DT, and $W_{ijkl}$ the
elements of the fourth-order KT and $MD$ is the mean diffusivity. As the DT,
KT has antipodal symmetry and thus only 15 Wijkl elemments are needed to fully
characterize the KT:

.. math::
   \begin{matrix} ( & W_{xxxx} & W_{yyyy} & W_{zzzz} & W_{xxxy} & W_{xxxz}
                    & ... \\
                    & W_{xyyy} & W_{yyyz} & W_{xzzz} & W_{yzzz} & W_{xxyy}
                    & ... \\
                    & W_{xxzz} & W_{yyzz} & W_{xxyz} & W_{xyyz} & W_{xyzz}
                    & & )\end{matrix}

In the following example we show how to fit the diffusion kurtosis model on
diffusion-weighted multi-shell datasets and how to estimate diffusion kurtosis
based statistics.

First, we import all relevant modules:
"""

import dipy.reconst.dki as dki
import dipy.reconst.dti as dti
import matplotlib.pyplot as plt
from dipy.data import fetch_cenir_multib
from dipy.data import read_cenir_multib
from dipy.segment.mask import median_otsu

"""
DKI requires multi-shell data, i.e. data acquired from more than one non-zero
b-value. Here, we use fetch to download a multi-shell dataset which parameters
are similar to the used on the Human Connectome Project (HCP). The total size
of the dowloaded data is 1760 MBytes, however you only need to fetch it once.
Parameter ``with_raw`` of function ``fetch_cenir_multib`` is set to ``False``
to only download eddy-current/motion corrected data:
"""

fetch_cenir_multib(with_raw=False)

"""
Next, we read the saved dataset. To decrease the influence of diffusion signal
taylor approximation componets larger than the fourth order (componets not
taken into account by the diffusion kurtosis tensor), we only select the
b-values up to 2000 $s.mm^{-2}$:
"""

bvals = [200, 400, 1000, 2000]

img, gtab = read_cenir_multib(bvals)

data = img.get_data()

"""
Function ``read_cenir_multib`` return img and gtab which contains respectively
a nibabel Nifti1Image object (where the data can extracted) and a GradientTable
object with information about the b-values and b-vectors.

Before fitting the data, we first mask and crop the data to avoid calculating
Tensors on the background of the image.
"""

maskdata, mask = median_otsu(data, 3, 1, True, vol_idx=range(10, 50), dilate=2)

"""
Now that we have loaded and prepared the datasets we can go forward with the
voxel reconstruction. This can be done by first instantiating the
DiffusionKurtosisModel in the following way:
"""

dkimodel = dki.DiffusionKurtosisModel(gtab)

"""
To fitting the data using the defined ``dkimodel``, we just need to call the
fit function of the DiffusionKurtosisModel:
"""

dkifit = dkimodel.fit(maskdata)

"""
The fit method creates a DiffusionKurtosisFit object which contains all the
diffusion and kurtosis fitting parameters and other DKI attributes. For
instance, since the diffusion kurtosis model estimates the diffusion tensor,
all diffusion standard tensor statistics can be computed from the
DiffusionKurtosisFit instance. As example, we show below how to extract the
fractional anisotropy (FA), the mean diffusivity (MD), the axial diffusivity
(AD) and the radial diffusivity (RD) from the DiffusionKurtosisiFit instance.
"""

FA = dkifit.fa
MD = dkifit.md
AD = dkifit.ad
RD = dkifit.rd

"""
Note that these four standard measures could also be computed from Dipy's DTI
module. For comparison purposes, we calculate below the FA, MD, AD, and RD
using Dipy's TensorModel.
"""

tenmodel = dti.TensorModel(gtab)
tenfit = tenmodel.fit(maskdata)

dti_FA = tenfit.fa
dti_MD = tenfit.md
dti_AD = tenfit.ad
dti_RD = tenfit.rd

"""
The DT based measured obtain from DKI and DTI can be easly visualized using
matplotlib. For example, the FA, MD, AD, and RD obtain from the DKI model
(upper panels) and the DTI model (lower panels) are ploted for an arbitary
selected axial slice.
"""

axial_slice = 40

fig1, ax = plt.subplots(2, 4, figsize=(12, 6),
                        subplot_kw={'xticks': [], 'yticks': []})

fig1.subplots_adjust(hspace=0.3, wspace=0.05)

ax.flat[0].imshow(FA[:, :, axial_slice], cmap='gray')
ax.flat[0].set_title('FA (DKI)')
ax.flat[1].imshow(MD[:, :, axial_slice], cmap='gray')
ax.flat[1].set_title('MD (DKI)')
ax.flat[2].imshow(AD[:, :, axial_slice], cmap='gray')
ax.flat[2].set_title('AD (DKI)')
ax.flat[3].imshow(RD[:, :, axial_slice], cmap='gray')
ax.flat[3].set_title('RD (DKI)')

ax.flat[4].imshow(dti_FA[:, :, axial_slice], cmap='gray')
ax.flat[4].set_title('FA (DTI)')
ax.flat[5].imshow(dti_MD[:, :, axial_slice], cmap='gray')
ax.flat[5].set_title('MD (DTI)')
ax.flat[6].imshow(dti_AD[:, :, axial_slice], cmap='gray')
ax.flat[6].set_title('AD (DTI)')
ax.flat[7].imshow(dti_RD[:, :, axial_slice], cmap='gray')
ax.flat[7].set_title('RD (DTI)')

plt.show()
fig1.savefig('Diffusion_tensor_measures_from_DTI_and_DKI.png')

"""
.. figure:: Diffusion_tensor_measures_from_DTI_and_DKI.png
   :align: center
   **Diffusion tensor measures obtain from the diffusion tensor estimated from
   DKI (upper panels) and DTI (lower panels).**.

From the figure, we can see that the standard diffusion measures from DKI has
similar constrasts than the standard diffusion measures from DTI.

In addition to the standard diffusion statistics, the DiffusionKurtosisFit
instance can be used to estimate standard non-Gaussian measures as the mean
kurtosis (MK), the axial kurtosis (AK) and the radial kurtosis (RK).
"""

MK = dkifit.mk(0, 2)
AK = dkifit.ak(0, 2)
RK = dkifit.rk(0, 2)

"""
Kurtosis measures are susceptible to high amplitude outliers. The impact of
high amplitude kurtosis outliers were removed on the above lines of codes by
introducing as an optional input the extremes of the typical values of kurtosis
(assumed here as the values on the range between 0 and 2)

Now we are ready to plot the kurtosis standard measures using matplotlib:
"""

fig2, ax = plt.subplots(1, 3, figsize=(12, 6),
                        subplot_kw={'xticks': [], 'yticks': []})

fig2.subplots_adjust(hspace=0.3, wspace=0.05)

ax.flat[0].imshow(MK[:, :, axial_slice], cmap='gray')
ax.flat[0].set_title('MK')
ax.flat[1].imshow(AK[:, :, axial_slice], cmap='gray')
ax.flat[1].set_title('AK')
ax.flat[2].imshow(RK[:, :, axial_slice], cmap='gray')
ax.flat[2].set_title('RK')

plt.show()
fig2.savefig('Kurtosis_tensor_standard_measures.png')

"""
.. figure:: Kurtosis_tensor_standard_measures.png
   :align: center
   **Kurtosis tensor standard measures obtain from the kurtosis tensor.**.

The non-Gaussian behaviour of the diffusion signal is larger when water
diffusion is restrited by compartments and barriers (e.g., myelin sheath).
Therefore, as the figure above shows, white matter kurtosis values are smaller
along the axial direction of fibers (smaller amplitudes shown in the AK map)
than for the radial directions (larger amplitudes shown in the RK map).

References:

.. [Jensen2005] Jensen JH, Helpern JA, Ramani A, Lu H, Kaczynski K (2005).
                Diffusional Kurtosis Imaging: The Quantification of
                Non_Gaussian Water Diffusion by Means of Magnetic Resonance
                Imaging. Magnetic Resonance in Medicine 53: 1432-1440
.. [Jensen2010] Jensen JH, Helpern JA (2010). MRI quantification of
                non-Gaussian water diffusion by kurtosis analysis. NMR in
                Biomedicine 23(7): 698-710
.. [Fierem2011] Fieremans E, Jensen JH, Helpern JA (2011). White matter
                characterization with diffusion kurtosis imaging. NeuroImage
                58: 177-188
.. [NetoHe2015] Neto Henriques R, Correia MM, Nunes RG, Ferreira HA (2015).
                Exploring the 3D geometry of the diffusion kurtosis tensor -
                Impact on the development of robust tractography procedures and
                novel biomarkers, NeuroImage 111: 85-99

.. include:: ../links_names.inc
"""
