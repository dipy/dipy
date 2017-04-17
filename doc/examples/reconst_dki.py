"""
=====================================================================
Reconstruction of the diffusion signal with the kurtosis tensor model
=====================================================================

The diffusion kurtosis model is an expansion of the diffusion tensor model
(see :ref:`example_reconst_dti`). In addition to the diffusion tensor (DT), the
diffusion kurtosis model quantifies the degree to which water diffusion in
biological tissues is non-Gaussian using the kurtosis tensor (KT)
[Jensen2005]_.

Measurements of non-Gaussian diffusion from the diffusion kurtosis model are of
interest because they can be used to charaterize tissue microstructural
heterogeneity [Jensen2010]_ and to derive concrete biophysical parameters, such
as the density of axonal fibres and diffusion tortuosity [Fierem2011]_.
Moreover, DKI can be used to resolve crossing fibers in tractography and to
obtain invariant rotational measures not limited to well-aligned fiber
populations [NetoHe2015]_.

The diffusion kurtosis model expresses the diffusion-weighted signal as:

.. math::

    S(n,b)=S_{0}e^{-bD(n)+\frac{1}{6}b^{2}D(n)^{2}K(n)}

where $\mathbf{b}$ is the applied diffusion weighting (which is dependent on
the measurement parameters), $S_0$ is the signal in the absence of diffusion
gradient sensitization, $\mathbf{D(n)}$ is the value of diffusion along
direction $\mathbf{n}$, and $\mathbf{K(n)}$ is the value of kurtosis along
direction $\mathbf{n}$. The directional diffusion $\mathbf{D(n)}$ and kurtosis
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

import numpy as np
import matplotlib.pyplot as plt
import dipy.reconst.dki as dki
import dipy.reconst.dti as dti
import dipy.reconst.dki_micro as dki_micro
from dipy.data import fetch_cenir_multib
from dipy.data import read_cenir_multib
from dipy.segment.mask import median_otsu
from scipy.ndimage.filters import gaussian_filter

"""
DKI requires multi-shell data, i.e. data acquired from more than one non-zero
b-value. Here, we use fetch to download a multi-shell dataset with parameters
that are similar to the data collected in the Human Connectome Project (HCP).
The total size of the downloaded data is 1760 MBytes, however you only need to
fetch it once. Parameter ``with_raw`` of function ``fetch_cenir_multib`` is set
to ``False`` to only download eddy-current/motion corrected data:
"""

fetch_cenir_multib(with_raw=False)

"""
Next, we read the saved dataset. To decrease the influence of diffusion signal
Taylor approximation components larger than the fourth order (componets not
taken into account by the diffusion kurtosis tensor), we only select the
b-values up to 2000 $s.mm^{-2}$:
"""

bvals = [200, 400, 1000, 2000]

img, gtab = read_cenir_multib(bvals)

data = img.get_data()

affine = img.affine

"""
Function ``read_cenir_multib`` return img and gtab which contains respectively
a nibabel Nifti1Image object (where the data can be extracted) and a
GradientTable object with information about the b-values and b-vectors.

Before fitting the data, we preform some data pre-processing. We first compute
a brain mask to avoid unnecessary calculations on the background of the image.
"""

maskdata, mask = median_otsu(data, 4, 2, False, vol_idx=[0, 1], dilate=1)

"""
Since the diffusion kurtosis models involves the estimation of a large number
of parameters [TaxCMW2015]_ and since the non-Gaussian components of the
diffusion signal are more sensitive to artefacts [NetoHe2012]_, suppression of
the effects of noise and artefacsts are fundamental before diffusion kurtosis
fitting. Following the suggestion of pioner DKI studies (e.g. [Jensen2005]_,
 [NetoHe2012]_, [Fierem2011]_), the suppression of noise and artefacts in
this example is done using a 3D gaussian smoothing procedure (with a gaussian
kernel with fwhm=1.25).

For this, we use Dipy's non-local mean filter (see
:ref:`example-denoise-nlmeans`).
"""

fwhm = 1.25
gauss_std = fwhm / np.sqrt(8 * np.log(2))  # converting fwhm to gaussian std
data_smooth = np.zeros(data.shape)
for v in range(data.shape[-1]):
    data_smooth[..., v] = gaussian_filter(data[..., v], sigma=gauss_std)

"""
Data smoothing has the drawback of corrupting the data spatial information and
increasing partial volume effects. We opted to smooth DKI data just for
illustration proposes and easier comparison between produced figures and
previous literature. However, it is important to note that alternative noise
suppression algorithms are available in dipy as the use of non-local mean
(see :ref:`example-denoise-nlmeans`) and PCA filtering
(see :ref:`example-denoise-nlmeans`) techniques.

Now that we have loaded and prepared the voxels to process we can go forward
with the voxel reconstruction. This can be done by first instantiating the
DiffusionKurtosisModel in the following way:
"""

dkimodel = dki.DiffusionKurtosisModel(gtab)

"""
To fit the data using the defined model object, we call the ``fit`` function of
this object:
"""

dkifit = dkimodel.fit(data_smooth, mask=mask)

"""
The fit method creates a DiffusionKurtosisFit object which contains all the
diffusion and kurtosis fitting parameters and other DKI attributes. For
instance, since the diffusion kurtosis model estimates the diffusion tensor,
all diffusion standard tensor statistics can be computed from the
DiffusionKurtosisFit instance. For example, we show below how to extract the
fractional anisotropy (FA), the mean diffusivity (MD), the axial diffusivity
(AD) and the radial diffusivity (RD) from the DiffusionKurtosisiFit instance.
"""

FA = dkifit.fa
MD = dkifit.md
AD = dkifit.ad
RD = dkifit.rd

"""
Note that these four standard measures could also be computed from Dipy's DTI
module. Theoretically, computing these measures from both models should be
analogous. However, since the diffusion kurtosis model involves a larger number
of parameters, diffusion statistics maps can look more noisy. On the other
hand, the diffusion statistics from the kurtosis model are expected to have
better accuracy [Veraar2011]_. For comparison purposes, we calculate below the
FA, MD, AD, and RD using Dipy's TensorModel.
"""

tenmodel = dti.TensorModel(gtab)
tenfit = tenmodel.fit(data_smooth, mask=mask)

dti_FA = tenfit.fa
dti_MD = tenfit.md
dti_AD = tenfit.ad
dti_RD = tenfit.rd

"""
The DT based measures can be easly visualized using matplotlib. For example,
the FA, MD, AD, and RD obtain from the diffusion kurtosis model (upper panels)
and the tensor model (lower panels) are plotted for a selected axial slice.
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

From the figure, we can see that the standard diffusion measures of the
HCP-like data obtained from the diffusion kurtosis model have similar contrasts
to the standard diffusion measures obtain from the tensor model. This can be
explained by the high quality of the dataset and the high performance of the
diffusion kurtosis fit procedure.

In addition to the standard diffusion statistics, the DiffusionKurtosisFit
instance can be used to estimate the non-Gaussian measures of mean kurtosis
(MK), the axial kurtosis (AK) and the radial kurtosis (RK).
"""

MK = dkifit.mk(0, 3)
AK = dkifit.ak(0, 3)
RK = dkifit.rk(0, 3)

"""
Kurtosis measures are susceptible to high amplitude outliers. The impact of
high amplitude kurtosis outliers were removed on the above lines of codes by
introducing as an optional input the extremes of the typical values of kurtosis
(assumed here as the values on the range between 0 and 3)

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

As mentioned above, DKI can also be used to derive concrete biophysical
parameters by applying microstructural models to DT and KT estimated from DKI.
For instance,  Fieremans et al. (2011) showed that DKI can be used to estimate
the contribution of hindered and restricted diffusion for well aligned fibers.
These tensors can be also interprested as the influences of intra and
extra-cellular compartments and can be used to estimate the axonal volume
fraction and diffusion extra-cellular tortuosity which can be used to
distinguish processes of axonal loss from processes of myelin degeneration
[Fieremans2012]_.

The model proposed by Fieremans and collegues can be defined in dipy by
instantiating the 'KurtosisMicrostructureModel' Object in the following way:
"""

dki_micro_model = dki_micro.KurtosisMicrostructureModel(gtab)

"""
Before fitting this microstructural model, it is usefull to indicate the
regions in which this model provides meanfully information (i.e. voxels of
well-aligned fibers). Following Fieremans et al. [Fieremans2011]_, a simpe way
to select this regions is to generate a well aligned fiber mask based on the
values of diffusion sphericity, planarity and linearity. Here we will follow
this selection criteria for a better comparision of our results with Fieremans
original article, however it is important to note that voxel well-aligned fiber
can be selected using other approaches such as based on predefined regions of
interest.
"""

well_aligned_mask = np.ones(data.shape[:-1], dtype='bool')

cl = tenfit.linearity.copy()
well_aligned_mask[cl < 0.4] = False
well_aligned_mask[np.isnan(cl)] = False

cp = tenfit.planarity.copy()
well_aligned_mask[cp > 0.3] = False
well_aligned_mask[np.isnan(cp)] = False

cs = tenfit.sphericity.copy()
well_aligned_mask[cs > 0.5] = False
well_aligned_mask[np.isnan(cs)] = False

print(np.sum(well_aligned_mask == True))

well_aligned_mask = np.ones(data.shape[:-1], dtype='bool')

ev1 = dkifit.evals[..., 0]
ev2 = dkifit.evals[..., 1]
ev3 = dkifit.evals[..., 1]

cs = ev3 / ev1
cp = (ev2 - ev3) / ev1
cl = (ev1 - ev2) / ev1

cl = dkifit.linearity
well_aligned_mask[cl < 0.4] = False
well_aligned_mask[np.isnan(cl)] = False

cp = dkifit.planarity
well_aligned_mask[cp > 0.2] = False
well_aligned_mask[np.isnan(cp)] = False

cs = dkifit.sphericity
well_aligned_mask[cs > 0.35] = False
well_aligned_mask[np.isnan(cs)] = False

print(np.sum(well_aligned_mask == True))

"""
Analgous to DKI, the data fit can be done by calling the ``fit`` function of
the model's object as follows:
"""

dki_micro_fit = dki_micro_model.fit(data_smooth, mask=well_aligned_mask)

"""
The KurtosisMicrostructureFit object created by this ``fit`` function can then
be used to extract model parameters as the axonal water fraction and diffusion
hindered tortuosity:
"""

AWF = dki_micro_fit.awf
TORT = dki_micro_fit.tortuosity

""" Below these parameters are ploted in top of the mean kurtosis maps:
"""
fig2, ax = plt.subplots(1, 2, figsize=(9, 3),
                        subplot_kw={'xticks': [], 'yticks': []})

ax[0].imshow(MK[:, :, 41], cmap=plt.cm.gray, interpolation='nearest')
ax[0].imshow(AWF[:, :, 41], cmap=plt.cm.Reds, alpha=.5,
             interpolation='nearest')

ax[1].imshow(MK[:, :, 41], cmap=plt.cm.gray, interpolation='nearest')
ax[1].imshow(TORT[:, :, 41], cmap=plt.cm.Reds, alpha=.5,
             interpolation='nearest')

"""
References:

.. [TaxCMW2015] Tax CMW, Otte WM, Viergever MA, Dijkhuizen RM, Leemans A
                (2014). REKINDLE: Robust extraction of kurtosis INDices with
                linear estimation. Magnetic Resonance in Medicine 73(2):
                794-808.
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
.. [Fierem2012] Fieremans E, Jensen JH, Helpern JA, Kim S, Grossman RI,
                Inglese M, Novikov DS. (2012). Diffusion distinguishes between
                axonal loss and demyelination in brain white matter.
                Proceedings of the 20th Annual Meeting of the International
                Society for Magnetic Resonance Medicine; Melbourne, Australia.
                May 5–11.
.. [NetoHe2012] Neto Henriques R, Ferreira H, Correia M, (2012). Diffusion
                kurtosis imaging of the healthy human brain. Master
                Dissertation Bachelor and Master Programin Biomedical
                Engineering and Biophysics, Faculty of Sciences.
                http://repositorio.ul.pt/bitstream/10451/8511/1/ulfc104137_tm_Rafael_Henriques.pdf
.. [NetoHe2015] Neto Henriques R, Correia MM, Nunes RG, Ferreira HA (2015).
                Exploring the 3D geometry of the diffusion kurtosis tensor -
                Impact on the development of robust tractography procedures and
                novel biomarkers, NeuroImage 111: 85-99
.. [Veraar2011] Veraart J, Poot DH, Van Hecke W, Blockx I, Van der Linden A,
                Verhoye M, Sijbers J (2011). More Accurate Estimation of
                Diffusion Tensor Parameters Using Diffusion Kurtosis Imaging.
                Magnetic Resonance in Medicine 65(1): 138-145

.. include:: ../links_names.inc
"""
