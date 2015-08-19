"""
=====================================================================
Reconstruction of the diffusion signal with the kurtosis tensor model
=====================================================================

The diffusion kurtois model is an expansion of the diffusion tensor model. In
addition to the diffusion tensor (DT), the diffusion kurtosis model quantifies
the degree to which water diffusion in biologic tissues is non-Gaussian using
the kurtosis tensor [Jensen2005]_. Measurements of non-Gaussian diffusion are
of interest because they can be used to charaterize tissue microstructural
heterogeneity [Jensen2010]_ and to derive concrete biophysical parameters as
the density of axonal fibres and diffusion tortuosity [Fierem2011]_. Moreover,
DKI can be used to resolve crossing fibers in tractography and to obtain
invariant rotational measures not limited to well aligned fiber populations
[NetoHe2015]_.

The diffusion kurtosis model relates the diffusion-weighted signal,
$S(\mathbf{n}, b)$, to the applied diffusion weighting, $\mathbf{b}$, the
signal in the absence of diffusion gradient sensitisation, $S_0$, and the
values of diffusion, $\mathbf{D(n)}$, and diffusion kurtosis, $\mathbf{K(n)}$,
along the spatial direction $\mathbf{n}$ [NetoHe2015]_:

.. math::
    S(n,b)=S_{0}e^{-bD(n)+\frac{1}{6}b^{2}D(n)^{2}K(n)}

$\mathbf{D(n)}$ and $\mathbf{K(n)}$ can be computed from the DT and KT using
the following equations:

.. math::
     D(n)=\sum_{i=1}^{3}\sum_{j=1}^{3}n_{i}n_{j}D_{ij}

and

.. math::
     K(n)=\frac{MD^{2}}{D(n)^{2}}\sum_{i=1}^{3}\sum_{j=1}^{3}\sum_{k=1}^{3}
     \sum_{l=1}^{3}n_{i}n_{j}n_{k}n_{l}W_{ijkl}

where $D_{ij}$ and $W_{ijkl}$ are the elements of the second-order DT and the
fourth-order KT tensors, respectively, and $MD$ is the mean diffusivity.
As the DT, KT has antipodal symmetry and thus only 15 Wijkl elemments are
needed to fully characterize the KT:

.. math::
   \begin{matrix} ( & W_{xxxx} & W_{yyyy} & W_{zzzz} & W_{xxxy} & W_{xxxz}
                    & ... \\
                    & W_{xyyy} & W_{yyyz} & W_{xzzz} & W_{yzzz} & W_{xxyy}
                    & ... \\
                    & W_{xxzz} & W_{yyzz} & W_{xxyz} & W_{xyyz} & W_{xyzz}
                    & & )\end{matrix}

In the following example we show how to reconstruct your diffusion multi-shell
datasets using the kurtosis tensor model.

First import Dipy's DKI module:

"""

import dipy.reconst.dki as dki

"""
DKI requires multi-shell data, i.e. data acquired from more than one non-zero
b-value. Here, we use fetch to download the raw HARDI of a dMRI dataset
cotaining three non-zero b-values. The size of the dataset is 188 MBytes,
however you only need to fetch it once.
"""

from dipy.data import fetch_sherbrooke_3shell

fetch_sherbrooke_3shell()

"""
Next, we read the saved dataset
"""

from dipy.data import read_sherbrooke_3shell

img, gtab = read_sherbrooke_3shell()

data = img.get_data()

"""
img contains a nibabel Nifti1Image object (with the data) and gtab contains a
GradientTable object with information about the b-values and b-vectors. The
b-values used on the loaded dataset are visualized below
"""

import matplotlib.pyplot as plt

plt.plot(gtab.bvals, label='b-values')
fig1 = plt.gcf()
plt.legend()
plt.show()
fig1.savefig('HARDI193_bvalues.png')

"""
.. figure:: HARDI193_bvalues.png
   :align: center
   **b-values of the loaded dataset**.

From the figure above, we can check that the loaded dataset contains three
non-zero b-values as required for DKI. However the highest b-value of 3500
$s.mm^{-2}$ is higher than normally used on DKI. Since DKI neglects diffusion
signal components higher than the 4th order KT, a upper bound of b-value < 3000
$s.mm^{-2}$ is normally implied to decrease the influence of higher order terms
[Jensen2010]_. Following this, we discard the b-value shell of 3500 $s.mm^{-2}$
before DKI fitting.
"""

select_ind = gtab.bvals < 3000
selected_bvals = gtab.bvals[select_ind]
selected_bvecs = gtab.bvecs[select_ind, :]
selected_data = data[:, :, :, select_ind]

"""
The selected b-values and gradient directions are then converted to Dipy's
GradientTable format.
"""

from dipy.core.gradients import gradient_table

gtab = gradient_table(selected_bvals, selected_bvecs)

"""
Before fitting the data some data pre-processing is done. First, we mask and
crop the data to avoid calculating Tensors on the background of the image.
"""

from dipy.segment.mask import median_otsu

maskdata, mask = median_otsu(selected_data, 3, 1, True,
                             vol_idx=range(10, 50), dilate=2)

"""
Now that we have prepared the datasets we can go forward with the voxel
reconstruction. This can be done by first instantiate the DiffusinKurtosisModel
in the following way.
"""

dkimodel = dki.DiffusionKurtosisModel(gtab)

"""
Fitting the data is very simple. We just need to call the fit method of the
DiffusinKurtosisModel in the following way:
"""

dkifit = dkimodel.fit(maskdata)

"""
The fit method creates a DiffusionKurtosisFit object which contains all the
diffusion and kurtosis fitting parameters and other DKI attributes. For
instance, all diffusion standard tensor statistics can be computed from the
DiffusionKurtosisFit instance as the fractional anisotropy (FA), the mean
diffusivity (MD), the axial diffusivity (AD) and the radial diffusivity (RD).
"""

FA = dkifit.fa
MD = dkifit.md
AD = dkifit.ad
RD = dkifit.rd

"""
Note that these four standard measures could also be computed from Dipy's DTI
module. For comparison purposes, we calculate below the FA, MD, AD, and RD
using Dipy's DTI module. For this, we first extract the data of the lower
b-value to decrease the effects of the signal components lager than the 2nd
order which are not taken into account by DTI.
"""

select_dti_ind = gtab.bvals < 1500
selected_dti_bvals = gtab.bvals[select_dti_ind]
selected_dti_bvecs = gtab.bvecs[select_dti_ind, :]
selected_dti_data = data[:, :, :, select_dti_ind]
gtab_for_dti = gradient_table(selected_dti_bvals, selected_dti_bvecs)
maskdata_for_dti, mask = median_otsu(selected_dti_data, 3, 1, True,
                                     vol_idx=range(10, 50), dilate=2)

""" Then, we carry on with the DTI fitting and standard tensor statistics
by instantiating Dipy's TensorModel """

import dipy.reconst.dti as dti

tenmodel = dti.TensorModel(gtab_for_dti)
tenfit = tenmodel.fit(maskdata_for_dti)

dti_FA = tenfit.fa
dti_MD = tenfit.md
dti_AD = tenfit.ad
dti_RD = tenfit.rd

"""
The DT based measured obtain from DKI and DTI can be easly visualized using
matplotlib. For example, we show below the middle axial slices of FA, MD, AD,
and RD obtain from the DKI model (upper panels) and the DTI model (lower
panels).
"""

axial_middle = FA.shape[2] / 2

fig2, ax = plt.subplots(2, 4, figsize=(12, 6),
                        subplot_kw={'xticks': [], 'yticks': []})

fig2.subplots_adjust(hspace=0.3, wspace=0.05)

ax.flat[0].imshow(FA[:, :, axial_middle], cmap='gray')
ax.flat[0].set_title('FA (DKI)')
ax.flat[1].imshow(MD[:, :, axial_middle], cmap='gray')
ax.flat[1].set_title('MD (DKI)')
ax.flat[2].imshow(AD[:, :, axial_middle], cmap='gray')
ax.flat[2].set_title('AD (DKI)')
ax.flat[3].imshow(RD[:, :, axial_middle], cmap='gray')
ax.flat[3].set_title('RD (DKI)')

ax.flat[4].imshow(dti_FA[:, :, axial_middle], cmap='gray')
ax.flat[4].set_title('FA (DTI)')
ax.flat[5].imshow(dti_MD[:, :, axial_middle], cmap='gray')
ax.flat[5].set_title('MD (DTI)')
ax.flat[6].imshow(dti_AD[:, :, axial_middle], cmap='gray')
ax.flat[6].set_title('AD (DTI)')
ax.flat[7].imshow(dti_RD[:, :, axial_middle], cmap='gray')
ax.flat[7].set_title('RD (DTI)')

plt.show()
fig2.savefig('Diffusion_tensor_measures_from_DTI_and_DKI.png')

"""
.. figure:: Diffusion_tensor_measures_from_DTI_and_DKI.png
   :align: center
   **Diffusion tensor measures obtain from the diffusion tensor estimated from
   DKI (upper panels) and DTI (lower panels).**.

From the figure, we can see that the DT standard diffusion measures from DKI
are noisier than the DTI measurements. This is a well known pitfall of DKI
[NetoHe2014]_. Since it involves the estimation of a larger number of
parameters, DKI is more sensitive to noise than DTI. Nevertheless, DT diffusion
based measures were shown to have better precision (i.e. less sensitive to
bias) [Veraa2011]_.

The standard kurtosis statistics can be computed from the DiffusinKurtosisFit
instance as the mean kurtosis (MK), the axial kurtosis (AD) and the radial
kurtosis (RK).

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
.. [NetoHe2014] Neto Henriques R, Ferreira HA, Correia MM (2012). Diffusion
                kurtosis imaging of the healthy human brain. Master
                Dissertation Bachelor and Master Program in Biomedical
                Engineering and Biophysics, Faculty of Sciences.
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
