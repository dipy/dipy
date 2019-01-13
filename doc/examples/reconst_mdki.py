"""
===============================================================================
Characterizing diffusion signal with the mean signal diffusion kurtosis imaging
===============================================================================

Several microstrutural models have been proposed to increase the specificity
of diffusion-weighted data; however improrper model assumptions are known to
compromise the validity of the model's estimates [Henriques2019]_. To avoid
misleading interpretation, it might be useful and enought to characterize
diffusion-weighted data using signal representation techniques. For example,
assuming that the degree the non-Gaussian diffusion decreases with tissue
degeneration, quantifying this diffusion property can provides usefull
information about the tissue degeneration degree at a microscopic level of the
tissue. Although this cannot be use to distinguish different mechanisms of
microstructural changes (e.g. axonal loss vs demyelination), the degree of
non-Gaussian diffusion can provide a usefull sensitive measure that can be
used for example to study the relationship between brain microstructure and
behaviour changes (e.g. [Price2017]_).

Diffusion Kurtosis Imaging is one of the conventional way to estimate the
degree of non-Gaussian diffusion (see :ref:`example_reconst_dki`). However,
as previously pointed [NetoHe2015]_, standard kurtosis measures do not only
depend on microstructural properties but also on mesoscopic properties such as
fiber dispersion or the intersection angle of crossing fibers.

In the following example, we show how one can process the diffusion kurtosis of
mean signals (also known as powder-averaged signals) and obtain a
characterization of non-Gaussian diffusion independently to the degree of fiber
organization [NetoHe2018]_. In the first part of this example, the properties
of the measures obtained from the mean signal diffusion kurtosis imaging
[NetoHe2018]_ are illustrated using synthetic data. Secondly, the mean signal
diffusion kurtosis imaging will be applied to real in-vivo MRI data.

Let's import all relevant modules:
"""

import numpy as np
import matplotlib.pyplot as plt
import dipy.reconst.dki as dki
import dipy.reconst.dti as dti
import dipy.reconst.mdki as mdki
from dipy.data import fetch_cfin_multib
from dipy.data import read_cfin_dwi
from dipy.segment.mask import median_otsu
from scipy.ndimage.filters import gaussian_filter

""" Simulations (WIP)

Note as the standard DKI, MSDKI requires multi-shell
data (i.e. data acquired from more than one non-zero b-value).
"""

"""
Now that the properties of MSDKI we illustrated, let's apply MSDKI to in-vivo
diffusion-weighted data. As the example for the standard DKI
(see :ref:`example_reconst_dki`), we use fetch to download a multi-shell
dataset which was kindly provided by Hansen and Jespersen (more details about
the data are provided in their paper [Hansen2016]_). The total size of the
downloaded data is 192 MBytes, however you only need to fetch it once.
"""

fetch_cfin_multib()

img, gtab = read_cfin_dwi()

data = img.get_data()

affine = img.affine

"""
Before fitting the data, we preform some data pre-processing. For illustration,
in this examplae we only mask the data to avoid unnecessary calculations on the
background of the image. However, if you want to suppress noise artefacts,
several denoising algorithms are available in DIPY_ (e.g. the non-local means
filter :ref:`example-denoise-nlmeans`).
"""

maskdata, mask = median_otsu(data, 4, 2, False, vol_idx=[0, 1], dilate=1)

"""
Now that we have loaded and pre-processed the data we can go forward
with DKI fitting. For this, the DKI model is first defined for the data's
GradientTable object by instantiating the DiffusionKurtosisModel object in the
following way:
"""

mdki_model = mdki.MeanDiffusionKurtosisModel(gtab)

"""
To fit the data using the defined model object, we call the ``fit`` function of
this object:
"""

mdki_fit = mdki_model.fit(data, mask=mask)

"""
From the above fit object we can extract, the parameters of the MSDKI can
be obtained such as the mean signal diffusion (MSD) and mean signal kurtosis
(MSK)
"""

MSD = mdki_fit.msd
MSK = mdki_fit.msk

"""
For comparison purposes, we also calculate below the mean diffusivity (MD) and
mean kurtosis (MK) from the standard diffusion kurtosis imaging.
"""

dki_model = dki.DiffusionKurtosisModel(gtab)
dki_fit = dki_model.fit(data, mask=mask)

MD = dki_fit.md
MK = dki_fit.mk(0, 3)


"""
The DT based measures can be easily visualized using matplotlib. For example,
the FA, MD, AD, and RD obtained from the diffusion kurtosis model (upper
panels) and the tensor model (lower panels) are plotted for a selected axial
slice.
"""

axial_slice = 9

fig1, ax = plt.subplots(2, 2, figsize=(6, 6),
                        subplot_kw={'xticks': [], 'yticks': []})

fig1.subplots_adjust(hspace=0.3, wspace=0.05)

ax.flat[0].imshow(MD[:, :, axial_slice].T, cmap='gray', vmin=0, vmax=2.0e-3,
                  origin='lower')
ax.flat[0].set_title('MD (DKI)')
ax.flat[1].imshow(MK[:, :, axial_slice].T, cmap='gray', vmin=0, vmax=2,
                  origin='lower')
ax.flat[1].set_title('MK (DKI)')
ax.flat[2].imshow(MSD[:, :, axial_slice].T, cmap='gray', vmin=0, vmax=2.0e-3,
                  origin='lower')
ax.flat[2].set_title('MSD (MSDKI)')
ax.flat[3].imshow(MSK[:, :, axial_slice].T, cmap='gray', vmin=0, vmax=2,
                  origin='lower')
ax.flat[3].set_title('MSK (MSDKI)')

plt.show()
fig1.savefig('Measures_from_DKI_and_MSDKI.png')

"""
.. figure:: Measures_from_DKI_and_MSDKI.png
   :align: center

   Diffusion tensor measures obtained from the diffusion tensor estimated
   from DKI (upper panels) and DTI (lower panels).

In addition to the standard diffusion statistics, the DiffusionKurtosisFit
instance can be used to estimate the non-Gaussian measures of mean kurtosis
(MK), the axial kurtosis (AK) and the radial kurtosis (RK).
"""


"""
References
----------
.. [NetoHe2019] Neto Henriques R, Jespersen SN, Shemesh N (2019). Microscopic
                anisotropy misestimation in spherical‐mean single diffusion
                encoding MRI. Magnetic Resonance in Medicine (In Press).
                doi: 10.1002/mrm.27606
.. [Price2017]  Price D, Tyler LK, Neto Henriques R, Campbell KR, Williams N,
                Treder M, Taylor J, Cam-CAN, Henson R (2017). Age-Related
                Delay in Visual and Auditory Evoked Responses is Mediated by
                White- and Gray-matter Differences. Nature Communications 8,
                15671. doi: 10.1038/ncomms15671.
.. [Jensen2005] Jensen JH, Helpern JA, Ramani A, Lu H, Kaczynski K (2005).
                Diffusional Kurtosis Imaging: The Quantification of
                Non_Gaussian Water Diffusion by Means of Magnetic Resonance
                Imaging. Magnetic Resonance in Medicine 53: 1432-1440
.. [NetoHe2015] Neto Henriques R, Correia MM, Nunes RG, Ferreira HA (2015).
                Exploring the 3D geometry of the diffusion kurtosis tensor -
                Impact on the development of robust tractography procedures and
                novel biomarkers, NeuroImage 111: 85-99
.. [NetoHe2018] Henriques RN, 2018. Advanced Methods for Diffusion MRI Data
                Analysis and their Application to the Healthy Ageing Brain
                (Doctoral thesis). Downing College, University of Cambridge.
                https://doi.org/10.17863/CAM.29356
.. [Hansen2016] Hansen, B, Jespersen, SN (2016). Data for evaluation of fast
                kurtosis strategies, b-value optimization and exploration of
                diffusion MRI contrast. Scientific Data 3: 160072
                doi:10.1038/sdata.2016.72

.. include:: ../links_names.inc
"""
