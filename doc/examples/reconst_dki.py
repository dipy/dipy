"""
=====================================================================
Reconstruction of the diffusion signal with the kurtosis tensor model
=====================================================================

The diffusion kurtosis model is an expansion of the diffusion tensor model
(see :ref:`sphx_glr_examples_built_reconstruction_reconst_dti.py`). In
addition to the diffusion tensor (DT), the diffusion kurtosis model
quantifies the degree to which water diffusion in biological tissues is
non-Gaussian using the kurtosis tensor (KT) [Jensen2005]_.

Measurements of non-Gaussian diffusion from the diffusion kurtosis model are of
interest because they can be used to characterize tissue microstructural
heterogeneity [Jensen2010]_. Moreover, DKI can be used to: 1) derive concrete
biophysical parameters, such as the density of axonal fibers and diffusion
tortuosity [Fierem2011]_ (see
:ref:`sphx_glr_examples_built_reconstruction_reconst_dki_micro.py`); and 2)
resolve crossing fibers in tractography and to obtain invariant rotational
measures not limited to well-aligned fiber populations [NetoHe2015]_.

The diffusion kurtosis model expresses the diffusion-weighted signal as:

.. math::

    S(n,b)=S_{0}e^{-bD(n)+\\frac{1}{6}b^{2}D(n)^{2}K(n)}

where $\mathbf{b}$ is the applied diffusion weighting (which is dependent on
the measurement parameters), $S_0$ is the signal in the absence of diffusion
gradient sensitization, $\mathbf{D(n)}$ is the value of diffusion along
direction $\mathbf{n}$, and $\mathbf{K(n)}$ is the value of kurtosis along
direction $\mathbf{n}$. The directional diffusion $\mathbf{D(n)}$ and kurtosis
$\mathbf{K(n)}$ can be related to the diffusion tensor (DT) and kurtosis tensor
(KT) using the following equations:

.. math::
     D(n)=\\sum_{i=1}^{3}\\sum_{j=1}^{3}n_{i}n_{j}D_{ij}

and

.. math::
     K(n)=\\frac{MD^{2}}{D(n)^{2}}\\sum_{i=1}^{3}\\sum_{j=1}^{3}\\sum_{k=1}^{3}
     \\sum_{l=1}^{3}n_{i}n_{j}n_{k}n_{l}W_{ijkl}

where $D_{ij}$ are the elements of the second-order DT, and $W_{ijkl}$ the
elements of the fourth-order KT and $MD$ is the mean diffusivity. As the DT,
KT has antipodal symmetry and thus only 15 Wijkl elements are needed to fully
characterize the KT:

.. math::
   \\begin{matrix} ( & W_{xxxx} & W_{yyyy} & W_{zzzz} & W_{xxxy} & W_{xxxz}
                    & ... \\\\
                    & W_{xyyy} & W_{yyyz} & W_{xzzz} & W_{yzzz} & W_{xxyy}
                    & ... \\\\
                    & W_{xxzz} & W_{yyzz} & W_{xxyz} & W_{xyyz} & W_{xyzz}
                    & & )\\end{matrix}

In the following example we show how to fit the diffusion kurtosis model on
diffusion-weighted multi-shell datasets and how to estimate diffusion kurtosis
based statistics.

First, we import all relevant modules:
"""

import numpy as np
import dipy.reconst.dki as dki
import dipy.reconst.dti as dti
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.segment.mask import median_otsu
from dipy.viz.plotting import compare_maps
from scipy.ndimage import gaussian_filter

###############################################################################
# DKI requires multi-shell data, i.e. data acquired from more than one
# non-zero b-value. Here, we use fetch to download a multi-shell dataset which
# was kindly provided by Hansen and Jespersen (more details about the data are
# provided in their paper [Hansen2016]_). The total size of the downloaded
# data is 192 MBytes, however you only need to fetch it once.

fraw, fbval, fbvec, t1_fname = get_fnames('cfin_multib')

data, affine = load_nifti(fraw)
bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
gtab = gradient_table(bvals, bvecs)

###############################################################################
# Function ``get_fnames`` downloads and outputs the paths of the data,
# ``load_nifti`` returns the data as a nibabel Nifti1Image object, and
# ``read_bvals_bvecs`` loads the arrays containing the information about the
# b-values and b-vectors. These later arrays are converted to the
# GradientTable object required for Dipy_'s data reconstruction.
#
# Before fitting the data, we perform some data pre-processing. We first
# compute a brain mask to avoid unnecessary calculations on the background
# of the image.

maskdata, mask = median_otsu(data, vol_idx=[0, 1], median_radius=4, numpass=2,
                             autocrop=False, dilate=1)

###############################################################################
# Since the diffusion kurtosis models involves the estimation of a large
# number of parameters [TaxCMW2015]_ and since the non-Gaussian components of
# the diffusion signal are more sensitive to artefacts [NetoHe2012]_, it might
# be favorable to suppress the effects of noise and artefacts before diffusion
# kurtosis fitting. In this example the effects of noise and artefacts are
# suppress by using 3D Gaussian smoothing (with a Gaussian kernel with
# fwhm=1.25) as suggested by pioneer DKI studies (e.g. [Jensen2005]_,
# [NetoHe2012]_). Although here the Gaussian smoothing is used so that results
# are comparable to these studies, it is important to note that more advanced
# noise and artifact suppression algorithms are available in DIPY_, e.g. the
# Marcenko-Pastur PCA denoising algorithm
# (:ref:`sphx_glr_examples_built_preprocessing_denoise_mppca.py`) and
# the Gibbs artefact suppression algorithm
# (:ref:`sphx_glr_examples_built_preprocessing_denoise_gibbs.py`).

fwhm = 1.25
gauss_std = fwhm / np.sqrt(8 * np.log(2))  # converting fwhm to Gaussian std
data_smooth = np.zeros(data.shape)
for v in range(data.shape[-1]):
    data_smooth[..., v] = gaussian_filter(data[..., v], sigma=gauss_std)

###############################################################################
# Now that we have loaded and pre-processed the data we can go forward
# with DKI fitting. For this, the DKI model is first defined for the data's
# GradientTable object by instantiating the DiffusionKurtosisModel object in
# the following way:

dkimodel = dki.DiffusionKurtosisModel(gtab)

###############################################################################
# To fit the data using the defined model object, we call the ``fit`` function
# of this object. For the purpose of this example, we will only fit a
# single slice of the data:

data_smooth = data_smooth[:, :, 9:10]
mask = mask[:, :, 9:10]
dkifit = dkimodel.fit(data_smooth, mask=mask)

###############################################################################
# The fit method creates a DiffusionKurtosisFit object, which contains all the
# diffusion and kurtosis fitting parameters and other DKI attributes. For
# instance, since the diffusion kurtosis model estimates the diffusion tensor,
# all standard diffusion tensor statistics can be computed from the
# DiffusionKurtosisFit instance. For example, we can extract the fractional
# anisotropy (FA), the mean diffusivity (MD), the axial diffusivity (AD) and
# the radial diffusivity (RD) from the DiffusionKurtosisiFit instance. Of
# course, these measures can also be computed from DIPY's ``TensorModel`` fit,
# and should be analogous; however, theoretically, the diffusion statistics
# from the kurtosis model are expected to have better accuracy, since DKI's
# diffusion tensor are decoupled from higher order terms effects
# [Veraar2011]_, [NetoHe2012]_. Below we compare the FA, MD, AD, and RD,
# computed from both DTI and DKI.

tenmodel = dti.TensorModel(gtab)
tenfit = tenmodel.fit(data_smooth, mask=mask)

fits = [tenfit, dkifit]
maps = ['fa', 'md', 'ad', 'rd']
fit_labels = ['DTI', 'DKI']
map_kwargs = [{'vmax': 0.7}, {'vmax': 2e-3}, {'vmax': 2e-3}, {'vmax': 2e-3}]
compare_maps(fits, maps, fit_labels=fit_labels, map_kwargs=map_kwargs,
             filename='Diffusion_tensor_measures_from_DTI_and_DKI.png')

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Diffusion tensor measures obtained from the diffusion tensor estimated
# from DKI (upper panels) and DTI (lower panels).
#
#
# DTI's diffusion estimates present lower values than DKI's estimates,
# showing that DTI's diffusion measurements are underestimated  by higher
# order effects.
#
# In addition to the standard diffusion statistics, the DiffusionKurtosisFit
# instance can be used to estimate the non-Gaussian measures of mean kurtosis
# (MK), the axial kurtosis (AK) and the radial kurtosis (RK).

maps = ['mk', 'ak', 'rk']
compare_maps([dkifit], maps, fit_labels=['DKI'],
             map_kwargs={'vmin': 0, 'vmax': 1.5},
             filename='Kurtosis_tensor_standard_measures.png')

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# DKI standard kurtosis measures.
#
#
# The non-Gaussian behaviour of the diffusion signal is expected to be higher
# when tissue water is confined by multiple compartments. MK is, therefore,
# higher in white matter since it is highly compartmentalized by myelin
# sheaths. These water diffusion compartmentalization is expected to be more
# pronounced perpendicularly to white matter fibers and thus the RK map
# presents higher amplitudes than the AK map.
#
# It is important to note that kurtosis estimates might present negative
# estimates in deep white matter regions (e.g. the band of dark voxels in the
# RK map above). These negative kurtosis values are artefactual and might be
# induced by:
# 1) low radial diffusivities of aligned white matter - since it is very hard
# to capture non-Gaussian information in radial direction due to it's low
# diffusion decays, radial kurtosis estimates (and consequently the mean
# kurtosis estimates) might have low robustness and tendency to exhibit
# negative values [NetoHe2012]_;
# 2) Gibbs artefacts - MRI images might be corrupted by signal oscillation
# artefact between tissue's edges if an inadequate number of high frequencies
# of the k-space is sampled. These oscillations might have different signs on
# images acquired with different diffusion-weighted and inducing negative
# biases in kurtosis parametric maps [Perron2015]_, [NetoHe2018]_.
#
# One can try to suppress this issue by using the more advance noise and
# artefact suppression algorithms, e.g., as mentioned above, the MP-PCA
# denoising (:ref:`sphx_glr_examples_built_preprocessing_denoise_mppca.py`)
# and Gibbs Unringing
# (:ref:`sphx_glr_examples_built_preprocessing_denoise_gibbs.py`)
# algorithms. Alternatively, one can overcome this artefact by computing the
# kurtosis values from powder-averaged diffusion-weighted signals. The details
# on how to compute the kurtosis from powder-average signals in dipy are
# described in follow the tutorial
# (:ref:`sphx_glr_examples_built_reconstruction_reconst_msdki.py`). Finally,
# one can use constrained optimization to ensure that the fitted parameters
# are physically plausible [DelaHa2020]_, as we will illustrate in the next
# section. Ideally though, artefacts such as Gibbs ringing should be corrected
# for as well as possible before using constrained optimization.
#
# Constrained optimization for DKI
# ================================
#
# When instantiating the DiffusionKurtosisModel, the model can be set up to use
# constraints with the option `fit_method='CLS'` (for ordinary least squares)
# or with `fit_method='CWLS'` (for weighted least squares). Constrained fitting
# takes more time than unconstrained fitting, but is generally recommended to
# prevent physically unplausible parameter estimates [DelaHa2020]_. For
# performance purposes it is recommended to use the MOSEK solver
# (https://www.mosek.com/) by setting ``cvxpy_solver='MOSEK'``. Different
# solvers can differ greatly in terms of runtime and solution accuracy, and in
# some cases solvers may show warnings about convergence or recommended option
# settings.
#
# .. note::
#    In certain atypical scenarios, the DKI+ constraints could potentially be
#    too restrictive. Always check the results of a constrained fit with their
#    unconstrained counterpart to verify that there are no unexpected
#    qualitative differences.

dkimodel_plus = dki.DiffusionKurtosisModel(gtab, fit_method='CLS')
dkifit_plus = dkimodel_plus.fit(data_smooth, mask=mask)

###############################################################################
# We can now compare the kurtosis measures obtained with the constrained fit to
# the measures obtained before, where we see that many of the artefactual
# voxels have now been corrected. In particular outliers caused by pure noise
# -- instead of for example acquisition artefacts -- can be corrected with
# this method.

compare_maps([dkifit_plus], ['mk', 'ak', 'rk'], fit_labels=['DKI+'],
             filename='Kurtosis_tensor_standard_measures_plus.png')

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# DKI standard kurtosis measures obtained with constrained optimization.
#
#
# When using constrained optimization, the expected range of the kurtosis
# measures is also naturally constrained, and so does not typically require
# additional clipping.
#
# Finally, constrained optimization obviates the need for smoothing in many
# cases:

dkifit_noisy = dkimodel.fit(data[:, :, 9:10], mask=mask)
dkifit_noisy_plus = dkimodel_plus.fit(data[:, :, 9:10], mask=mask)

compare_maps([dkifit_noisy, dkifit_noisy_plus], ['mk', 'ak', 'rk'],
             fit_labels=['DKI', 'DKI+'], map_kwargs={'vmin': 0, 'vmax': 1.5},
             filename='Kurtosis_tensor_standard_measures_noisy.png')

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# DKI standard kurtosis measures obtained on unsmoothed data with constrained
# optimization.
#
#
# Mean kurtosis tensor and kurtosis fractional anisotropy
# =======================================================
#
# As pointed by previous studies [Hansen2013]_, axial, radial and mean kurtosis
# depends on the information of both diffusion and kurtosis tensor. DKI
# measures that only depend on the kurtosis tensor include the mean of the
# kurtosis tensor [Hansen2013]_, and the kurtosis fractional anisotropy
# [GlennR2015]_. These measures are computed and illustrated below:

compare_maps([dkifit_plus], ['mkt', 'kfa'], fit_labels=['DKI+'],
             map_kwargs=[{'vmin': 0, 'vmax': 1.5}, {'vmin': 0, 'vmax': 1}],
             filename='Measures_from_kurtosis_tensor_only.png')

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# DKI measures obtained from the kurtosis tensor only.
#
#
# As reported by [Hansen2013]_, the mean of the kurtosis tensor (MKT) produces
# similar maps than the standard mean kurtosis (MK). On the other hand,
# the kurtosis fractional anisotropy (KFA) maps shows that the kurtosis tensor
# have different degrees of anisotropy than the FA measures from the diffusion
# tensor.
#
# References
# ----------
# .. [Jensen2005] Jensen JH, Helpern JA, Ramani A, Lu H, Kaczynski K (2005).
#                 Diffusional Kurtosis Imaging: The Quantification of
#                 Non_Gaussian Water Diffusion by Means of Magnetic Resonance
#                 Imaging. Magnetic Resonance in Medicine 53: 1432-1440
# .. [Jensen2010] Jensen JH, Helpern JA (2010). MRI quantification of
#                 non-Gaussian water diffusion by kurtosis analysis. NMR in
#                 Biomedicine 23(7): 698-710
# .. [Fierem2011] Fieremans E, Jensen JH, Helpern JA (2011). White matter
#                 characterization with diffusion kurtosis imaging. NeuroImage
#                 58: 177-188
# .. [Veraar2011] Veraart J, Poot DH, Van Hecke W, Blockx I, Van der Linden A,
#                 Verhoye M, Sijbers J (2011). More Accurate Estimation of
#                 Diffusion Tensor Parameters Using Diffusion Kurtosis Imaging.
#                 Magnetic Resonance in Medicine 65(1): 138-145
# .. [NetoHe2012] Neto Henriques R, Ferreira H, Correia M, (2012). Diffusion
#                 kurtosis imaging of the healthy human brain. Master
#                 Dissertation Bachelor and Master Programin Biomedical
#                 Engineering and Biophysics, Faculty of Sciences.
#                 https://repositorio.ul.pt/bitstream/10451/8511/1/ulfc104137_tm_Rafael_Henriques.pdf
# .. [Hansen2013] Hansen B, Lund TE, Sangill R, and Jespersen SN (2013).
#                 Experimentally and computationally393fast method for
#                 estimation of a mean kurtosis. Magnetic Resonance in
#                 Medicine 69, 1754–1760.394doi:10.1002/mrm.24743
# .. [GlennR2015] Glenn GR, Helpern JA, Tabesh A, Jensen JH (2015).
#                 Quantitative assessment of diffusional387kurtosis anisotropy.
#                 NMR in Biomedicine28, 448–459. doi:10.1002/nbm.3271
# .. [NetoHe2015] Neto Henriques R, Correia MM, Nunes RG, Ferreira HA (2015).
#                 Exploring the 3D geometry of the diffusion kurtosis tensor -
#                 Impact on the development of robust tractography procedures
#                 and novel biomarkers, NeuroImage 111: 85-99
# .. [Perron2015] Perrone D, Aelterman J, Pižurica A, Jeurissen B, Philips W,
#                 Leemans A, (2015). The effect of Gibbs ringing artifacts on
#                 measures derived from diffusion MRI. Neuroimage 120, 441-455.
#                 https://doi.org/10.1016/j.neuroimage.2015.06.068.
# .. [TaxCMW2015] Tax CMW, Otte WM, Viergever MA, Dijkhuizen RM, Leemans A
#                 (2014). REKINDLE: Robust extraction of kurtosis INDices with
#                 linear estimation. Magnetic Resonance in Medicine 73(2):
#                 794-808.
# .. [Hansen2016] Hansen, B, Jespersen, SN (2016). Data for evaluation of fast
#                 kurtosis strategies, b-value optimization and exploration of
#                 diffusion MRI contrast. Scientific Data 3: 160072
#                 doi:10.1038/sdata.2016.72
# .. [NetoHe2018] Neto Henriques R (2018). Advanced Methods for Diffusion MRI
#                 Data Analysis and their Application to the Healthy Ageing
#                 Brain (Doctoral thesis). https://doi.org/10.17863/CAM.29356
# .. [DelaHa2020] Dela Haije et al. "Enforcing necessary non-negativity
#                 constraints for common diffusion MRI models using sum of
#                 squares programming". NeuroImage 209, 2020, 116405.
