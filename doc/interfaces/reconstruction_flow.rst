.. _reconstruction_flow:

==============
Reconstruction
==============

This tutorial walks through the steps to perform reconstruction using DIPY.
Multiple reconstruction methods are available in DIPY.

You can try these methods using your own data; we will be using the data in
DIPY. You can check how to :ref:`fetch the DIPY data<data_fetch>`.

-----------------------------------------
Constrained Spherical Deconvolution (CSD)
-----------------------------------------

This method is mainly useful with datasets with gradient directions acquired in
Cartesian coordinates that can be resampled to spherical coordinates so as to
use SD methods.

The basic idea of spherical deconvolution methods lies in the fact that the
underlying fiber distribution can be obtained by deconvolving the measured
diffusion signal with a fiber response function, provided that we are able to
accurately estimate the latter.

In this way, the reconstruction of the fiber orientation distribution function
(fODF) in CSD involves two steps:

* Estimation of the fiber response function.
* Use the response function to reconstruct the fODF.

We will be using the ``stanford_hardi`` dataset for the CSD command line
interface demonstration purposes.

We will start by creating the directories to which to save the peaks volume
(e.g.: ``recons_csd_output``) and mask file (e.g.: ``stanford_hardi_mask``)::

    mkdir recons_csd_output stanford_hardi_mask

The workflow for the CSD reconstruction method requires the paths to the
diffusion input file, b-values file, b-vectors file and mask file. The optional
arguments can also be provided. In this case, we will specify the FA threshold
for calculating the response function (``fa_thr``), spherical harmonics order
(l) used in the CSA fit (``sh_order_max``), whether to use parallelization in
peak-finding during the calibration procedure or not (``parallel``), and the
output directory (``out_dir``).

To get the mask file, we will use the median Otsu thresholding method by
calling the ``dipy_median_otsu`` command::

    dipy_median_otsu data/stanford_hardi/HARDI150.nii.gz --vol_idx 10-50 --out_dir "stanford_hardi_mask"

Then, to perform the CSD reconstruction we will run the ``dipy_fit_csd``
command as::

    dipy_fit_csd data/stanford_hardi/HARDI150.nii.gz data/stanford_hardi/HARDI150.bval data/stanford_hardi/HARDI150.bvec stanford_hardi_mask/brain_mask.nii.gz --fa_thr 0.7 --sh_order_max 8 --parallel --out_dir "recons_csd_output"

This command will save the CSD metrics to the specified output directory.

----------------------------------
Mean Apparent Propagator (MAP)-MRI
----------------------------------

The MAP-MRI basis allows for the computation of directional indices, such as
the Return To the Axis Probability (RTAP), the Return To the Plane Probability
(RTPP), and the parallel and perpendicular Non-Gaussianity.

The estimation of analytical Orientation Distribution Function (ODF) and a
variety of scalar indices from noisy DWIs requires that the fitting of the
MAPMRI basis is regularized.

We will use the ``cfin_multib`` dataset in DIPY to showcase this reconstruction
method. You can also use your own data!

We will create the output directory in which to save the MAP-MRI metrics (e.g.:
``recons_mapmri_output``)::

    mkdir recons_mapmri_output

The Mean Apparent Propagator reconstruction method requires the paths to the
diffusion input file, b-values file, b-vectors file, small delta value, big
delta value. The optional parameters can also be provided. In this case, we
will be specifying the threshold used to find b0 volumes (``b0_threshold``)
and the output directory (``out_dir``).

To run the MAP-MRI reconstruction method on the data, execute the
``dipy_fit_mapmri`` command, e.g.::

    dipy_fit_mapmri data/cfin_multib/__DTI_AX_ep2d_2_5_iso_33d_20141015095334_4.nii data/cfin_multib/__DTI_AX_ep2d_2_5_iso_33d_20141015095334_4.bval data/cfin_multib/__DTI_AX_ep2d_2_5_iso_33d_20141015095334_4.bvec 0.0157 0.0365 --b0_threshold 80.0 --out_dir recons_mapmri_output

This command will save the MAP-MRI metrics to the specified output directory.

------------------------------
Diffusion Tensor Imaging (DTI)
------------------------------

The diffusion tensor model is a model that describes the diffusion within a
voxel. First proposed by Basser and colleagues [Basser1994]_, it has been very
influential in demonstrating the utility of diffusion MRI in characterizing the
micro-structure of white matter tissue and of the biophysical properties of
tissue, inferred from local diffusion properties and it is still very commonly
used.

The diffusion tensor models the diffusion signal as:

.. math::

    \frac{S(\mathbf{g}, b)}{S_0} = e^{-b\mathbf{g}^T \mathbf{D} \mathbf{g}}

Where $\mathbf{g}$ is a unit vector in 3D space indicating the direction of
measurement and b are the parameters of measurement, such as the strength and
duration of the diffusion-weighting gradient. $S(\mathbf{g}, b)$ is the
measured diffusion-weighted signal and $S_0$ is the signal conducted in a
measurement with no diffusion weighting. $\mathbf{D}$ is a positive-definite
quadratic form, which contains six free parameters to be fit. These six
parameters are:

.. math::

   \mathbf{D} = \begin{pmatrix} D_{xx} & D_{xy} & D_{xz} \\
                       D_{yx} & D_{yy} & D_{yz} \\
                       D_{zx} & D_{zy} & D_{zz} \\ \end{pmatrix}

This matrix is a variance/covariance matrix of the diffusivity along the three
spatial dimensions. Note that we can assume that diffusivity has antipodal
symmetry, so elements across the diagonal are equal. For example:
$D_{xy} = D_{yx}$. This is why there are only 6 free parameters to estimate
here.

We will use the ``stanford_hardi`` dataset in DIPY to showcase this
reconstruction method. As with any other workflow in DIPY, you can also use
your own data!

We will first create a directory in which to save the output volumes(e.g.:
``recons_dti_output``)::

    mkdir recons_dti_output

To run the Diffusion Tensor Imaging reconstruction method, we need to specify
the paths to the diffusion input file, b-values file, b-vectors file and mask
file, followed by optional arguments. In this case, we will be specifying the
list of metrics to save (``save_metrics``), the output directory (``out_dir``),
and the name of the tensors volume to be saved (``out_tensor``).

The DTI reconstruction is performed by calling the ``dipy_fit_dti`` command,
e.g.::

    dipy_fit_dti data/stanford_hardi/HARDI150.nii.gz data/stanford_hardi/HARDI150.bval data/stanford_hardi/HARDI150.bvec stanford_hardi_mask/brain_mask.nii.gz --save_metrics "md" "mode" "tensor" --out_dir "recons_dti_output" --out_tensor "dti_tensors.nii.gz"

This command will save the DTI metrics to the specified output directory. The tensors will be saved as a 4D data with last dimension representing (Dxx, Dxy, Dyy, Dxz, Dyz, Dzz).

--------------------------------
Diffusion Kurtosis Imaging (DKI)
--------------------------------

The diffusion kurtosis model is an expansion of the diffusion tensor model. In
addition to the diffusion tensor (DT), the diffusion kurtosis model quantifies
the degree to which water diffusion in biological tissues is non-Gaussian using
the kurtosis tensor (KT) [Jensen2005]_.

Measurements of non-Gaussian diffusion from the diffusion kurtosis model are of
interest because they can be used to characterize tissue microstructural
heterogeneity [Jensen2010]_.

Moreover, DKI can be used to:

* Derive concrete biophysical parameters, such as the density of axonal fibers
  and diffusion tortuosity [Fierem2011]_.

* Resolve crossing fibers in tractography and to obtain invariant rotational
  measures not limited to well-aligned fiber populations [NetoHe2015]_.

We will use the ``cfin_multib`` dataset in DIPY to showcase this reconstruction
method. You can also use your own data!

We will create the directories in which to save the DKI metrics (e.g.:
``recons_dki_output``) and mask file (e.g.: ``cfin_multib_mask``)::

    mkdir recons_dki_output cfin_multib_mask

The Diffusion Kurtosis Imaging reconstruction method requires the paths to the
diffusion input file, b-values file, b-vectors file and mask file. The optional
parameters can also be provided. In this case, we will be specifying the threshold
used to find b0 volumes (``b0_threshold``) and the output
directory (``out_dir``).

To get the mask file, we will use the median Otsu thresholding method by calling
the ``dipy_median_otsu`` command::

    dipy_median_otsu data/cfin_multib/__DTI_AX_ep2d_2_5_iso_33d_20141015095334_4.nii --vol_idx 3-6 --out_dir "cfin_multib_mask"

To run the DKI reconstruction method on the data, execute the ``dipy_fit_dki``
command, e.g.::

    dipy_fit_dki data/cfin_multib/__DTI_AX_ep2d_2_5_iso_33d_20141015095334_4.nii data/cfin_multib/__DTI_AX_ep2d_2_5_iso_33d_20141015095334_4.bval data/cfin_multib/__DTI_AX_ep2d_2_5_iso_33d_20141015095334_4.bvec cfin_multib_mask/brain_mask.nii.gz --b0_threshold 70.0 --out_dir recons_dki_output

This command will save the DKI metrics to the specified output directory.

--------------------------
Constant Solid Angle (CSA)
--------------------------

We are using ``stanford_hardi`` dataset. As with any other workflow in DIPY,
you can also use your own data!

We will create a directory in which to save the peaks volume (e.g.:
``recons_csa_output``)::

    mkdir recons_csa_output

The workflow for the CSA reconstruction method requires the paths to the
diffusion input file, b-values file, b-vectors file and mask file. The optional
arguments can also be provided. In this case, we will be specifying whether or
not to save pam volumes as single nifti files (``extract_pam_values``) and the
output directory (``out_dir``).

Then, to perform the CSA reconstruction we will run the ``dipy_fit_csa`` command
as::

    dipy_fit_csa data/stanford_hardi/HARDI150.nii.gz data/stanford_hardi/HARDI150.bval data/stanford_hardi/HARDI150.bvec stanford_hardi_mask/brain_mask.nii.gz --extract_pam_values --out_dir "recons_csa_output"

This command will save the CSA metrics to the specified output directory.

-----------------------------------
Intravoxel Incoherent Motion (IVIM)
-----------------------------------

The intravoxel incoherent motion (IVIM) model describes diffusion and perfusion
in the signal acquired with a diffusion MRI sequence that contains multiple low
b-values. The IVIM model can be understood as an adaptation of the work of
Stejskal and Tanner [Stejskal65]_ in biological tissue, and was proposed by Le
Bihan [LeBihan84]_. The model assumes two compartments: a slow moving
compartment, where particles diffuse in a Brownian fashion as a consequence of
thermal energy, and a fast moving compartment (the vascular compartment), where
blood moves as a consequence of a pressure gradient. In the first compartment,
the diffusion coefficient is $\mathbf{D}$ while in the second compartment, a
pseudo diffusion term $\mathbf{D^*}$ is introduced that describes the
displacement of the blood elements in an assumed randomly laid out vascular
network, at the macroscopic level. According to [LeBihan84]_, $\mathbf{D^*}$ is
greater than $\mathbf{D}$.

We will be using the ``ivim`` dataset for the IVIM command line interface
demonstration purposes.

We will start by creating the directories in which to save the output volumes
(e.g.: ``recons_ivim_output``) and mask file (e.g.: ``ivim_mask``)::

    mkdir recons_ivim_output ivim_mask

In order to run the IVIM reconstruction method, we need to specify the locations
of the diffusion input file, b-values file, b-vectors file and mask file
followed by the optional arguments. In this case, we will be  specifying the
value to split the bvals to estimate D for the two-stage process of fitting
(``split_b_D``) and the output directory (``out_dir``).

To get the mask file, we will use the median Otsu thresholding method by calling
the ``dipy_median_otsu`` command::

    dipy_median_otsu data/ivim/HARDI150.nii.gz --vol_idx 10-50 --out_dir "ivim_mask"

Then, to perform the IVIM reconstruction we will run the command as::

    dipy_fit_ivim data/ivim/ivim.nii.gz data/ivim/ivim.nii.gz.bval data/ivim/ivim.nii.gz.bvec ivim_mask/brain_mask.nii.gz --split_b_D 250 --out_dir "recons_ivim_output"

This command will save the IVIM metrics to the directory ``recons_ivim_output``.

In case the output directory was not specified, the output volumes will be
saved to the current directory by default.


References
----------

.. [Basser1994] Basser PJ, Mattielo J, LeBihan (1994). MR diffusion tensor
                spectroscopy and imaging.

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

.. [Stejskal65] Stejskal, E. O.; Tanner, J. E. (1 January 1965).
                "Spin Diffusion Measurements: Spin Echoes in the Presence
                of a Time-Dependent Field Gradient". The Journal of Chemical
                Physics 42 (1): 288. Bibcode: 1965JChPh..42..288S.
                doi:10.1063/1.1695690.

.. [LeBihan84] Le Bihan, Denis, et al. "Separation of diffusion
               and perfusion in intravoxel incoherent motion MR
               imaging." Radiology 168.2 (1988): 497-505.