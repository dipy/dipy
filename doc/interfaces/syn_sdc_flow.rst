.. _syn_sdc_flow:

==================================================================
Fieldmap-less Susceptibility Distortion Correction with SyN
==================================================================

This tutorial walks through the steps to correct susceptibility-induced
distortions of diffusion MRI data with ``dipy_correct_syn_sdc`` when neither a
fieldmap nor a reverse phase-encoded b0 was acquired.

You can try this using your own data; we will be using the data in DIPY.
You can check how to :ref:`fetch the DIPY data<data_fetch>`.

--------------------------------------------
Susceptibility Distortion Correction (SDC)
--------------------------------------------

Echo-planar images are displaced along the phase-encoding (PE) direction by
the off-resonance field caused by the difference in magnetic susceptibility
between tissue and air, mostly near the sinuses and the ear canals. The
displacement can be estimated by registering the distorted b0 to an
undistorted T1-weighted (T1w) or T2-weighted (T2w) image of the same subject,
with the deformation restricted to the PE axis :footcite:p:`Huang2008`,
:footcite:p:`Wang2017`.

A T1w image is intensity-inverted to resemble a b0, while the intensities of
a T2w image are mapped to the b0 through the joint histogram of the two images
inside the brain. The registration uses symmetric diffeomorphic
registration (SyN) :footcite:p:`Avants2008` with the local cross-correlation
metric, and every update is projected onto the PE axis.

We will use the ``stanford_hardi`` dataset, its T1w image ``stanford_t1`` and
the tissue probability maps ``stanford_pve_maps``, from which a brain mask is
computed. The anatomical image should be skull-stripped, or a brain mask given
with ``--anat_mask_file``.

You may want to create an output directory to save the results::

    mkdir sdc_output

The command takes the DWI and the anatomical image, and the voxel axis of the
DWI along which phase encoding was performed (here, the anterior-posterior
axis, voxel axis 1)::

    dipy_correct_syn_sdc data/stanford_hardi/HARDI150.nii.gz data/stanford_hardi/t1.nii.gz --anat_mask_file brain_mask.nii.gz --pe_axis 1 --anat_contrast t1 --out_dir "sdc_output"

The distortion is estimated from the volume given by ``--b0_index`` (0 by
default) and applied to every volume. The command writes the corrected DWI
(``corrected.nii.gz``) and the displacement along the PE axis, in voxels
(``field.nii.gz``), to the output directory.

With a T2w image, use ``--anat_contrast t2``. The polarity of phase encoding
does not need to be specified.

----------
References
----------

.. footbibliography::
