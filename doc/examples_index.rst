.. _examples:

========
Examples
========

.. toctree::
   :maxdepth: 1

   note_about_examples

.. contents::
   :depth: 2


-----------
Quick Start
-----------

- :ref:`example_quick_start`
- :ref:`example_tracking_introduction_eudx`

-------------
Preprocessing
-------------

Gradients & Spheres
~~~~~~~~~~~~~~~~~~~

- :ref:`example_gradients_spheres`

Brain Extraction
~~~~~~~~~~~~~~~~

- :ref:`example_brain_extraction_dwi`

Basic SNR estimation
~~~~~~~~~~~~~~~~~~~~

- :ref:`example_snr_in_cc`

Reslice
~~~~~~~

- :ref:`example_reslice_datasets`

---------
Denoising
---------

- :ref:`example_denoise_patch2self`
- :ref:`example_denoise_nlmeans`
- :ref:`example_denoise_localpca`
- :ref:`example_denoise_mppca`
- :ref:`example_denoise_gibbs`

--------------
Reconstruction
--------------

Below, an overview of all reconstruction models available on DIPY.

Note: Some reconstruction models do not have a tutorial yet

.. include:: reconstruction_models_list.rst

Constrained Spherical Deconvolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :ref:`example_reconst_csd`
- :ref:`example_reconst_mcsd`

Fiber Orientation Estimated using Continuous Axially Symmetric Tensors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :ref:`example_reconst_forecast`

Simple Harmonic Oscillator based Reconstruction and Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :ref:`example_reconst_shore`
- :ref:`example_reconst_shore_metrics`

Mean Apparent Propagator (MAP)-MRI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :ref:`example_reconst_mapmri`

Studying diffusion time-dependence using qt-dMRI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :ref:`example_reconst_qtdmri`

Diffusion Tensor Imaging
~~~~~~~~~~~~~~~~~~~~~~~~

- :ref:`example_reconst_dti`
- :ref:`example_restore_dti`

Free-water Diffusion Tensor Imaging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :ref:`example_reconst_fwdti`

Diffusion Kurtosis Imaging
~~~~~~~~~~~~~~~~~~~~~~~~~~

- :ref:`example_reconst_dki`
- :ref:`example_reconst_msdki`

White Matter Tract Integrity Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- :ref:`example_reconst_dki_micro`

Q-Ball Constant Solid Angle
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :ref:`example_reconst_csa`


Diffusion Spectrum Imaging
~~~~~~~~~~~~~~~~~~~~~~~~~~

- :ref:`example_reconst_dsi`
- :ref:`example_reconst_dsi_metrics`


Generalized Q-Sampling Imaging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :ref:`example_reconst_gqi`

DSI with Deconvolution
~~~~~~~~~~~~~~~~~~~~~~

- :ref:`example_reconst_dsid`

Sparse Fascicle Model
~~~~~~~~~~~~~~~~~~~~~

- :ref:`example_reconst_sfm`

Intravoxel incoherent motion (IVIM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :ref:`example_reconst_ivim`


Statistical evaluation
~~~~~~~~~~~~~~~~~~~~~~

- :ref:`example_kfold_xval`

Intra-Voxel Signal Reconstruction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :ref:`example_reconst_sh`


Q-space Trajectory Imaging
~~~~~~~~~~~~~~~~~~~~~~~~~~

- :ref:`example_reconst_qti`


Robust and Unbiased Model-BAsed Spherical Deconvolution (RUMBA-SD)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :ref:`example_reconst_rumba`


----------------------
Contextual enhancement
----------------------

- :ref:`example_contextual_enhancement`
- :ref:`example_fiber_to_bundle_coherence`

--------------
Fiber tracking
--------------

- :ref:`example_tracking_introduction_eudx`
- :ref:`example_tracking_deterministic`
- :ref:`example_tracking_probabilistic`
- :ref:`example_tracking_bootstrap_peaks`
- :ref:`example_tracking_stopping_criterion`
- :ref:`example_tracking_pft`
- :ref:`example_tracking_sfm`
- :ref:`example_tracking_rumba`
- :ref:`example_linear_fascicle_evaluation`
- :ref:`example_surface_seed`

------------------------------------
Streamline analysis and connectivity
------------------------------------

- :ref:`example_streamline_tools`
- :ref:`example_streamline_length`
- :ref:`example_cluster_confidence`
- :ref:`example_path_length_map`
- :ref:`example_afq_tract_profiles`
- :ref:`example_bundle_assignment_maps`
- :ref:`example_bundle_shape_similarity`

------------
Registration
------------

Image-based Registration
~~~~~~~~~~~~~~~~~~~~~~~~

- :ref:`example_affine_registration_3d`
- :ref:`example_affine_registration_masks`
- :ref:`example_syn_registration_2d`
- :ref:`example_syn_registration_3d`
- :ref:`example_register_binary_fuzzy`

Streamline-based Registration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :ref:`example_bundle_registration`
- :ref:`example_streamline_registration`

------------
Segmentation
------------

Streamline Clustering
~~~~~~~~~~~~~~~~~~~~~

- :ref:`example_segment_quickbundles`
- :ref:`example_segment_extending_clustering_framework`
- :ref:`example_segment_clustering_features`
- :ref:`example_segment_clustering_metrics`

Brain Segmentation
~~~~~~~~~~~~~~~~~~

- :ref:`example_brain_extraction_dwi`

Tissue Classification
~~~~~~~~~~~~~~~~~~~~~

- :ref:`example_tissue_classification`

Bundle Extraction
~~~~~~~~~~~~~~~~~

- :ref:`example_bundle_extraction`

-----------
Simulations
-----------

- :ref:`example_simulate_multi_tensor`
- :ref:`example_reconst_dsid`
- :ref:`example_simulate_dki`

---------------
Multiprocessing
---------------

- :ref:`example_reconst_csd_parallel`
- :ref:`example_reconst_csa_parallel`

------------
File Formats
------------

- :ref:`example_streamline_formats`

-------------
Visualization
-------------

- :ref:`example_viz_advanced`
- :ref:`example_viz_slice`
- :ref:`example_viz_bundles`
- :ref:`example_viz_roi_contour`

---------
Workflows
---------

- :ref:`example_workflow_creation`
- :ref:`example_combined_workflow_creation`
