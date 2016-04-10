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
- :ref:`example_tracking_quick_start`

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

Robust noise estimation with PIESNO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :ref:`example_piesno`

Denoising
~~~~~~~~~

- :ref:`example_denoise_nlmeans`

Reslice
~~~~~~~

- :ref:`example_reslice_datasets`

--------------
Reconstruction
--------------

Constrained Spherical Deconvolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :ref:`example_reconst_csd`

Simple Harmonic Oscillator based Reconstruction and Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :ref:`example_reconst_shore`
- :ref:`example_reconst_shore_metrics`


Diffusion Tensor Imaging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :ref:`example_reconst_dti`
- :ref:`example_restore_dti`


Diffusion Kurtosis Imaging (NEW)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- :ref:`example_reconst_dki`


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
~~~~~~~~~~~~~~~~~~~~~~

- :ref:`example_sfm_reconst`

Statistical evaluation
~~~~~~~~~~~~~~~~~~~~~~

- :ref:`example_kfold_xval`

------------------------------------------------
Contextual enhancement (NEW)
------------------------------------------------

- :ref:`example_contextual_enhancement`


--------------------
Fiber tracking (NEW)
--------------------

- :ref:`example_introduction_to_basic_tracking`
- :ref:`example_probabilistic_fiber_tracking`
- :ref:`example_deterministic_fiber_tracking`
- :ref:`example_tracking_tissue_classifier`
- :ref:`example_sfm_tracking`
- :ref:`example_tracking_eudx_tensor`
- :ref:`example_tracking_eudx_odf`

-------------------------------------
Fiber tracking validation
-------------------------------------

- :ref:`example_linear_fascicle_evaluation`


------------------------------------
Streamline analysis and connectivity
------------------------------------

- :ref:`example_streamline_tools`
- :ref:`example_streamline_length`


------------------
Registration
------------------

Image-based Registration (NEW)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- :ref:`example_affine_registration_3d`
- :ref:`example_syn_registration_2d`
- :ref:`example_syn_registration_3d`

Streamline-based Registration (NEW)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- :ref:`example_bundle_registration`

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

-----------
Simulations
-----------

- :ref:`example_simulate_multi_tensor`
- :ref:`example_reconst_dsid`

---------------
Multiprocessing
---------------

- :ref:`example_reconst_csd_parallel`
- :ref:`example_reconst_csa_parallel`

------------
File Formats
------------

- :ref:`example_streamline_formats`

-------------------
Visualization (NEW)
-------------------

- :ref:`example_viz_advanced`
- :ref:`example_viz_slice`
- :ref:`example_viz_bundles`
- :ref:`example_viz_widgets`

.. In order to build the examples, you'll need (on Debian)
    sudo apt-get install python-tables python-matplotib python-vtk
