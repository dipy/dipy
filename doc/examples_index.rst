.. _examples:

========
Examples
========

.. toctree::
   :maxdepth: 1

   note_about_examples


-----------
Quick Start
-----------



-------------
Preprocessing
-------------

Gradients
~~~~~~~~~

Spheres
~~~~~~~

Brain Extraction
~~~~~~~~~~~~~~~~

- :ref:`example_brain_extraction_dwi`

SNR estimation
~~~~~~~~~~~~~~

- :ref:`example_snr_in_cc`

Reslice
~~~~~~~

- :ref:`example_reslice_datasets`


--------------
Reconstruction
--------------

Constrained Spherical Deconvolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :ref:`example_reconst_csd`


Diffusion Tensor Imaging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :ref:`example_reconst_dti`


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

--------------
Fiber tracking
--------------

- :ref:`example_tracking_eudx_tensor`
- :ref:`example_tracking_eudx_odf`
- :ref:`example_probabilistic_tracking_odfs`

-------------------
Streamline analysis
-------------------

- lengths
- curvature
- streamline distances
- resampling
- roi intersections
- connectivity

------------
Segmentation
------------

Streamline Clustering
~~~~~~~~~~~~~~~~~~~~~

- :ref:`example_segment_quickbundles`

Brain Segmentation
~~~~~~~~~~~~~~~~~~
- :ref:`example_brain_extraction_dwi`


Simulations
~~~~~~~~~~~

---------------
Multiprocessing
---------------

- :ref:`example_reconst_csd_parallel`
- :ref:`example_reconst_csa_parallel`
   
------------
File Formats
------------

- :ref:`example_streamline_formats`
   

.. In order to build the examples, you'll need (on Debian)
    sudo apt-get install python-tables python-matplotib python-vtk
