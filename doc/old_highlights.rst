.. _old_highlights:

****************
Older Highlights
****************

**DIPY 0.13.0** is now available. New features include:

- Faster local PCA implementation.
- Fixed different issues with OpenMP and Windows / OSX.
- Replacement of cvxopt by cvxpy.
- Replacement of Pytables by h5py.
- Updated API to support latest numpy version (1.14).
- New user interfaces for visualization.
- Large documentation update.

**DIPY 0.12.0** is now available. New features include:

- IVIM Simultaneous modeling of perfusion and diffusion.
- MAPL, tissue microstructure estimation using Laplacian-regularized MAP-MRI.
- DKI-based microstructural modelling.
- Free water diffusion tensor imaging.
- Denoising using Local PCA.
- Streamline-based registration (SLR).
- Fiber to bundle coherence (FBC) measures.
- Bayesian MRF-based tissue classification.
- New API for integrated user interfaces.
- New hdf5 file (.pam5) for saving reconstruction results.
- Interactive slicing of images, ODFs and peaks.
- Updated API to support latest numpy versions.
- New system for automatically generating command line interfaces.
- Faster computation of cross correlation for image registration.

**DIPY 0.11.0** is now available. New features include:

- New framework for contextual enhancement of ODFs.
- Compatibility with numpy (1.11).
- Compatibility with VTK 7.0 which supports Python 3.x.
- Faster PIESNO for noise estimation.
- Reorient gradient directions according to motion correction parameters.
- Supporting Python 3.3+ but not 3.2.
- Reduced memory usage in DTI.
- DSI now can use datasets with multiple b0s.
- Fixed different issues with Windows 64bit and Python 3.5.

**DIPY 0.10.1** is now available. New features in this release include:

- Compatibility with new versions of scipy (0.16) and numpy (1.10).
- New cleaner visualization API, including compatibility with VTK 6, and functions to create your own interactive visualizations.
- Diffusion Kurtosis Imaging (DKI): Google Summer of Code work by Rafael Henriques.
- Mean Apparent Propagator (MAP) MRI for tissue microstructure estimation.
- Anisotropic Power Maps from spherical harmonic coefficients.
- A new framework for affine registration of images.

DIPY was an **official exhibitor** for OHBM 2015.

.. raw :: html

  <div style="width: 80% max-width=800px">
    <a href="http://www.humanbrainmapping.org/i4a/pages/index.cfm?pageID=3625" target="_blank"><img alt=" " class="align-center" src="_static/hbm2015_exhibitors.jpg" style="width: 90%;max-height: 90%"></a>
  </div>


**DIPY 0.9.2** is now available for :ref:`download <installation>`. Here is a summary of the new features.

* Anatomically Constrained Tissue Classifiers for Tracking
* Massive speedup of Constrained Spherical Deconvolution (CSD)
* Recursive calibration of response function for CSD
* New experimental framework for clustering
* Improvements and 10X speedup for Quickbundles
* Improvements in Linear Fascicle Evaluation (LiFE)
* New implementation of Geodesic Anisotropy
* New efficient transformation functions for registration
* Sparse Fascicle Model supports acquisitions with multiple b-values


**DIPY 0.8.0** is now available for :ref:`download <installation>`. The new
release contains state-of-the-art algorithms for diffusion MRI registration, reconstruction, denoising, statistical evaluation, fiber tracking and validation of tracking.

For more information about DIPY_, read the `DIPY paper`_  in Frontiers in Neuroinformatics.

.. raw :: html

  <div style="width: 80% max-width=800px">
    <a href="http://www.frontiersin.org/Neuroinformatics/10.3389/fninf.2014.00008/abstract" target="_blank"><img alt=" " class="align-center" src="_static/dipy_paper_logo.jpg" style="width: 90%;max-height: 90%"></a>
  </div>

So, how similar are your bundles to the real anatomy? Learn how to optimize your analysis as we did to create the fornix of the figure above, by reading the tutorials in our :ref:`gallery <examples>`.


In DIPY_ we care about methods which can solve complex problems efficiently and robustly. QuickBundles is one of the many state-of-the-art algorithms found in DIPY. It can be used to simplify large datasets of streamlines. See our :ref:`gallery <examples>` of examples and try QuickBundles with your data. Here is a video of QuickBundles applied on a simple dataset.

.. raw:: html

    <iframe width="420" height="315" src="http://www.youtube.com/embed/kstL7KKqu94" frameborder="0" allowfullscreen></iframe>


.. include:: links_names.inc