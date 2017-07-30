.. _old_highlights:

****************
Older Highlights
****************

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

For more information about dipy_, read the `DIPY paper`_  in Frontiers in Neuroinformatics.

.. raw :: html

  <div style="width: 80% max-width=800px">
    <a href="http://www.frontiersin.org/Neuroinformatics/10.3389/fninf.2014.00008/abstract" target="_blank"><img alt=" " class="align-center" src="_static/dipy_paper_logo.jpg" style="width: 90%;max-height: 90%"></a>
  </div>

So, how similar are your bundles to the real anatomy? Learn how to optimize your analysis as we did to create the fornix of the figure above, by reading the tutorials in our :ref:`gallery <examples>`.


In dipy_ we care about methods which can solve complex problems efficiently and robustly. QuickBundles is one of the many state-of-the art algorithms found in DIPY. It can be used to simplify large datasets of streamlines. See our :ref:`gallery <examples>` of examples and try QuickBundles with your data. Here is a video of QuickBundles applied on a simple dataset.

.. raw:: html

    <iframe width="420" height="315" src="http://www.youtube.com/embed/kstL7KKqu94" frameborder="0" allowfullscreen></iframe>


.. include:: links_names.inc