.. _home:

###########################
Diffusion Imaging In Python
###########################

DIPY_ is a **free** and **open source** software project for computational neuroanatomy,
focusing mainly on **diffusion** *magnetic resonance imaging* (dMRI) analysis. It implements a
broad range of algorithms for denoising, registration, reconstruction, tracking, clustering,
visualization, and statistical analysis of MRI data.

**********
Highlights
**********

**DIPY 0.14.0** is now available. New features include:

- RecoBundles: anatomically relevant segmentation of bundles
- New super fast clustering algorithm: QuickBundlesX
- New tracking algorithm: Particle Filtering Tracking.
- New tracking algorithm: Probabilistic Residual Bootstrap Tracking.
- Integration of the Streamlines API for reading, saving and processing tractograms.
- Fiber ORientation Estimated using Continuous Axially Symmetric Tensors (Forecast).
- New command line interfaces.
- Deprecated fvtk (old visualization framework).
- A range of new visualization improvements.
- Large documentation update.

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

See :ref:`Older Highlights <old_highlights>`.


*************
Announcements
*************

- :ref:`DIPY 0.14 <release0.14>` released May 1, 2018.
- :ref:`DIPY 0.13 <release0.13>` released October 24, 2017.
- :ref:`DIPY 0.12 <release0.12>` released June 26, 2017.
- :ref:`DIPY 0.11 <release0.11>` released February 21, 2016.
- :ref:`DIPY 0.10 <release0.10>` released December 4, 2015.
- :ref:`DIPY 0.9.2 <release0.9>` released, March 18, 2015.
- :ref:`DIPY 0.8.0 <release0.8>` released, January 6, 2015.
- DIPY_ was an official exhibitor in `OHBM 2015 <http://ohbm.loni.usc.edu>`_.
- DIPY was featured in `The Scientist Magazine <http://www.the-scientist.com/?articles.view/articleNo/41266/title/White-s-the-Matter>`_, Nov, 2014.
- `DIPY paper`_ accepted in Frontiers of Neuroinformatics, January 22nd, 2014.

See some of our :ref:`Past Announcements <old_news>`


***************
Getting Started
***************

Here is a quick snippet showing how to calculate `color FA` also known as the
DEC map. We use a Tensor model to reconstruct the datasets which are
saved in a Nifti file along with the b-values and b-vectors which are saved as
text files. Finally, we save our result as a Nifti file ::

    fdwi = 'dwi.nii.gz'
    fbval = 'dwi.bval'
    fbvec = 'dwi.bvec'

    from dipy.io.image import load_nifti, save_nifti
    from dipy.io import read_bvals_bvecs
    from dipy.core.gradients import gradient_table
    from dipy.reconst.dti import TensorModel

    data, affine = load_nifti(fdwi)
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    gtab = gradient_table(bvals, bvecs)

    tenmodel = TensorModel(gtab)
    tenfit = tenmodel.fit(data)

    save_nifti('colorfa.nii.gz', tenfit.color_fa, affine)

As an exercise, you can try to calculate `color FA` with your datasets. You will need
to replace the filepaths `fimg`, `fbval` and `fbvec`. Here is what
a slice should look like.

.. image:: _static/colorfa.png
    :align: center

**********
Next Steps
**********

You can learn more about how you to use DIPY_ with  your datasets by reading the examples in our :ref:`documentation`.

.. We need the following toctree directive to include the documentation
.. in the document hierarchy - see http://sphinx.pocoo.org/concepts.html
.. toctree::
   :hidden:

   documentation
   stateoftheart

*******
Support
*******

We acknowledge support from the following organizations:

- The department of Intelligent Systems Engineering of Indiana University.

- The Gordon and Betty Moore Foundation and the Alfred P. Sloan Foundation, through the
  University of Washington eScience Institute Data Science Environment.

- Google supported DIPY through the Google Summer of Code Program during
  Summer 2015 and 2016.




.. include:: links_names.inc
