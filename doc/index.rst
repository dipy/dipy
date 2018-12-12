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

**DIPY 0.15.0** is now available. New features include:

- Updated RecoBundles for automatic anatomical bundle segmentation.
- New Reconstruction Model: qtau-dMRI.
- New command line interfaces (e.g. dipy_slr).
- New continuous integration with AppVeyor CI.
- Nibabel Streamlines API now used almost everywhere for better memory management.
- Compatibility with Python 3.7.
- Many tutorials added or updated (5 New).
- Large documentation update.
- Moved visualization module to a new library: FURY.
- Closed 287 issues and merged 93 pull requests. 

See :ref:`Older Highlights <old_highlights>`.


*************
Announcements
*************

- :ref:`DIPY 0.15 <release0.15>` released December 12, 2018.
- :ref:`DIPY 0.14 <release0.14>` released May 1, 2018.
- :ref:`DIPY 0.13 <release0.13>` released October 24, 2017.
- :ref:`DIPY 0.12 <release0.12>` released June 26, 2017.

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
