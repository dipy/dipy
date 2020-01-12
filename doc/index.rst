.. _home:

###########################
Diffusion Imaging In Python
###########################

DIPY_ is the paragon 3D/4D+ imaging library in Python. Contains generic methods for
spatial normalization, signal processing, machine learning, statistical analysis
and visualization of medical images. Additionally, it contains 
specialized methods for computational anatomy including diffusion, 
perfusion and structural imaging. 



**********
Highlights
**********

**DIPY 1.1.1** is now available. New features include:

- New module for deep learning ``DIPY.NN`` (uses TensorFlow 2.0).
- Improved DKI performance and increased utilities.
- Non-linear and RESTORE fits from DTI compatible now with DKI.
- Numerical solutions for estimating axial, radial and mean kurtosis.
- Added Kurtosis Fractional Anisotropy by Glenn et al. 2015.
- Added Mean Kurtosis Tensor by Hansen et al. 2013.
- Nibabel minimum version is 3.0.0.
- Azure CI added and Appveyor CI removed.
- New command line interfaces for LPCA, MPPCA and Gibbs Unringing.
- New MTMS CSD tutorial added.
- Horizon refactored and updated to support StatefulTractograms.
- Speeded up all cython modules by using a smarter configuration setting.
- All tutorials updated to API changes and 2 new tutorials added.
- Large documentation update.
- Closed 126 issues and merged 50 pull requests.


See :ref:`Older Highlights <old_highlights>`.


*************
Announcements
*************
- DIPY Workshop - Titanium Edition (March 16-20, 2020) is now open for registration:
- :doc:`DIPY 1.1.1 <release_notes/release1.1>` released January 10, 2020.
- :doc:`DIPY 1.0 <release_notes/release1.0>` released August 5, 2019.
- :doc:`DIPY 0.16 <release_notes/release0.16>` released March 10, 2019.


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
to replace the filepaths `fdwi`, `fbval` and `fbvec`. Here is what
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

- The National Institute of Biomedical Imaging and Bioengineering, NIH.

- The Gordon and Betty Moore Foundation and the Alfred P. Sloan Foundation, through the
  University of Washington eScience Institute Data Science Environment.

- Google supported DIPY through the Google Summer of Code Program during
  Summer 2015, 2016 and 2018.

- The International Neuroinformatics Coordination Facility.




.. include:: links_names.inc
