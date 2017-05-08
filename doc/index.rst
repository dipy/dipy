.. _home:

###########################
Diffusion Imaging In Python
###########################

Dipy_ is a **free** and **open source** software project for computational neuroanatomy,
focusing mainly on **diffusion** *magnetic resonance imaging* (dMRI) analysis. It implements a
broad range of algorithms for denoising, registration, reconstruction, tracking, clustering,
visualization, and statistical analysis of MRI data.

**********
Highlights
**********

**Dipy 0.11.0** is now available. New features include:

- New framework for contextual enhancement of ODFs.
- Compatibility with numpy (1.11).
- Compatibility with VTK 7.0 which supports Python 3.x.
- Faster PIESNO for noise estimation.
- Reorient gradient directions according to motion correction parameters.
- Supporting Python 3.3+ but not 3.2.
- Reduced memory usage in DTI.
- DSI now can use datasets with multiple b0s.
- Fixed different issues with Windows 64bit and Python 3.5.

**Dipy 0.10.1** is now available. New features in this release include:

- Compatibility with new versions of scipy (0.16) and numpy (1.10).
- New cleaner visualization API, including compatibility with VTK 6, and functions to create your own interactive visualizations.
- Diffusion Kurtosis Imaging (DKI): Google Summer of Code work by Rafael Henriques.
- Mean Apparent Propagator (MAP) MRI for tissue microstructure estimation.
- Anisotropic Power Maps from spherical harmonic coefficients.
- A new framework for affine registration of images.

See :ref:`older highlights <old_highlights>`.


*************
Announcements
*************

- :ref:`Dipy 0.11 <release0.11>` released February 21, 2016.
- :ref:`Dipy 0.10 <release0.10>` released December 4, 2015.
- :ref:`Dipy 0.9.2 <release0.9>` released, March 18, 2015.
- :ref:`Dipy 0.8.0 <release0.8>` released, January 6, 2015.
- Dipy_ was an official exhibitor in `HBM 2015 <http://ohbm.loni.usc.edu>`_.
- Dipy was featured in `The Scientist Magazine <http://www.the-scientist.com/?articles.view/articleNo/41266/title/White-s-the-Matter>`_, Nov, 2014.
- `Dipy paper`_ accepted in Frontiers of Neuroinformatics, January 22nd, 2014.

See some of our :ref:`past announcements <old_news>`


***************
Getting Started
***************

Here is a simple example showing how to fit a DTI model to diffusion MRI data::

    # The dipy.data module includes example data-sets:
    import dipy.data as dpd
    fimg, fbval, fbvec = dpd.get_data('small_101D')

    # Print fimg, fbval and fbvec to the console:
    print(fimg, fbval, fbvec)

    # Read the data in, using nibabel
    import nibabel as nib
    img = nib.load(fimg)
    data = img.get_data()

    # Create a gradient table object from the b-values and b-vectors:
    import dipy.core.gradients as dpg
    gtab = dpg.gradient_table(fbval, fbvec)

    # We will use DTI as the model for the data:
    import dipy.reconst.dti as dti

    # We initialize a model object:
    ten = dti.TensorModel(gtab)

    # Fitting the model to the data:
    fit = ten.fit(data)

    # We calculate the FA from the model fit:
    fa = dti.fractional_anisotropy(fit.evals)

    # An RGB map of the principal diffusion direction can be computed:
    cfa = dti.color_fa(fa, fit.evecs)

    # And displayed using Matplotlib:
    import matplotlib.pyplot as plt
    plt.imshow(cfa[cfa.shape[0]//2])
    plt.show()

As an exercise try to calculate the `color FA` with your datasets, by replacing
fimg, fbval, and fbvec with the names of files with your data. Here is what an
image of a middle slice through the brain should look like.

.. image:: _static/colorfa.png
    :align: center

**********
Next Steps
**********

You can learn more about how you to use Dipy_ with  your datasets by reading the examples in our :ref:`documentation`.

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

- The Gordon and Betty Moore Foundation and the Alfred P. Sloan Foundation, through the
  University of Washington eScience Institute Data Science Environment.

- Google supported the work of Rafael Neto Henriques and Julio Villalon through the Google
  Summer of Code Program, Summer 2015.

.. include:: links_names.inc
