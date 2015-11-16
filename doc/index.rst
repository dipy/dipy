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

**Dipy 0.10** is now available. New features in this release include: 


- FVTK 2.0: a new visualization framework. For example, see: 
- DKI: Thanks to support through the Google Summer of Code, to fund work done by Rafael Henriques, we now have the only open-source implementation of DKI.
- Compatibility with new versions of scipy (0.16) and numpy (1.10)


See :ref:`older highlights <old_highlights>`.


*************
Announcements
*************

- **Dipy 0.10** released November ??, 2015.
- **Dipy 0.9.2** released, March 18th, 2015.
- The creators of Dipy_ will attend both ISMRM and HBM 2015. Come and meet us!
- **Dipy 0.8.0** released, 6 January, 2015.
- Dipy_ will be an official exhibitor in `HBM 2015 <http://ohbm.loni.usc.edu>`_. Don't miss our booth!
- Dipy was featured in `The Scientist Magazine <http://www.the-scientist.com/?articles.view/articleNo/41266/title/White-s-the-Matter>`_, Nov, 2014.
- `Dipy paper`_ accepted in Frontiers of Neuroinformatics, January 22nd, 2014.

See some of our :ref:`past announcements <old_news>`


***************
Getting Started
***************

Here is a simple example showing how to calculate `color FA`. We
use a single Tensor model to reconstruct the datasets which are saved in a
Nifti file along with the b-values and b-vectors which are saved as text files.
In this example we use only a few voxels with 101 gradient directions::

    from dipy.data import get_data
    fimg, fbval, fbvec = get_data('small_101D')

    import nibabel as nib
    img = nib.load(fimg)
    data = img.get_data()

    from dipy.io import read_bvals_bvecs
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)

    from dipy.core.gradients import gradient_table
    gtab = gradient_table(bvals, bvecs)

    from dipy.reconst.dti import TensorModel
    ten = TensorModel(gtab)
    tenfit = ten.fit(data)

    from dipy.reconst.dti import fractional_anisotropy
    fa = fractional_anisotropy(tenfit.evals)

    from dipy.reconst.dti import color_fa
    cfa = color_fa(fa, tenfit.evecs)

As an exercise try to calculate the `color FA` with your datasets. Here is what
a slice should look like.

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
