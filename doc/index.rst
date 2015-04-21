.. _home:

###########################
Diffusion Imaging In Python
###########################

Dipy_ is a **free** and **open source** software project focusing mainly on **diffusion** *magnetic resonance imaging* (dMRI) analysis.
Nonetheless, as we solve problems in dMRI some of the solutions are applicable to the greater medical imaging and image processing communities.
See for example our registration and denoising tutorials.


**********
Highlights
**********

**Dipy** is an **official exhibitor** for OHBM 2015. Come and meet us!

.. raw :: html

	<div style="width: 80% max-width=800px">
		<a href="http://www.frontiersin.org/Neuroinformatics/10.3389/fninf.2014.00008/abstract" target="_blank"><img alt=" " class="align-center" src="_static/hbm2015_exhibitors.jpg" style="width: 90%;max-height: 90%">
        </a>
	</div>


**Dipy 0.9.2** is now available for :ref:`download <installation>`. Here is a summary of the new features.

* Anatomically Constrained Tissue Classifiers for Tracking
* Massive speedup of Constrained Spherical Deconvolution (CSD)
* Recursive calibration of response function for CSD
* New experimental framework for clustering
* Improvements and 10X speedup for Quickbundles
* Improvements in Linear Fascicle Evaluation (LiFE)
* New implementation of Geodesic Anisotropy 
* New efficient transformation functions for registration
* Sparse Fascicle Model supports acquisitions with multiple b-values

See :ref:`older highlights <old_highlights>`.


*************
Announcements
*************

- **Dipy 0.9.2** released, March 18th, 2015.
- The creators of Dipy_ will attend both ISMRM and HBM 2015. Come and meet us!
- **Dipy 0.8.0** released, 6 January, 2015.
- Dipy_ will be an official exhibitor in `HBM 2015 <http://ohbm.loni.usc.edu>`_. Don't miss our booth!
- Dipy was featured in `The Scientist Magazine <http://www.the-scientist.com/?articles.view/articleNo/41266/title/White-s-the-Matter>`_, Nov, 2014.
- `Dipy paper`_ accepted in Frontiers of Neuroinformatics, January 22nd, 2014.
- **Dipy 0.7.1** Released!, January 16th, 2014.
- **Dipy 0.7.0** Released!, December 23rd, 2013.
- A team of Dipy developers **wins** the `IEEE ISBI HARDI challenge <http://hardi.epfl.ch/static/events/2013_ISBI/workshop.html#results>`_, 7 April, 2013.

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

As an exercise try to calculate the `color FA` with your datasets. Here is how
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




.. include:: links_names.inc
