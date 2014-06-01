.. _home:

###########################
Diffusion Imaging In Python
###########################

Dipy_ is a **free** and **open source** software project for
**diffusion** *magnetic resonance imaging* (dMRI) **analysis**.


**********
Highlights
**********

**Dipy 0.7.1** is now available for :ref:`download <installation>` with **3X** more tutorials than 0.6.0! In addition, a `journal paper`_ focusing on teaching the fundamentals of Dipy is now available in Frontiers of Neuroinformatics.

.. raw :: html

	<div style="width: 80% max-width=800px">
		<a href="http://www.frontiersin.org/Neuroinformatics/10.3389/fninf.2014.00008/abstract" target="_blank"><img alt=" " class="align-center" src="_static/dipy_paper_logo.jpg" style="width: 90%;max-height: 90%">
        </a>
	</div>

So, how similar are your bundles to the real anatomy? Learn how to optimize your analysis as we did to create the fornix of the figure above, by reading the tutorials in our :ref:`gallery <examples>`.

See :ref:`older highlights <old_highlights>`.


*************
Announcements
*************

- The creators of Dipy will attend both ISMRM and HBM 2014. Come and meet us!
- A new **hands on Dipy** seminar to 50 neuroscientists from Canada, as part of QBIN's "Do's and dont's of diffusion MRI" workshop, 8 April, 2014.
- `Dipy paper`_ accepted in Frontiers of Neuroinformatics, 22 January, 2014.
- **Dipy 0.7.1** Released!, 16 January, 2014.
- **Dipy 0.7.0** Released!, 23 December, 2013.
- **Spherical Deconvolution** algorithms are now included in the current development version 0.7.0dev. See the examples in :ref:`gallery <examples>`, 24 June 2013.
- A team of Dipy developers **wins** the `IEEE ISBI HARDI challenge <http://hardi.epfl.ch/static/events/2013_ISBI/workshop.html#results>`_, 7 April, 2013.
- **Hands on Dipy** seminar took place at the dMRI course of the CREATE-MIA summer school, 5-7 June, McGill, Montreal, 2013.
- **Dipy 0.6.0** Released!, 30 March, 2013.
- **Dipy 3rd Sprint**, Berkeley, CA, 8-18 April, 2013.
- **IEEE ISBI HARDI challenge** 2013 chooses **Dipy**, February, 2013.


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
