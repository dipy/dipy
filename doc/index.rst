.. _home:

###########################
Diffusion Imaging In Python
###########################

Dipy_ is a **free** and **open source** software project for
**diffusion** *magnetic resonance imaging* (dMRI) **analysis**. 


**********
Highlights
**********

In Dipy_ we care about methods which can solve complex problems efficiently and
robustly. Here is QuickBundles one of the state-of-the art applications found
in Dipy_ which can be used to simplify large datasets of streamlines. See our
examples and try QuickBundles with your data :ref:`examples_index`.

.. raw:: html

    <iframe width="420" height="315" src="http://www.youtube.com/embed/kstL7KKqu94" frameborder="0" allowfullscreen></iframe>



***************
Getting Started
***************

Here is a simple example for the calculation of fractional anisotropy (FA)::

  import numpy as np
  import nibabel as nib
  from dipy.reconst.dti import TensorModel
  from dipy.data import get_data
  from dipy.io import read_bvals_bvecs
  from dipy.core.gradients import gradient_table
  fimg, fbval, fbvec = get_data('small_101D')
  img = nib.load(fimg)
  data = img.get_data()
  bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
  gtab = gradient_table(bvals, bvecs)
  ten = TensorModel(gtab)
  tenfit = ten.fit(data)
  FA = tenfit.fa



:ref:`Download <installation>` Dipy_ and try it for yourself.

If you want to learn more about how you can create these with your datasets read the examples in our :ref:`documentation`.

.. We need the following toctree directive to include the documentation
.. in the document hierarchy - see http://sphinx.pocoo.org/concepts.html
.. toctree::
   :hidden:

   documentation
   stateoftheart

.. include:: links_names.inc
