.. _home:

####
Dipy
####

Dipy_ is an **free** and **open source** software project for
**diffusion** *magnetic resonance imaging* **analysis**. 


Dipy is multiplatform and will run under any standard operating systems such as *Windows*, *Linux* and  *Mac OS X*.


**Join in the fun** and enjoy the `video <http://www.youtube.com/watch?v=tNB0sM7JJqg>`_  we made for the Summer Exhibition in London for the celebration of the 350 years of the Royal Society.

An Example
~~~~~~~~~~

Here is a tiny dipy code snippet::

  >>> import numpy as np
  >>> import nibabel as nib
  
  >>> from dipy.reconst.dti import TensorModel
  >>> from dipy.data import get_data
  >>> from dipy.io import read_bvals_bvecs
  >>> from dipy.core.gradients import gradient_table

  >>> fimg, fbval, fbvec = get_data('small_101D')
  >>> img = nib.load(fimg)
  >>> data = img.get_data()

  >>> bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
  >>> gtab = gradient_table(bvals, bvecs)
  
  >>> ten = TensorModel(gtab)
  >>> tenfit = ten.fit(data)
  
  >>> FA = tenfit.fa


:ref:`Download <installation>` dipy and try it for yourself.

If you want to learn more how you can create these with your datasets read the examples in our :ref:`documentation` .

.. We need the following toctree directive to include the documentation
.. in the document hierarchy - see http://sphinx.pocoo.org/concepts.html
.. toctree::
   :hidden:

   documentation
   stateoftheart

.. include:: links_names.inc
