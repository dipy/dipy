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
robustly. QuickBundles is one of the many state-of-the art applications found
in Dipy_ which can be used to simplify large datasets of streamlines. See our
examples and try QuickBundles with your data :ref:`examples_index`. Here is a
video of QuickBundles applied on a simple dataset.

.. raw:: html

    <iframe width="420" height="315" src="http://www.youtube.com/embed/kstL7KKqu94" frameborder="0" allowfullscreen></iframe>


***************
Getting Started
***************

Here is a simple example showing how to calculate fractional anisotropy (FA). We
use a single Tensor model to reconstruct the datasets which are saved in the
Nifti file along with the b-values and b-vectors which are saved as text files.
For quick execution we use only a few voxels with 101 gradients::

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

We recommend to copy a past this example in an IPython_ console. IPython_ helps interacting
with the datasets easily. For example it is easy to find the size of the
dataset which is given by `data.shape`.

:ref:`Download <installation>` Dipy_ and try it for yourself.

If you want to learn more about how you can create these with your datasets read the examples in our :ref:`documentation`.

.. We need the following toctree directive to include the documentation
.. in the document hierarchy - see http://sphinx.pocoo.org/concepts.html
.. toctree::
   :hidden:

   documentation
   stateoftheart

.. include:: links_names.inc
