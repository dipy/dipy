

Dipy_ is an **international**, **free**, **open**, **clear** and **welcoming** software project for **diffusion** magnetic resonance imaging **analysis**, **research** and **development**.  

Depends on a few standard libraries: python_ (the core language), numpy_ (for numerical computation), 
scipy_ (for more specific mathematical operations), and cython_ (for extra speed). Optionally, it can use python-vtk_ (for visualisation), pytables_ (for handling large datasets), matplotlib_ (for scientific plotting), and ipython_ (for interaction with the code and its results).

Dipy is multiplatform and will run under any standard operating systems such as *Windows*, *Linux*, *Mac OS X*.  

Just some of our **killer** applications are:

- Reconstruction algorithms e.g. GQI, DTI  
- Tractography generation algorithms e.g. EuDX
- Intelligent downsampling of tracks
- Ultra fast tractography clustering
- Resampling datasets with anisotropic voxels to isotropic
- Visualizing multiple brains simultaneously
- Finding track correspondence between different brains
- Warping tractographies into another space e.g. MNI space
- Reading many different file formats e.g. Trackvis or Nifti
- Dealing with huge tractographies without memory restrictions
- Playing with datasets interactively without storing
- And much more and even more to come next ...

Here is a tiny code snippet to get you started

::

  >>> import numpy as np
  >>> from dipy.reconst.dti import Tensor
  >>> from dipy.data import get_data
  >>> fimg,fbval,fbvec=get_data('small_101D')
  >>> import nibabel as nib
  >>> img=nib.load(fimg)
  >>> data=img.get_data()
  >>> bvals=np.loadtxt(fbvals)
  >>> gradients=np.loadtxt(fbvecs).T
  >>> ten=dti.Tensor(data,bvals,gradients,thresh=50)  
  >>> FA=ten.fa()
  >>> MASK = FA < 0.2
  
In this very small example we loaded a small dataset with their data, bvecs and bvals, 
calculated the Tensors and fractional anisotropy (FA) and then created a mask to remove the 
regions with low anisotropy. 

**Join in the fun** and enjoy the `video <http://www.youtube.com/watch?v=tNB0sM7JJqg>`_  we made for the Summer Exhibition in London for the celebration of the 350 years of the Royal Society.

.. figure:: _static/simplified_tractography.png
   :align: center
      
   **This is a depiction of a tractography skeleton created using dipy**. 
   
   Using skeletal tracks we can very easily have a fast visual description of our
   datasets. If you want to learn more how you can create these with your datasets 
   read the examples in our :ref:`documentation` .


.. We need the following toctree directive to include the documentation
.. in the document hierarchy - see http://sphinx.pocoo.org/concepts.html
.. toctree::
   :hidden:

   documentation


.. include:: ./links_names.txt