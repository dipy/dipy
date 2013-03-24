.. _stateoftheart:

============================
A quick overview of features
============================

Here are just a few of the state-of-the-art technologies and algorithms which are provided in Dipy_:

- Reconstruction algorithms e.g. DSI, GQI, DTI, QBI.
- Fiber tracking algorithms e.g. Deterministic, Probabilistic.
- Simple interactive visualization of ODFs and streamlines.
- Apply different operations on streamlines.
- Simplify large datasets of streamlines using QuickBundles clustering.
- Reslice datasets with anisotropic voxels to isotropic.
- Calculate distances/correspondences between streamlines.
- Deal with huge tractographies without memory restrictions (.dpy).

With the help of some external tools you can also:

- Warp/Register tractographies into another space e.g. MNI space (with nipy or
  fsl).
- Read many different file formats e.g. Trackvis or Nifti (with nibabel).
- Examine your datasets interactively (with ipython).

For a full list of the features implemented in the most recent release cycle, check out the release notes

.. toctree::
   :maxdepth: 1

   release0.6

=================
Systems supported
=================

Dipy_ is multiplatform and will run under any standard operating systems such
as *Windows*, *Linux* and  *Mac OS X*. Every single new code addition is being tested on
a number of different builbots and can be monitored online `here <http://nipy.bic.berkeley.edu/builders>`_.


.. include:: links_names.inc
