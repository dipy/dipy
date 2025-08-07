.. _stateoftheart:

============================
A quick overview of features
============================

Here are just a few of the state-of-the-art :ref:`technologies <examples>` and algorithms which are provided in DIPY_:

- Reconstruction algorithms: CSD, DSI, GQI, DTI, DKI, QBI, SHORE and MAPMRI.
- Fiber tracking algorithms: deterministic and probabilistic.
- Simple interactive visualization of ODFs and streamlines.
- Apply different operations on streamlines (selection, resampling, registration).
- Simplify large datasets of streamlines using QuickBundles clustering.
- Reslice datasets with anisotropic voxels to isotropic.
- Calculate distances/correspondences between streamlines.
- Deal with huge streamline datasets without memory restrictions (using the .dpy file format).
- Visualize streamlines in the same space as anatomical images.

With the help of some external tools you can also:

- Read many different file formats e.g. Trackvis or Nifti (with nibabel).
- Examine your datasets interactively (with ipython).

For more information on specific algorithms we recommend starting by looking at DIPY's :ref:`gallery <examples>` of examples.

For a full list of the features implemented in the most recent release cycle, check out the release notes.

.. toctree::
   :maxdepth: 1

   release_notes/release1.11
   release_notes/release1.10
   release_notes/release1.9
   release_notes/release1.8
   release_notes/release1.7
   release_notes/release1.6
   release_notes/release1.5
   release_notes/release1.4.1
   release_notes/release1.4
   release_notes/release1.3
   release_notes/release1.2
   release_notes/release1.1
   release_notes/release1.0
   release_notes/release0.16
   release_notes/release0.15
   release_notes/release0.14
   release_notes/release0.13
   release_notes/release0.12
   release_notes/release0.11
   release_notes/release0.10
   release_notes/release0.9
   release_notes/release0.8
   release_notes/release0.7
   release_notes/release0.6

=================
Systems supported
=================

DIPY_ is multiplatform and will run under any standard operating systems such
as *Windows*, *Linux* and  *Mac OS X*. Every single new code addition is being tested on
a number of different buildbots and can be monitored online `here <https://github.com/dipy/dipy/actions>`_.


.. include:: links_names.inc
