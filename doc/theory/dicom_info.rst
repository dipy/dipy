===================
 DICOM information
===================

DICOM_ is a large and sometimes confusing imaging data format.  

In the other pages in this series we try and document our understanding
of various aspects of DICOM relevant to converting to formats such as
NIfTI_ .

There are a large number of DICOM_ image conversion programs already,
partly because it is a complicated format with features that vary from
manufacturer to manufacturer.

We use the excellent PyDICOM_ as our back-end for reading DICOM.

Here is a selected list of other tools and relevant resources:

* Grassroots DICOM : GDCM_ .  It is C++ code wrapped with swig_ and so
  callable from Python.  ITK_ apparently uses it for DICOM conversion.
  BSD_ license.
* dcm2nii_ - a BSD_ licensed converter by Chris Rorden. As usual, Chris
  has done an excellent job of documentation, and it is well
  battle-tested.  There's a nice set of example data to test against and
  a list of other DICOM software.  The `MRIcron install`_ page points to
  the source code.  Chris has also put effort into extracting diffusion
  parameters from the DICOM images.
* SPM8_ - SPM has a stable and robust general DICOM conversion tool
  implemented in the ``spm_dicom_convert.m`` and ``spm_dicom_headers.m``
  scripts.  The conversions don't try to get the diffusion parameters.
  The code is particularly useful because it has been well-tested and is
  written in Matlab_ - and so is relatively easy to read.  GPL_ license.
  We've described some of the algorithms that SPM uses for DICOM
  conversion in :ref:`spm-dicom`.
* DICOM2Nrrd_: a command line converter to convert DICOM images to Nrrd_
  format.  You can call the command from within the Slicer_ GUI.  It
  does have algorithms for getting diffusion information from the DICOM
  headers, and has been tested with Philips, GE and Siemens data. It's
  not clear whether it yet supports the :ref:`dicom-mosaic`.  BSD_ style
  license.
* The famous Philips cookbook: http://www.archive.org/details/DicomCookbook
* http://dicom.online.fr/fr/dicomlinks.htm

===============
 Sample images
===============

* http://www.barre.nom.fr/medical/samples/
* http://pubimage.hcuge.ch:8080/
* Via links from the dcm2nii_ page.
 
.. include:: ../links_names.txt
