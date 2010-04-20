.. _dicom-fields:

==============
 DICOM fields
==============

In which we pick out some interesting fields in the DICOM header.

We're getting the information mainly from the standard `DICOM object
definitions`_

We won't talk about the orientation, patient position-type fields here
because we've covered those somewhat in :ref:`dicom-orientation`.

Fields for ordering DICOM files into images
===========================================

You'll see some discussion of this in :ref:`spm-dicom`. 

Section 7.3.1: general series module

* Modality (0008,0060) - Type of equipment that originally acquired the
  data used to create the images in this Series. See C.7.3.1.1.1 for
  Defined Terms.
* Series Instance UID (0020,000E) - Unique identifier of the Series.
* Series Number (0020,0011) - A number that identifies this Series.
* Series Time (0008,0031) - Time the Series started.

Section C.7.6.1:

* Instance Number (0020,0013) - A number that identifies this image.
* Acquisition Number (0020,0012) - A number identifying the single
  continuous gathering of data over a period of time that resulted in
  this image.
* Acquisition Time (0008,0032) - The time the acquisition of data that
  resulted in this image started

Section C.7.6.2.1.2:

Slice Location (0020,1041) is defined as the relative position of the
image plane expressed in mm. This information is relative to an
unspecified implementation specific reference point.

Section C.8.3.1 MR Image Module

* Slice Thickness (0018,0050) - Nominal reconstructed slice thickness,
  in mm.

Section C.8.3.1 MR Image Module

* Spacing Between Slices (0018,0088) - Spacing between slices, in
  mm. The spacing is measured from the center-tocenter of each slice.
* Temporal Position Identifier (0020,0100) - Temporal order of a dynamic
  or functional set of Images.
* Number of Temporal Positions (0020,0105) - Total number of temporal
  positions prescribed.
* Temporal Resolution (0020,0110) - Time delta between Images in a
  dynamic or functional set of images

Multi-frame images
==================

An image for which the pixel data is a continuous stream of sequential frames.

Section C.7.6.6: Multi-Frame Module

* Number of Frames (0028,0008) - Number of frames in a Multi-frame
  Image.
* Frame Increment Pointer (0028,0009) - Contains the Data Element Tag of
  the attribute that is used as the frame increment in Multi-frame pixel
  data.


.. include:: ../links_names.txt
