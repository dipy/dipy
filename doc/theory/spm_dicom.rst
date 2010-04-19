.. _spm-dicom:

======================
 SPM DICOM conversion
======================

These are some notes on the algorithms that SPM_ uses to convert from
DICOM_ to nifti_.  There are other notes in :ref:`dicom-mosaic`. 

The relevant SPM files are ``spm_dicom_headers.m``,
``spm_dicom_dict.mat`` and ``spm_dicom_convert.m``.  These notes refer
the version in SPM8, as of around January 2010.

``spm_dicom_dict.mat``
======================

This is obviously a Matlab ``.mat`` file.  It contains variables
``group`` and ``element``, and ``values``, where ``values`` is a struct
array, one element per (group, element) pair, with fields ``name`` and
``vr`` (the last a cell array).


``spm_dicom_headers.m``
=======================

Reads the given DICOM files into a struct.  It looks like this was
written by John Ahsburner (JA).  Relevant fixes are:

File opening
------------

When opening the DICOM file, SPM (subfunction ``readdicomfile``) 

#. opens as little endian
#. reads 4 characters starting at pos 128
#. checks if these are ``DICM``; if so then continues file read;
   otherwise, tests to see if this is what SPM calls *truncated DICOM
   file format* - lacking 128 byte lead in and ``DICM`` string:

   #. Seeks to beginning of file
   #. Reads two unsigned short values into ``group`` and ``tag``
   #. If the (``group``, ``element``) pair exist in
      ``spm_dicom_dict.mat``, then set file pointer to 0 and continue
      read with ``read_dicom`` subfunction..
   #. If ``group`` == 8 and ``element`` == 0, this is apparently the
      signature for a 'GE Twin+excite' for which JA notes there is no
      documentation; set file pointer to 0 and continue read with
      ``read_dicom`` subfunction.
   #. Otherwise - crash out with error saying that this is not DICOM file.

tag read for Philips Integra
----------------------------

The ``read_dicom`` subfunction reads a tag, then has a loop during which
the tag is processed (by setting values into the return structure).  At
the end of the loop, it reads the next tag.  The loop breaks when the
current tag is empty, or is the item delimitation tag (group=FFFE,
element=E00D).

After it has broken out of the loop, if the last tag was (FFFE, E00D)
(item delimitation tag), and the tag length was not 0, then SPM sets the
file pointer back by 4 bytes from the current position.  JA comments
that he didn't find that in the standard, but that it seemed to be
needed for the Philips Integra.

Tag length
----------

Tag lengths as read in ``read_tag`` subfunction.  If current format is
explicit (as in 'explicit little endian'):

#. For VR of \x00\x00, then group, element must be (FFFE, E00D) (item
   delimitation tag). JA comments that GE 'ImageDelimitationItem' has
   no VR, just 4 0 bytes.  In this case the tag length is zero, and we
   read another two bytes ahead.

There's a check for not-even tag length.  If not even:

#. 4294967295 appears to be OK - and decoded as Inf for tag length. 
#. 13 appears to mean 10 and is reset to be 10
#. Any other odd number is not valid and gives a tag length of 0

``SQ`` VR type (Sequnce of items type)
--------------------------------------

tag length of 13 set to tag length 10.


``spm_dicom_convert.m``
=======================

Written by John Ashburner and Jesper Andersson. 

File categorization
-------------------

SPM makes a special case of Siemens 'spectroscopy images'.  These are
images that have 'SOPClassUID' == '1.3.12.2.1107.5.9.1' and the private
tag of (29, 1210); for these it pulls out the affine, and writes a
volume of ones corresponding to the acquisition planes. 

For images that are not spectroscopy:

* Discards images that do not have any of ('MR', 'PT', 'CT') in 'Modality' field.
* Discards images lacking any of 'StartOfPixelData', 'SamplesperPixel',
  'Rows', 'Columns', 'BitsAllocated', 'BitsStored', 'HighBit',
  'PixelRespresentation'
* Discards images lacking any of 'PixelSpacing', 'ImagePositionPatient',
  'ImageOrientationPatient' - presumably on the basis that SPM cannot
  reconstruct the affine.
* Fields 'SeriesNumber', 'AcquisitionNumber' and 'InstanceNumber' are
  set to 1 if absent.

Next SPM distinguishes between :ref:`dicom-mosaic` and standard DICOM. 

Mosaic images are those with the Siemens private tag::

  (0029, 1009) [CSA Image Header Version]          OB: '20100114'

and a readable CSA header (see :ref:`dicom-mosaic`), and with
non-empty fields from that header of 'AcquisitionMatrixText',
'NumberOfImagesInMosaic', and with non-zero 'NumberOfImagesInMosaic'.  The
rest are standard DICOM.

For converting mosaic format, see :ref:`dicom-mosaic`.  The rest of this
page refers to standard (slice by slice) DICOMs.

.. _spm-volume-sorting:

Sorting files into volumes
--------------------------

First pass
~~~~~~~~~~

Take first header, put as start of first volume.   For each subsequent header:

#. Get ``ICE_Dims`` if present.  Look for Siemens 'CSAImageHeaderInfo',
   check it has a 'name' field, then pull dimensions out of 'ICE_Dims'
   field in form of 9 integers separated by '_', where 'X' in this
   string replaced by '-1' - giving 'ICE1'

Then, for each currently identified volume: 

#. If we have ICE1 above, and we do have 'CSAIMageHeaderInfo', with a
   'name', in the first header in this volume, then extract ICE dims in
   the same way as above, for the first header in this volume, and check
   whether all but ICE1[6:8] are the same as ICE2.  Set flag that all
   ICE dims are identical for this volume.  Set this flag to True if we
   did not have ICE1 or CSA information.
#. Match the current header to the current volume iff the following match:

   #. SeriesNumber
   #. Rows
   #. Columns
   #. ImageOrientationPatient (to tolerance of sum squared difference 1e-4)
   #. PixelSpacing (to tolerance of sum squared difference 1e-4)
   #. ICE dims as defined above
   #. ImageType (iff imagetype exists in both)zv
   #. SequenceName (iff sequencename exists in both)
   #. SeriesInstanceUID (iff exists in both)
   #. EchoNumbers (iff exists in both)

#. If the current header matches the current volume, insert it there,
   otherwise make a new volume for this header

.. _spm-second-pass:

Second pass
~~~~~~~~~~~

We now have a list of volumes, where each volume is a list of headers
that may match.

For each volume:

#. Estimate the z direction cosine by (effectively) finding the cross
   product of the x and y direction cosines contained in
   'ImageOrientationPatient' - call this ``z_dir_cos``
#. For each header in this volume, get the z coordinate by taking the
   dot product of the 'ImagePositionPatient' vector and ``z_dir_cos``
   (see :ref:`dicom-z-from-slice`).
#. Sort the headers according to this estimated z coordinate. 
#. If this volume is more than one slice, and there are any slices with
   the same z coordinate (as defined above), run the
   :ref:`dicom-img-resort` on this volume - on the basis that it may
   have caught more than one volume-worth of slices.  Return one or more
   volume's worth of lists.

Final check
~~~~~~~~~~~

For each volume, recalculate z coordinate as above.  Calculate the z
gaps.  Subtract the mean of the z gaps from all z gaps.  If the average of the
(gap-mean(gap)) is greater than 1e-4, then print a warning that there
are missing DICOM files.

.. _dicom-img-resort:

Possible volume resort
~~~~~~~~~~~~~~~~~~~~~~

This step happens if there were volumes with slices having the same z
coordinate in the :ref:`spm-second-pass` step above.  The resort is on the
set of DICOM headers that were in the volume, for which there were
slices with identical z coordinates.  We'll call the list of headers
that the routine is still working on - ``work_list``.

#. If there is no 'InstanceNumber' field for the first header in
   ``work_list``, bail out.
#. Print a message about the 'AcquisitionNumber' not changing from
   volume to volume.  This may be a relic from previous code, because
   this version of SPM does not use the 'AcquisitionNumber' field except
   for making filenames.
#. Calculate the z coordinate as for :ref:`spm-second-pass`, for each
   DICOM header.
#. Sort the headers by 'InstanceNumber' 
#. If any headers have the same 'InstanceNumber', then discard all but
   the first header with the same number.  At this point the remaining
   headers in ``work_list`` will have different 'InstanceNumber's, but
   may have the same z coordinate.
#. Now sort by z coordinate
#. If there are ``N`` headers, make a ``N`` length vector of flags
   ``is_processed``, for which all values == False
#. Make an output list of header lists, call it ``hdr_vol_out``, set to empty. 
#. While there are still any False elements in ``is_processed``:

   #. Find first header for which corresponding ``is_processed`` is
      False - call this ``hdr_to_check``
   #. Collect indices (in ``work_list``) of headers which have the same
      z coordinate as ``hdr_to_check``, call this list
      ``z_same_indices``.
   #. Sort ``work_list[z_same_indices]`` by 'InstanceNumber'
   #. For each index in ``z_same_indices`` such that ``i`` indexes the
      indices, and ``zsind`` is ``z_same_indices[i]``: append header
      corresponding to ``zsind`` to ``hdr_vol_out[i]``.  This assumes
      that the original ``work_list`` contained two or more volumes,
      each with an identical set of z coordinates.
   #. Set corresponding ``is_processed`` flag to True for all ``z_same_indices``. 

#. Finally, if the headers in ``work_list`` have 'InstanceNumber's that
   cannot be sorted to a sequence ascending in units of 1, or if any
   of the lists in ``hdr_vol_out`` have different lengths, emit a
   warning about missing DICOM files.

Writing DICOM volumes
---------------------

This means - writing DICOM volumes from standard (slice by slice) DICOM
datasets rather than :ref:`dicom-mosaic`.

Making the affine
~~~~~~~~~~~~~~~~~

We need the (4,4) affine $A$ going from voxel (array) coordinates in the
DICOM pixel data, to mm coordinates in the :ref:`dicom-pcs`.

This section tries to explain how SPM achieves this, but I don't
completely understand their method.  See :ref:`dicom-3d-affines` for
what I believe to be a simpler explanation.

First define the constants, matrices and vectors as in
:ref:`dicom-affine-defs`.

$N$ is the number of slices in the volume.

Then define the following matrices:

.. math::

   R = \left(\begin{smallmatrix}1 & a & 1 & 0\\1 & b & 0 & 1\\1 & c & 0 & 0\\1 & d & 0 & 0\end{smallmatrix}\right)
   
   L = \left(\begin{smallmatrix}T^{1}_{{1}} & e & F_{{11}} \Delta{r} & F_{{12}} \Delta{c}\\T^{1}_{{2}} & f & F_{{21}} \Delta{r} & F_{{22}} \Delta{c}\\T^{1}_{{3}} & g & F_{{31}} \Delta{r} & F_{{32}} \Delta{c}\\1 & h & 0 & 0\end{smallmatrix}\right)

For a volume with more than one slice (header), then $a=1; b=1, c=N, d=1$. $e, f, g$ are the values from $T^N$,
and $h == 1$.

For a volume with only one slice (header) $a=0, b=0, c=1, d=0$ and $e,
f, g, h$ are $n_1 \Delta{s}, n_2 \Delta{s}, n_3 \Delta{s}, 0$.

The full transform appears to be $A_{spm} = R L^{-1}$.

Now, SPM, don't forget, is working in terms of Matlab array indexing,
which starts at (1,1,1) for a three dimensional array, whereas DICOM
expects a (0,0,0) start (see :ref:`dicom-slice-affine`).  In this
particular part of the SPM DICOM code, somewhat confusingly, the (0,0,0)
to (1,1,1) indexing is dealt with in the $A$ transform, rather than the
``analyze_to_dicom`` transformation used by SPM in other places. So, the
transform $A_{spm}$ goes from (1,1,1) based voxel indices to mm.  To
get the (0, 0, 0)-based transform we want, we need to pre-apply the
transform to take 0-based voxel indices to 1-based voxel indices:

.. math::

   A = R L^{-1} \left(\begin{smallmatrix}1 & 0 & 0 & 1\\0 & 1 & 0 & 1\\0 & 0 & 1 & 1\\0 & 0 & 0 & 1\end{smallmatrix}\right)

This formula with the definitions above result in the single and multi
slice formulae in :ref:`dicom-3d-affine-formulae`.

See :download:`derivations/spm_dicom_orient.py` for the derivations and
some explanations.

Writing the voxel data
~~~~~~~~~~~~~~~~~~~~~~

Just apply scaling and offset from 'RescaleSlope' and 'RescaleIntercept'
for each slice and write volume.

.. include:: ../links_names.txt
