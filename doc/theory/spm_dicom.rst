.. _spm-dicom:

===============
 SPM and DICOM
===============

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
   #. If ``group`` == 8 and ``element`` == 0, this is apparently a 'GE
      Twin+excite' for which JA notes there is no documentation; set
      file pointer ot 0 and continue read with ``read_dicom``
      subfunction.
   #. Otherwise - crash out with error saying that this is not DICOM file.

tag read for Philips Integra
----------------------------

The ``read_dicom`` subfunction reads a tag, then has a loop during which
the tag is processed (by setting values into the return structure).  At
the end of the loop, it reads the next tag.  The loop breaks when the
current tag is empty, or is (group=FFFE, element=E00D).  

After it has broken out of the loop, if the last tag was (FFFE, E00D),
and the tag length was not 0, then SPM sets the file pointer back by 4
bytes from the current position.  JA comments that he didn't find that
in the standard, but that it seemed to be needed for the Philips
Integra.

Tag length
----------

Tag lengths as read in ``read_tag`` subfunction.  If current format is
explicit (as in 'explicit little endian') then tag length depends on VR
(value representation).  In this case, fixes are:

#. For VR of 'UN' or group of FFFE, it looks like we need a seek of 2
   bytes forward in the file.
#. For VR of \x00\x00, then we only know about (FFFE, E00D).  JA
   comments that GE 'ImageDelimitationItem' has no VR, just 4 0 bytes.
   In this case the tag length is zero, and we read another two bytes
   ahead.

Otherwise (implicit endianness) tag length is always uint.

There's a check for not-even tag length.  If not even:

#. 4294967295 appears to be OK - and decoded as Inf for tag length. 
#. 13 appears to mean 10 and is reset to be 10
#. Any other odd number is not valid and gives a tag length of 0

``spm_dicom_convert.m``
=======================

