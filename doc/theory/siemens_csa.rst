.. _siemens_csa:

======================================
 Siemens format DICOM with CSA header
======================================

Recent Siemens DICOM images have useful information stored in a private
header.  We'll call this the *CSA header*.

.. _csa-header:

CSA header
==========

See this Siemens `Syngo DICOM conformance`_ statement, and a GDCM_ 
`Siemens header dump`_.

.. _`Siemens header dump`: http://sourceforge.net/apps/mediawiki/gdcm/index.php?title=Gdcmdump#SIEMENS_CSA_Header
.. _`Syngo DICOM conformance`: http://www.medical.siemens.com/siemens/en_GLOBAL/rg_marcom_FBAs/files/brochures/DICOM/rs/syngoImaging_DCS_VB30A_External.pdf

The CSA header is stored in DICOM private tags.  In the images we are
looking at, there are several relevant tags::

  (0029, 1008) [CSA Image Header Type]             OB: 'IMAGE NUM 4 '
  (0029, 1009) [CSA Image Header Version]          OB: '20100114'
  (0029, 1010) [CSA Image Header Info]             OB: Array of 11560 bytes
  (0029, 1018) [CSA Series Header Type]            OB: 'MR'
  (0029, 1019) [CSA Series Header Version]         OB: '20100114'
  (0029, 1020) [CSA Series Header Info]            OB: Array of 80248 bytes

In our case we want to read the 'CSAImageHeaderInfo'.

From the SPM_ (SPM8) code ``spm_dicom_headers.m``

The CSAImageHeaderInfo and the CSA Series Header Info fields are of the
same format.  The fields can be of two types, CSA1 and CSA2.

Both are always little-endian, whatever the machine endian is.

The CSA2 format begins with the string 'SV10', the CSA1 format does
not. 

The code below keeps track of the position *within the CSA header
stream*.  We'll call this ``csa_position``.  At this point (after
reading the 8 bytes of the header), ``csa_position == 8``.   There's a
variable that sets the last byte position in the file that is sensibly
still CSA header, and we'll call that ``csa_max_pos``.

CSA1
====

Start header
------------

#. n_tags, uint32, number of tags.  Number of tags should apparently be
   between 1 and 128.  If this is not true we just abort and move to
   ``csa_max_pos``.
#. unused, uint32, apparently has value 77

Each tag
--------

#. name : S64, null terminated string 64 bytes
#. vm : int32
#. vr : S4, first 3 characters only
#. syngodt : int32
#. nitems : int32
#. xx : int32 - apparently either 77 or 205

``nitems`` gives the number of items in the tag.  The items follow
directly after the tag.

Each item
---------

1. xx : int32 * 4 .  The first of these seems to be the length of the
   item in bytes, modified as below.

At this point SPM does a check, by calculating the length of this item
``item_len`` with ``xx[0]`` - the ``nitems`` of the *first* read tag.
If ``item_len`` is less than 0 or greater than
``csa_max_pos-csa_position`` (the remaining number of bytes to read in
the whole header) then we break from the item reading loop,
setting the value below to ''. 

Then we calculate ``item_len`` rounded up to the nearest 4 byte boundary
tp get ``next_item_pos``. 

2. value : uint8, ``item_len``. 

We set the stream position to ``next_item_pos``. 

CSA2
====

Start header
------------

#. hdr_id : S4 == 'SV10'
#. unused1 : uint8, 4
#. n_tags, uint32, number of tags.  Number of tags should apparently be
   between 1 and 128.  If this is not true we just abort and move to
   ``csa_max_pos``.
#. unused2, uint32, apparently has value 77

Each tag
--------

#. name : S64, null terminated string 64 bytes
#. vm : int32
#. vr : S4, first 3 characters only
#. syngodt : int32
#. nitems : int32
#. xx : int32 - apparently either 77 or 205

``nitems`` gives the number of items in the tag.  The items follow
directly after the tag.

Each item
---------

1. xx : int32 * 4 .  The first of these seems to be the length of the
   item in bytes, modified as below.

Now there's a different length check from CSA1.  ``item_len`` is given
just by ``xx[1]``.  If ``item_len`` > ``csa_max_pos - csa_position``
(the remaining bytes in the header), then we just read the remaning
bytes in the header (as above) into ``value`` below, as uint8, move the
filepointer to the next 4 byte boundary, and give up reading. 

2. value : uint8, ``item_len``. 

We set the stream position to the next 4 byte boundary. 

.. include:: ../links_names.txt
