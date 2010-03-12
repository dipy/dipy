=======================
 Siemens mosaic format
=======================

Siemens mosaic format is a way of storing a 3D image in a DICOM_ image
file.   The standard DICOM_ image format only knows how to store 2D
files.  For example, a 3D image in DICOM is usually stored as a series
of 2D slices.  Mosaic format stored the 3D image slices as a grid - or
mosaic.

.. _csa-header:

Siemens private header
======================

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

DICOM orientation for mosaic
============================

See :ref:`dicom-pcs` and :ref:`dicom-orientation`.  To define the voxel
to millimeter mapping, in terms of the :ref:`dicom-pcs`, we need a 4 x 4
affine homogenous transform matrix, which can in turn be thought of as
the (3,3) component, $RS$, and a (3,1) translation vector $\mathbf{t}$.
$RS$ can in turn be thought of as the dot product of a (3,3) rotation
matrix $R$ and a scaling matrix $S$, where ``S = diag(s)`` and
$\mathbf{s}$ is a (3,) vector of voxel sizes.  $\mathbf{t}$ is a (3,1)
translation vector, defining the coordinate in millimeters of the
first voxel in the voxel volume (the voxel given by
``voxel_array[0,0,0]``).

In the case of the mosaic, we have the first two columns of $R$ from the
``ImageOrientationPatient`` DICOM field.  To make a full rotation
matrix, we can generate the last column from the cross product of the
first two.  However, Siemens defines, in its private header, a
``SliceNormalVector`` which gives the third column, but possibly with a
z flip, so that $R$ then is orthogaonal, but not a rotation matrix (it
has a determinant of < 0).

We can get the first two values of $\mathbf{s}$ with the
``PixelSpacing`` field, and the last (z scaling) value with the
``SpacingBetweenSlices``.

The SPM_ DICOM conversion code notes that, for mosaic DICOM images, the
$\mathbf{t}$ vector - which should come from the
``ImagePositionPatient`` field, is not correct for the mosaic format.
Comments in the code imply that ``ImagePositionPatient`` has been
derived from the (correct) position of the center of the first slice
(once the mosaic has been unpacked), but then adjusted to point to the
top left voxel, where the slice size used for this adjustment is the
size of the mosaic, before it has been unpacked.  Let's call the correct
position in millimeters of the center of the first slice $\mathbf{c} =
[c_x, c_y, c_z]$.  We have the derived $RS$ matrix from the calculations
above. The unpacked (eventual, real) slice dimensions are $[rd_x, rd_y]$
and the mosaic dimensions are $[md_x, md_y]$.  The
``ImagePositionPatient`` vector $\mathbf{i}$ resulted from:

.. math::

   \mathbf{i} = \mathbf{c} + RS 
      \begin{bmatrix} -(md_x-1) / 2\\
                      -(md_y-1) / 2\\
                      0 \end{bmatrix}

To correct this we reverse the translation, and add the correct
translation for the unpacked slice size $[rd_x, rd_y]$, giving the true
image position $\mathbf{t}$:

.. math::

   \mathbf{t} = \mathbf{i} - 
                (RS \begin{bmatrix} -(md_x-1) / 2\\
                                      -(md_y-1) / 2\\
                                      0 \end{bmatrix}) +
                (RS \begin{bmatrix} -(rd_x-1) / 2\\
                                      -(rd_y-1) / 2\\
                                      0 \end{bmatrix})


Because of the final zero in the voxel translations, this simplifies to:

.. math::

   \mathbf{t} = \mathbf{i} + 
                M \begin{bmatrix} (md_x - rd_x) / 2 \\
                                  (md_y - rd_y) / 2 \end{bmatrix}

where: 

.. math::

   M = \begin{bmatrix} rs_{11} & rs_{12} \\
                       rs_{21} & rs_{22} \\
                       rs_{31} & rs_{32} \end{bmatrix}

which is of course the contents of the ``ImagePositionPatient`` field in
the DICOM header.

.. include:: ../links_names.txt
