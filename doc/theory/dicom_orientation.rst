================================
 Defining the DICOM orientation
================================

.. _dicom-pcs:

DICOM patient coordinate system
===============================

First we define the standard DICOM patient-based coordinate system.
This is what DICOM means by x, y and z axes in its orientation
specification.  From section C.7.6.2.1.1 of the `DICOM object
definitions`_ (2009):

   If Anatomical Orientation Type (0010,2210) is absent or has a value
   of BIPED, the x-axis is increasing to the left hand side of the
   patient. The y-axis is increasing to the posterior side of the
   patient. The z-axis is increasing toward the head of the patient.

(we'll ignore the quadupeds for now). 

In a way it's funny to call this the 'patient-based' coordinate system.
'Doctor-based coordinate system' is a better name.  Think of a doctor
looking at the patient from the foot of the scanner bed.  Imagine the
doctor's right hand held in front of her like Spiderman about to shoot a
web, with her palm towards the patient, defining a right-handed
coordinate system.  Her thumb points to her right (the patient's left),
her index finger points down, and the middle finger points at the
patient.

.. _dicom-pixel-array:

DICOM pixel data
================

C.7.6.3.1.4 - Pixel Data 
   Pixel Data (7FE0,0010) for this image. The order of pixels sent for
   each image plane is left to right, top to bottom, i.e., the upper
   left pixel (labeled 1,1) is sent first followed by the remainder of
   row 1, followed by the first pixel of row 2 (labeled 2,1) then the
   remainder of row 2 and so on.

The resulting pixel array then has size ('Rows', 'Columns'), with
row-major storage (rows first, then columns).  We'll call this the DICOM
*pixel array*.

Pixel spacing
=============

Section 10.7.1.3:  Pixel Spacing
  The first value is the row spacing in mm, that is the spacing between
  the centers of adjacent rows, or vertical spacing.  The second value
  is the column spacing in mm, that is the spacing between the centers
  of adjacent columns, or horizontal spacing.

.. _dicom-orientation:

DICOM voxel to patient coordinate system mapping
================================================

See: 

* http://www.dclunie.com/medical-image-faq/html/part2.html
* http://fixunix.com/dicom/50449-image-position-patient-image-orientation-patient.html

See `wikipedia direction cosine`_ for a definition of direction cosines. 

From section C.7.6.2.1.1 of the `DICOM object definitions`_ (2009):

   The Image Position (0020,0032) specifies the x, y, and z coordinates
   of the upper left hand corner of the image; it is the center of the
   first voxel transmitted. Image Orientation (0020,0037) specifies the
   direction cosines of the first row and the first column with respect
   to the patient.  These Attributes shall be provide as a pair. Row
   value for the x, y, and z axes respectively followed by the Column
   value for the x, y, and z axes respectively.

From Section C.7.6.1.1.1 we see that the 'positive row axis' is left to
right, and is the direction of the rows, given by the direction of last
pixel in the first row from the first pixel in that row.  Similarly the
'positive column axis' is top to bottom and is the direction of the
columns, given by the direction of the last pixel in the first column
from the first pixel in that column.

Let's rephrase: the first three values of 'Image Orientation Patient'
are the direction cosine for the 'positive row axis'.  That is, they
express the direction change in (x, y, z), in the DICOM patient
coordinate system (DPCS), as you move along the row.  That is, as you
move from one column to the next.  That is, as the *column* array index
changes. Similarly, the second triplet of values of 'Image Orientation
Patient' (``img_ornt_pat[3:]`` in Python), are the direction cosine for
the 'positive column axis', and express the direction you move, in the
DPCS, as you move from row to row, and therefore as the *row* index
changes.

Further down section C.7.6.2.1.1 (RCS below is the *reference coordinate
system* - see `DICOM object definitions`_ section 3.17.1):

   The Image Plane Attributes, in conjunction with the Pixel Spacing
   Attribute, describe the position and orientation of the image slices
   relative to the patient-based coordinate system. In each image frame
   the Image Position (Patient) (0020,0032) specifies the origin of the
   image with respect to the patient-based coordinate system. RCS and
   the Image Orientation (Patient) (0020,0037) attribute values specify
   the orientation of the image frame rows and columns. The mapping of
   pixel location (i, j) to the RCS is calculated as follows:

   .. math::

      \begin{bmatrix} P_x\\
                      P_y\\
                      P_z\\
                      1 \end{bmatrix} = 
      \begin{bmatrix} X_x\Delta{i} & Y_x\Delta{j} & 0 & S_x \\ 
                      X_y\Delta{i} & Y_y\Delta{j} & 0 & S_y \\
                      X_z\Delta{i} & Y_z\Delta{j} & 0 & S_z \\
                      0   & 0   & 0 & 1 \end{bmatrix}
      \begin{bmatrix} i\\
                      j\\
                      0\\
                      1 \end{bmatrix} 
      = M 
      \begin{bmatrix} i\\
                      j\\
                      0\\
                      1 \end{bmatrix} 
      
   Where:

   #. $P_{xyz}$ : The coordinates of the voxel (i,j) in the frame's
      image plane in units of mm.
   #. $S_{xyz}$ : The three values of the Image Position (Patient)
      (0020,0032) attributes. It is the location in mm from the origin
      of the RCS.
   #. $X_{xyz}$ : The values from the row (X) direction cosine of the
      Image Orientation (Patient) (0020,0037) attribute.
   #. $Y_{xyz}$ : The values from the column (Y) direction cosine of the
      Image Orientation (Patient) (0020,0037) attribute.
   #. $i$ : Column index to the image plane. The first column is index
      zero.
   #. $\Delta{i}$: Column pixel resolution of the Pixel Spacing
      (0028,0030) attribute in units of mm.
   #. $j$ : Row index to the image plane. The first row index is zero.
   #. $\Delta{j}$ - Row pixel resolution of the Pixel Spacing
      (0028,0030) attribute in units of mm.

.. _ij-transpose:

(i, j), columns, rows in DICOM
==============================

We stop to ask ourselves, what does DICOM mean by voxel (i, j)?

Isn't that obvious?  Oh dear, no it isn't.  See the
:ref:`dicom-orientation` formula above.  In particular, you'll see:

* $i$ : Column index to the image plane. The first column is index zero.
* $j$ : Row index to the image plane. The first row index is zero.

That is, if we have the :ref:`dicom-pixel-array` as defined above, and
we call that ``pixel_array``, then voxel (i, j) in the notation above is
given by ``pixel_array[j, i]``.

What does this mean?  It means that, if we want to apply the formula
above to array indices in ``pixel_array``, we first have to apply a
column / row flip to the indices.  Say $M_{pixar}$ (sorry) is the affine
to go from array indices in ``pixel_array`` to mm in the DPCS.  Then,
given $M$ above:

.. math::

   M_{pixar} = M \left(\begin{smallmatrix}0 & 1 & 0 & 0\\1 & 0 & 0 & 0\\0 & 0 & 1 & 0\\0 & 0 & 0 & 1\end{smallmatrix}\right)

The affines below do not take this premultiplication into account - that
is, they assume that you have either already applied it, or that you
have transposed ``pixel_array``.

.. _dicoms-and-affines:

Getting the affine transformation from DICOM files and file lists
=================================================================

Let us say, we have a single DICOM file, or a list of DICOM files that
we believe to be a set of slices from the same volume.  We'll call the
first the *single slice* case, and the second, *multi slice*.

In the *multi slice* case, we can assume that the
'ImageOrientationPatient' field is the same for all the slices.

We want to get the affine transformation matrix $A$ that maps from voxel
coordinates in the DICOM file(s), to mm in the :ref:`dicom-pcs`.  

By voxel coordinates, we mean *transposed coordinates* - see
:ref:`ij-transpose` above.

In the single slice case, the voxel coordinates are just the indices
into the pixel array, with the third (slice) coordinate always being 0.
In the multi-slice case, we have arranged the slices in ascending or
descending order, where slice numbers range from 0 to $N-1$ - where $N$
is the number of slices - and the slice coordinate is a number on this
scale.

We know, from the formula above, that the first, second and fourth
columns in $A$ are given directly by the formula in
:ref:`dicom-orientation` - from the 'ImageOrientationPatient',
(reversed) 'PixelSpacing' and 'ImagePositionPatient' field of the first
(or only) slice.

Our job then is to fill the first three rows of the third column of $A$.
Let's call this the vector $\mathbf{s}$ with values  $s_1, s_2, s_3$.

.. _dicom-affine-defs:

DICOM affine Definitions
------------------------

Let $O$ be the DICOM orientation patient field, reorganized to the (3,2)
matrix it represents (see :ref:`dicom-orientation`).  Let $T^1$ be the 3
element vector of the 'ImagePositionPatient' field of the first header
in the list of headers for this volume.  Let $T^N$ be the
'ImagePositionPatient' vector for the last header in the list for this
volume, if there is more than one header in the volume.  Let
$\Delta_{cols}$ and $\Delta_{rows}$ be the two values in the
'PixelSpacing' field.  Let $\Delta_{slices}$ be the value for the
'SliceThickness' field, if present, otherwise $\Delta_{slices} = 1$.  Let
vector $\mathbf{c} = (c_1, c_2, c_3)$ be the result of taking the cross
product of the two columns of $O$.

Derivations
-----------

For the single slice case we just fill $\mathbf{s}$ with $\mathbf{c} \cdot
\Delta_{slices}$ - on the basis that the Z dimension should be
right-handed orthogonal to the X and Y directions.

For the multi-slice case, we can fill in $\mathbf{s}$ by using the information
from $T^N$, because $T^N$ is the translation needed to take the
first voxel in the last (slice index = $N-1$) slice to mm space.  So:

.. math:: 

   \left(\begin{smallmatrix}T^N\\1\end{smallmatrix}\right) = A \left(\begin{smallmatrix}0\\0\\-1 + N\\1\end{smallmatrix}\right)

From this it follows that:

.. math::

   \begin{Bmatrix}s_{{1}} : \frac{T^{1}_{{1}} - T^{N}_{{1}}}{1 - N}, & s_{{2}} : \frac{T^{1}_{{2}} - T^{N}_{{2}}}{1 - N}, & s_{{3}} : \frac{T^{1}_{{3}} - T^{N}_{{3}}}{1 - N}\end{Bmatrix}

and therefore:

.. _dicom-affine-formulae:

DICOM affine formulae
---------------------

.. math::

   A_{multi} = \left(\begin{smallmatrix}O_{{11}} \Delta_{cols} & O_{{12}} \Delta_{rows} & \frac{T^{1}_{{1}} - T^{N}_{{1}}}{1 - N} & T^{1}_{{1}}\\O_{{21}} \Delta_{cols} & O_{{22}} \Delta_{rows} & \frac{T^{1}_{{2}} - T^{N}_{{2}}}{1 - N} & T^{1}_{{2}}\\O_{{31}} \Delta_{cols} & O_{{32}} \Delta_{rows} & \frac{T^{1}_{{3}} - T^{N}_{{3}}}{1 - N} & T^{1}_{{3}}\\0 & 0 & 0 & 1\end{smallmatrix}\right)
   
   A_{single} = \left(\begin{smallmatrix}O_{{11}} \Delta_{cols} & O_{{12}} \Delta_{rows} & c_{{1}} \Delta_{slices} & T^{1}_{{1}}\\O_{{21}} \Delta_{cols} & O_{{22}} \Delta_{rows} & c_{{2}} \Delta_{slices} & T^{1}_{{2}}\\O_{{31}} \Delta_{cols} & O_{{32}} \Delta_{rows} & c_{{3}} \Delta_{slices} & T^{1}_{{3}}\\0 & 0 & 0 & 1\end{smallmatrix}\right)

See :download:`derivations/spm_dicom_orient.py` for the derivations and
some explanations.

.. _dicom-z-from-slice:

Working out the Z coordinates for a set of slices
=================================================

We may have the problem (see e.g. :ref:`spm-volume-sorting`) of trying
to sort a set of slices into anatomical order.  For this we want to use
the orientation information to tell us where the slices are in space,
and therefore, what order they should have.

To do this sorting, we need something that is proportional, plus a
constant, to the voxel coordinate for the slice (the value for the slice
index).

Our DICOM might have the 'SliceLocation' field (0020,1041).
'SliceLocation' seems to be proportianal to slice location, at least for
some GE and Philips DICOMs I was looking at.  But, there is a more
reliable way (that doesn't depend on this field), and uses only the very
standard 'ImageOrientationPatient' and 'ImagePositionPatient' fields.

Consider the case where we have a set of slices, of unknown order, from
the same volume.

Now let us say we have one of these slices - slice $i$.  We have the
affine for this slice from the calculations above, for a single slice
($A_{single}$).

Now let's say we have another slice $j$ from the same volume.  It will
have the same affine, except that the 'ImagePositionPatient' field will
change to reflect the different position of this slice in space. Let us
say that there a translation of $d$ slices between $i$ and $j$.  If
$A_i$ ($A$ for slice $i$) is $A_{single}$ then $A_j$ for $j$ is given
by:

.. math::

   A_j = A_{single} \left(\begin{smallmatrix}1 & 0 & 0 & 0\\0 & 1 & 0 & 0\\0 & 0 & 1 & d\\0 & 0 & 0 & 1\end{smallmatrix}\right)

and 'ImagePositionPatient' for $j$ is:

.. math::

   T^j = \left(\begin{smallmatrix}T^{1}_{{1}} + c_{{1}} d \Delta_{slices}\\T^{1}_{{2}} + c_{{2}} d \Delta_{slices}\\T^{1}_{{3}} + c_{{3}} d \Delta_{slices}\end{smallmatrix}\right)

Remember that the third column of $A$ gives the vector resulting from a
unit change in the slice voxel coordinate.  So, the
'ImagePositionPatient' of slice - say slice $j$ - can be thought of the
addition of two vectors $T^j = \mathbf{a} + \mathbf{b}$, where
$\mathbf{a}$ is the position of the first voxel in some slice (here
slice 1, therefore $\mathbf{a} = T^1$) and $\mathbf{b}$ is $d$ times the
third colum of $A$.  Obviously $d$ can be negative or positive. This
leads to various ways of recovering something that is proportional to
$d$ plus a constant.  SPM takes the inner product of $T^j$ with the unit
vector component of third column of $A_j$ - in the descriptions here,
this is the vector $\mathbf{c}$:

.. math::

   T^j \cdot \mathbf{c} = \left(\begin{smallmatrix}c_{{1}} T^{1}_{{1}} + c_{{2}} T^{1}_{{2}} + c_{{3}} T^{1}_{{3}} + d \Delta_{slices} c_{{1}}^{2} + d \Delta_{slices} c_{{2}}^{2} + d \Delta_{slices} c_{{3}}^{2}\end{smallmatrix}\right)

The unknown $T^1$ terms pool into a constant, and the operation has the
neat feature that, because the $c_N^2$ terms, by definition, sum to 1,
the whole can be expressed as $\lambda + \Delta_{slices} d$ - i.e. it is
equal to the slice voxel size ($\Delta_{slices}$) multiplied by $d$,
plus a constant.

Again, see :download:`derivations/spm_dicom_orient.py` for the derivations.

.. include:: ../links_names.txt

