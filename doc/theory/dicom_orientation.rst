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

In a way it's funny to call this the 'patient-based' coordinate system,
because it's better understood by thinking of a doctor looking at the
patient from the foot of the scanner bed.  Imagine the doctor's right
hand held in front of her like spider-man about to shoot a web, with her
palm towards the patient, defining a right-handed coordinate system.
Her thumb points to her right (the patient's left), her index finger
points down, and the middle finger points at the patient.

.. _dicom-orientation:

DICOM voxel to patient coordinate system mapping
================================================

See: 

* http://www.dclunie.com/medical-image-faq/html/part2.html
* http://fixunix.com/dicom/50449-image-position-patient-image-orientation-patient.html

From section C.7.6.2.1.1 of the `DICOM object definitions`_ (2009):

   The Image Position (0020,0032) specifies the x, y, and z coordinates
   of the upper left hand corner of the image; it is the center of the
   first voxel transmitted. Image Orientation (0020,0037) specifies the
   direction cosines of the first row and the first column with respect
   to the patient.  These Attributes shall be provide as a pair. Row
   value for the x, y, and z axes respectively followed by the Column
   value for the x, y, and z axes respectively.

See C.7.6.16.2.3.1 for some further complications for sampled images
(where Volumetric Properties (0008,9206) has a value of ``SAMPLED``);
we'll ignore this case also for now.

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


.. _dicoms-and-affines:

Getting the affine transformation from DICOM files and file lists
=================================================================

Let us say, we have a single DICOM file, or a list of DICOM files that
we believe to be a set of slices from the same volume.  We'll call the
first the *single slice* case, and the second, *multi slice*.

In the *multi slice* case, we can assume that the
'ImageOrientationPatient' field is the same for all the slices.

We want to get the affine transformation matrix $A$ that maps from voxel
coordinates in the DICOM file(s), to mm in the :ref:`dicom-pcs`.  In the
single slice case, the voxel coordinates are just the indices into the
pixel array, with the third (z) coordinate always being 0.  In the
multi-slice case, we have arranged the slices in ascending or descending
order in Z.  The z coordinate refers to slice in this case, with 0 being
the first slice, and NZ-1 being the last slice.

We know, from the formula above, that the first, second and fourth
columns in $A$ are given directly by the formula in
:ref:`dicom-orientation` - from the 'ImageOrientationPatient',
'PixelSpacing' and 'ImagePositionPatient' field of the first (or only)
slice.

Our job then is to fill the first three rows of the third column of $A$.
Let's call this the vector $AZ$ with values  $AZ_1, AZ_2, AZ_3$.

.. _dicom-affine-defs:

DICOM affine Definitions
------------------------

Let ``DOP`` be the DICOM orientation patient field, reorganized to the
(3,2) matrix it represents (see :ref:`dicom-orientation`).  Let $IPP^0$
be the 3 element vector of the 'ImagePositionPatient' field of the first
header in the list of headers for this volume.  Let $IPP^N$ be the
'ImagePositionPatient' vector for the last header in the list for this
volume, if there is more than one header in the volume.  Let ``XS`` and
``YS`` be the two values in the 'PixelSpacing' field.  Let ``ZS`` be the
value for the 'SliceThickness' field, if present, otherwise ``ZS == 1``.
Let vector ``CP = [cp1, cp2, cp3]`` be the result of taking the cross
product of the two columns of ``DOP``. 

Derivations
-----------

For the single slice case we just fill $AZ$ with $CP \cdot ZS$ - on the
basis that the Z dimension should be right-handed orthogonal to the X
and Y directions.

For the multi-slice case, we can fill in $AZ$ by using the information
from $IPP^N$, because $IPP^N$ is the translation needed to take the
first voxel in the last ($z=NZ-1$) slice to mm space.  So:

.. math:: 

   \left(\begin{smallmatrix}IPP^N\\1\end{smallmatrix}\right) = A \left(\begin{smallmatrix}0\\0\\-1 + NZ\\1\end{smallmatrix}\right)

From this it follows that:

.. math::

   \begin{Bmatrix}AZ_{{1}} : \frac{IPP^{0}_{{1}} - IPP^{N}_{{1}}}{1 - NZ}, & AZ_{{2}} : \frac{IPP^{0}_{{2}} - IPP^{N}_{{2}}}{1 - NZ}, & AZ_{{3}} : \frac{IPP^{0}_{{3}} - IPP^{N}_{{3}}}{1 - NZ}\end{Bmatrix}

and therefore:

.. _dicom-affine-formulae:

DICOM affine formulae
---------------------

.. math::

   A_{multi} = \left(\begin{smallmatrix}DOP_{{11}} XS & DOP_{{12}} YS & \frac{IPP^{0}_{{1}} - IPP^{N}_{{1}}}{1 - NZ} & IPP^{0}_{{1}}\\DOP_{{21}} XS & DOP_{{22}} YS & \frac{IPP^{0}_{{2}} - IPP^{N}_{{2}}}{1 - NZ} & IPP^{0}_{{2}}\\DOP_{{31}} XS & DOP_{{32}} YS & \frac{IPP^{0}_{{3}} - IPP^{N}_{{3}}}{1 - NZ} & IPP^{0}_{{3}}\\0 & 0 & 0 & 1\end{smallmatrix}\right)
   
   A_{single} = \left(\begin{smallmatrix}DOP_{{11}} XS & DOP_{{12}} YS & CP_{{1}} ZS & IPP^{0}_{{1}}\\DOP_{{21}} XS & DOP_{{22}} YS & CP_{{2}} ZS & IPP^{0}_{{2}}\\DOP_{{31}} XS & DOP_{{32}} YS & CP_{{3}} ZS & IPP^{0}_{{3}}\\0 & 0 & 0 & 1\end{smallmatrix}\right)

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
constant, to the $z$ voxel coordinate for the slice.

Consider the case where we have a set of slices, of unknown order, from
the same volume.

Now let us say we have one of these slices - slice $i$.  We have the
affine for this slice from the calculations above, for a single slice
($A_{single}$).

Now let's say we have another slice $j$ from the same volume.  It will
have the same affine, except that the 'ImagePositionPatient' field will
change to reflect the different position of this slice in space. Let us
say that there a translation of $N_z$ voxels (slices) between $i$ and
$j$.  If the formula for slice $i$ - $A_i = A_{single}$ then the formula for
$j$ is given by:

.. math::

   A_j = A_{single} \left(\begin{smallmatrix}1 & 0 & 0 & 0\\0 & 1 & 0 & 0\\0 & 0 & 1 & N_{z}\\0 & 0 & 0 & 1\end{smallmatrix}\right)

and 'ImagePositionPatient' for $j$ is:

.. math::

   IPP_j = \left(\begin{smallmatrix}IPP^{0}_{{1}} + CP_{{1}} N_{z} ZS\\IPP^{0}_{{2}} + CP_{{2}} N_{z} ZS\\IPP^{0}_{{3}} + CP_{{3}} N_{z} ZS\end{smallmatrix}\right)

Remember that the third column of $A$ gives the vector resulting from a
unit change in the z voxel coordinate.  So, the 'ImagePositionPatient'
of any slice, can be thought of as the addition of the position of the
first voxel in some slice (here $IPP^0$) to $N_z$ times the third colum
of $A$; obviously $N_z$ can be negative or positive. This leads to
various ways of recovering something that is proportional to $N_z$ with
a constant.  SPM takes the dot product of $IPP_j$ with the unit vector
component of third column of $A_j$ - in the descriptions here, this is
the vector $CP$.  This gives:

.. math::

   IPP_j^T CP = \left(\begin{smallmatrix}CP_{{1}} IPP^{0}_{{1}} + CP_{{2}} IPP^{0}_{{2}} + CP_{{3}} IPP^{0}_{{3}} + N_{z} ZS CP_{{1}}^{2} + N_{z} ZS CP_{{2}}^{2} + N_{z} ZS CP_{{3}}^{2}\end{smallmatrix}\right)

The unknown $IPP^0$ terms pool into a constant, and the operation has
the neat feature that, because the $CP_N^2$ terms, by definition, sum to
1, the whole can be expresed as $\lambda + ZS N_z$ - i.e. it is equal to
the voxel size * $N_z$ plus a constant.

Obviously we could also do an element-wise divide of $IPP_j$ by $CP$,
and take the mean of the result, giving:

.. math::

   \frac{CP_{{1}} CP_{{2}} IPP^{0}_{{3}} + CP_{{1}} CP_{{3}} IPP^{0}_{{2}} + CP_{{2}} CP_{{3}} IPP^{0}_{{1}} + 3 CP_{{1}} CP_{{2}} CP_{{3}} N_{z} ZS}{3 CP_{{1}} CP_{{2}} CP_{{3}}}

This is also equal to the voxel size * $N_z$ plus a constant. 

Again, see :download:`derivations/spm_dicom_orient.py` for the derivations.

.. include:: ../links_names.txt
