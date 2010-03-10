================================
 Defining the DICOM orientation
================================
    
See : http://medical.nema.org/dicom/2004/04_03PU.PDF

  The direction cosines of the first row and the first column with
  respect to the patient. See C.7.6.2.1.1 and C.7.6.16.2.3.1 for further
  explanation.  Required if Frame Type (0008,9007) Value 1 of this frame
  is ORIGINAL and Volumetric Properties (0008,9206) of this frame is
  other than DISTORTED. May be present otherwise.

RCS below is the *reference coordinate system*

From: C.7.6.2.1.1

   Image Orientation (0020,0037) specifies the direction cosines of the
   first row and the first column with respect to the patient. These
   Attributes shall be provide as a pair. Row value for the x, y, and z
   axes respectively followed by the Column value for the x, y, and z
   axes respectively.

   The Image Plane Attributes, in conjunction with the Pixel Spacing
   Attribute, describe the position and orientation of the image slices
   relative to the patient-based coordinate system. In each image frame
   the Image Position (Patient) (0020,0032) specifies the origin of the
   image with respect to the patient-based coordinate system. RCS and
   the Image Orientation (Patient) (0020,0037) attribute values specify
   the orientation of the image frame rows and columns. The mapping of
   pixel location (i, j) to the RCS is calculated as follows:

P X∆i Y∆j 0 Si i
xx	x	x  Py=Xy∆i Yy∆j 0 Syj=Mj
P X∆i Y∆j 0 S0 0 zz	z	z  10 00111

    Where:

    #. Pxyz : The coordinates of the voxel (i,j) in the frame's image
       plane in units of mm.
    #. Sxyz : The three values of the Image Position (Patient)
       (0020,0032) attributes. It is the location in mm from the origin
       of the RCS.
    #. Xxyz : The values from the row (X) direction cosine of the Image
       Orientation (Patient) (0020,0037) attribute.
    #. Yxyz : The values from the column (Y) direction cosine of the
       Image Orientation (Patient) (0020,0037) attribute.
    #. i : Column index to the image plane. The first column is index
       zero. 
    #. ∆i : Column pixel resolution of the Pixel Spacing (0028,0030)
       attribute in units of mm. 
    #. j : Row index to the image plane. The first row index is
       zero. 
    #. ∆j - Row pixel resolution of the Pixel Spacing (0028,0030)
       attribute in units of mm.
