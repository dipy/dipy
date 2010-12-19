.. _faq:

==========================
Frequently Asked Questions
==========================

-----------
Theoretical
-----------

1. What is a b-value?

The b-value $b$ or *diffusion weighting* is a function of the
strength, duration and temporal spacing and timing parameters of the
specific paradigm. This function is derived from the Bloch-Torrey
equations. In the case of the classical Stejskal-Tanner
pulsed gradient spin-echo (PGSE) sequence, at the time of readout
$b=\gamma^{2}G^{2}\delta^{2}\left(\Delta-\frac{\delta}{3}\right)$
where $\gamma$ is the gyromagnetic radio, $\delta$ denotes the pulse
width, $G$ is the gradient amplitude and $\Delta$ the centre to
centre spacing. $\gamma$ is a constant, but we can change the other
three parameters and in that way control the b-value.

2. What is q-space?
  
  Q-space is the space of one or more 3D spin displacement wave vectors
$\mathbf{q}$ as shown in equation \ref{eq:fourier}. The vector $\mathbf{q}$
parametrises the space of diffusion gradients. It is related to the
applied magnetic gradient $\mathbf{g}$ by the formula $\mathbf{q}=(2\pi)^{-1}\gamma\delta\mathbf{g}$.
Every single vector $\mathbf{q}$ has the same orientation as the
direction of diffusion gradient $\mathbf{g}$ and length proportional
to the strength $g$ of the gradient field. Every single point in
q-space corresponds to a possible 3D volume of the brain for a specific
gradient direction and strength. Therefore if for example we have
programmed the scanner to apply 60 gradient directions then our data
should have 60 diffusion volumes with each volume obtained for a specific
gradient. A Diffusion Weighted Image (DWI) is the volume acquired
from only one direction gradient.
  
3. What DWI stands for?
   
  Diffusion Weighted Imaging (DWI) is MRI imaging designed to be sensitive
to diffusion. A diffusion weighted image is a volume of voxel data gathered by applying only one gradient direction
using a diffusion sequence. We expect that the signal in every voxel
should be low if there is greater mobility of water molecules along
the specified gradient direction and it should be high if there is
less movement in that direction. Yes, it is counterintuitive correct!
  
4. Why dMRI and not DTI?

  Diffusion MRI (dMRI or dwMRI) are prefered terms if you want to speak about diffusion weighted MRI in general. 
  DTI (diffusion tensor imaging) is just one of the many ways you can reconstruct the voxel from your measured signal. 
  There are plenty of others for example DSI, GQI, QBI etc.     

5. What is the best practice for registration of diffusion datasets?

  Registration can be tricky. But this is what usually works for us for normal healthy adult subjects. 
  We register the FAs to the FMRIB_FA_1mm template which is in MNI space using FSL's flirt & fnirth. Then you can apply the 
  warping displacements in any other scalar volumes that you have or the invert displacements for tractography. 

6. What is the difference between image coordinates and world coordinates?

  Image coordinates have >0 integer values and express the voxel's centers. Then you can apply an affine transform (using stored in the nifti file) that 
  takes the image coordinates and transforms them to millimeter (mm) real world space. You have now floating point precision and your data have real dimensions.
  
7. Why tracks and not tracts?

  Tractography is only an approximation or simulation if you prefer of the real tracts (brain neural fiber pathways or brain nerves). 
  Therefore we prefer to call these simulated tracts as tracks (trajectories or curves) so, that others will be clear 
  that they are not the real tracts (fibers) but on. We hope that in the future tractography could reach a point that what you see on
  your screen is a nearly identical representation of what is in the brain. However, the field is not yet at this level of detail.    

8. Why deterministic and not probabilistic tractography?

  We wanted to create first a tractographic method which will help us & you to get closer to datasets in a very efficient way. Therefore, we
  created first EuDX (Euler Delta Crossings) which is a tracking method which can work both with model or modelfree input and resolve also
  crossing fibers of high order. Also it is very fast to calculate (~2 minutes for 1 million tracks ). We hope that at a later stage we will 
  be able to test more methods e.g. probabilistic, global & graph-theoretic.
  
9. We made the mistake in our lab to generate datasets with nonisotropic voxelsizes what do we do?
  
  You need to resample your raw data to an isotropic size. Have a look at the module dipy.align.noniso2iso
  
10. Why anisotropic voxel sizes are a bad idea in diffusion?
  
  If for example you have 2x2x4 mm^3 voxels, the last dimension will
  be averaged over the double distance and less detail will be captured compared
  to the other two dimensions. Furthermore, with very anisotropic voxels 
  the uncertainty on orientation estimates will depend on the position of 
  the subject in the scanner.

---------
Practical
---------

1. Why python and not matlab or some other language?

  python is free, batteries included, very well designed,  painless to read and easy to use. There is nothing else like it. Give it a go. Once with a python always with python. 
  
2. Isn't python slow?

  True, some times python can be slow if you are using for example multiple nested for loops. In that case we use cython which takes up to C speed.
  
3. What numerical libraries do you use in  python?

  The best ever designed numerical library numpy.   
  
2. Which python console do your recommend?

  ipython

3. What do you use for visualization?

  We use fosvtk(fvtk) this depends on python-vtk 
  from dipy.viz import fvtk

4. What about interactive visualization?

  There is already interaction on the fvtk module but we have started a new project 
  only for visualization which we plan to integrate in dipy in the near future for more information 
  have a look at http://fos.me

5. Which file formats do you support?
  
  Dicom (Siemens), Nifti (.nii) , Trackvis (.trk), Dipy (.dpy), Numpy (.npy, ,npz), text and all other formats supported by nibabel,nifti and pydicom.
  
6. What is Dpy?

  Dpy is and hdf5 file format which we use in dipy to store tractography and other information. This allows us to store huge tractographies and load different parts of the datasets directly from the disk like it was in memory.

7. Which python editor should I use?

  Any text editor would do the job but we prefer the following Aptana, Emacs, Vim and Eclipse (with PyDev).
  
8. I have problems reading my dicom files using nibabel, what should I do?

  Use Chris Roden's dcm2nii to transform them to nifti files.  
  http://www.cabiatl.com/mricro/mricron/dcm2nii.html
  Or you can make your own reader using pydicom   
  http://code.google.com/p/pydicom/
  and then use nibabel to store the data as niftis.
  
  
